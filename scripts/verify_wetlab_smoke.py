"""Verify the actual reference-loader batch and diagnose (not certify) clock lag."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def motion_diagnostic(source):
    rows = [json.loads(l) for l in (source / "robot/arm.jsonl").open()]
    t = np.array([r["host_timestamp_ns"] for r in rows], np.float64) / 1e9
    xyz = np.array([r["tcp_pose"][:3] for r in rows])
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Robot host timestamps must be strictly increasing")
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt
    robot_t = (t[1:] + t[:-1]) / 2
    csv = np.genfromtxt(source / "wrist_right.csv", delimiter=",", names=True)
    key = next(k for k in ("timestamp_s", "epoch_s", "epoch", "timestamp") if k in csv.dtype.names)
    video_t = csv[key]
    ids = np.flatnonzero((video_t >= t[0] - 3.1) & (video_t <= t[-1] + 3.1))
    cap = cv2.VideoCapture(str(source / "wrist_right.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ids[0]))
    flow_t, motion, prev = [], [], None
    cv2.setNumThreads(1)
    for index in ids:
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Unable to decode diagnostic frame")
        gray = cv2.cvtColor(cv2.resize(frame, (256,144)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev,gray,None,.5,3,15,3,5,1.2,0)
            # Global wrist-camera motion is only a proxy: hand/object movement also contributes.
            magnitude = np.linalg.norm(flow, axis=-1)
            motion.append(float(np.median(magnitude)))
            flow_t.append((video_t[index] + video_t[index-1])/2)
        prev = gray
    cap.release()
    motion = gaussian_filter1d(np.array(motion),2)
    speed = gaussian_filter1d(speed,2)
    lags = np.arange(-90,91)/30
    result = {}
    for name, mask in (("whole",np.ones(len(speed),bool)),
                       ("first_half",robot_t < np.median(robot_t)),
                       ("second_half",robot_t >= np.median(robot_t))):
        scores = []
        for lag in lags:
            x = np.interp(robot_t[mask]+lag,flow_t,motion)
            y = speed[mask]
            scores.append(float(np.corrcoef(x,y)[0,1]) if x.std() > 1e-10 and y.std() > 1e-10 else 0.)
        best = int(np.argmax(scores))
        result[name] = dict(best_lag_ms=float(lags[best]*1000),best_correlation=scores[best],
                            zero_lag_correlation=scores[90],scores=scores)
    return dict(method="median wrist optical-flow magnitude vs measured TCP speed",
                sign="add lag to Linux robot timestamp to query Windows video timestamp",
                certified=False, offset_applied=False, lag_grid_ms=(lags*1000).tolist(),results=result,
                limitation="Motion proxy is not an independent synchronization marker; no offset correction authorized by this diagnostic.")


def batch_check():
    from n0vtla.training import config, data_loader
    from n0vtla import transforms
    from compute_canonical_norm import action_horizon_sequence, apply_delta, ROBOT_DELTA_MASKS
    import pyarrow.parquet as pq

    cfg = config.get_config("vtla_tactile_posttrain")
    loader = data_loader.create_data_loader(cfg, framework="pytorch", shuffle=False)
    obs, act = next(iter(loader))
    assert tuple(act.shape) == (64,50,32) and bool(act.isfinite().all())
    assert bool(obs.state.isfinite().all())
    masks = {k:dict(true=int(v.sum()),total=len(v)) for k,v in obs.image_masks.items()}
    assert masks["base_0_rgb"]["true"] == 64
    assert masks["right_wrist_0_rgb"]["true"] == 64
    assert masks["left_wrist_0_rgb"]["true"] == 0
    mask = transforms.make_bool_mask(9,-1,9,-1,-12)
    error = 0.
    for p in (Path(cfg.data.repo_id)/"data").rglob("*.parquet"):
        tab = pq.read_table(p)
        state = np.asarray(tab["observation.state"].to_pylist(),np.float32)
        action = np.asarray(tab["action"].to_pylist(),np.float32)
        horizon = action_horizon_sequence(action)
        expected = apply_delta(horizon,state,ROBOT_DELTA_MASKS["aloha"])
        for i in range(len(state)):
            delta = transforms.DeltaActions(mask)({"state":state[i],"actions":horizon[i].copy()})
            np.testing.assert_allclose(delta["actions"],expected[i],atol=1e-6)
            absolute = transforms.AbsoluteActions(mask)(delta)["actions"]
            error = max(error,float(np.max(np.abs(absolute-horizon[i]))))
    return dict(action_shape=list(act.shape),state_shape=list(obs.state.shape),finite=True,
                action_range=[float(act.min()),float(act.max())],image_masks=masks,
                delta_stats_match=True,delta_inverse_max_error=error,
                config=cfg.name,batch_size=cfg.batch_size,horizon=cfg.model.action_horizon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["batch","motion"])
    parser.add_argument("--source",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = batch_check() if args.mode == "batch" else motion_diagnostic(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
