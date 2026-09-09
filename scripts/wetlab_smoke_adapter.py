"""Single-recording robot plumbing smoke, NOT a deployment-certified converter.

Keep the reference 32-D model, delta transform, 30 Hz grid and 50-step horizon.
An explicit flag acknowledges the dataset's unverified cross-host clock offset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from itw_pressure import PAD_IDS, SCHEMA, write_aligned_rgb, write_pressure_video
from itw_tactile_smoke_adapter import (
    _fixed_size_list_array, _hf_schema_metadata, _info_json, _quantile_stats, _write_jsonl,
)

TASK = "Pick the cap and place it in the white container."
LAYOUT = "right_eef_mm_columns6d_10_19_revo2_raw6_20_26_v1"


def causal_indices(source_ns, query_ns, max_age_ns):
    source = np.asarray(source_ns, np.int64)
    query = np.asarray(query_ns, np.int64)
    if not len(source) or np.any(np.diff(source) < 0):
        raise ValueError("Source timestamps must be nonempty and nondecreasing")
    index = np.searchsorted(source, query, side="right") - 1
    age = query - source[np.maximum(index, 0)]
    valid = (index >= 0) & (age >= 0) & (age <= max_age_ns)
    return np.maximum(index, 0), age, valid


def contiguous_runs(valid, min_length=50):
    ids = np.flatnonzero(valid)
    return [g for g in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
            if len(g) >= min_length]


def pack_commands(arm_aa, hand):
    arm, hand = np.asarray(arm_aa, np.float64), np.asarray(hand, np.float64)
    if arm.ndim != 2 or arm.shape[1] != 6 or hand.shape != arm.shape:
        raise ValueError("Expected matching (N,6) arm and hand targets")
    if not np.isfinite(arm).all() or not np.isfinite(hand).all():
        raise ValueError("Nonfinite command")
    if np.any((hand < 0) | (hand > 1000)):
        raise ValueError("Motor command outside 0..1000")
    rot = Rotation.from_rotvec(np.deg2rad(arm[:, 3:])).as_matrix()
    out = np.zeros((len(arm), 32), np.float32)
    out[:, 10:13] = arm[:, :3]
    out[:, 13:19] = rot[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    out[:, 20:26] = hand
    return out


def decode_commands(canonical):
    from n0vtla.policies.rotation_utils import rot6d_to_matrix

    x = np.asarray(canonical)
    rot = rot6d_to_matrix(x[:, 13:19].astype(np.float64))
    arm = np.concatenate([x[:, 10:13], np.rad2deg(Rotation.from_matrix(rot).as_rotvec())], axis=1)
    return arm, x[:, 20:26].copy()


def fit_smoke_pressure(archive, eligible):
    # Same P5/P99.9, four-frame sampling and positive-median floor as itw_pressure.
    # This one-recording TRAIN smoke asset is not full-dataset normalization.
    rng = np.random.default_rng(42)
    chosen = eligible[rng.integers(0, len(eligible), size=min(4, len(eligible)))]
    baseline, scales, threshold = [], [], []
    for pad in PAD_IDS:
        a = np.asarray(archive[f"tactile_{pad}"][chosen], np.float64).reshape(-1)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite tactile fitting values")
        lo, hi = np.percentile(a, [5, 99.9])
        scale = max(float(hi - lo), 1e-6)
        med = np.median(a)
        baseline.append(float(lo))
        scales.append(scale)
        threshold.append(max(float(med + 4 * np.median(np.abs(a - med))), float(lo) + .1 * scale))
    positive = [x for x in scales if x > 1e-5]
    floor = max(float(np.median(positive)) * .05 if positive else 1e-3, 1e-5)
    scales = [max(x, floor) for x in scales]
    threshold = [max((t - b) / s, .1) for t, b, s in zip(threshold, baseline, scales)]
    return dict(schema=SCHEMA, fit_split="train", smoke_only=True,
                normal_baseline=[0.] * 15 + baseline, normal_scale=[1.] * 15 + scales,
                contact_threshold=[.1] * 15 + threshold, absent_left_identity=True,
                fit_npz_indices=chosen.tolist(), seed=42, samples_per_recording=4)


def convert(source, output, allow_unverified_sync):
    if not allow_unverified_sync:
        raise ValueError("Physical clock alignment is unverified; require --allow-unverified-sync-smoke")
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "robot/manifest.json").read_text())
    if manifest["task"]["name"] != "cap_to_tray" or manifest["trial"]["label"] != "success":
        raise ValueError("Expected successful cap_to_tray trial")
    command_rows = [json.loads(l) for l in (source / "robot/arm_cmd.jsonl").open()]
    commands = [r for r in command_rows if r.get("accepted") is True and r.get("rc") == 0 and r.get("clutch")]
    hands = [json.loads(l) for l in (source / "robot/hand.jsonl").open()]
    hands = [r for r in hands if r.get("t_cmd_ns") is not None and r.get("target") is not None]
    # The delivered "right_hand_data.npz" is a dead channel on this rig (single right
    # arm/hand; robot/manifest.json declares tactile_left absent). The live 880-taxel
    # signal is actually stored under the "left"-named file. Read from it, but keep
    # every downstream hand="right" label: physically this is still the right glove,
    # only the delivered filename is swapped. See ROBOT_POSTTRAIN_OPEN_ISSUES.md §3.1.
    tactile_source = source / "left_hand_data.npz"
    with h5py.File(source / "episode_30hz.h5", "r") as f, np.load(tactile_source, allow_pickle=False) as z:
        pad_std = max(float(np.std(np.asarray(z[f"tactile_{pad}"], np.float64))) for pad in PAD_IDS)
        if pad_std < 1e-4:
            raise ValueError(
                f"{tactile_source.name} looks dead (max per-pad std {pad_std:.2e}); "
                "the live/dead file assignment may have changed upstream -- verify "
                "before trusting this conversion"
            )
        t = f["time/timestamp_ns"][:]
        ai, aa, av = causal_indices([r["host_timestamp_ns"] for r in commands], t, 150_000_000)
        # A target must already have been issued AND its logged row available.
        hi, ha, hv = causal_indices([max(r["t_cmd_ns"], r["host_timestamp_ns"]) for r in hands], t, 150_000_000)
        hand_command_age = t - np.array([r["t_cmd_ns"] for r in hands], np.int64)[hi]
        hv &= (hand_command_age >= 0) & (hand_command_age <= 150_000_000)
        ti, ta, tv = causal_indices(np.rint(z["timestamps"] * 1e9), t, 50_000_000)
        valid = av & hv & tv & f["action/right/clutch"][:].astype(bool)
        video_indices = {}
        camera_gaps = {}
        for name in ("rgb_head", "wrist_right"):
            # Delivered H5 mapping is validated against original CSV rows.
            csv = np.genfromtxt(source / f"{name}.csv", delimiter=",", names=True)
            ids = f[f"obs/video/{name}/src_idx"][:]
            epoch_key = next(k for k in ("timestamp_s", "epoch_s", "epoch", "timestamp") if k in csv.dtype.names)
            csv_t = np.rint(csv[epoch_key] * 1e9).astype(np.int64)
            if np.any(ids < 0) or np.any(ids >= len(csv_t)):
                raise ValueError("Video index outside timestamp table")
            gap = t - csv_t[ids]
            if not np.allclose(csv_t[ids] / 1e9, f[f"obs/video/{name}/epoch"][:], rtol=0, atol=2e-6):
                raise ValueError("H5/CSV timestamp mapping mismatch")
            valid &= f[f"valid/{name}"][:].astype(bool) & (gap >= 0) & (gap <= 50_000_000)
            video_indices[name] = ids
            camera_gaps[name] = dict(min_ms=float(gap.min()/1e6), max_ms=float(gap.max()/1e6))
        valid &= (t >= manifest["trial"]["engaged_host_ns"]) & (t < manifest["trial"]["released_host_ns"])
        runs = contiguous_runs(valid)
        if not runs:
            raise ValueError("No continuous causal 50-frame segment")
        chosen = np.concatenate(runs)
        norm = fit_smoke_pressure(z, np.unique(ti[chosen]))
        norm["source_episode"] = source.name
        arm = np.array([r["target_aa"] for r in commands])[ai]
        hand = np.array([r["target"] for r in hands])[hi]
        actions = pack_commands(arm, hand)
        restored_arm, restored_hand = decode_commands(actions)
        rotation_error = (Rotation.from_rotvec(np.deg2rad(restored_arm[:, 3:])).inv() *
                          Rotation.from_rotvec(np.deg2rad(arm[:, 3:]))).magnitude()
        audit = dict(source_episode=source.name, layout=LAYOUT, smoke_only=True,
                     physical_sync_verified=False, cross_host_offset_ms_applied=0,
                     cross_host_evidence=manifest["clock_domains"]["windows_unix_utc"],
                     state_contract="last causally available issued arm and hand commands",
                     action_contract="absolute issued targets; loader subtracts current EEF state",
                     source_rows=len(t), selected_rows=len(chosen),
                     runs=[[int(g[0]), int(g[-1]), len(g)] for g in runs],
                     max_command_age_ms=150, max_sensor_age_ms=50, camera_gaps=camera_gaps,
                     max_arm_age_ms=float(aa[chosen].max()/1e6),
                     max_hand_command_age_ms=float(hand_command_age[chosen].max()/1e6),
                     max_tactile_age_ms=float(ta[chosen].max()/1e6),
                     position_roundtrip_error_mm=float(np.max(np.abs(restored_arm[:, :3]-arm[:, :3]))),
                     rotation_roundtrip_error_rad=float(rotation_error.max()),
                     hand_roundtrip_error=float(np.max(np.abs(restored_hand-hand))),
                     train_only_one_episode=True, end_padding="unchanged reference loader repeat-last",
                     tactile_units="raw NPZ; physical calibration not certified",
                     tactile_source_file=tactile_source.name,
                     tactile_source_note="delivered right_hand_data.npz is a dead channel on this rig; "
                     "live signal read from left_hand_data.npz instead, labeled as right-hand data "
                     "because that is the physically instrumented hand")
    meta = output / "meta"
    meta.mkdir(parents=True)
    (meta / "audit.json").write_text(json.dumps(audit, indent=2))
    (meta / "tactile_normalization.json").write_text(json.dumps(norm, indent=2))
    episodes, stats, total, data_bytes, video_bytes = [], [], 0, 0, 0
    streams = {"observation.image.third_view": "rgb_head", "observation.image.right_wrist_view": "wrist_right",
               "observation.image.right_wrist_right_tactile": "right"}
    for e, g in enumerate(runs):
        n = len(g)
        x = actions[g]
        mask = np.zeros_like(x, dtype=bool)
        mask[:, 10:19] = True
        mask[:, 20:26] = True
        columns = {"observation.state": x, "action": x.copy(), "action_mask": mask,
                   "timestamp": np.arange(n, dtype=np.float32)/30,
                   "frame_index": np.arange(n, dtype=np.int64), "episode_index": np.full(n,e,np.int64),
                   "index": np.arange(total,total+n,dtype=np.int64), "task_index": np.zeros(n,np.int64)}
        arrays = [_fixed_size_list_array(v,32,pa.bool_() if k == "action_mask" else pa.float32())
                  if v.ndim == 2 else pa.array(v) for k,v in columns.items()]
        table = pa.Table.from_arrays(arrays,names=list(columns)).replace_schema_metadata(_hf_schema_metadata())
        dst = output / "data/chunk-000" / f"episode_{e:06d}.parquet"
        dst.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(table,dst,compression="zstd")
        data_bytes += dst.stat().st_size
        for key,name in streams.items():
            dst = output / "videos/chunk-000" / key / f"episode_{e:06d}.mp4"
            if name == "right":
                video_bytes += write_pressure_video(tactile_source,dst,ti[g],norm,hand="right")
            else:
                video_bytes += write_aligned_rgb(source/f"{name}.mp4",dst,video_indices[name][g])
        episodes.append(dict(episode_index=e,tasks=[TASK],length=n))
        stats.append(dict(episode_index=e,stats={k:_quantile_stats(v) for k,v in columns.items()}))
        np.savez_compressed(meta / f"source_map_{e:06d}.npz",timestamp_ns=t[g],h5_rows=g,
                            arm_command_index=ai[g],hand_row_index=hi[g],tactile_index=ti[g],
                            **{k+"_index":v[g] for k,v in video_indices.items()})
        total += n
    _write_jsonl(meta/"episodes.jsonl",episodes)
    _write_jsonl(meta/"episodes_stats.jsonl",stats)
    _write_jsonl(meta/"tasks.jsonl",[dict(task_index=0,task=TASK)])
    info = _info_json(len(runs),total,data_bytes,video_bytes,list(streams))
    info["robot_type"] = "xarm6_revo2_smoke_only"
    (meta/"info.json").write_text(json.dumps(info,indent=2))
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--allow-unverified-sync-smoke",action="store_true")
    args = parser.parse_args()
    convert(args.source,args.output,args.allow_unverified_sync_smoke)
