"""Two diagnostics of a trained dual-arm (Task 4) checkpoint, through the real inference path (create_trained_policy):

A. SAMPLING SPREAD. The policy denoises fresh Gaussian noise on every call, so one observation yields a different 50-step
   chunk each time. For each sampled frame, K noise draws -> mean pairwise distance between the chunks (xyz mm, rotation
   deg, hand motor units), split into the first 10 steps (what a receding-horizon executor actually uses) and all 50, plus
   whether averaging the K chunks lands closer to the demonstrated chunk than a single sample does.
B. INPUT SENSITIVITY. With the SAME noise, replace one input and measure how much the chunk changes (same units):
   tactile -> "no contact" (current := baseline, per hand and both), tactile -> another frame's tactile; as scale references
   the wrist/head RGB -> black. A change much smaller than the sampling spread of A means the input barely steers the output.
   Reported separately for frames with strong tactile contact vs the rest. Training frames; a probe, not a benchmark.

  JAX_PLATFORMS=cpu python scripts/check_dual_policy_sensitivity.py --checkpoint <ckpt dir> \
      --dataset DATA/canonical_wetlab_task4_dual_train --asset-id task4_dual_pour_train --output out.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dual_policy_replay import ARMS, HANDS, geodesic_deg, to_hwc_uint8, to_tactile_stack  # noqa: E402


def chunk_dist(a: np.ndarray, b: np.ndarray, steps: slice) -> dict:
    """Mean over the given steps of the per-step distance between two (50, 32) absolute chunks."""
    out = {}
    for side, (lo, rlo, rhi) in ARMS.items():
        out[f"xyz_{side[0].upper()}"] = float(np.linalg.norm(a[steps, lo:lo + 3] - b[steps, lo:lo + 3], axis=1).mean())
        out[f"rot_{side[0].upper()}"] = float(geodesic_deg(a[steps, rlo:rhi], b[steps, rlo:rhi]).mean())
    for side, (x, y) in HANDS.items():
        out[f"hand_{side[0].upper()}"] = float(np.abs(a[steps, x:y] - b[steps, x:y]).mean())
    return out


def mean_dicts(ds: list[dict]) -> dict:
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]} if ds else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--num-frames", type=int, default=48)
    ap.add_argument("--k-samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    os.environ["VTLA_DATASET_PATH"] = str(args.dataset)
    os.environ["VTLA_ASSET_ID"] = args.asset_id

    from n0vtla.policies import policy_config as _pc
    from n0vtla.training import config as _config
    from n0vtla.training import data_loader as _data

    config = _config.get_config("vtla_tactile_posttrain")
    data_config = config.data.create(config.assets_dirs, config.model)
    policy = _pc.create_trained_policy(config, args.checkpoint)
    ds = _data.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    total = len(ds)
    rng = np.random.default_rng(args.seed)
    picks = np.sort(rng.choice(total, size=args.num_frames, replace=False))
    others = rng.choice(total, size=args.num_frames)          # frames whose tactile is swapped in

    def build(idx: int) -> tuple[dict, np.ndarray, np.ndarray, dict]:
        raw = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in ds[int(idx)].items()}
        obs = {"observation.state": np.asarray(raw["observation.state"], np.float32), "prompt": "Perform the task."}
        for view in ("third_view", "left_wrist_view", "right_wrist_view"):
            obs[f"observation.image.{view}"] = to_hwc_uint8(raw[f"observation.image.{view}"])
        contact = {}
        for hand, view in (("left", "left_wrist_left_tactile"), ("right", "right_wrist_right_tactile")):
            st = to_tactile_stack(raw[f"observation.image.{view}"])
            obs[f"observation.image.{view}"] = st
            contact[hand] = float(np.abs(st[1].astype(float) - st[0].astype(float)).mean())
        return obs, np.asarray(raw["action"], np.float32), ~np.asarray(raw["action_is_pad"]).astype(bool), contact

    def infer(obs: dict, noise: np.ndarray) -> np.ndarray:
        return np.asarray(policy.infer(obs, noise=noise)["actions"], np.float32)

    def variant(obs: dict, name: str, other: dict) -> dict:
        o = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        tl, tr = "observation.image.left_wrist_left_tactile", "observation.image.right_wrist_right_tactile"
        if name in ("tactile_off_both", "tactile_off_left"):
            o[tl] = np.stack([o[tl][0], o[tl][0]])
        if name in ("tactile_off_both", "tactile_off_right"):
            o[tr] = np.stack([o[tr][0], o[tr][0]])
        if name == "tactile_swapped":
            o[tl], o[tr] = other[tl], other[tr]
        if name == "black_left_wrist":
            o["observation.image.left_wrist_view"] = np.zeros_like(o["observation.image.left_wrist_view"])
        if name == "black_right_wrist":
            o["observation.image.right_wrist_view"] = np.zeros_like(o["observation.image.right_wrist_view"])
        if name == "black_head":
            o["observation.image.third_view"] = np.zeros_like(o["observation.image.third_view"])
        return o

    names = ["tactile_off_both", "tactile_off_left", "tactile_off_right", "tactile_swapped",
             "black_left_wrist", "black_right_wrist", "black_head"]
    first, allsteps = slice(0, 10), slice(0, 50)
    spread = {"first10": [], "all50": []}
    avg_gain = {"single": [], "mean_of_K": []}
    sens = {n: {"contact": [], "rest": []} for n in names}
    contact_scores = []
    records = []
    for i, idx in enumerate(picks):
        obs, gt, ok, contact = build(idx)
        other_obs, _, _, _ = build(others[i])
        score = max(contact.values())
        contact_scores.append(score)
        records.append((obs, gt, ok, other_obs, score))
    thr = float(np.percentile(contact_scores, 70))
    for i, (obs, gt, ok, other_obs, score) in enumerate(records):
        draws = [rng.standard_normal((50, 32)).astype(np.float32) for _ in range(args.k_samples)]
        chunks = [infer(obs, nz) for nz in draws]
        pairs = list(itertools.combinations(range(len(chunks)), 2))
        spread["first10"].append(mean_dicts([chunk_dist(chunks[a], chunks[b], first) for a, b in pairs]))
        spread["all50"].append(mean_dicts([chunk_dist(chunks[a], chunks[b], allsteps) for a, b in pairs]))
        gt_c = np.where(ok[:, None], gt, chunks[0])
        avg_gain["single"].append(mean_dicts([chunk_dist(c, gt_c, slice(0, 50)) for c in chunks]))
        avg_gain["mean_of_K"].append(chunk_dist(np.mean(chunks, axis=0), gt_c, slice(0, 50)))
        ref = chunks[0]
        group = "contact" if score >= thr else "rest"
        for n in names:
            sens[n][group].append(chunk_dist(infer(variant(obs, n, other_obs), draws[0]), ref, allsteps))
        print(f"[{i + 1}/{len(records)}] frame {picks[i]} contact_score {score:.2f} ({group})", flush=True)

    result = {
        "n_frames": len(records), "k_samples": args.k_samples, "contact_threshold_top30pct_mean_abs_pixel_diff": thr,
        "A_sampling_spread_mean_pairwise": {k: mean_dicts(v) for k, v in spread.items()},
        "A_single_sample_vs_mean_of_K_error_to_demo": {k: mean_dicts(v) for k, v in avg_gain.items()},
        "B_change_with_same_noise": {n: {g: mean_dicts(v) for g, v in groups.items() if v} for n, groups in sens.items()},
        "note": "Training frames, one checkpoint; units: xyz mm, rot deg, hand motor units (0-1000). B uses identical noise, "
                "so differences come only from the changed input; compare them with A's sampling spread.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
