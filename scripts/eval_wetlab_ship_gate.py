"""Offline "ship gate" from the SingleBicycle/tacwam-wetlab-tasks dataset card, §6.

Not an accuracy eval -- a pre-flight noise-vs-signal sanity check the card
explicitly recommends before ever touching the real robot: run the trained
policy open-loop on real observations, and compare the PREDICTED action
chunk's own row-to-row |dxyz| step size and direction-reversal rate against the
demonstrations' own measured statistics (mean 3.77mm, p50 2.83, p95 10.86mm,
reversal 7.6%, measured at 30Hz inside the clutch window). A checkpoint whose
predicted step is ~2x the demonstrations' or whose reversal rate is 40-50% is
noise-dominated -- that exact signature cost tacWAM seven live robot trials on
2026-09-01. "This gate takes five minutes on one GPU" (dataset card).

The card's own instructions say "run ... on at least two training episodes";
this evaluates canonical_wetlab_v2_holdout by default instead -- the block-level
held-out set this project uses for checkpoint-selection/deployment gating (see
ROBOT_POSTTRAIN_OPEN_ISSUES.md #3.8/#5.1.8), for the same check on genuinely
unseen episodes rather than ones the model was trained on.

Reference numbers are for our 30Hz native horizon (N0-VTLA's own contract, see
ROBOT_POSTTRAIN_OPEN_ISSUES.md #6) -- do NOT compare against tacWAM's own
10Hz-cadence numbers (9.38mm / 5.1%), which are a different temporal stride.

Usage:
  python scripts/eval_wetlab_ship_gate.py \
    --config vtla_tactile_posttrain \
    --checkpoint /DATA2/qianqian/N0-VTLA/checkpoints/vtla_tactile_posttrain/wetlab_v2_train_full_run1/15000 \
    --dataset-root /DATA2/qianqian/n0vtla_robot_audit/canonical_wetlab_v2_holdout \
    --output /DATA2/qianqian/n0vtla_robot_audit/ship_gate_15000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Measured on the demonstrations themselves, 30Hz, inside the clutch window
# (dataset card §6). Only valid for our 30Hz-horizon contract -- see module docstring.
REF_MEAN_STEP_MM = 3.77
REF_P50_STEP_MM = 2.83
REF_P95_STEP_MM = 10.86
REF_REVERSAL_RATE = 0.076
NOISE_DOMINATED_STEP_RATIO = 2.0  # card: "~2x the demonstrations'"
NOISE_DOMINATED_REVERSAL_LOW = 0.40  # card: "40-50%"
NOISE_DOMINATED_REVERSAL_HIGH = 0.50

XYZ_SLICE = slice(10, 13)  # right_eef_mm_columns6d_10_19_revo2_raw6_20_26_v1 layout


def sample_frame_indices(episodes: list[dict], stride: int, margin: int) -> list[tuple[int, int]]:
    """(episode_index, frame_index) pairs, skipping `margin` frames at each
    episode's start/end (avoids feeding the model a frame right at a boundary
    where there's little runway left to judge a predicted trajectory)."""
    pairs = []
    for ep in episodes:
        length = ep["length"]
        for frame in range(margin, max(margin + 1, length - margin), stride):
            pairs.append((ep["episode_index"], frame))
    return pairs


def step_stats(xyz: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Row-to-row step sizes and direction-reversal count, matching the dataset
    card's own reproduction method (its §0 script)."""
    d = np.diff(xyz, axis=0)
    steps = np.linalg.norm(d, axis=1)
    u, v = d[:-1], d[1:]
    n = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    ok = n > 1e-6
    reversals = int(((u[ok] * v[ok]).sum(1) / n[ok] < 0).sum())
    return steps, reversals, int(ok.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="vtla_tactile_posttrain")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=20, help="Frames between sampled starting points per episode.")
    parser.add_argument("--margin", type=int, default=10, help="Frames to skip at each episode's start/end.")
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="e.g. cuda:0. Defaults to cuda if available, else cpu.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from n0vtla.policies import policy_config
    from n0vtla.training import config as _config
    from n0vtla.training.data_loader import LocalLeRobotV3Dataset

    policy = policy_config.create_trained_policy(
        _config.get_config(args.config), args.checkpoint, pytorch_device=args.device,
    )
    dataset = LocalLeRobotV3Dataset(args.dataset_root)
    episodes = [json.loads(l) for l in (args.dataset_root / "meta" / "episodes.jsonl").open()]

    pairs = sample_frame_indices(episodes, args.stride, args.margin)
    rng = np.random.default_rng(args.seed)
    if len(pairs) > args.max_samples:
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), size=args.max_samples, replace=False))]

    offsets = np.concatenate([[0], np.cumsum([ep["length"] for ep in episodes])[:-1]])
    all_steps, total_reversals, total_pairs, per_sample = [], 0, 0, []
    for episode_index, frame_index in pairs:
        global_index = int(offsets[episode_index]) + frame_index
        item = dataset[global_index]
        # A live serving client never has future ground-truth actions to send; drop it here too
        # so DeltaActions' "actions" not in data no-op path matches real inference exactly,
        # instead of choking on a single-frame (no horizon axis) ground-truth action.
        item.pop("action", None)
        out = policy.infer(item)
        actions = np.asarray(out["actions"])  # (horizon, action_dim), absolute physical units
        xyz = actions[:, XYZ_SLICE]
        steps, reversals, n_pairs = step_stats(xyz)
        all_steps.append(steps)
        total_reversals += reversals
        total_pairs += n_pairs
        per_sample.append(dict(episode_index=episode_index, frame_index=frame_index,
                                mean_step_mm=float(steps.mean()), max_step_mm=float(steps.max())))

    steps = np.concatenate(all_steps)
    mean_step = float(steps.mean())
    reversal_rate = total_reversals / max(total_pairs, 1)
    verdict = "NOISE_DOMINATED" if (mean_step > NOISE_DOMINATED_STEP_RATIO * REF_MEAN_STEP_MM or
                                     NOISE_DOMINATED_REVERSAL_LOW <= reversal_rate <= NOISE_DOMINATED_REVERSAL_HIGH or
                                     reversal_rate > NOISE_DOMINATED_REVERSAL_HIGH) else "ok_so_far"

    report = dict(
        checkpoint=str(args.checkpoint), dataset_root=str(args.dataset_root),
        n_samples=len(pairs), n_step_pairs=len(steps),
        predicted=dict(mean_step_mm=mean_step, median_step_mm=float(np.median(steps)),
                        p95_step_mm=float(np.percentile(steps, 95)), max_step_mm=float(steps.max()),
                        reversal_rate=reversal_rate),
        reference_30hz_demonstrations=dict(mean_step_mm=REF_MEAN_STEP_MM, median_step_mm=REF_P50_STEP_MM,
                                            p95_step_mm=REF_P95_STEP_MM, reversal_rate=REF_REVERSAL_RATE,
                                            source="SingleBicycle/tacwam-wetlab-tasks dataset card #6"),
        verdict=verdict,
        verdict_note=("NOT a pass/fail certification -- a checkpoint clearing this gate has only been "
                      "shown to predict demonstration-scale, non-oscillating motion, not a successful "
                      "task. A checkpoint failing it should not go anywhere near the real robot."),
        per_sample=per_sample,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"mean_step_mm={mean_step:.2f} (ref {REF_MEAN_STEP_MM}) | "
          f"p95_step_mm={report['predicted']['p95_step_mm']:.2f} (ref {REF_P95_STEP_MM}) | "
          f"reversal_rate={reversal_rate:.3f} (ref {REF_REVERSAL_RATE}) | verdict={verdict}")
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
