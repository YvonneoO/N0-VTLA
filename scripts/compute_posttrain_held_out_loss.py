#!/usr/bin/env python
"""Held-out flow-matching action loss of ONE post-train checkpoint on its held-out split.

The same quantity train_pytorch.py optimizes (model(observation, actions) -> per-sample loss),
computed with no gradient on episodes the run never trained on (smoke_test_v2 dev, task1 val),
using the run's own training-set normalization (--asset-id). Per-sample loss depends on a random
(noise, time) draw, so --repeats draws are averaged per sample, seeded by row index so every
checkpoint sees the SAME draws (paired comparison across checkpoints). Model construction, loader
factory and loss follow compute_held_out_action_loss.py (3b), minus its tactile-activity
correlation part (needs raw-source data we do not have on VISION).

Run from the repo root with the n0vtla env:
  python scripts/compute_posttrain_held_out_loss.py \
    --checkpoint checkpoints/vtla_tactile_posttrain/task1_posttrain_s1_20pct_10k/10000 \
    --checkpoint-label task1_20pct --dataset-root <.../canonical_wetlab_task1_val> \
    --asset-id task1_tuberack_train --output out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="vtla_tactile_posttrain")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-label", required=True)
    p.add_argument("--dataset-root", type=Path, required=True, help="canonical held-out dataset dir")
    p.add_argument("--asset-id", required=True, help="asset id of the TRAIN set (its norm_stats.json is used)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=5, help="score every Nth dataset row")
    p.add_argument("--repeats", type=int, default=4, help="(noise,time) draws averaged per scored row")
    p.add_argument("--max-rows", type=int, default=0, help="smoke: stop after this many rows (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    os.environ["VTLA_DATASET_PATH"] = str(args.dataset_root)
    os.environ["VTLA_ASSET_ID"] = args.asset_id
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import jax
    import safetensors.torch
    import torch

    import scripts.train_n0vtla  # noqa: F401  (side effect: patches PI0Pytorch -> N0VTLAPolicy)
    import n0vtla.models.model
    import n0vtla.models.pi0_config
    import n0vtla.models_pytorch.pi0_pytorch
    from n0vtla.training import config as _config
    from n0vtla.training import data_loader as _data

    episodes = [json.loads(l) for l in (args.dataset_root / "meta" / "episodes.jsonl").open()]
    lengths = [ep["length"] for ep in episodes]
    total_rows = sum(lengths)
    boundaries = np.concatenate([[0], np.cumsum(lengths)])

    device = torch.device(args.device)
    config = _config.get_config(args.config)
    object.__setattr__(config, "batch_size", 1)
    model_cfg = config.model
    if not isinstance(model_cfg, n0vtla.models.pi0_config.Pi0Config):
        raise TypeError(f"unexpected model config type {type(model_cfg)}")
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    print(f"=== loading {args.checkpoint} ===", flush=True)
    model = n0vtla.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    missing, unexpected = safetensors.torch.load_model(model, str(args.checkpoint / "model.safetensors"), device=str(device))
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch -- missing={missing}, unexpected={unexpected}")
    model.eval()

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    n_ds = len(loader._data_loader.torch_loader.dataset)
    if n_ds != total_rows:
        raise RuntimeError(f"dataset length {n_ds} != episodes.jsonl total {total_rows}")
    print(f"=== held-out rows={total_rows} episodes={len(episodes)} stride={args.stride} repeats={args.repeats} ===", flush=True)

    per_episode: dict[int, list[float]] = {i: [] for i in range(len(episodes))}
    n_scored = 0
    for row, (observation, actions) in enumerate(loader):
        if row >= total_rows or (args.max_rows and row >= args.max_rows):
            break
        if row % args.stride:
            continue
        ep = int(np.searchsorted(boundaries, row, side="right") - 1)
        observation = jax.tree.map(lambda x: x.to(device), observation)
        actions_t = actions.to(torch.float32).to(device)
        draws = []
        for r in range(args.repeats):
            torch.manual_seed(args.seed * 100_000 + row * 10 + r)
            with torch.no_grad():
                losses = model(observation, actions_t)
            draws.append(float(losses.reshape(losses.shape[0], -1).mean(dim=1).item()))
        per_episode[ep].append(float(np.mean(draws)))
        n_scored += 1
        if n_scored % 100 == 0:
            print(f"  scored {n_scored} rows (row {row}/{total_rows})", flush=True)

    ep_rows = [dict(episode_index=i, length=lengths[i], n_samples=len(v),
                    mean_action_loss=float(np.mean(v)) if v else float("nan")) for i, v in per_episode.items()]
    ep_means = np.array([r["mean_action_loss"] for r in ep_rows if r["n_samples"]])
    all_losses = np.array([x for v in per_episode.values() for x in v])
    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_label=args.checkpoint_label,
        dataset_root=str(args.dataset_root), asset_id=args.asset_id,
        stride=args.stride, repeats=args.repeats, seed=args.seed, n_episodes=len(episodes),
        n_scored_rows=n_scored, mean_action_loss_micro=float(all_losses.mean()),
        mean_action_loss_macro=float(ep_means.mean()),
        episode_se=float(ep_means.std(ddof=1) / np.sqrt(len(ep_means))) if len(ep_means) > 1 else float("nan"),
        per_episode=ep_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"{args.checkpoint_label}: micro={report['mean_action_loss_micro']:.5f} macro={report['mean_action_loss_macro']:.5f} "
          f"±{report['episode_se']:.5f} ({n_scored} rows)", flush=True)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
