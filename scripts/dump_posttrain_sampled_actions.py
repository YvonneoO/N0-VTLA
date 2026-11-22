#!/usr/bin/env python
"""Dump the SAMPLED action chunk of ONE checkpoint for every Nth row of a dataset, in physical units, together with the demonstrated chunk and a
per-row tactile activity flag. Run it once per tactile variant of the SAME episodes (real / predicted / all-zero glove) with the same checkpoint and
seed: the noise draw is seeded per row (seed*100000 + row*10 + r), so the three dumps differ only through the tactile input. Compare them with
scripts/compare_tactile_action_sensitivity.py.

Same conventions as compute_posttrain_open_loop_error.py (physical units: xyz mm, rot6d dimensionless, hand motor counts).
  python scripts/dump_posttrain_sampled_actions.py --checkpoint <dir> --dataset-root <canonical dataset> --asset-id <train asset> --output out.npz
Saved arrays: rows (scored dataset rows), episode (episode index per row), pred (n, H, 32) mean of --samples draws, gt (n, H, 32),
tactile_act (n,) = max |tactile current - tactile baseline| over the real tactile views (images in [-1,1]; 0.067 = an input of x=0.3 in policy units).
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
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--asset-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--max-rows", type=int, default=0)
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

    import scripts.train_n0vtla  # noqa: F401  (patches PI0Pytorch -> N0VTLAPolicy)
    import n0vtla.models.pi0_config
    import n0vtla.models_pytorch.pi0_pytorch
    from n0vtla.training import config as _config
    from n0vtla.training import data_loader as _data
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compute_posttrain_open_loop_error import load_action_stats, unnormalize

    episodes = [json.loads(l) for l in (args.dataset_root / "meta" / "episodes.jsonl").open()]
    lengths = [ep["length"] for ep in episodes]
    total_rows = sum(lengths)
    boundaries = np.concatenate([[0], np.cumsum(lengths)])
    config = _config.get_config(args.config)
    object.__setattr__(config, "batch_size", 1)
    q01, q99, _ = load_action_stats(Path(config.assets_base_dir) / config.name / args.asset_id)
    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    device = torch.device(args.device)
    print(f"=== loading {args.checkpoint} ===", flush=True)
    model = n0vtla.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    missing, unexpected = safetensors.torch.load_model(model, str(args.checkpoint / "model.safetensors"), device=str(device))
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch -- missing={missing}, unexpected={unexpected}")
    model.eval()
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    if len(loader._data_loader.torch_loader.dataset) != total_rows:
        raise RuntimeError("dataset length != episodes.jsonl total")
    print(f"=== rows={total_rows} episodes={len(episodes)} stride={args.stride} samples={args.samples} ===", flush=True)

    rows, eps, preds, gts, acts = [], [], [], [], []
    first = True
    for row, (observation, actions) in enumerate(loader):
        if row >= total_rows or (args.max_rows and row >= args.max_rows):
            break
        if row % args.stride:
            continue
        observation = jax.tree.map(lambda x: x.to(device), observation)
        act = 0.0
        for k, img in observation.images.items():
            if k.endswith("_tactile") and (k + ".baseline") in observation.images:
                m = observation.image_masks.get(k) if getattr(observation, "image_masks", None) is not None else None
                if m is None or bool(m.reshape(-1)[0]):
                    act = max(act, float((img.float() - observation.images[k + ".baseline"].float()).abs().max()))
        if first:
            print("tactile keys:", [k for k in observation.images if "tactile" in k], flush=True)
            first = False
        samples = []
        for r in range(args.samples):
            torch.manual_seed(args.seed * 100_000 + row * 10 + r)
            with torch.no_grad():
                out = model.sample_actions(device, observation, num_steps=args.num_steps)
            samples.append(unnormalize(out[0].float().cpu().numpy().astype(np.float64), q01, q99))
        rows.append(row)
        eps.append(int(np.searchsorted(boundaries, row, side="right") - 1))
        preds.append(np.mean(samples, axis=0).astype(np.float32))
        gts.append(unnormalize(actions[0].float().cpu().numpy().astype(np.float64), q01, q99).astype(np.float32))
        acts.append(act)
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} rows (row {row}/{total_rows})", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, rows=np.array(rows), episode=np.array(eps), pred=np.stack(preds), gt=np.stack(gts), tactile_act=np.array(acts, np.float32))
    print(f"Wrote: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
