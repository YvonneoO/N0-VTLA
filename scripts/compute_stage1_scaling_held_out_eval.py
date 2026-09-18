#!/usr/bin/env python
"""Held-out Stage-1 (online) eval for the data-scaling study, one checkpoint at a time.

WHY THIS EXISTS (not just reading each run's own training loss off wandb): the
20/40/60/80pct scaling runs all train for the SAME number of steps (14000) on
DIFFERENT-sized episode pools drawn from ONE shuffle (see build_itw_scaling_splits.py).
A smaller pool gets recycled far more times over those 14000 steps than a larger one,
so a smaller-fraction run's own training loss is biased low by repetition/memorization
of its own pool -- it is NOT a fair proxy for "how well did this checkpoint learn to
predict future tactile state in general". The only fair comparison is every
checkpoint's loss on the SAME episodes none of them ever trained on: the "held_out"
key build_itw_scaling_splits.py writes (the complement of the largest built fraction,
so it is by construction absent from every 20/40/60/80pct split).

This script scores exactly one checkpoint per invocation (run it once per data
fraction, pointing --checkpoint-dir at that fraction's exp dir -- the LATEST step
found there is loaded, matching load_stage1_checkpoint's own resume behavior) and
writes one small JSON report. Collect the reports across fractions and plot
data_fraction (x, log scale) vs. mean_stage1_nce or mean_stage1_total (y) for the
scaling-law curve.

Reuses train_stage1_online.py's own model-construction, checkpoint-loading, and loss
computation UNCHANGED (imported, not copied) so the held-out number is computed by
the exact same code path as training, just in eval mode with one full held-out pass
instead of a shuffled infinite training stream.

Usage:
    python scripts/compute_stage1_scaling_held_out_eval.py \
        --checkpoint-dir /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA_scaling_extract/checkpoints/vtla_stage1_predictor_pretrain/stage1_online_20pct \
        --checkpoint-label 20pct --data-fraction 0.2 \
        --episode-list-json /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/itw_scaling_splits.json \
        --raw-root /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/raw \
        --normalization /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/normalization_pad30_per_task_scale.json \
        --base-checkpoint /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla/n0-vtla-base \
        --output /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/stage1_scaling_held_out/20pct.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-config", default="vtla_stage1_predictor_pretrain")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                         help="Exp-level checkpoint dir (e.g. .../stage1_online_20pct); "
                              "the LATEST step subdir found there is loaded.")
    parser.add_argument("--checkpoint-label", required=True, help="e.g. '20pct' -- goes in the output JSON.")
    parser.add_argument("--data-fraction", type=float, required=True, help="e.g. 0.2 -- the x-axis value.")
    parser.add_argument("--episode-list-json", type=Path, required=True,
                         help="Manifest from build_itw_scaling_splits.py (must have a 'held_out' split).")
    parser.add_argument("--episode-list-key", default="held_out")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True,
                         help="n0-vtla-base dir or model.safetensors path (VTLA_PRETRAINED_CHECKPOINT).")
    parser.add_argument("--future-frame-offset", type=int, default=50)
    parser.add_argument("--default-prompt", default="Perform the task.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")

    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_pytorch import setup_ddp  # noqa: E402
    from train_stage1_predictor import (  # noqa: E402
        STAGE1_TRAINABLE_PREFIXES,
        _Stage1Wrapper,
        load_stage1_checkpoint,
        load_stage1_policy_weights,
    )

    import n0vtla.training.itw_online_dataset as _online  # noqa: E402
    from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy  # noqa: E402
    from n0vtla.training.config import get_config  # noqa: E402

    config = get_config(args.stage1_config)
    model_cfg = config.model
    if not getattr(model_cfg, "stage1_pretrain_enabled", False):
        raise ValueError(f"{args.stage1_config!r} does not have stage1_pretrain_enabled=True")
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    manifest = json.loads(args.episode_list_json.read_text())
    try:
        rel_paths = manifest["splits"][args.episode_list_key]
    except KeyError as e:
        raise ValueError(f"key {args.episode_list_key!r} not found in {args.episode_list_json}'s splits "
                          f"(available: {sorted(manifest.get('splits', {}))})") from e
    episode_dirs = [args.raw_root / rel for rel in rel_paths]
    missing = [str(p) for p in episode_dirs if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} held-out episode dirs missing under "
                                 f"{args.raw_root}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"Held-out set ({args.episode_list_key}): {len(episode_dirs)} episodes")

    device = torch.device(args.device)
    use_ddp, _local_rank, _device = setup_ddp()
    assert not use_ddp, "single-GPU eval only -- do not launch this via torchrun"

    loader = _online.create_stage1_data_loader(
        config, episode_dirs, str(args.normalization),
        future_frame_offset=args.future_frame_offset, default_prompt=args.default_prompt,
    )
    torch_loader = loader._data_loader.torch_loader
    num_batches = len(torch_loader)
    if num_batches == 0:
        raise ValueError("No complete batch available from the held-out set at this batch size")
    print(f"One held-out pass: {num_batches} batches of batch_size={config.batch_size}")

    policy = N0VTLAPolicy(model_cfg).to(device)
    base_weights = args.base_checkpoint / "model.safetensors" if args.base_checkpoint.is_dir() else args.base_checkpoint
    missing, unexpected = load_stage1_policy_weights(policy, base_weights, device, strict=False)
    allowed_missing = ("tactile_encoder.", "tactile_predictor.", "tactile_recon_head.", "z_proj.", "z_gate")
    missing_base = [name for name in missing if not name.startswith(allowed_missing)]
    if missing_base or unexpected:
        raise ValueError(f"Incompatible base checkpoint: missing_base={missing_base}, unexpected={unexpected}")

    for name, p in policy.named_parameters():
        p.requires_grad = name.startswith(STAGE1_TRAINABLE_PREFIXES)

    model = _Stage1Wrapper(policy)
    # load_stage1_checkpoint unconditionally loads optimizer state onto whatever
    # optimizer it's handed -- never stepped here, just satisfies that signature.
    dummy_optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-8)
    loaded_step = load_stage1_checkpoint(model, dummy_optim, args.checkpoint_dir, device)
    print(f"Loaded checkpoint step {loaded_step} from {args.checkpoint_dir}")

    model.eval()
    infos = []
    with torch.no_grad():
        for observation, _actions in itertools.islice(loader, num_batches):
            import jax

            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            model(observation)
            raw_policy = model.policy
            info = dict(raw_policy._last_loss_parts)
            if info.get("stage1_valid_count", 0) == 0:
                continue
            infos.append(info)

    if not infos:
        raise RuntimeError("Every held-out batch had zero valid future-tactile targets")

    total_valid = sum(i["stage1_valid_count"] for i in infos)
    weighted_means = {
        k: float(sum(i[k] * i["stage1_valid_count"] for i in infos) / total_valid)
        for k in ("stage1_nce", "stage1_recon", "stage1_total")
    }
    report = {
        "checkpoint_label": args.checkpoint_label,
        "data_fraction": args.data_fraction,
        "checkpoint_step": loaded_step,
        "episode_list_key": args.episode_list_key,
        "num_held_out_episodes": len(episode_dirs),
        "num_batches_scored": len(infos),
        "num_batches_skipped_empty": num_batches - len(infos),
        "total_valid_samples": int(total_valid),
        "mean_stage1_valid_frac": float(np.mean([i["stage1_valid_frac"] for i in infos])),
        **{f"mean_{k}": v for k, v in weighted_means.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
