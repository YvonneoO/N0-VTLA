#!/usr/bin/env python
"""Merges base + Stage-1 tactile delta + Stage-2 expert-alignment delta into ONE full
model.safetensors, so it can be handed directly to train_pytorch.py's existing (unmodified)
single-file checkpoint loader for post-train (vtla_tactile_posttrain) -- see this project's
Stage-2 design notes for why post-train's own loader is left untouched rather than teaching it
to compose three checkpoints itself.

NOT a training script -- this never runs the model, just layers three safetensors files' state
dicts in order (base, then Stage-1's trainable subset, then Stage-2's trainable subset) and
writes the union to a new file. CPU-only, no GPU/DDP needed.

Usage:
    python scripts/merge_stage2_checkpoint_for_posttrain.py \\
        --base-checkpoint checkpoints/n0-vtla-base \\
        --stage1-checkpoint checkpoints/vtla_stage1_predictor_pretrain/online_full_v3 \\
        --stage2-checkpoint checkpoints/vtla_stage2_align_expert/my_stage2_run \\
        --output checkpoints/merged_for_posttrain/my_stage2_run

Each of --stage1-checkpoint/--stage2-checkpoint may be either a specific numeric step directory
or the parent experiment directory (in which case the highest step present is used).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import safetensors.torch


def _resolve_step_dir(path: Path) -> Path:
    """If `path` already has a model.safetensors directly inside it, use it as-is (a specific
    step dir). Otherwise treat it as an experiment dir and pick the highest numeric step."""
    if (path / "model.safetensors").is_file():
        return path
    steps = [
        int(d.name) for d in path.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if not steps:
        raise FileNotFoundError(f"No checkpoint (model.safetensors or numeric step dirs) found in {path}")
    return path / str(max(steps))


def _load_state(path: Path) -> dict:
    step_dir = _resolve_step_dir(path)
    weights_file = step_dir / "model.safetensors"
    with safetensors.safe_open(weights_file, framework="pt", device="cpu") as f:
        state = {name: f.get_tensor(name) for name in f.keys()}
    logging.info(f"Loaded {len(state)} tensors from {weights_file}")
    return state


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--stage1-checkpoint", required=True, type=Path)
    parser.add_argument("--stage2-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_file = args.output / "model.safetensors"
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"{output_file} already exists; pass --overwrite to replace it")

    base_path = args.base_checkpoint / "model.safetensors" if (args.base_checkpoint / "model.safetensors").is_file() else args.base_checkpoint
    with safetensors.safe_open(base_path, framework="pt", device="cpu") as f:
        merged = {name: f.get_tensor(name) for name in f.keys()}
    logging.info(f"Loaded {len(merged)} tensors from base checkpoint {base_path}")

    stage1_state = _load_state(args.stage1_checkpoint)
    # Stage-1's checkpoint carries tactile_recon_head.* too (Stage-1-only auxiliary head) --
    # drop it here since post-train's model never constructs that submodule either.
    stage1_applied = {k: v for k, v in stage1_state.items() if not k.startswith("tactile_recon_head.")}
    missing_in_base1 = [k for k in stage1_applied if k not in merged]
    if missing_in_base1:
        raise ValueError(f"Stage-1 delta has keys not present in base checkpoint: {missing_in_base1}")
    merged.update(stage1_applied)
    logging.info(f"Applied {len(stage1_applied)} Stage-1 tensors on top of base "
                 f"(dropped {len(stage1_state) - len(stage1_applied)} tactile_recon_head tensors)")

    stage2_state = _load_state(args.stage2_checkpoint)
    missing_in_base2 = [k for k in stage2_state if k not in merged]
    if missing_in_base2:
        raise ValueError(f"Stage-2 delta has keys not present in base+Stage-1 checkpoint: {missing_in_base2}")
    merged.update(stage2_state)
    logging.info(f"Applied {len(stage2_state)} Stage-2 tensors on top")

    args.output.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(merged, output_file)
    logging.info(f"Wrote merged checkpoint ({len(merged)} tensors) to {output_file}")


if __name__ == "__main__":
    main()
