#!/usr/bin/env python
"""Builds fixed, NESTED data-fraction splits of the ITW raw episode corpus for the
Stage-1 human-data pretraining scaling study, saved as one JSON manifest.

Why fixed + nested, not a fresh random sample per run: `train_stage1_online.py`'s
existing `VTLA_ITW_MAX_EPISODES` re-samples with `random.Random(seed).sample(...)`
every launch -- fine for a smoke-test cap, but for a scaling study we want every run
at a given fraction to see the EXACT same episode set (reproducible, inspectable) and
each lower fraction to be a strict subset of the next (20% subset of 40% subset of
60% subset of 80% subset of 100%), so differences in the resulting scaling curve are
attributable to how much data was added, not to which episodes got resampled.

Also writes a "held_out" split: the complement of the LARGEST built fraction (default
80%) in the same shuffled order, i.e. shuffled[n_80pct:]. Because 20/40/60/80pct are
all PREFIXES of one deterministic shuffle (same seed), this tail slice is, by
construction, never included in ANY of them -- a scaling-law comparison (checkpoint
performance vs. data fraction) must score every checkpoint on episodes none of them
trained on, or a smaller-fraction run's lower training loss is confounded by seeing
its smaller pool repeated more often over the same step budget, not by learning
better. Re-running this script with --overwrite and the SAME --seed/--fractions
reproduces the 20/40/60/80pct keys byte-identically (pure functions of raw_root +
seed) and is safe even while jobs that already read the old manifest are running --
they loaded their fixed episode list into memory at startup and never re-read the file.

This corpus (this account's own `yqq/data/raw`) is NOT the same date range as the
already-completed prj-02-phai-lab full-corpus run (tujian_v5_recent, itw07-23..itw09-10)
-- it's missing itw07-28/29/30 and has itw09-12/14/15 instead, per user decision on
2026-09-17: use whatever this account actually has as this study's own 100% reference,
rather than trying to reconstruct the exact old corpus.

Usage:
    python scripts/build_itw_scaling_splits.py \
        --raw-root /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/raw \
        --output /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/itw_scaling_splits.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from n0vtla.training.itw_online_dataset import list_episode_dirs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    raw_root = Path(args.raw_root)
    episodes = list_episode_dirs(raw_root)
    total = len(episodes)
    print(f"Total episode dirs under {raw_root}: {total}")
    if total == 0:
        raise ValueError(f"No episode dirs found under {raw_root}")

    rel_paths = [str(p.relative_to(raw_root)) for p in episodes]
    shuffled = rel_paths.copy()
    random.Random(args.seed).shuffle(shuffled)

    splits: dict[str, list[str]] = {}
    for frac in args.fractions:
        n = round(frac * total)
        key = f"{round(frac * 100)}pct"
        splits[key] = shuffled[:n]
        print(f"{key}: {n} episodes")

    n_max = round(max(args.fractions) * total)
    splits["held_out"] = shuffled[n_max:]
    print(f"held_out: {len(splits['held_out'])} episodes (complement of the "
          f"{round(max(args.fractions) * 100)}pct split)")

    manifest = {
        "raw_root": str(raw_root),
        "seed": args.seed,
        "total_episodes": total,
        "nested": True,
        "note": "Fixed subsets for the Stage-1 data-scaling study -- see this script's "
                "module docstring for why this account's own corpus (not the older "
                "prj-02-phai-lab tujian_v5_recent date range) is the 100% reference here.",
        "splits": splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {output}")


if __name__ == "__main__":
    main()
