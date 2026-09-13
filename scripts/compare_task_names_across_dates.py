#!/usr/bin/env python3
"""Does the itw task-name VOCABULARY (task_info.json's "name" field) actually differ
across dates, or did the earlier per-task-scale coverage check (0% on itw07-23 against
a table fit on itw08-03..itw08-31) just get unlucky with a small 15-episode sample?

Per-task-scale itself is a date-independent concept -- the same real-world task
performed on two different days should carry the same task name and deserve the same
scale. This script checks that assumption directly against real data: sample task
names from two date sets and report the overlap, rather than assuming a "different
date range" explanation is the whole story.

Usage: python scripts/compare_task_names_across_dates.py <raw_root> \\
           --dates-a itw07-23,itw07-25 --dates-b itw08-10,itw08-11 [--sample 300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n0vtla.training.itw_online_dataset import list_episode_dirs, read_task_name_for_norm  # noqa: E402


def _names_for(raw_root: Path, dates: str, sample: int) -> tuple[set[str], int, int]:
    episodes = list_episode_dirs(raw_root, date_dirs=dates.split(","))
    episodes = episodes[:sample]
    names: set[str] = set()
    n_missing = 0
    for ep in episodes:
        name = read_task_name_for_norm(ep)
        if name:
            names.add(name)
        else:
            n_missing += 1
    return names, len(episodes), n_missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--dates-a", required=True)
    parser.add_argument("--dates-b", required=True)
    parser.add_argument("--sample", type=int, default=300)
    args = parser.parse_args()

    names_a, count_a, missing_a = _names_for(args.raw_root, args.dates_a, args.sample)
    names_b, count_b, missing_b = _names_for(args.raw_root, args.dates_b, args.sample)
    overlap = names_a & names_b

    print(f"dates_a={args.dates_a}: {count_a} episodes scanned ({missing_a} missing name), {len(names_a)} distinct task names")
    print(f"dates_b={args.dates_b}: {count_b} episodes scanned ({missing_b} missing name), {len(names_b)} distinct task names")
    print(f"overlap: {len(overlap)} shared task names between the two date sets")
    if overlap:
        for name in sorted(overlap)[:15]:
            print(f"  shared: {name!r}")
    else:
        print("  no overlap at all -- sampling a few from each side for comparison:")
        for name in sorted(names_a)[:8]:
            print(f"  a: {name!r}")
        for name in sorted(names_b)[:8]:
            print(f"  b: {name!r}")
    print("TASK_NAME_COMPARISON_DONE")


if __name__ == "__main__":
    main()
