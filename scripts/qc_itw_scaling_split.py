#!/usr/bin/env python
"""One-off QC scan: verify every episode in a build_itw_scaling_splits.py manifest split
has its expected video files present (not just the dir/csv sidecar), and optionally write
a filtered manifest with bad episodes removed.

Triggered by job 561220 (the 80pct Stage-1 run) crashing on
itw09-15/5361ec0a-1ac8-4804-8e6d-163d1043a0c9: wrist_right.csv existed but
wrist_right.mp4 never finished writing/transferring -- a real incomplete-episode gap
in this account's raw data mirror, not a code bug. 80pct is a strict superset of the
already-completed 60pct/40pct/20pct runs (nested-prefix design), so this episode was
newly pulled in only at 80pct; smaller runs never touched it. Since a crash 30+ min
into an 8-GPU job is expensive, this scans the WHOLE split up front so all bad
episodes (if more than one) are found in a single pass instead of one crash-resubmit
cycle per bad episode.

Usage:
    python scripts/qc_itw_scaling_split.py \
        --raw-root /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/raw \
        --manifest /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/itw_scaling_splits.json \
        --split 80pct \
        --write-filtered /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/itw_scaling_splits.json \
        --in-place
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_FILES = ("rgb_head.mp4", "wrist_left.mp4", "wrist_right.mp4", "depth_head.mkv", "episode.h5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True, help="Which manifest['splits'] key to scan.")
    parser.add_argument("--in-place", action="store_true",
                         help="Rewrite --manifest with the bad episodes removed from --split "
                              "(all other keys untouched). Without this, only reports.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rel_paths = manifest["splits"][args.split]
    print(f"Scanning {len(rel_paths)} episodes in split {args.split!r}...")

    bad: list[str] = []
    for i, rel in enumerate(rel_paths):
        ep_dir = args.raw_root / rel
        missing = [f for f in REQUIRED_FILES if not (ep_dir / f).is_file()]
        if missing:
            bad.append(rel)
            print(f"  BAD: {rel} missing {missing}")
        if (i + 1) % 5000 == 0:
            print(f"  ...{i + 1}/{len(rel_paths)} scanned, {len(bad)} bad so far")

    print(f"Done: {len(bad)}/{len(rel_paths)} episodes bad in split {args.split!r}")
    if bad:
        print("Bad episode list:", json.dumps(bad, indent=2))

    if args.in_place:
        manifest["splits"][args.split] = [r for r in rel_paths if r not in set(bad)]
        manifest.setdefault("qc_removed", {})[args.split] = bad
        args.manifest.write_text(json.dumps(manifest, indent=2))
        print(f"Rewrote {args.manifest}: split {args.split!r} now has "
              f"{len(manifest['splits'][args.split])} episodes ({len(bad)} removed).")


if __name__ == "__main__":
    main()
