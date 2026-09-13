#!/usr/bin/env python3
"""One-off check: which key does a real tacWAM-format task_info.json use for the
task/language field -- "name" (tacWAM's own convention, confirmed via
tacwam/tujian_v2.py:766 `task_payload.get("name", "")` and
tacwam/cosmos_tactile/pad_data.py:407 `task.get("name", "")`, both of which also feed
the per-task-scale normalization manifest's `task_scale` keys) or "steps" (what
scripts/itw_pressure.py's read_task-equivalent currently prefers)?

Prints only the key names and, for "name"/"steps" specifically, their values -- not a
raw dump of the whole file -- so this is safe to run against real data and safe to
read from a job log.

Usage: python scripts/inspect_task_info_schema.py <episode_dir> [<episode_dir> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    for arg in sys.argv[1:]:
        path = Path(arg) / "task_info.json"
        if not path.exists():
            print(f"{arg}: task_info.json MISSING")
            continue
        info = json.loads(path.read_text(encoding="utf-8"))
        print(f"{arg}: keys={sorted(info.keys())}")
        if "name" in info:
            print(f"  name={info['name']!r}")
        if "steps" in info:
            print(f"  steps={info['steps']!r}")


if __name__ == "__main__":
    main()
