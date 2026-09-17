"""Backfill a wandb run from an already-completed (or in-progress) local training log,
for a run that was launched with wandb_enabled=False and only has the local
`STEP <n> loss=<x> lr=<y>` text log to show for it.

Not a live logger -- this parses the log ONCE and pushes every step as a historical
point, then finishes the run. Re-running against a log that has grown (e.g. a still-
running job) after this creates a wholly separate run rather than resuming a finished
one, since a finished run cannot be appended to.

Usage:
  python scripts/wandb_backfill_from_log.py \
    --log logs/wetlab_v2_train_full_run1.log \
    --exp-name wetlab_v2_train_full_run1 \
    --project n0vtla
"""
from __future__ import annotations

import argparse
import re

import wandb

STEP_RE = re.compile(r"STEP (\d+) loss=([0-9.eE+-]+)(?: lr=([0-9.eE+-]+))?")


def parse_log(path: str) -> list[tuple[int, float, float | None]]:
    points: dict[int, tuple[float, float | None]] = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            loss = float(m.group(2))
            lr = float(m.group(3)) if m.group(3) else None
            points[step] = (loss, lr)  # last occurrence wins (dedupes tqdm re-prints)
    return [(step, *vals) for step, vals in sorted(points.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True)
    parser.add_argument("--exp-name", required=True, help="wandb run name")
    parser.add_argument("--project", default="n0vtla")
    args = parser.parse_args()

    points = parse_log(args.log)
    if not points:
        raise SystemExit(f"no 'STEP <n> loss=...' lines found in {args.log}")

    run = wandb.init(project=args.project, name=args.exp_name, job_type="backfill")
    for step, loss, lr in points:
        payload = {"loss": loss}
        if lr is not None:
            payload["lr"] = lr
        wandb.log(payload, step=step)
    run.summary["backfilled_points"] = len(points)
    run.summary["final_step"] = points[-1][0]
    run.summary["final_loss"] = points[-1][1]
    wandb.finish()
    print(f"backfilled {len(points)} points ({points[0][0]}..{points[-1][0]}) to {args.project}/{args.exp_name}")


if __name__ == "__main__":
    main()
