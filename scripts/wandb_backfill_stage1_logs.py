"""Incrementally backfill a wandb run from the segment logs of a Stage-1 (mid-train) run launched with
wandb_enabled=False (e.g. the TouchScale mid-trains, deploy/vision/n0vtla_midtrain_touchscale_zzh.sbatch).

Reads the `step=<n> k=v k=v ...` lines that train_stage1_online.py logs, from every segment log whose header
names EXP_NAME (`=== <EXP_NAME> split=...`), in job-id order (a resumed segment re-logs steps after its
checkpoint; the later segment wins). The wandb run has a FIXED id (<exp>-bf), so re-running later resumes
the same run and only uploads steps newer than what it already has -- safe to call after every segment.

Usage: python scripts/wandb_backfill_stage1_logs.py --exp-name ts_all_20k --logs logs/midtrain_ts_*.log \
         [--min-job 991346] [--project n0vtla]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import wandb

STEP_RE = re.compile(r"\bstep=(\d+) (.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_./]+)=([-+0-9.eE]+|nan|inf)")
JOB_RE = re.compile(r"_(\d+)\.log$")


def parse(logs: list[Path], exp: str, min_job: int) -> tuple[dict[int, dict[str, float]], list[str]]:
    header = f"=== {exp} split="
    picked = []
    for p in logs:
        m = JOB_RE.search(p.name)
        if not m or int(m.group(1)) < min_job:
            continue
        text = p.read_text(errors="replace")
        if header in text:
            picked.append((int(m.group(1)), p, text))
    points: dict[int, dict[str, float]] = {}
    for _, _, text in sorted(picked):
        for line in text.splitlines():
            m = STEP_RE.search(line)
            if m:
                points[int(m.group(1))] = {k: float(v) for k, v in KV_RE.findall(m.group(2))}
    return points, [p.name for _, p, _ in sorted(picked)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--logs", nargs="+", type=Path, required=True)
    ap.add_argument("--min-job", type=int, default=0, help="ignore logs of older jobs (e.g. smokes)")
    ap.add_argument("--project", default="n0vtla")
    args = ap.parse_args()

    points, used = parse(args.logs, args.exp_name, args.min_job)
    if not points:
        raise SystemExit(f"no step lines for {args.exp_name} in {len(args.logs)} logs")
    run = wandb.init(project=args.project, id=f"{args.exp_name}-bf", name=args.exp_name, resume="allow",
                     job_type="backfill", config={"backfilled_from_logs": True, "exp_name": args.exp_name})
    start = run.step  # next step this run accepts (0 for a new run)
    new = [(s, v) for s, v in sorted(points.items()) if s >= start]
    for s, v in new:
        wandb.log(v, step=s)
    run.summary["backfill_logs"] = used
    run.summary["backfill_last_step"] = max(points)
    print(f"{args.exp_name}: {len(points)} step lines in {used}; uploaded {len(new)} new "
          f"(steps {new[0][0] if new else '-'}..{new[-1][0] if new else '-'}); run {run.entity}/{args.project}/{run.id}",
          flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
