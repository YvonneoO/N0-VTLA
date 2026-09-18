#!/usr/bin/env python
"""Aggregates compute_stage1_scaling_held_out_eval.py's per-checkpoint JSON reports into
one scaling-law line chart: data_fraction (x, log scale) vs. held-out Stage-1 loss (y).

Every report was scored on the SAME QC'd held_out episode set (see
build_itw_scaling_splits.py / qc_itw_scaling_split.py), so this is a fair
data-quantity-vs-performance comparison -- unlike each run's own training loss,
which is confounded by smaller data fractions being recycled more often over the
same fixed step budget (see compute_stage1_scaling_held_out_eval.py's docstring).

Usage:
    python scripts/plot_stage1_scaling_curve.py \
        --reports /path/to/stage1_scaling_held_out/*.json \
        --output stage1_scaling_curve.png
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", nargs="+", required=True,
                         help="JSON report paths or glob patterns (e.g. 'stage1_scaling_held_out/*.json').")
    parser.add_argument("--metric", default="mean_stage1_total",
                         choices=["mean_stage1_total", "mean_stage1_nce", "mean_stage1_recon"])
    parser.add_argument("--output", type=Path, default=Path("stage1_scaling_curve.png"))
    parser.add_argument("--title", default="Stage-1 held-out loss vs. human ITW data fraction")
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.reports:
        matched = glob.glob(pattern)
        paths.extend(Path(p) for p in matched) if matched else paths.append(Path(pattern))
    if not paths:
        raise FileNotFoundError(f"No report files matched: {args.reports}")

    rows = []
    for p in sorted(set(paths)):
        report = json.loads(p.read_text())
        rows.append(report)
    rows.sort(key=lambda r: r["data_fraction"])

    print(f"{'label':<12} {'fraction':>9} {'step':>7} {'n_episodes':>11} {'mean_stage1_nce':>16} "
          f"{'mean_stage1_recon':>18} {'mean_stage1_total':>18}")
    for r in rows:
        print(f"{r['checkpoint_label']:<12} {r['data_fraction']:>9.2f} {r['checkpoint_step']:>7} "
              f"{r['num_held_out_episodes']:>11} {r['mean_stage1_nce']:>16.4f} "
              f"{r['mean_stage1_recon']:>18.4f} {r['mean_stage1_total']:>18.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = [r["data_fraction"] for r in rows]
    values = [r[args.metric] for r in rows]
    labels = [r["checkpoint_label"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fractions, values, marker="o", linewidth=2)
    for x, y, label in zip(fractions, values, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Human ITW data fraction (log scale)")
    ax.set_ylabel(args.metric.replace("mean_", "held-out ").replace("_", " "))
    ax.set_title(args.title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
