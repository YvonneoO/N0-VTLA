#!/usr/bin/env python
"""Figures + markdown tables from analyze_stage1_predictions.py's metrics.json.

  python scripts/plot_stage1_prediction_metrics.py --metrics OUT/metrics.json --out OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"base": "#8a8a8a", "20pct": "#9ecae1", "40pct": "#6baed6", "60pct": "#3182bd", "80pct": "#08519c", "100pct": "#d95f02"}
NAME = {"sv2": "smoke_test_v2", "task1": "Task1", "task2": "Task2", "task3": "Task3"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    M = json.loads(args.metrics.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in ("task1", "task2", "task3", "sv2") if t in M]

    # ---- AUC vs threshold: rows = label family, cols = task ----
    for fam, title in (("future_contact", "future contact state  (mean|tac_{t+H}-tac_0| > tau)"),
                       ("contact_change", "contact change  (||tac_{t+H}-tac_t|| > tau)")):
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, t in zip(axes, tasks):
            c = M[t]["contact"][fam]
            q = np.array(c["quantiles"]) * 100
            for lab in M[t]["labels"]:
                ax.plot(q, c["auc"][lab]["z"], "-o", ms=3, color=COL[lab], label=lab if lab != "100pct" else "100% (ref)")
            ax.plot(q, c["auc"]["100pct"]["cur"], "--", color="k", lw=1, label="persistence probe")
            ax.plot(q, c["auc"]["raw_current_state"], ":", color="#555", lw=1.2, label="raw current state")
            ax.axhline(0.5, color="#bbb", lw=0.8)
            ax.set_title(NAME[t], fontsize=10)
            ax.set_xlabel("threshold tau (train-split quantile, %)")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("val ROC-AUC")
        axes[-1].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"Frame-level detection of {title}: linear probe on predicted latent z", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out / f"auc_vs_tau_{fam}.png", dpi=150)
        plt.close(fig)

    # ---- headline retrieval: expected top-10 in a pool of 256, false negatives removed ----
    for proto, ttl in (("p256_fn", "pool 256, false negatives removed"), ("p256_fn_top50", "pool 256, false negatives removed, contact-change top 50%")):
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, t in zip(axes, tasks):
            labs = M[t]["labels"]
            g = M[t]["retrieval"]["pooled"]
            xs = np.arange(len(labs))
            m = np.array([g[l][f"z/{proto}"]["top10"]["mean"] * 100 for l in labs])
            lo = np.array([g[l][f"z/{proto}"]["top10"]["ci95"][0] * 100 for l in labs])
            hi = np.array([g[l][f"z/{proto}"]["top10"]["ci95"][1] * 100 for l in labs])
            ax.bar(xs, m, color=[COL[l] for l in labs], yerr=[m - lo, hi - m], capsize=2)
            pers = np.mean([g[l][f"cur/{proto}"]["top10"]["mean"] * 100 for l in labs])
            chance = g[labs[0]][f"z/{proto}"]["top10"]["chance"] * 100
            ax.axhline(chance, color="r", ls="--", lw=1, label=f"chance {chance:.1f}%")
            ax.axhline(pers, color="k", ls=":", lw=1, label=f"persistence {pers:.1f}%")
            ax.set_xticks(xs)
            ax.set_xticklabels([l.replace("pct", "%") for l in labs], fontsize=8)
            ax.set_title(NAME[t], fontsize=10)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("top-10 accuracy (%)")
        fig.suptitle(f"Future-latent retrieval, {ttl} (pooled robot-train + val)", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out / f"retrieval_{proto}.png", dpi=150)
        plt.close(fig)

    # ---- markdown tables ----
    lines = []
    for t in tasks:
        labs = M[t]["labels"]
        lines.append(f"\n### {NAME[t]}  (n = {M[t]['n']})\n")
        lines.append("| protocol (pooled) | " + " | ".join(l.replace("pct", "%") for l in labs) + " | persistence (mean) | chance |")
        lines.append("|---|" + "---|" * (len(labs) + 2))
        g = M[t]["retrieval"]["pooled"]
        for proto in ("full", "full_fn", "p256", "p256_fn", "p256_fn_top50", "p256_fn_top25"):
            for k in (1, 10):
                row = [f"{g[l][f'z/{proto}'][f'top{k}']['mean'] * 100:.2f}" for l in labs]
                pers = np.mean([g[l][f"cur/{proto}"][f"top{k}"]["mean"] * 100 for l in labs])
                ch = g[labs[0]][f"z/{proto}"][f"top{k}"]["chance"] * 100
                lines.append(f"| {proto} top-{k} % | " + " | ".join(row) + f" | {pers:.2f} | {ch:.2f} |")
    (args.out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
