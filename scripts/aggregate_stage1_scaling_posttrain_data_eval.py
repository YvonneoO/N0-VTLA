#!/usr/bin/env python
"""CPU aggregation for compute_stage1_scaling_posttrain_data_eval.py embeddings.

For every checkpoint label and every split (train, val, and POOLED = train+val scored in
ONE joint contrastive pool), computes the same retrieval readouts as the held-out v2 eval:
i2t/t2i top-1/5/10/100, MRR, pool_nce, mean positive cosine, plus their episode-clustered
bootstrap 95% CIs (frames within an episode are near-duplicates, so the effective sample size
is the number of episodes, not frames) and PAIRED differences vs --reference (default `base`,
the no-Stage-1 model) using the same episode resamples for both sides.

Input files: <inputs>/<label>_<split>.pt written by the collector. Every checkpoint must have
been run on the identical sample set (same stride/cap), which the collector guarantees.

Usage:
    python scripts/aggregate_stage1_scaling_posttrain_data_eval.py --inputs DIR \
        --labels base 20pct 40pct 60pct 100pct --output-json out.json --output-csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

KS = (1, 5, 10, 100)


def per_query(hz: torch.Tensor, hs: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-query metric vectors within ONE joint pool (row i's positive is column i)."""
    hz, hs = hz.float(), hs.float()
    sim = hz @ hs.t()
    diag = sim.diagonal()
    out: dict[str, np.ndarray] = {}
    for name, m in (("i2t", sim), ("t2i", sim.t())):
        rank = (m > diag[:, None]).sum(1)
        for k in KS:
            out[f"{name}_top{k}"] = (rank < k).float().numpy()
        out[f"{name}_mrr"] = (1.0 / (rank.float() + 1)).numpy()
    nce_i2t = torch.logsumexp(sim, dim=1) - diag
    nce_t2i = torch.logsumexp(sim, dim=0) - diag
    out["pool_nce"] = (0.5 * (nce_i2t + nce_t2i)).numpy()
    out["pos_cos"] = diag.numpy()
    return out


def cluster_bootstrap(vals: dict[str, np.ndarray], ep: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Bootstrap distribution of each metric's mean, resampling whole episodes."""
    uniq, inv = np.unique(ep, return_inverse=True)
    n_ep = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    denom = weights @ n_ep
    return {k: (weights @ np.bincount(inv, weights=v.astype(np.float64), minlength=len(uniq))) / denom
            for k, v in vals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--reference", default="base")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data: dict[str, dict[str, dict]] = {}
    for label in args.labels:
        data[label] = {}
        for split in args.splits:
            d = torch.load(args.inputs / f"{label}_{split}.pt", map_location="cpu")
            data[label][split] = d

    results: dict = {}
    rows = []
    groups = [(s, [s]) for s in args.splits] + [("pooled", list(args.splits))]
    for gname, parts in groups:
        # Same episode / frame set for every checkpoint is required for paired differences.
        keys0 = None
        per_label = {}
        for label in args.labels:
            hz = torch.cat([data[label][s]["hz"] for s in parts])
            hs = torch.cat([data[label][s]["hzs"] for s in parts])
            ep = np.concatenate([
                np.array([f"{s}:{int(e)}" for e in data[label][s]["episode_index"]]) for s in parts
            ])
            fr = np.concatenate([data[label][s]["frame_in_episode"].numpy() for s in parts])
            key = (tuple(ep.tolist()), tuple(fr.tolist()))
            if keys0 is None:
                keys0 = key
            elif key != keys0:
                raise RuntimeError(f"{label}/{gname}: sample set differs from {args.labels[0]}'s -- not comparable")
            per_label[label] = (per_query(hz, hs), ep, len(hz))
        ep = per_label[args.labels[0]][1]
        n_eps = len(np.unique(ep))
        weights = rng.multinomial(n_eps, np.full(n_eps, 1.0 / n_eps), size=args.bootstrap).astype(np.float64)
        boots = {label: cluster_bootstrap(pq, ep, weights) for label, (pq, _e, _n) in per_label.items()}
        results[gname] = {}
        for label, (pq, _e, n) in per_label.items():
            entry = {"pool_size": n, "n_episodes": n_eps, "chance_top1": 1.0 / n, "metrics": {}, "delta_vs_reference": {}}
            for k, v in pq.items():
                lo, hi = np.percentile(boots[label][k], [2.5, 97.5])
                entry["metrics"][k] = {"mean": float(v.mean()), "ci95": [float(lo), float(hi)]}
                if label != args.reference and args.reference in boots:
                    diff = boots[label][k] - boots[args.reference][k]
                    dlo, dhi = np.percentile(diff, [2.5, 97.5])
                    entry["delta_vs_reference"][k] = {
                        "mean": float(v.mean() - per_label[args.reference][0][k].mean()),
                        "ci95": [float(dlo), float(dhi)],
                    }
            results[gname][label] = entry
            rows.append({
                "split": gname, "label": label, "pool_size": n, "n_episodes": n_eps,
                **{f"{k}": entry["metrics"][k]["mean"] for k in entry["metrics"]},
                **{f"{k}_ci_lo": entry["metrics"][k]["ci95"][0] for k in entry["metrics"]},
                **{f"{k}_ci_hi": entry["metrics"][k]["ci95"][1] for k in entry["metrics"]},
            })
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for gname in results:
        print(f"\n== {gname} ==  (i2t top-1 %, [95% CI, episode bootstrap]; delta vs {args.reference})")
        for label, e in results[gname].items():
            m = e["metrics"]["i2t_top1"]
            d = e["delta_vs_reference"].get("i2t_top1")
            ds = f"  d={d['mean'] * 100:+.2f} [{d['ci95'][0] * 100:+.2f},{d['ci95'][1] * 100:+.2f}]" if d else ""
            print(f"  {label:>8} pool={e['pool_size']:>6} eps={e['n_episodes']:>4} "
                  f"{m['mean'] * 100:6.2f} [{m['ci95'][0] * 100:.2f},{m['ci95'][1] * 100:.2f}]{ds}")


if __name__ == "__main__":
    main()
