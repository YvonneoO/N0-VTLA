#!/usr/bin/env python
"""Extra, pool-size-independent readouts on the predictions saved by collect_stage1_multi_ckpt_predictions.py.

Motivation: top-k retrieval accuracy is tiny in absolute terms (identity retrieval among near-duplicate,
stochastic futures). These metrics are defined up front and each comes with a ceiling / chance reference.

Per task (pooled robot train+val, false negatives = same episode within +-H frames removed) and checkpoint:

 1. pct_rank   percentile rank of the true partner among the candidates (== retrieval ROC-AUC per query;
               chance 50%, higher is better). Independent of pool size. Mean / median + episode-bootstrap CI.
 2. near_hit   neighbourhood-consistent retrieval: N_i = the 10% of candidates whose target z* is closest to
               the true z*_i (including itself). near1 = P(top-1 retrieved candidate in N_i); near10 =
               mean fraction of the top-10 retrieved that are in N_i. Chance = 10%.
 3. oracle     ceiling: the query is the target z* of the NEAREST-IN-TIME sample of the same episode (a few
               frames away) instead of the prediction; same protocol. Also reported: median time gap.
               skill = (metric - chance) / (oracle - chance): fraction of the attainable gap that is closed.
 4. probe      linear-probe frame-level contact detection at fixed thresholds (train quantile 0.5 / 0.8):
               balanced accuracy and F1 on val, for z, the persistence embedding, per checkpoint.

Queries: z (predicted), cur (persistence), oracle. Subsets: all, top50 (contact-change top half).

Usage:
  python scripts/analyze_stage1_extra_metrics.py --inputs DIR --tasks task1 task2 task3 sv2 --out OUTDIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_stage1_predictions import allowed_mask, boot_ci, load, pca_probe_scores  # noqa: E402


def nearest_in_time(ep: np.ndarray, fr: np.ndarray):
    """For each sample, index of the nearest-in-time other sample of the same episode (-1 if none) + gap."""
    n = len(ep)
    nb = np.full(n, -1)
    gap = np.full(n, np.nan)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(fr[idx])]
        f = fr[idx]
        for p in range(len(idx)):
            cand = []
            if p > 0:
                cand.append((f[p] - f[p - 1], idx[p - 1]))
            if p < len(idx) - 1:
                cand.append((f[p + 1] - f[p], idx[p + 1]))
            g, j = min(cand)
            nb[idx[p]], gap[idx[p]] = j, g
    return nb, gap


def rank_metrics(q: torch.Tensor, hs: torch.Tensor, allow: np.ndarray, valid: np.ndarray | None = None):
    """Per-query percentile rank + neighbourhood-consistent hits. Positive of row i is hs[i]."""
    n = q.shape[0]
    sim = (q.float() @ hs.float().t())
    pos = sim.diag().clone()
    al = torch.as_tensor(allow)
    m = al.sum(1).clamp(min=1).float()
    pct = (((sim < pos[:, None]) & al).sum(1).float() + 0.5 * ((sim == pos[:, None]) & al).sum(1).float()) / m
    # top-10 (full pool) for reference
    gt = ((sim > pos[:, None]) & al).sum(1)
    top10 = (gt < 10).float()
    # neighbourhood consistency: candidate set S_i = allowed negatives + the positive itself
    S = al.clone()
    S[torch.arange(n), torch.arange(n)] = True
    zz = hs.float() @ hs.float().t()
    ninf = torch.tensor(float("-inf"))
    zzS = torch.where(S, zz, ninf)
    size = S.sum(1)
    k = torch.ceil(0.1 * size.float()).long().clamp(min=1)
    srt = torch.sort(zzS, dim=1, descending=True).values
    thr = srt.gather(1, (k - 1)[:, None])
    nmask = (zzS >= thr) & S
    simS = torch.where(S, sim, ninf)
    top1 = simS.argmax(1)
    near1 = nmask[torch.arange(n), top1].float()
    t10 = simS.topk(min(10, n), dim=1).indices
    near10 = nmask.gather(1, t10).float().mean(1)
    chance_near = (nmask.sum(1).float() / size.float())
    out = {"pct": pct.numpy(), "top10": top10.numpy(), "near1": near1.numpy(), "near10": near10.numpy(),
           "chance_near": chance_near.numpy()}
    if valid is not None:
        out = {k_: np.where(valid, v, np.nan) for k_, v in out.items()}
    return out


def summarise(vals: np.ndarray, ep: np.ndarray, rng, b: int):
    mean, lo, hi = boot_ci(vals, ep, rng, b)
    return {"mean": mean, "ci95": [lo, hi], "median": float(np.nanmedian(vals))}


def bal_acc_f1(y: np.ndarray, s: np.ndarray):
    pred = s > 0
    tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    fp, fn = ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return float(0.5 * (tpr + tnr)), float(f1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3", "sv2"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--probe-quantiles", type=float, nargs="+", default=[0.5, 0.8])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results: dict = {}

    for task in args.tasks:
        try:
            tr, va = load(args.inputs, task, "train", args.tag), load(args.inputs, task, "val", args.tag)
        except FileNotFoundError:
            print(f"[{task}] missing inputs, skipped")
            continue
        labels, H = tr["labels"], int(tr["future_frame_offset"])
        R = results.setdefault(task, {"labels": labels, "H": H, "rank": {}, "probe": {}})
        ds = [tr, va]
        ep = np.concatenate([np.array([f"{d['split']}:{int(e)}" for e in d["episode_index"]]) for d in ds])
        fr = np.concatenate([d["frame_in_episode"].numpy() for d in ds])
        dl2 = np.concatenate([d["d_l2"].numpy() for d in ds])
        nb, gap = nearest_in_time(ep, fr)
        valid_or = nb >= 0
        R["oracle_time_gap_frames"] = {"median": float(np.nanmedian(gap)), "p90": float(np.nanpercentile(gap, 90))}
        subsets = {"all": np.ones(len(ep), bool), "top50": dl2 >= np.quantile(dl2, 0.5)}
        for sub, m in subsets.items():
            R["rank"][sub] = {}
            ep_m, fr_m = ep[m], fr[m]
            allow = allowed_mask(ep_m, fr_m, H)
            nb_m, _ = nearest_in_time(ep_m, fr_m)          # oracle neighbour restricted to the subset
            valid = nb_m >= 0
            nbi = np.where(valid, nb_m, 0)
            mt = torch.as_tensor(m)
            for k, label in enumerate(labels):
                hz = torch.cat([d["hz"][k] for d in ds]).float()[mt]
                hs = torch.cat([d["hzs"][k] for d in ds]).float()[mt]
                hc = torch.cat([d["hcur"][k] for d in ds]).float()[mt]
                cell = {}
                for qname, q, vmask in (("z", hz, None), ("cur", hc, None),
                                        ("oracle", hs[torch.as_tensor(nbi)], valid)):
                    r = rank_metrics(q, hs, allow, vmask)
                    cell[qname] = {
                        "pct_rank": summarise(r["pct"], ep_m, rng, args.bootstrap),
                        "top10_full": summarise(r["top10"], ep_m, rng, args.bootstrap),
                        "near_top1": summarise(r["near1"], ep_m, rng, args.bootstrap),
                        "near_top10": summarise(r["near10"], ep_m, rng, args.bootstrap),
                        "chance_near": float(np.nanmean(r["chance_near"])),
                        "chance_top10": float(10 / max(allow.sum(1).mean() + 1, 1)),
                    }
                R["rank"][sub][label] = cell
            print(f"[{task}/{sub}] rank done", flush=True)

        # ---- fixed-threshold probe readouts (train -> val) ----
        stats_tr = {"future_contact": tr["fut_mean_abs"].numpy(), "contact_change": tr["d_l2"].numpy()}
        stats_va = {"future_contact": va["fut_mean_abs"].numpy(), "contact_change": va["d_l2"].numpy()}
        for lbl_name in stats_tr:
            for q in args.probe_quantiles:
                tau = np.quantile(stats_tr[lbl_name], q)
                ytr, yva = (stats_tr[lbl_name] > tau).astype(int), (stats_va[lbl_name] > tau).astype(int)
                cell = {}
                for k, label in enumerate(labels):
                    cell[label] = {}
                    for feat, key in (("z", "hz"), ("cur", "hcur")):
                        s = pca_probe_scores(tr[key][k].float().numpy(), ytr, va[key][k].float().numpy())
                        ba, f1 = bal_acc_f1(yva, s)
                        cell[label][feat] = {"balanced_acc": ba, "f1": f1}
                R["probe"].setdefault(lbl_name, {})[f"q{q}"] = {"pos_rate_val": float(yva.mean()), "by_ckpt": cell}
        print(f"[{task}] probe done", flush=True)

    (args.out / "extra_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out / 'extra_metrics.json'}")


if __name__ == "__main__":
    main()
