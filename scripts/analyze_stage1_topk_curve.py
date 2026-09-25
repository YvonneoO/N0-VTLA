#!/usr/bin/env python
"""Recall@K curves (K = 1 ... 1000) of the future-latent retrieval, per task / checkpoint / query type.

TIE HANDLING. Many samples (no tactile change) have IDENTICAL target latents z*. The earlier protocol
(`gt < k`, strict ">") ranks the positive optimistically among exact ties, which inflates top-k for queries
whose positive sits in a tied cluster (hundreds of candidates). Here we report both:
  opt   strict ">" (== the earlier v3/v4 convention)
  fair  random tie-breaking, implemented by adding tiny independent jitter to every similarity (averaged
        over several draws) -- the positive is uniformly placed among the candidates it ties with.

Same protocol as analyze_stage1_extra_metrics.py (pooled robot train+val, same-episode +-H false negatives
removed, full candidate pool). Queries: z (predicted), cur (persistence), oracle (true target of the
nearest-in-time sample). recall@K = P(true partner is among the K most similar candidates); chance = K/pool.

  python scripts/analyze_stage1_topk_curve.py --inputs DIR --tasks task1 task2 task3 sv2 --out OUTDIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_stage1_extra_metrics import nearest_in_time  # noqa: E402
from analyze_stage1_predictions import allowed_mask, load  # noqa: E402

KS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def ranks(q: torch.Tensor, hs: torch.Tensor, allow: np.ndarray, jitter: float = 0.0, gen=None) -> np.ndarray:
    sim = q.float() @ hs.float().t()
    if jitter > 0:
        sim = sim + jitter * torch.randn(sim.shape, generator=gen)
    pos = sim.diag().clone()
    return (((sim > pos[:, None]) & torch.as_tensor(allow)).sum(1)).numpy()   # 0 = best


def p256_top(gt: np.ndarray, m: np.ndarray, k: int) -> np.ndarray:
    from scipy.stats import hypergeom
    draws = np.minimum(255, m)
    return np.where(m > 0, hypergeom.cdf(k - 1, np.maximum(m, 1), gt, draws), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3", "sv2"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res: dict = {"ks": list(KS)}
    for task in args.tasks:
        try:
            tr, va = load(args.inputs, task, "train"), load(args.inputs, task, "val")
        except FileNotFoundError:
            continue
        labels, H = tr["labels"], int(tr["future_frame_offset"])
        ds = [tr, va]
        ep = np.concatenate([np.array([f"{d['split']}:{int(e)}" for e in d["episode_index"]]) for d in ds])
        fr = np.concatenate([d["frame_in_episode"].numpy() for d in ds])
        dl2 = np.concatenate([d["d_l2"].numpy() for d in ds])
        R = res.setdefault(task, {"labels": labels})
        for sub, m in {"all": np.ones(len(ep), bool), "top50": dl2 >= np.quantile(dl2, 0.5)}.items():
            allow = allowed_mask(ep[m], fr[m], H)
            nb, _ = nearest_in_time(ep[m], fr[m])
            valid = nb >= 0
            nbi = torch.as_tensor(np.where(valid, nb, 0))
            mt = torch.as_tensor(m)
            pool = allow.sum(1) + 1
            cell = {"chance": [float(np.mean(np.minimum(k / pool, 1.0))) for k in KS]}
            for k_i, label in enumerate(labels):
                hz = torch.cat([d["hz"][k_i] for d in ds]).float()[mt]
                hs = torch.cat([d["hzs"][k_i] for d in ds]).float()[mt]
                hc = torch.cat([d["hcur"][k_i] for d in ds]).float()[mt]
                for qn, q, v in (("z", hz, None), ("cur", hc, None), ("oracle", hs[nbi], valid)):
                    r = ranks(q, hs, allow)
                    sel = np.ones(len(r), bool) if v is None else v
                    cell.setdefault(label, {})[qn] = [float((r[sel] < k).mean()) for k in KS]          # opt (strict >)
                    gen = torch.Generator().manual_seed(0)
                    rj = [ranks(q, hs, allow, 1e-5, gen) for _ in range(3)]
                    cell.setdefault(label + "/fair", {})[qn] = [float(np.mean([(x[sel] < k).mean() for x in rj])) for k in KS]
                    if qn != "oracle":
                        mm = allow.sum(1)
                        cell.setdefault(label + "/p256", {})[qn] = {
                            "opt": {k: float(np.nanmean(p256_top(r, mm, k))) for k in (1, 10)},
                            "fair": {k: float(np.nanmean([np.nanmean(p256_top(x, mm, k)) for x in rj])) for k in (1, 10)}}
            R[sub] = cell
        print(f"[{task}] done", flush=True)
    (args.out / "topk_curve.json").write_text(json.dumps(res, indent=2))
    print("Wrote", args.out / "topk_curve.json")


if __name__ == "__main__":
    main()
