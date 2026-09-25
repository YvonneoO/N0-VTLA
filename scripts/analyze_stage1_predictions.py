#!/usr/bin/env python
"""Offline metrics on the predictions saved by collect_stage1_multi_ckpt_predictions.py.

The metric set is FIXED up front (nothing here is chosen after looking at results); every number is
reported next to its chance level and next to a persistence baseline (the same protocol with the CURRENT
tactile embedding as the query instead of the predicted future latent).

Per (task, split-group) and checkpoint (base, 20/40/60/80/100%):

A. Retrieval of the real future latent z* from the predicted latent z (i2t), several protocols:
     full        all candidates of the split (== the v3 protocol)
     full_fn     + false-negative removal: candidates from the SAME episode within +-H frames of the query
                 (H = future_frame_offset) are dropped (their targets are near-identical to the true one)
     p256        expected top-k in a pool of 256 candidates (1 positive + 255 random negatives), computed
                 EXACTLY via the hypergeometric distribution (no sampling noise)
     p256_fn     p256 + false-negative removal            <- headline
     p256_fn_top50 / _top25   p256_fn restricted (queries AND candidates) to the contact-change subset:
                 samples whose ||tac_{t+H}-tac_t|| is above the 50th / 75th percentile of the split
   Chance for top-k is k / (#candidates in the query's pool); the ratio to chance is reported.
   Queries: `z` (predicted), `cur` (persistence: current-tactile embedding in the ckpt's own space).
   Episode-clustered bootstrap CIs (frames of an episode are near-duplicates).
B. Space drift: cosine between the z* of different checkpoints for the same samples, and the correlation
   of their sample-similarity structures (RSA) -- is "the target moves with the checkpoint" a problem?
C. Fixed target space: ridge from z (or cur) to the frozen DINOv2 feature of the future diff (f_star,
   identical for every checkpoint), fit on the TRAIN split, retrieval on the VAL split.
D. Frame-level contact detection: label = 1[stat > tau], tau swept over quantiles of the TRAIN split, for
   (i) future contact state (mean|tac_{t+H}-tac_0|) and (ii) contact change (||tac_{t+H}-tac_t||).
   Scores: PCA(128)+logistic probe on z (fit on train, scored on val), the same on the persistence
   embedding, the raw current contact state, and (for non-base) the recon-head magnitude. ROC-AUC vs tau.
E. Region level (8x8 cells): ridge z -> grid_tgt, per-sample correlation on high-change val samples,
   plus MSE skill vs predicting the train-mean grid.

Usage:
  python scripts/analyze_stage1_predictions.py --inputs DIR --tasks task1 task2 task3 sv2 --out OUTDIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hypergeom

KS = (1, 5, 10)


def load(inputs: Path, task: str, split: str, tag: str = ""):
    return torch.load(inputs / f"{task}_{split}{tag}.pt", map_location="cpu")


def allowed_mask(ep: np.ndarray, fr: np.ndarray, h: int) -> np.ndarray:
    """allowed[i, j]: candidate j may be a NEGATIVE for query i (i.e. not a near-duplicate target)."""
    _, ep = np.unique(ep, return_inverse=True)
    same = ep[:, None] == ep[None, :]
    near = np.abs(fr[:, None] - fr[None, :]) <= h
    a = ~(same & near)
    np.fill_diagonal(a, False)
    return a


def retrieval(q: torch.Tensor, cand: torch.Tensor, allow: np.ndarray, pool256: bool) -> dict[str, np.ndarray]:
    """Per-query top-k indicators/expectations. q, cand: (N, D) L2-normalised; row i's positive is cand[i].
    `allow[i, j]`: candidate j may act as a NEGATIVE for query i (diagonal must be False)."""
    sim = (q.float() @ cand.float().t()).numpy()
    pos = np.diag(sim).copy()
    gt = ((sim > pos[:, None]) & allow).sum(1)           # allowed negatives that beat the positive
    m = allow.sum(1)                                     # allowed negatives
    out = {}
    for k in KS:
        if pool256:
            draws = np.minimum(255, m)
            out[f"top{k}"] = np.where(m > 0, hypergeom.cdf(k - 1, np.maximum(m, 1), gt, draws), np.nan)
            pool = draws + 1
        else:
            out[f"top{k}"] = (gt < k).astype(np.float64)
            pool = m + 1
        out[f"chance{k}"] = np.minimum(k / pool, 1.0)
    return out


def boot_ci(vals: np.ndarray, ep: np.ndarray, rng: np.random.Generator, b: int = 1000):
    uniq, inv = np.unique(ep, return_inverse=True)
    ok = ~np.isnan(vals)
    s = np.bincount(inv, weights=np.where(ok, vals, 0.0), minlength=len(uniq))
    c = np.bincount(inv, weights=ok.astype(float), minlength=len(uniq))
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=b).astype(float)
    est = (w @ s) / np.maximum(w @ c, 1)
    return float(np.nanmean(vals)), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def pca_probe_scores(xtr, ytr, xte, n_comp=128, c=1.0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if ytr.min() == ytr.max():
        return np.full(len(xte), np.nan)
    pca = PCA(n_components=min(n_comp, xtr.shape[1], xtr.shape[0] - 1), random_state=0).fit(xtr)
    sc = StandardScaler().fit(pca.transform(xtr))
    clf = LogisticRegression(C=c, max_iter=300).fit(sc.transform(pca.transform(xtr)), ytr)
    return clf.decision_function(sc.transform(pca.transform(xte)))


def pca_reduce(xtr, xva, n_comp=256):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_comp, xtr.shape[0] - 1, xtr.shape[1]), random_state=0).fit(xtr)
    return pca.transform(xtr), pca.transform(xva)


def ridge_fit(x, y, lams, ep, folds=4, seed=0):
    """Ridge via the dual/normal equations with episode-grouped CV for lambda. x:(N,D) y:(N,P)."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mu_x, mu_y = x.mean(0), y.mean(0)
    xc, yc = x - mu_x, y - mu_y
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {u: i % folds for i, u in zip(range(len(uniq)), uniq[perm])}
    f = np.array([fold_of[e] for e in ep])
    best, best_l = np.inf, lams[0]
    for lam in lams:
        err = 0.0
        for k in range(folds):
            tr, va = f != k, f == k
            if va.sum() == 0 or tr.sum() == 0:
                continue
            a = xc[tr].T @ xc[tr] + lam * np.eye(xc.shape[1])
            w = np.linalg.solve(a, xc[tr].T @ yc[tr])
            err += ((xc[va] @ w - yc[va]) ** 2).sum()
        if err < best:
            best, best_l = err, lam
    a = xc.T @ xc + best_l * np.eye(xc.shape[1])
    w = np.linalg.solve(a, xc.T @ yc)
    return w, mu_x, mu_y, best_l


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3", "sv2"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
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
        labels = tr["labels"]
        H = int(tr["future_frame_offset"])
        R = results.setdefault(task, {"labels": labels, "H": H})
        groups = {"train": [tr], "val": [va], "pooled": [tr, va]}
        R["n"] = {g: int(sum(len(d["episode_index"]) for d in ds)) for g, ds in groups.items()}

        # ---------- A. retrieval ----------
        R["retrieval"] = {}
        for g, ds in groups.items():
            ep = np.concatenate([np.array([f"{d['split']}:{int(e)}" for e in d["episode_index"]]) for d in ds])
            fr = np.concatenate([d["frame_in_episode"].numpy() for d in ds])
            # episodes must be distinct across splits for the FN mask; encode split in the id (done above)
            dl2 = np.concatenate([d["d_l2"].numpy() for d in ds])
            subsets = {"all": np.ones(len(ep), bool), "top50": dl2 >= np.quantile(dl2, 0.5), "top25": dl2 >= np.quantile(dl2, 0.75)}
            R["retrieval"][g] = {}
            masks = {}
            for sub, m in subsets.items():
                n_m = int(m.sum())
                masks[sub] = {True: allowed_mask(ep[m], fr[m], H), False: ~np.eye(n_m, dtype=bool)}
            for k, label in enumerate(labels):
                hz = torch.cat([d["hz"][k] for d in ds]).float()
                hs = torch.cat([d["hzs"][k] for d in ds]).float()
                hc = torch.cat([d["hcur"][k] for d in ds]).float()
                entry = {}
                for qname, q in (("z", hz), ("cur", hc)):
                    for pname, fn, p256, sub in (("full", False, False, "all"), ("full_fn", True, False, "all"),
                                                 ("p256", False, True, "all"), ("p256_fn", True, True, "all"),
                                                 ("p256_fn_top50", True, True, "top50"), ("p256_fn_top25", True, True, "top25")):
                        m = subsets[sub]
                        mt = torch.as_tensor(m)
                        r = retrieval(q[mt], hs[mt], masks[sub][fn], p256)
                        cell = {}
                        for kk in KS:
                            mean, lo, hi = boot_ci(r[f"top{kk}"], ep[m], rng, args.bootstrap)
                            ch = float(np.nanmean(r[f"chance{kk}"]))
                            cell[f"top{kk}"] = {"mean": mean, "ci95": [lo, hi], "chance": ch, "x_chance": mean / ch if ch > 0 else None}
                        cell["n_queries"] = int(m.sum())
                        entry[f"{qname}/{pname}"] = cell
                R["retrieval"][g][label] = entry
        print(f"[{task}] A done", flush=True)

        # ---------- B. target-space drift ----------
        pooled = groups["pooled"]
        hs_all = torch.stack([torch.cat([d["hzs"][k] for d in pooled]).float() for k in range(len(labels))])  # (K,N,D)
        cos = {}
        for k, label in enumerate(labels):
            cos[label] = float((hs_all[k] * hs_all[-1]).sum(-1).mean())      # vs the last label (100%)
        sub = torch.randperm(hs_all.shape[1], generator=torch.Generator().manual_seed(0))[:2000]
        sims = [(hs_all[k][sub] @ hs_all[k][sub].t()).flatten() for k in range(len(labels))]
        rsa = {labels[k]: float(np.corrcoef(sims[k].numpy(), sims[-1].numpy())[0, 1]) for k in range(len(labels))}
        R["space_drift"] = {"mean_cos_zstar_vs_last": cos, "rsa_corr_vs_last": rsa, "last": labels[-1]}

        # ---------- C. fixed target space + D. contact detection + E. region ----------
        ep_tr = np.array([int(e) for e in tr["episode_index"]])
        ep_va = np.array([f"v{int(e)}" for e in va["episode_index"]])
        fr_va = va["frame_in_episode"].numpy()
        ftr, fva = tr["f_star"].float().numpy(), va["f_star"].float().numpy()
        ftr_n = ftr / np.linalg.norm(ftr, axis=1, keepdims=True).clip(1e-6)
        fva_n = fva / np.linalg.norm(fva, axis=1, keepdims=True).clip(1e-6)
        gtr, gva = tr["grid_tgt"].float().flatten(1).numpy(), va["grid_tgt"].float().flatten(1).numpy()
        hi = va["d_l2"].numpy() >= np.quantile(tr["d_l2"].numpy(), 0.5)
        R["fixed_target"], R["contact"], R["region"] = {}, {}, {}
        stats_tr = {"future_contact": tr["fut_mean_abs"].numpy(), "contact_change": tr["d_l2"].numpy()}
        stats_va = {"future_contact": va["fut_mean_abs"].numpy(), "contact_change": va["d_l2"].numpy()}
        for lbl_name in stats_tr:
            R["contact"][lbl_name] = {"quantiles": args.quantiles, "tau": [float(np.quantile(stats_tr[lbl_name], q)) for q in args.quantiles],
                                      "pos_rate_val": [float((stats_va[lbl_name] > np.quantile(stats_tr[lbl_name], q)).mean()) for q in args.quantiles],
                                      "auc": {}}
        lams = [1.0, 10.0, 100.0, 1000.0]
        allow_va = allowed_mask(ep_va, fr_va, H)
        for k, label in enumerate(labels):
            for feat in ("z", "cur"):
                key = "hz" if feat == "z" else "hcur"
                xtr, xva = tr[key][k].float().numpy(), va[key][k].float().numpy()
                xtr_p, xva_p = pca_reduce(xtr, xva)
                # C: ridge to fixed DINOv2 target, retrieval on val
                w, mx, my, lam = ridge_fit(xtr_p, ftr_n, lams, ep_tr)
                pred = torch.as_tensor((xva_p - mx) @ w + my)
                pred = torch.nn.functional.normalize(pred, dim=-1)
                r_fn = retrieval(pred, torch.as_tensor(fva_n), allow_va, True)
                R["fixed_target"].setdefault(label, {})[feat] = {
                    "lambda": lam, **{f"p256_fn_top{kk}": {"mean": float(np.nanmean(r_fn[f"top{kk}"])), "chance": float(np.nanmean(r_fn[f"chance{kk}"]))} for kk in KS}}
                # E: region grid
                wg, mxg, myg, _ = ridge_fit(xtr_p, gtr, lams, ep_tr)
                pg = (xva_p - mxg) @ wg + myg
                corr = np.array([np.corrcoef(pg[i], gva[i])[0, 1] if pg[i].std() > 0 and gva[i].std() > 0 else np.nan for i in range(len(pg))])
                mse = ((pg - gva) ** 2).mean()
                mse_mean = ((gtr.mean(0, keepdims=True) - gva) ** 2).mean()
                R["region"].setdefault(label, {})[feat] = {"corr_highchange": float(np.nanmean(corr[hi])), "corr_all": float(np.nanmean(corr)), "mse_skill_vs_train_mean": float(1 - mse / mse_mean)}
                # D: contact detection AUC
                for lbl_name in stats_tr:
                    aucs = []
                    for q in args.quantiles:
                        tau = np.quantile(stats_tr[lbl_name], q)
                        ytr, yva = (stats_tr[lbl_name] > tau).astype(int), (stats_va[lbl_name] > tau).astype(int)
                        aucs.append(roc_auc(yva, pca_probe_scores(xtr, ytr, xva)))
                    R["contact"][lbl_name]["auc"].setdefault(label, {})[feat] = aucs
        # raw persistence + recon-head magnitude scores
        for lbl_name in stats_tr:
            aucs_raw = []
            for q in args.quantiles:
                tau = np.quantile(stats_tr[lbl_name], q)
                aucs_raw.append(roc_auc((stats_va[lbl_name] > tau).astype(int), va["cur_mean_abs"].numpy()))
            R["contact"][lbl_name]["auc"]["raw_current_state"] = aucs_raw
            gp = va["grid_pred"].float().abs().flatten(2).mean(-1).numpy()   # (K, N)
            for k, label in enumerate(labels):
                if label == "base" or np.isnan(gp[k]).all():
                    continue
                aucs_rc = [roc_auc((stats_va[lbl_name] > np.quantile(stats_tr[lbl_name], q)).astype(int), gp[k]) for q in args.quantiles]
                R["contact"][lbl_name]["auc"].setdefault(label, {})["recon_head"] = aucs_rc
        print(f"[{task}] B-E done", flush=True)

    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
