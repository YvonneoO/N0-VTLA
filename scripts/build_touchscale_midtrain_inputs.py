"""Inputs for mid-train (Stage-1 online) on the TouchScale corpus (touchscale_data/, one episode.h5 per episode).

Writes into OUT_DIR:
  manifest.json   {"raw_root": ROOT, "splits": {"all": [...], "old100": [...]}, ...} for VTLA_ITW_EPISODE_LIST_JSON.
                  all    = every usable episode (index: complete & idx_ok).
                  old100 = every usable episode whose DATE is one of the dates of the old data-scaling manifest
                           (docs/artifacts/itw_scaling_splits.json, 20/40/60/80pct keys) -- "the old 100%" by date.
  index_<key>.json  per-split sample-index cache (episode path, aligned frame count) for VTLA_ITW_INDEX_CACHE,
                  so training does not re-align ~60k episodes on every rank and every resumed segment.
  normalization_pad30_per_task_scale_touchscale.json
                  the per-task-scale normalization with task_scale entries ADDED for tasks the original table
                  lacks (tasks recorded after it was fit). Existing entries, baselines, default_scale untouched.
                  New entries follow tacWAM's audit recipe: q99.9 of the positive residual (raw - per-pad
                  baseline) over sampled frames of the task's recordings, only for tasks with >= 5 recordings
                  (others keep falling back to default_scale). The recipe is first re-applied to existing tasks
                  and the ratio to their table value is reported, as a check that it reproduces the table.

Usage: python scripts/build_touchscale_midtrain_inputs.py ROOT NORM_JSON OLD_MANIFEST OUT_DIR [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import itw_pressure as ip  # noqa: E402

MIN_RECORDINGS = 5
QUANTILE = 99.9


def _frames(ep: str) -> tuple[str, int | None, str]:
    try:
        m, _ = ip.aligned_timeline_h5(Path(ep))
        n = len(m["master_timestamp_ns"])
        return ep, (n if n >= 51 else None), "" if n >= 51 else f"only {n} frames"
    except Exception as e:  # noqa: BLE001
        return ep, None, str(e)[:120]


def _residuals(args) -> np.ndarray:
    ep, baseline, frames_per_ep, seed = args
    rng = np.random.default_rng(seed)
    out = []
    with h5py.File(Path(ep) / "episode.h5", "r") as f:
        n = f["obs/tactile/raw/left/t_epoch"].shape[0]
        take = np.sort(rng.choice(n, size=min(frames_per_ep, n), replace=False))
        for hi, hand in enumerate(ip.HANDS):
            for pi, pad in enumerate(ip.PAD_IDS):
                r = f[f"obs/tactile/raw/{hand}/pad_{pad}"][take].astype(np.float64) - baseline[hi * 15 + pi]
                out.append(r[r > 0])
    return np.concatenate(out)


def task_scale(pool, eps: list[str], baseline, rng, max_eps: int, frames_per_ep: int) -> float:
    pick = list(rng.choice(eps, size=min(max_eps, len(eps)), replace=False))
    vals = np.concatenate(pool.map(_residuals, [(e, baseline, frames_per_ep, i) for i, e in enumerate(pick)]))
    return float(np.percentile(vals, QUANTILE)) if len(vals) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("norm", type=Path)
    ap.add_argument("old_manifest", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-eps-per-task", type=int, default=20)
    ap.add_argument("--frames-per-ep", type=int, default=40)
    ap.add_argument("--n-check", type=int, default=12, help="existing tasks re-fitted to check the recipe")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(args.root / "index/episode_index.csv", low_memory=False)
    use = idx[idx.complete & idx.idx_ok.eq(True)].sort_values(["day", "uuid"]).reset_index(drop=True)
    old = json.loads(args.old_manifest.read_text())
    old_dates = sorted({p.split("/")[0] for v in old["splits"].values() for p in v})
    present = sorted(set(old_dates) & set(use.day))
    splits = {"all": use.rel_dir.tolist(), "old100": use[use.day.isin(old_dates)].rel_dir.tolist()}
    hours = {"all": float(use.duration_s.sum() / 3600), "old100": float(use[use.day.isin(old_dates)].duration_s.sum() / 3600)}
    manifest = {"raw_root": str(args.root), "splits": splits, "hours": hours,
                "old100_dates_requested": old_dates, "old100_dates_present": present,
                "old100_dates_missing": [d for d in old_dates if d not in present],
                "note": "old100 = all usable TouchScale episodes on the dates of docs/artifacts/itw_scaling_splits.json"}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest))
    print(f"manifest: all {len(splits['all'])} eps {hours['all']:.1f} h | old100 {len(splits['old100'])} eps "
          f"{hours['old100']:.1f} h on {len(present)}/{len(old_dates)} dates (missing {manifest['old100_dates_missing']})", flush=True)

    norm = ip.load_normalization(args.norm)
    baseline = np.asarray(norm["normal_baseline"], np.float64)
    table = dict(norm["task_scale"])
    rng = np.random.default_rng(0)
    with Pool(args.workers) as pool:
        # index cache for the full corpus; per-split caches are subsets in the same order
        results = pool.map(_frames, [str(args.root / r) for r in splits["all"]], chunksize=64)
        n_of = {ep: n for ep, n, _ in results}
        bad = [(ep, why) for ep, n, why in results if n is None]
        print(f"alignment: {len(results) - len(bad)} ok, {len(bad)} skipped; first reasons {bad[:3]}", flush=True)
        for key, rels in splits.items():
            rows = [[str(args.root / r), n_of[str(args.root / r)]] for r in rels if n_of[str(args.root / r)] is not None]
            (args.out_dir / f"index_{key}.json").write_text(json.dumps({"episodes": rows}))
            print(f"index_{key}.json: {len(rows)} episodes, {sum(r[1] for r in rows)} samples", flush=True)

        by_task = use.groupby("task_name").rel_dir.apply(lambda s: [str(args.root / r) for r in s]).to_dict()
        existing = [t for t in by_task if t in table and len(by_task[t]) >= MIN_RECORDINGS]
        checks = []
        for t in list(rng.choice(existing, size=min(args.n_check, len(existing)), replace=False)):
            s = task_scale(pool, by_task[t], baseline, rng, args.max_eps_per_task, args.frames_per_ep)
            checks.append(s / table[t])
            print(f"  check existing task: refit/table = {s / table[t]:.3f}", flush=True)
        print(f"recipe check on {len(checks)} existing tasks: refit/table median {np.median(checks):.3f} "
              f"min {np.min(checks):.3f} max {np.max(checks):.3f}", flush=True)
        added, too_few = {}, []
        for t, eps in sorted(by_task.items()):
            if t in table:
                continue
            if len(eps) < MIN_RECORDINGS:
                too_few.append(t)
                continue
            added[t] = task_scale(pool, eps, baseline, rng, args.max_eps_per_task, args.frames_per_ep)
    new = dict(norm)
    new["task_scale"] = {**table, **{k: v for k, v in added.items() if np.isfinite(v) and v > 0}}
    new["touchscale_extension"] = {"source_normalization": str(args.norm), "recipe": f"q{QUANTILE} positive residual",
                                   "min_recordings": MIN_RECORDINGS, "added_tasks": len(added),
                                   "tasks_too_few_recordings": len(too_few), "recipe_check_refit_over_table": checks}
    out = args.out_dir / "normalization_pad30_per_task_scale_touchscale.json"
    out.write_text(json.dumps(new, ensure_ascii=False, indent=1))
    cov = use.task_name.isin(new["task_scale"]).mean()
    print(f"normalization: {len(table)} -> {len(new['task_scale'])} task scales (+{len(added)}; {len(too_few)} tasks "
          f"under {MIN_RECORDINGS} recordings stay on default_scale); episode coverage {cov:.1%}; added scale "
          f"range {min(added.values()):.4g}..{max(added.values()):.4g}" if added else "no tasks added", flush=True)
    print("TOUCHSCALE_INPUTS_DONE")


if __name__ == "__main__":
    main()
