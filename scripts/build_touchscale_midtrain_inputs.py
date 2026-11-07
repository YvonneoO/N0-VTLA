"""Inputs for mid-train (Stage-1 online) on the TouchScale corpus (touchscale_data/, one episode.h5 per episode).

Writes into OUT_DIR:
  manifest.json   {"raw_root": ROOT, "splits": {"all": [...], "old100": [...]}, ...} for VTLA_ITW_EPISODE_LIST_JSON.
                  all    = every usable episode (index: complete & idx_ok).
                  old100 = every usable episode whose DATE is one of the dates of the old data-scaling manifest
                           (docs/artifacts/itw_scaling_splits.json, 20/40/60/80pct keys) -- "the old 100%" by date.
  index_<key>.json  per-split sample-index cache (episode path, aligned frame count) for VTLA_ITW_INDEX_CACHE,
                  so training does not re-align ~60k episodes on every rank and every resumed segment.
Per-task tactile scales for tasks the original table lacks: scripts/extend_task_scale_touchscale.py.

Usage: python scripts/build_touchscale_midtrain_inputs.py ROOT OLD_MANIFEST OUT_DIR [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import itw_pressure as ip  # noqa: E402

def _frames(ep: str) -> tuple[str, int | None, str]:
    try:
        m, _ = ip.aligned_timeline_h5(Path(ep))
        n = len(m["master_timestamp_ns"])
        return ep, (n if n >= 51 else None), "" if n >= 51 else f"only {n} frames"
    except Exception as e:  # noqa: BLE001
        return ep, None, str(e)[:120]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("old_manifest", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--workers", type=int, default=16)
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

    print("TOUCHSCALE_INPUTS_DONE")


if __name__ == "__main__":
    main()
