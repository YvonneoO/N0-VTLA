"""Inputs for mid-train (Stage-1 online) on a raw tacWAM-layout ITW tree (<root>/<date>/<uuid>/, as synced from S3).

Writes into OUT_DIR:
  manifest.json      {"raw_root": ROOT, "splits": {KEY: [<date>/<uuid>, ...]}, ...} for VTLA_ITW_EPISODE_LIST_JSON.
                     KEY (default old100) = every episode dir on DATES that passes the loader's alignment checks.
  index_<KEY>.json   [[abs path, aligned frames], ...] for VTLA_ITW_INDEX_CACHE, so the 8 ranks and every resumed
                     segment do not re-align ~40k episodes.
DATES default: the dates of docs/artifacts/itw_scaling_splits.json (the old 100%, itw07-23..itw09-15).
Also prints the per-task-scale coverage of NORM (fraction of episodes whose task has its own scale).
--episode-list MANIFEST.json [--list-key K]: keep only episodes whose uuid (last path component) is in that manifest's
  splits[K] (e.g. the zzh TouchScale old100 list, to continue a zzh run on this copy); the dates then default to the
  dates of those episodes, and the number of listed uuids not found here is reported.

  python scripts/build_itw_raw_midtrain_inputs.py ROOT OUT_DIR --norm NORM.json [--key old100] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from n0vtla.training import itw_online_dataset as D  # noqa: E402

MIN_FRAMES = 51
OLD_MANIFEST = Path(__file__).resolve().parents[1] / "docs/artifacts/itw_scaling_splits.json"


def frames(ep: str) -> tuple[str, int | None, str]:
    try:
        m, _ = D.episode_timeline(Path(ep))
        n = len(m["master_timestamp_ns"])
        return ep, (n if n >= MIN_FRAMES else None), "" if n >= MIN_FRAMES else f"only {n} frames"
    except Exception as e:  # noqa: BLE001
        return ep, None, f"{type(e).__name__}: {str(e)[:100]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--norm", type=Path, required=True)
    ap.add_argument("--key", default="old100")
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--episode-list", type=Path, default=None)
    ap.add_argument("--list-key", default="old100")
    args = ap.parse_args()
    keep = None
    if args.episode_list is not None:
        listed = json.loads(args.episode_list.read_text())["splits"][args.list_key]
        keep = {Path(r).name for r in listed}
        list_dates = sorted({d for r in listed for d in Path(r).parts if d.startswith("itw")})
        print(f"ITB|episode list {args.episode_list}[{args.list_key}]: {len(listed)} entries, {len(keep)} uuids, "
              f"dates {list_dates}", flush=True)
    old = json.loads(OLD_MANIFEST.read_text())
    dates = args.dates or (list_dates if keep is not None and list_dates else
                           sorted({p.split("/")[0] for v in old["splits"].values() for p in v}))
    missing_dates = [d for d in dates if not (args.root / d).is_dir()]
    eps = D.list_episode_dirs(args.root, date_dirs=[d for d in dates if d not in missing_dates])
    if keep is not None:
        here = {e.name for e in eps}
        eps = [e for e in eps if e.name in keep]
        print(f"ITB|kept {len(eps)} episode dirs from the list; {len(keep - here)} listed uuids not found here", flush=True)
    print(f"ITB|{len(dates)} dates ({len(missing_dates)} missing: {missing_dates}), {len(eps)} episode dirs", flush=True)
    with Pool(args.workers) as pool:
        res = pool.map(frames, [str(e) for e in eps], chunksize=32)
    ok = [(ep, n) for ep, n, _ in res if n is not None]
    bad = [(ep, why) for ep, n, why in res if n is None]
    reasons = Counter(why.split(":")[0] if ":" in why else why.split(" ")[0] for _, why in bad)
    print(f"ITB|alignment: {len(ok)} ok, {len(bad)} skipped; reasons {reasons.most_common(6)}; e.g. {bad[:2]}", flush=True)
    rel = [str(Path(ep).relative_to(args.root)) for ep, _ in ok]
    frames_total = sum(n for _, n in ok)
    per_date = Counter(r.split("/")[0] for r in rel)
    cov = D.task_scale_coverage([Path(ep) for ep, _ in ok], json.loads(args.norm.read_text()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"index_{args.key}.json").write_text(json.dumps({"episodes": [[ep, n] for ep, n in ok]}))
    manifest = {"raw_root": str(args.root), "splits": {args.key: rel}, "dates": dates, "missing_dates": missing_dates,
                "summary": {"episodes": len(rel), "frames": frames_total, "hours": frames_total / 30 / 3600,
                            "skipped": len(bad), "per_date": dict(per_date), "task_scale_coverage": cov},
                "note": "raw tacWAM-layout ITW synced from S3 (no depth); key = every aligned episode on these dates"
                        + (f", restricted to the uuids of {args.episode_list}[{args.list_key}]" if keep is not None else "")}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest))
    print(f"ITB|{args.key}: {len(rel)} episodes, {frames_total} samples = {frames_total / 30 / 3600:.1f} h; "
          f"task-scale coverage {cov}", flush=True)
    print("ITW_RAW_INPUTS_DONE", flush=True)


if __name__ == "__main__":
    main()
