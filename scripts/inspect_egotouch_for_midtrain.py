"""Survey of the EgoTouch 3-view@224 HDF5 build as a candidate mid-train (Stage-1 online) corpus.

Read-only. Reports what the mid-train loader would need from it:
  * layout of one file (every dataset: shape, dtype, chunks, compression; group attrs);
  * per sequence: frames, fps, size on disk, task, wrist-quality flags, pressure scale attrs
    (tactile_max_*), grid value range, fraction of frames with contact, frames usable as
    (current, future=+FUTURE) pairs;
  * the split files next to the data (train/val/test_seen/test_unseen sizes, overlap with built files);
  * which 21x21 cells are ever non-zero per hand (the active sensor footprint), for the tactile adapter;
  * time to read one random frame of the 3 views + both grids (online-loader cost).

  python scripts/inspect_egotouch_for_midtrain.py DATA_ROOT OUT_JSON [--workers 8] [--future 50]
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

VIEWS = ("chest_color", "left_color", "right_color")
GRIDS = ("left_pressure_grid", "right_pressure_grid")


def layout(path: Path) -> dict:
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = dict(shape=list(obj.shape), dtype=str(obj.dtype), chunks=list(obj.chunks or []),
                                 compression=obj.compression)
            else:
                attrs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in obj.attrs.items()}
                if attrs:
                    out[name + "/@attrs"] = {k: (str(v)[:200] if not isinstance(v, (int, float, bool, list)) else v)
                                             for k, v in attrs.items()}
        f.visititems(visit)
    return out


def per_seq(args) -> dict:
    path, future = args
    rec = dict(path=str(path), bytes=path.stat().st_size)
    try:
        with h5py.File(path, "r") as f:
            m = f["metadata"].attrs
            rec.update(task=str(m.get("task_name", "")), fps=float(m.get("fps", -1)),
                       left_wrist_issue=bool(m.get("left_wrist_has_issue", False)),
                       right_wrist_issue=bool(m.get("right_wrist_has_issue", False)))
            n = {v: (f["images"][v].shape[0] if v in f["images"] else 0) for v in VIEWS}
            rec["frames_per_view"] = n
            p = f["pressure"]
            pa = p.attrs
            rec["separate_normalization"] = bool(pa.get("separate_normalization", False))
            rec["tactile_max"] = [float(pa.get("tactile_max_left", np.nan)), float(pa.get("tactile_max_right", np.nan))]
            fp = {}
            for g in GRIDS:
                a = p[g][...].reshape(p[g].shape[0], -1)
                bad = ~np.isfinite(a)
                fin = np.where(bad, 0.0, a)
                pos = fin[fin > 0]
                rng = np.random.default_rng(len(a))
                fp[g] = dict(n=int(a.shape[0]), nan_frame_frac=float(bad.any(1).mean()), nan_cell_frac=float(bad.mean()),
                             max=float(fin.max()), min=float(fin.min()),
                             pos_q=[float(x) for x in np.percentile(pos, [50, 90, 99, 99.9])] if len(pos) else [],
                             pos_sample=rng.choice(pos, size=min(2000, len(pos)), replace=False).tolist() if len(pos) else [],
                             contact_frac=float((fin.max(1) > 0).mean()),
                             active=(fin > 0).any(0).astype(np.uint8).tolist())
            rec["grids"] = fp
            nmin = min(list(n.values()) + [fp[g]["n"] for g in GRIDS])
            rec["frames"] = nmin
            rec["pairs"] = max(0, nmin - future)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    return rec


def _task_spread(ok: list[dict]) -> dict:
    """Per-task median of the per-seq q99.9 of positive values (both hands): is force scale task-dependent?"""
    by = {}
    for s in ok:
        q = [s["grids"][g]["pos_q"][3] for g in GRIDS if s["grids"][g]["pos_q"]]
        if q:
            by.setdefault(s["task"], []).append(max(q))
    med = np.array([np.median(v) for v in by.values() if len(v) >= 3])
    return dict(n_tasks_ge3_seqs=int(len(med)), task_median_q999_q10_50_90=[float(x) for x in np.percentile(med, [10, 50, 90])] if len(med) else [])


def read_cost(paths: list[Path], rng, k: int = 30) -> dict:
    ts = []
    for path in rng.choice(paths, size=min(k, len(paths)), replace=False):
        with h5py.File(path, "r") as f:
            n = f["images/chest_color"].shape[0]
            t = int(rng.integers(n))
            t0 = time.perf_counter()
            for v in VIEWS:
                if v in f["images"]:
                    _ = f["images"][v][t]
            for g in GRIDS:
                _ = f["pressure"][g][t]
            ts.append(time.perf_counter() - t0)
    return dict(n=len(ts), median_s=float(np.median(ts)), p90_s=float(np.percentile(ts, 90)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--future", type=int, default=50)
    ap.add_argument("--no-layout", action="store_true", help="skip printing the layout lines (already seen)")
    args = ap.parse_args()
    files = sorted(args.root.glob("*/*/*.hdf5"))
    print(f"{len(files)} hdf5 files under {args.root}", flush=True)
    res = dict(root=str(args.root), n_files=len(files), layout_example=layout(files[0]))
    if not args.no_layout:
        for k, v in res["layout_example"].items():
            print(f"EGT|L {k} {json.dumps(v)[:300]}", flush=True)

    with Pool(args.workers) as pool:
        seqs = pool.map(per_seq, [(p, args.future) for p in files], chunksize=8)
    ok = [s for s in seqs if "error" not in s]
    bad = [s for s in seqs if "error" in s]
    fps = sorted({s["fps"] for s in ok})
    frames = np.array([s["frames"] for s in ok])
    tm = np.array([s["tactile_max"] for s in ok], float)
    mismatch = sum(1 for s in ok if len(set(s["frames_per_view"].values()) | {s["grids"][g]["n"] for g in GRIDS}) > 1)
    act = {g: np.array([s["grids"][g]["active"] for s in ok]).any(0) for g in GRIDS}
    res["summary"] = dict(
        ok=len(ok), errors=len(bad), first_errors=[(s["path"], s["error"]) for s in bad[:5]],
        fps_values=fps, total_bytes=int(sum(s["bytes"] for s in seqs)),
        frames_total=int(frames.sum()), hours_total=float(frames.sum() / (fps[0] if fps else 30) / 3600),
        frames_min_med_max=[int(frames.min()), int(np.median(frames)), int(frames.max())],
        seqs_shorter_than_future_plus1=int((frames <= args.future).sum()),
        pairs_total=int(sum(s["pairs"] for s in ok)),
        frame_count_mismatch_seqs=mismatch,
        wrist_issue_seqs=dict(left=sum(s["left_wrist_issue"] for s in ok), right=sum(s["right_wrist_issue"] for s in ok)),
        separate_normalization_seqs=sum(s["separate_normalization"] for s in ok),
        tactile_max_left_min_med_max=np.nanpercentile(tm[:, 0], [0, 50, 100]).tolist(),
        tactile_max_right_min_med_max=np.nanpercentile(tm[:, 1], [0, 50, 100]).tolist(),
        grid_max_overall={g: float(max(s["grids"][g]["max"] for s in ok)) for g in GRIDS},
        grid_min_overall={g: float(min(s["grids"][g]["min"] for s in ok)) for g in GRIDS},
        contact_frac_median={g: float(np.median([s["grids"][g]["contact_frac"] for s in ok])) for g in GRIDS},
        active_cells={g: int(act[g].sum()) for g in GRIDS},
        active_mask_21x21={g: act[g].reshape(21, 21).astype(int).tolist() for g in GRIDS},
        n_tasks=len({s["task"] for s in ok}),
        nan_frame_frac_median={g: float(np.median([s["grids"][g]["nan_frame_frac"] for s in ok])) for g in GRIDS},
        seqs_with_any_nan={g: int(sum(s["grids"][g]["nan_frame_frac"] > 0 for s in ok)) for g in GRIDS},
        seqs_all_nan={g: int(sum(s["grids"][g]["nan_frame_frac"] >= 1 for s in ok)) for g in GRIDS},
        nan_cell_frac_mean={g: float(np.mean([s["grids"][g]["nan_cell_frac"] for s in ok])) for g in GRIDS},
        seqs_no_contact={g: int(sum(s["grids"][g]["contact_frac"] == 0 for s in ok)) for g in GRIDS},
        pos_value_q50_90_99_999_pooled={g: [float(x) for x in np.percentile(np.concatenate(
            [s["grids"][g]["pos_sample"] for s in ok if s["grids"][g]["pos_sample"]]), [50, 90, 99, 99.9])] for g in GRIDS},
        seq_max_q10_50_90={g: [float(x) for x in np.percentile([s["grids"][g]["max"] for s in ok], [10, 50, 90])] for g in GRIDS},
        seqs_max_le_1={g: int(sum(s["grids"][g]["max"] <= 1.0 for s in ok)) for g in GRIDS},
        task_seq_max_spread=_task_spread(ok),
    )
    splits = {}
    built = {str(p.relative_to(args.root)) for p in files}
    for sp in sorted(args.root.glob("split*.json")):
        d = json.loads(sp.read_text())
        info = {}
        for k, v in d.items():
            if isinstance(v, list):
                rel = set()
                for x in v:
                    parts = Path(str(x)).parts
                    if args.root.name in parts:
                        rel.add(str(Path(*parts[parts.index(args.root.name) + 1:])))
                    else:
                        rel.add(str(x))
                info[k] = dict(n=len(v), in_built=len(rel & built))
        splits[sp.name] = info
    res["splits"] = splits
    res["read_cost_one_frame_3views_2grids"] = read_cost(files, np.random.default_rng(0))
    res["per_seq"] = [{k: v for k, v in s.items() if k != "grids"} | (
        {"grids": {g: {kk: vv for kk, vv in s["grids"][g].items() if kk not in ("active", "pos_sample")} for g in GRIDS}}
        if "grids" in s else {}) for s in seqs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res))
    for k, v in res["summary"].items():
        if k == "active_mask_21x21":
            if not args.no_layout:
                for g, m in v.items():
                    print(f"EGT|S active {g} " + "/".join("".join(map(str, r)) for r in m), flush=True)
        else:
            print(f"EGT|S {k} {json.dumps(v)}", flush=True)
    for k, v in res["splits"].items():
        print(f"EGT|P {k} {json.dumps(v)}", flush=True)
    print(f"EGT|R {json.dumps(res['read_cost_one_frame_3views_2grids'])}", flush=True)
    print("EGOTOUCH_INSPECT_DONE")


if __name__ == "__main__":
    main()
