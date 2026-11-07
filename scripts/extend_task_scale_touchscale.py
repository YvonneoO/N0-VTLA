"""Add per-task tactile scales for TouchScale tasks the original table lacks, with tacWAM's OWN algorithm.

The original table (normalization_pad30_per_task_scale.json) was produced by tacWAM
scripts/cosmos3/audit_tactile_normalization_by_task.py + build_per_task_tactile_normalization.py
(github haohww/tacWAM, commit f957bc0). This script runs the same procedure; the only change is the
reader (TouchScale stores glove data in episode.h5 obs/tactile/raw/<hand>/pad_<id> instead of
<hand>_hand_data.npz tactile_<id>; same values, same units):
  - recordings sorted by uuid, one rng = default_rng(42) drawn in that order;
  - per recording and hand: 16 frame indices rng.integers(0, n) (with replacement, sorted);
  - residual = raw - normal_baseline[pad], non-finite dropped, pooled over both hands and all 15 pads;
  - task scale = max(q99.9 of the positive residuals, 1e-6); tasks with < 5 loaded recordings or no
    positive residual are left out (they fall back to default_scale).
Task name = task_info.json["name"], the field training looks the table up with.

Existing table entries, baselines, default_scale are kept unchanged (scales are only ADDED). The same
procedure is also applied to the existing tasks and refit/table is reported; it differs from 1 only
because the TouchScale recording set and the random draws differ from the original fit.

Usage: python scripts/extend_task_scale_touchscale.py ROOT NORM_JSON OUT_JSON [--workers 16]
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HANDS = ("left", "right")
PAD_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
SAMPLES_PER_RECORDING = 16
SEED = 42
SCALE_QUANTILE = 99.9
MIN_SCALE = 1e-6
MIN_RECORDINGS_PER_TASK = 5


def _task_name(ep: Path) -> str:
    try:
        return str(json.loads((ep / "task_info.json").read_text()).get("name", ""))
    except Exception:  # noqa: BLE001  (same as tacWAM: unreadable -> "")
        return ""


def _lengths(ep: str) -> tuple[str, str, list[int] | None]:
    try:
        with h5py.File(Path(ep) / "episode.h5", "r") as f:
            return ep, _task_name(Path(ep)), [int(f[f"obs/tactile/raw/{h}/t_epoch"].shape[0]) for h in HANDS]
    except Exception:  # noqa: BLE001
        return ep, _task_name(Path(ep)), None


def _residuals(args) -> np.ndarray | None:
    ep, idx_per_hand, baseline = args
    out = []
    try:
        with h5py.File(Path(ep) / "episode.h5", "r") as f:
            for hi, hand in enumerate(HANDS):
                idx = idx_per_hand[hi]
                if idx.size == 0:
                    continue
                for slot, pad in enumerate(PAD_IDS):
                    raw = np.asarray(f[f"obs/tactile/raw/{hand}/pad_{pad}"][:][idx], np.float32)
                    v = raw[np.isfinite(raw)].reshape(-1).astype(np.float64, copy=False)
                    if v.size:
                        out.append(v - baseline[hi * len(PAD_IDS) + slot])
    except Exception:  # noqa: BLE001
        return None
    return np.concatenate(out) if out else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("norm", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    idx = pd.read_csv(args.root / "index/episode_index.csv", low_memory=False)
    use = idx[idx.complete & idx.idx_ok.eq(True)]
    eps = sorted((str(args.root / r) for r in use.rel_dir), key=lambda p: Path(p).name)   # sorted by uuid
    norm = json.loads(args.norm.read_text())
    baseline = np.asarray(norm["normal_baseline"], np.float64)
    table = dict(norm["task_scale"])

    with Pool(args.workers) as pool:
        lens = pool.map(_lengths, eps, chunksize=64)
        rng = np.random.default_rng(SEED)
        jobs, names = [], []
        for ep, name, n in lens:          # draws in uuid order, exactly like tacWAM's sequential loop
            if n is None:
                continue
            per_hand = [np.sort(rng.integers(0, k, size=min(SAMPLES_PER_RECORDING, k), dtype=np.int64)) if k > 0
                        else np.zeros((0,), np.int64) for k in n]
            jobs.append((ep, per_hand, baseline))
            names.append(name)
        res = pool.map(_residuals, jobs, chunksize=16)

    by_task: dict[str, list[np.ndarray]] = {}
    n_rec: dict[str, int] = {}
    for name, r in zip(names, res):
        if r is None:
            continue
        by_task.setdefault(name, []).append(r)
        n_rec[name] = n_rec.get(name, 0) + 1
    print(f"recordings: {len(eps)} requested, {sum(n_rec.values())} loaded, {len(by_task)} task names", flush=True)

    scale = {}
    for name, chunks in by_task.items():
        pos = np.concatenate(chunks)
        pos = pos[pos > 0.0]
        if n_rec[name] >= MIN_RECORDINGS_PER_TASK and pos.size:
            scale[name] = max(float(np.percentile(pos, SCALE_QUANTILE)), MIN_SCALE)

    ratio = np.array([scale[t] / table[t] for t in scale if t in table])
    print(f"same procedure on {ratio.size} existing tasks: refit/table median {np.median(ratio):.3f}, "
          f"p10 {np.percentile(ratio, 10):.3f}, p90 {np.percentile(ratio, 90):.3f}, "
          f"min {ratio.min():.3f}, max {ratio.max():.3f}", flush=True)
    added = {t: s for t, s in scale.items() if t not in table}
    too_few = sorted(t for t in by_task if t not in table and t not in added)
    new = dict(norm)
    new["task_scale"] = {**table, **added}
    new["touchscale_extension"] = {
        "source_normalization": str(args.norm),
        "algorithm": "tacWAM audit_tactile_normalization_by_task.py + build_per_task_tactile_normalization.py "
                     "(haohww/tacWAM f957bc0), episode.h5 reader",
        "samples_per_recording": SAMPLES_PER_RECORDING, "seed": SEED, "scale_quantile": SCALE_QUANTILE,
        "min_recordings_per_task": MIN_RECORDINGS_PER_TASK, "added_tasks": len(added),
        "tasks_too_few_recordings": len(too_few),
        "refit_over_table_existing": {t: scale[t] / table[t] for t in scale if t in table},
    }
    args.out.write_text(json.dumps(new, ensure_ascii=False, indent=1))
    ep_names = pd.Series(names)
    print(f"task scales {len(table)} -> {len(new['task_scale'])} (+{len(added)}); {len(too_few)} new tasks under "
          f"{MIN_RECORDINGS_PER_TASK} recordings stay on default_scale {norm['default_scale']:.4g}; episode coverage "
          f"{ep_names.isin(new['task_scale']).mean():.1%}; added range "
          f"{min(added.values()):.4g}..{max(added.values()):.4g}; table range "
          f"{min(table.values()):.4g}..{max(table.values()):.4g}", flush=True)
    print("TASK_SCALE_DONE")


if __name__ == "__main__":
    main()
