#!/usr/bin/env python3
"""Smoke test of EgoTouchOnlineTactileDataset on the real HDF5 build (CPU).

Checks, on a seeded random subset of sequences:
  * every sample decodes; keys, shapes, dtypes match what the TouchScale loader emits;
  * the region map: share of the grid's pressure mass that lands in a Tujian pad slot (only the dropped pinky
    base should be missing), and that no active cell is left unmapped except the pinky base;
  * how often frame 0 (the tactile baseline) already shows contact;
  * time per sample; a PNG contact sheet (3 RGB views + left/right tactile [baseline, current, future]).
Summary lines start with "EGS|".

  python scripts/smoke_test_egotouch_loader.py MANIFEST KEY INDEX_CACHE OUT_PNG [--sequences 40] [--samples 120]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n0vtla.training import egotouch_online_dataset as E  # noqa: E402


def region_stats(path: Path) -> dict:
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        for hand in ("left", "right"):
            g = f["pressure"][f"{hand}_pressure_grid"][...]
            active = np.isfinite(g).any(0)
            g = np.nan_to_num(g, nan=0.0)
            mapped = np.zeros((21, 21), bool)
            for pad in E.REGIONS_RIGHT:
                rs, cs = E.region(hand, pad)
                mapped[rs, cs] = True
            dropped = np.zeros((21, 21), bool)
            c0, c1 = (17, 19) if hand == "right" else (1, 3)
            dropped[9:12, c0:c1 + 1] = True
            total = float(g.sum())
            out[hand] = dict(mass_mapped=float(g[:, mapped].sum()) / total if total else np.nan,
                             mass_pinky_base=float(g[:, dropped].sum()) / total if total else np.nan,
                             unmapped_active=int((active & ~mapped & ~dropped).sum()),
                             mapped_inactive=int((mapped & ~active).sum()),
                             frame0_max=float(g[0].max()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("key")
    ap.add_argument("index_cache", type=Path)
    ap.add_argument("out_png", type=Path)
    ap.add_argument("--sequences", type=int, default=40)
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = json.loads(args.manifest.read_text())
    root = Path(m["raw_root"])
    paths = [root / r for r in m["splits"][args.key]]
    print(f"EGS|manifest {args.key}: {len(paths)} sequences", flush=True)
    t0 = time.time()
    full = E.EgoTouchOnlineTactileDataset(paths, index_cache=args.index_cache)
    print(f"EGS|full index from cache: {len(full.episodes)} sequences, {len(full)} samples in {time.time() - t0:.1f}s",
          flush=True)

    rng = random.Random(args.seed)
    sub = sorted(rng.sample(paths, min(args.sequences, len(paths))))
    ds = E.EgoTouchOnlineTactileDataset(sub)
    rs = [region_stats(p) for p in ds.episodes]
    for hand in ("left", "right"):
        v = [r[hand] for r in rs]
        print(f"EGS|regions {hand}: mass in pad slots median {np.nanmedian([x['mass_mapped'] for x in v]):.3f} "
              f"min {np.nanmin([x['mass_mapped'] for x in v]):.3f}; pinky base median "
              f"{np.nanmedian([x['mass_pinky_base'] for x in v]):.3f}; unmapped active cells max "
              f"{max(x['unmapped_active'] for x in v)}; mapped-but-no-sensor cells {v[0]['mapped_inactive']}; "
              f"frame0 max>0.2 in {np.mean([x['frame0_max'] > 0.2 for x in v]):.0%} of sequences", flush=True)

    idxs = rng.sample(range(len(ds)), min(args.samples, len(ds)))
    rng.shuffle(idxs)
    ok, err, first, keep = 0, 0, None, []
    t0 = time.time()
    for i in idxs:
        try:
            s = ds[i]
        except Exception:  # noqa: BLE001
            err += 1
            traceback.print_exc()
            continue
        ok += 1
        if first is None:
            first = s
            for k, v in s.items():
                print(f"EGS|key {k} {getattr(v, 'shape', v)} {getattr(v, 'dtype', '')}", flush=True)
        if len(keep) < 6:
            keep.append(s)
    dt = (time.time() - t0) / max(ok, 1)
    print(f"EGS|samples {ok} ok, {err} errors, {dt:.3f}s/sample (random access, cold per-sequence cache)", flush=True)

    tl = "observation.image.left_wrist_left_tactile"
    tr = "observation.image.right_wrist_right_tactile"
    bad_shape = [k for k in (tl, tr) if first is None or first[k].shape != (3, 224, 224, 3) or first[k].dtype != np.uint8]
    bad_rgb = [k for k in E.RGB_VIEWS if first is None or first[k].shape != (224, 224, 3)]
    print(f"EGS|shape check: tactile bad {bad_shape}, rgb bad {bad_rgb}; letterbox rows black "
          f"{int((first['observation.image.third_view'].max(axis=(1, 2)) == 0).sum()) if first else -1}/224", flush=True)

    import cv2

    rows = []
    for s in keep:
        tiles = [s[k] for k in E.RGB_VIEWS] + [s[tl][j] for j in range(3)] + [s[tr][j] for j in range(3)]
        rows.append(np.concatenate(tiles, axis=1))
    if rows:
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out_png), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
        print(f"EGS|contact sheet {args.out_png} (cols: chest, left wrist, right wrist, L tac b/c/f, R tac b/c/f)")
    good = err == 0 and not bad_shape and not bad_rgb
    print("EGOTOUCH_SMOKE_OK" if good else "EGOTOUCH_SMOKE_FAIL", flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
