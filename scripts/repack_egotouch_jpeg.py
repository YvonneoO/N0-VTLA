"""Repack the EgoTouch 3-view@224 HDF5 build for fast random-frame reads in mid-train.

The source stores each view as gzip chunks of (104 frames, 28, 28, 1): reading ONE frame decompresses 104 frames of
every chunk it touches, which made the online loader data-bound (smoke 991397: 11 s/step on 2 GPUs). The repack keeps
the same <Scene>/<task>/<seq>.hdf5 layout and content but stores:
  images/<view>               one JPEG per frame (vlen uint8, RGB encoded with quality Q), chunks of 64 frames
  pressure/<hand>_pressure_grid  float32 (T, 21, 21), uncompressed, chunks of 64 frames (values unchanged, NaN kept)
  metadata/ + timestamps      copied
The source build is not modified. split*.json are copied next to the output.

  python scripts/repack_egotouch_jpeg.py SRC DST --estimate 40      # sample 40 sequences -> projected size, PSNR
  python scripts/repack_egotouch_jpeg.py SRC DST --workers 32       # full repack, resumable (skips finished files)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np

VIEWS = ("chest_color", "left_color", "right_color")
GRIDS = ("left_pressure_grid", "right_pressure_grid")
CHUNK = 64


def encode(frame_rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.reshape(-1)


def decode(buf: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.asarray(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def estimate_one(args) -> dict:
    path, quality, k = args
    rng = np.random.default_rng(abs(hash(str(path))) % 2**32)
    with h5py.File(path, "r") as f:
        n = f["images/chest_color"].shape[0]
        ts = np.sort(rng.choice(n, size=min(k, n), replace=False))
        nbytes, ps = 0, []
        for v in VIEWS:
            for t in ts:
                img = f["images"][v][int(t)]
                buf = encode(img, quality)
                nbytes += buf.size
                ps.append(psnr(img, decode(buf)))
    return dict(frames=n, sampled=len(ts), bytes_per_frame_3views=nbytes / len(ts), psnr=float(np.mean(ps)),
                src_bytes=path.stat().st_size)


def repack_one(args) -> tuple[str, str]:
    src, dst, quality = args
    if dst.exists():
        return str(src), "exists"
    tmp = dst.with_suffix(".hdf5.tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(src, "r") as fi, h5py.File(tmp, "w") as fo:
            fi.copy(fi["metadata"], fo, "metadata")
            if "timestamps" in fi:
                fi.copy(fi["timestamps"], fo, "timestamps")
            gi = fo.create_group("images")
            vlen = h5py.vlen_dtype(np.uint8)
            for v in VIEWS:
                frames = fi["images"][v][...]                      # sequential whole-view read: one pass over chunks
                d = gi.create_dataset(v, shape=(len(frames),), dtype=vlen, chunks=(min(CHUNK, len(frames)),))
                for t in range(len(frames)):
                    d[t] = encode(frames[t], quality)
                d.attrs["format"] = f"jpeg_rgb_q{quality}"
                d.attrs["shape"] = list(frames.shape[1:])
            gp = fo.create_group("pressure")
            for k, v in fi["pressure"].attrs.items():
                gp.attrs[k] = v
            for g in GRIDS:
                a = fi["pressure"][g][...]
                gp.create_dataset(g, data=a, chunks=(min(CHUNK, len(a)), 21, 21))
            fo.attrs["repacked_from"] = str(src)
        os.replace(tmp, dst)
        return str(src), "ok"
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return str(src), f"FAIL {type(e).__name__}: {e}"[:200]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--estimate", type=int, default=0, help="sample this many sequences and project the size")
    ap.add_argument("--frames-per-seq", type=int, default=20)
    args = ap.parse_args()
    files = sorted(args.src.glob("*/*/*.hdf5"))
    print(f"JPK|{len(files)} source files", flush=True)
    if args.estimate:
        rng = np.random.default_rng(0)
        pick = [files[i] for i in rng.choice(len(files), size=min(args.estimate, len(files)), replace=False)]
        with Pool(args.workers) as pool:
            res = pool.map(estimate_one, [(p, args.quality, args.frames_per_seq) for p in pick])
        bpf = np.average([r["bytes_per_frame_3views"] for r in res], weights=[r["frames"] for r in res])
        total_frames = 0
        for p in files:
            with h5py.File(p, "r") as f:
                total_frames += f["images/chest_color"].shape[0]
        grids = total_frames * 2 * 21 * 21 * 4
        proj = total_frames * bpf + grids
        src_total = sum(p.stat().st_size for p in files)
        print(f"JPK|q{args.quality}: {bpf / 1024:.1f} KiB per frame (3 views), PSNR {np.mean([r['psnr'] for r in res]):.1f} dB "
              f"(min {min(r['psnr'] for r in res):.1f}); {total_frames} frames -> images {total_frames * bpf / 2**30:.1f} GiB "
              f"+ grids {grids / 2**30:.1f} GiB = {proj / 2**30:.1f} GiB projected (source build {src_total / 2**30:.1f} GiB)",
              flush=True)
        print("JPK|ESTIMATE_DONE", flush=True)
        return
    args.dst.mkdir(parents=True, exist_ok=True)
    for sp in args.src.glob("split*.json"):
        shutil.copy2(sp, args.dst / sp.name)
    jobs = [(p, args.dst / p.relative_to(args.src), args.quality) for p in files]
    t0, done, fails = time.time(), 0, []
    with Pool(args.workers) as pool:
        for src, status in pool.imap_unordered(repack_one, jobs, chunksize=1):
            done += 1
            if status.startswith("FAIL"):
                fails.append((src, status))
                print(f"JPK|{status} {src}", flush=True)
            if done % 100 == 0 or done == len(jobs):
                print(f"JPK|{done}/{len(jobs)} done, {len(fails)} failed, {time.time() - t0:.0f}s", flush=True)
    out = sum(p.stat().st_size for p in args.dst.glob("*/*/*.hdf5"))
    print(f"JPK|output {out / 2**30:.1f} GiB in {len(list(args.dst.glob('*/*/*.hdf5')))} files; {len(fails)} failed", flush=True)
    (args.dst / "repack_report.json").write_text(json.dumps({"quality": args.quality, "failed": fails}))
    print("JPK|REPACK_DONE" if not fails else "JPK|REPACK_INCOMPLETE", flush=True)


if __name__ == "__main__":
    main()
