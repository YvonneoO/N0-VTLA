"""Manifest + sample-index caches for mid-train (Stage-1 online) on the EgoTouch 3-view@224 HDF5 build.

Writes into OUT_DIR:
  manifest.json      {"raw_root": ROOT, "splits": {"train_val": [...], "test": [...]}, ...} (paths relative to ROOT)
                     for VTLA_ITW_EPISODE_LIST_JSON (+ VTLA_ITW_RAW_ROOT=ROOT).
                     train_val = SPLIT_FILE's train + val (mid-train data); test = test_seen + test_unseen, kept out of
                     mid-train (Ego2Contact's in-distribution test set, available for a held-out check).
  index_<key>.json   [[abs path, frames], ...] for VTLA_ITW_INDEX_CACHE; sequences under 51 frames are left out.

Split files list absolute paths of the machine they were made on; they are rebased onto ROOT by the ROOT folder name.

  python scripts/build_egotouch_midtrain_inputs.py ROOT OUT_DIR [--split-file split_full.json] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n0vtla.training.egotouch_online_dataset import sequence_frames  # noqa: E402

MIN_FRAMES = 51


def rel(p: str, root: Path) -> str:
    parts = Path(p).parts
    return str(Path(*parts[parts.index(root.name) + 1:])) if root.name in parts else str(p)


def frames(path: str) -> tuple[str, int | None, str]:
    try:
        return path, sequence_frames(Path(path)), ""
    except Exception as e:  # noqa: BLE001
        return path, None, str(e)[:120]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--split-file", default="split_full.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    split = json.loads((args.root / args.split_file).read_text())
    splits = {"train_val": sorted(rel(p, args.root) for k in ("train", "val") for p in split[k]),
              "test": sorted(rel(p, args.root) for k in ("test_seen", "test_unseen") for p in split[k])}
    assert not set(splits["train_val"]) & set(splits["test"]), "train/val and test overlap"
    with Pool(args.workers) as pool:
        res = dict((p, (n, why)) for p, n, why in pool.map(
            frames, [str(args.root / r) for v in splits.values() for r in v], chunksize=8))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, rels in splits.items():
        rows, skipped = [], []
        for r in rels:
            n, why = res[str(args.root / r)]
            if n is None or n < MIN_FRAMES:
                skipped.append([r, n, why])
            else:
                rows.append([str(args.root / r), n])
        (args.out_dir / f"index_{key}.json").write_text(json.dumps({"episodes": rows}))
        splits[key] = [rel(p, args.root) for p, _ in rows]      # manifest = exactly the indexed sequences
        frames_total = sum(n for _, n in rows)
        summary[key] = dict(sequences=len(rows), skipped=skipped, frames=frames_total, hours=frames_total / 30 / 3600,
                            samples=frames_total)
        print(f"EGB|{key}: {len(rows)} sequences, {frames_total} frames = {frames_total / 30 / 3600:.2f} h; "
              f"skipped {len(skipped)}: {skipped[:6]}", flush=True)
    manifest = {"raw_root": str(args.root), "splits": splits, "summary": summary,
                "source_split_file": str(args.root / args.split_file),
                "note": "EgoTouch 3-view@224 HDF5 (Ego2Contact build); train_val = train+val, test = test_seen+test_unseen"}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest))
    print("EGOTOUCH_INPUTS_DONE", flush=True)


if __name__ == "__main__":
    main()
