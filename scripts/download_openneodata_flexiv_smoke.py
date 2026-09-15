#!/usr/bin/env python
"""Downloads a SMALL local slice (a few episodes) of NeoteAIEmbodied/OpenNeoData's `flexiv`
platform -- the actual NeoData the paper's Stage 2/3 (and n0-vtla-base itself) were trained on
-- and repacks it into the local LeRobot-v3 directory layout this repo's
LeRobotFlexivTactileDataConfig/LeRobotFlexivVisionDataConfig (n0vtla/training/config.py:269-457)
already expect. This is a smoke-test-sized slice for validating scripts/train_stage2_align_
expert.py end-to-end, NOT a real training corpus (OpenNeoData has 20,851 flexiv episodes; the
paper's own Stage 2/3 trained at that scale, which we deliberately do not reproduce here -- see
this project's Stage-2 design notes).

Column-naming note: OpenNeoData's flexiv schema uses `observation.images.<view>` (PLURAL
"images"), but this repo's `_FLEXIV_TACTILE_KEYS` constant (config.py:35-38) and the Flexiv
data configs' repack transforms were written against `observation.image.<view>` (SINGULAR --
apparently an earlier/internal NeoData naming convention). This script renames every such
column (in both the data and episode-metadata parquet files) and the corresponding `videos/`
subdirectory names, so the downloaded slice matches what those existing configs already expect
with zero changes to them.

Only chunk-000/file-000 is fetched (the first N episodes by episode_index, sorted) -- OpenNeoData
packs many episodes per data/video file (v3 "packed" format), so this is already enough data for
a smoke test without needing multiple chunk/file downloads. Requires HF access to
NeoteAIEmbodied/OpenNeoData (gated dataset; source `env.sh` first for HF_TOKEN, same convention
as every other `hf`/HF Hub use in this repo).

Usage (CPU-only, e.g. via sbatch):
    python scripts/download_openneodata_flexiv_smoke.py \\
        --output /scratch/project/prj-02-phai-lab/yqq/N0-VTLA/data/openneodata_flexiv_smoke \\
        --num-episodes 2
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "NeoteAIEmbodied/OpenNeoData"
PLATFORM = "flexiv"
OLD_PREFIX = "observation.images."
NEW_PREFIX = "observation.image."


def _rename_col(name: str) -> str:
    return name.replace(OLD_PREFIX, NEW_PREFIX) if OLD_PREFIX in name else name


def _download(filename: str, cache_dir: Path) -> Path:
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, cache_dir=str(cache_dir))
    return Path(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=None,
                         help="HF hub download cache (default: HF_HOME env or ~/.cache/huggingface)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.output)
    (args.output / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (args.output / "data" / "chunk-000").mkdir(parents=True)

    import tempfile
    cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="openneodata_dl_"))

    logging.info(f"Downloading meta/episodes + info/tasks for {PLATFORM} from {REPO_ID}...")
    episodes_src = _download(f"{PLATFORM}/meta/episodes/chunk-000/file-000.parquet", cache_dir)
    info_src = _download(f"{PLATFORM}/meta/info.json", cache_dir)
    tasks_src = _download(f"{PLATFORM}/meta/tasks.parquet", cache_dir)
    stats_src = _download(f"{PLATFORM}/meta/stats.json", cache_dir)

    episodes_table = pq.read_table(episodes_src)
    episodes_all = sorted(episodes_table.to_pylist(), key=lambda row: row["episode_index"])
    selected = episodes_all[: args.num_episodes]
    if not selected:
        raise ValueError("No episodes found")
    data_chunk_files = {(row["data/chunk_index"], row["data/file_index"]) for row in selected}
    if data_chunk_files != {(0, 0)}:
        raise NotImplementedError(
            f"Selected episodes span data files {data_chunk_files}, expected only chunk 0/file 0 -- "
            "increase --num-episodes cautiously or extend this script to fetch multiple data files."
        )
    max_row = max(row["dataset_to_index"] for row in selected)
    logging.info(f"Selected {len(selected)} episodes (indices "
                 f"{[row['episode_index'] for row in selected]}), {max_row} total frames")

    info = json.loads(info_src.read_text())
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    logging.info(f"Video keys: {video_keys}")

    # Data parquet: slice to the selected episodes' rows, rename observation.images.* columns.
    logging.info("Downloading data/chunk-000/file-000.parquet...")
    data_src = _download(f"{PLATFORM}/data/chunk-000/file-000.parquet", cache_dir)
    data_table = pq.read_table(data_src).slice(0, max_row)
    data_table = data_table.rename_columns([_rename_col(name) for name in data_table.column_names])
    pq.write_table(data_table, args.output / "data" / "chunk-000" / "file-000.parquet")

    # Episodes metadata: keep only selected rows, rename the videos/observation.images.* columns.
    episodes_table_trimmed = episodes_table.slice(0, len(selected))
    episodes_table_trimmed = episodes_table_trimmed.rename_columns(
        [_rename_col(name) for name in episodes_table_trimmed.column_names]
    )
    pq.write_table(episodes_table_trimmed, args.output / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # Video files: one file-000.mp4 per camera/tactile key, renamed subdirectory.
    for key in video_keys:
        renamed_key = _rename_col(key)
        video_dst_dir = args.output / "videos" / renamed_key / "chunk-000"
        video_dst_dir.mkdir(parents=True, exist_ok=True)
        video_filename = f"{PLATFORM}/videos/{key}/chunk-000/file-000.mp4"
        logging.info(f"Downloading {video_filename}...")
        video_src = _download(video_filename, cache_dir)
        shutil.copy(video_src, video_dst_dir / "file-000.mp4")

    # info.json: rename feature keys, patch episode/frame counts to match the trimmed slice.
    renamed_features = {_rename_col(k): v for k, v in info["features"].items()}
    info["features"] = renamed_features
    info["total_episodes"] = len(selected)
    info["total_frames"] = max_row
    info["total_tasks"] = info.get("total_tasks")  # left as-is; task_index values are unaffected
    info["splits"] = {"train": f"0:{len(selected)}"}
    (args.output / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    shutil.copy(tasks_src, args.output / "meta" / "tasks.parquet")
    shutil.copy(stats_src, args.output / "meta" / "stats.json")

    logging.info(f"Wrote local OpenNeoData flexiv smoke slice ({len(selected)} episodes, "
                 f"{max_row} frames) to {args.output}")


if __name__ == "__main__":
    main()
