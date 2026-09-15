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

State/action representation note: OpenNeoData's flexiv platform ships state/action in TWO
representations per frame: `observation.state`/`action` (raw joint space, 8-dim: j1..j7 +
gripper) and `observation.eef_pose`/`action.eef_pose` (10-dim: x,y,z,r1..r6,gripper --
Cartesian position + rot6d rotation + gripper). This repo's canonical Flexiv pipeline
(n0vtla/policies/canonical_schema.py's 32-dim padded EEF layout, FlexivEEFInputs/
FlexivEEFOutputs, and LeRobotFlexivTactileDataConfig's docstring) -- and, more importantly,
the actual Stage-1 checkpoint this Stage-2 script warm-starts from (config
vtla_stage1_predictor_pretrain uses LeRobotCanonicalTaskTactileDataConfig with
use_delta_eef_actions=True, "on-disk state and action use the canonical padded 32-dimensional
EEF layout") -- all expect the EEF-pose representation under the plain `observation.state`/
`action` column names, NOT joint space. So this script DROPS the raw joint columns (and their
per-episode `stats/observation.state/*` / `stats/action/*` entries) and promotes
`observation.eef_pose` -> `observation.state` / `action.eef_pose` -> `action` (and their stats
counterparts) to those same names, in the data parquet, the episode-metadata parquet, and
info.json's feature list.

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

# Raw joint-space columns (and their per-episode stats/* entries) to drop -- see the
# state/action representation note above. Prefixes end in "/" so they don't also match the
# eef_pose stats columns (e.g. "stats/action/min" vs "stats/action.eef_pose/min").
_DROP_EXACT = {"observation.state", "action"}
_DROP_STATS_PREFIXES = ("stats/observation.state/", "stats/action/")

# eef_pose -> canonical name promotion (applied before the image-prefix rename below).
_EEF_POSE_RENAME = {"observation.eef_pose": "observation.state", "action.eef_pose": "action"}
_EEF_POSE_STATS_PREFIXES = {
    "stats/observation.eef_pose/": "stats/observation.state/",
    "stats/action.eef_pose/": "stats/action/",
}


def _is_raw_joint_column(name: str) -> bool:
    return name in _DROP_EXACT or name.startswith(_DROP_STATS_PREFIXES)


def _promote_eef_pose(name: str) -> str:
    if name in _EEF_POSE_RENAME:
        return _EEF_POSE_RENAME[name]
    for old_prefix, new_prefix in _EEF_POSE_STATS_PREFIXES.items():
        if name.startswith(old_prefix):
            return new_prefix + name[len(old_prefix):]
    return name


def _rename_col(name: str) -> str:
    return name.replace(OLD_PREFIX, NEW_PREFIX) if OLD_PREFIX in name else name


def _canonicalize_columns(table: pa.Table) -> pa.Table:
    """Drop raw joint-space columns, promote eef_pose columns to the canonical
    observation.state/action names, then apply the existing image-prefix rename. See the
    state/action representation note in this module's docstring."""
    keep = [n for n in table.column_names if not _is_raw_joint_column(n)]
    table = table.select(keep)
    new_names = [_rename_col(_promote_eef_pose(n)) for n in table.column_names]
    return table.rename_columns(new_names)


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

    # Data parquet: slice to the selected episodes' rows, drop raw joint state/action + promote
    # eef_pose -> observation.state/action, rename observation.images.* columns.
    logging.info("Downloading data/chunk-000/file-000.parquet...")
    data_src = _download(f"{PLATFORM}/data/chunk-000/file-000.parquet", cache_dir)
    data_table = pq.read_table(data_src).slice(0, max_row)
    data_table = _canonicalize_columns(data_table)
    pq.write_table(data_table, args.output / "data" / "chunk-000" / "file-000.parquet")

    # Episodes metadata: keep only selected rows, same column drop/promote/rename as above
    # (including the per-episode stats/observation.state|action|eef_pose/* fields).
    episodes_table_trimmed = episodes_table.slice(0, len(selected))
    episodes_table_trimmed = _canonicalize_columns(episodes_table_trimmed)
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

    # info.json: drop raw joint state/action features, promote eef_pose -> canonical names,
    # rename image feature keys, patch episode/frame counts to match the trimmed slice.
    kept_features = {k: v for k, v in info["features"].items() if not _is_raw_joint_column(k)}
    renamed_features = {_rename_col(_promote_eef_pose(k)): v for k, v in kept_features.items()}
    info["features"] = renamed_features
    info["total_episodes"] = len(selected)
    info["total_frames"] = max_row
    info["total_tasks"] = info.get("total_tasks")  # left as-is; task_index values are unaffected
    info["splits"] = {"train": f"0:{len(selected)}"}
    (args.output / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    shutil.copy(tasks_src, args.output / "meta" / "tasks.parquet")
    # Dataset-wide stats.json: same drop/promote/rename as the data/episodes parquet columns,
    # applied to its top-level keys, so it doesn't keep describing the dropped joint columns.
    stats = json.loads(stats_src.read_text())
    kept_stats = {k: v for k, v in stats.items() if not _is_raw_joint_column(k)}
    renamed_stats = {_rename_col(_promote_eef_pose(k)): v for k, v in kept_stats.items()}
    (args.output / "meta" / "stats.json").write_text(json.dumps(renamed_stats, indent=2))

    logging.info(f"Wrote local OpenNeoData flexiv smoke slice ({len(selected)} episodes, "
                 f"{max_row} frames) to {args.output}")


if __name__ == "__main__":
    main()
