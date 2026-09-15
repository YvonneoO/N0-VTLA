#!/usr/bin/env python
"""Downloads a SMALL local slice (a few episodes) of one NeoteAIEmbodied/OpenNeoData platform
and repacks it into the local LeRobot-v3 directory layout this repo's
LeRobotCanonicalTaskTactileDataConfig (n0vtla/training/config.py:461) already expects -- the
same canonical data config vtla_stage1_predictor_pretrain and vtla_stage2_align_expert both use.
This is a smoke-test-sized slice for validating scripts/train_stage2_align_expert.py end-to-end,
NOT a real training corpus (OpenNeoData's `flexiv` platform alone has 20,851 episodes; the full
7-platform corpus is ~9TB -- see this project's Stage-2 design notes for the real-scale download
tool).

Despite the filename (kept for now -- the 2 sbatch scripts that already reference it as
"flexiv smoke data" still describe today's default `--platform flexiv` usage), this script
works for ANY of OpenNeoData's 7 platforms via `--platform` (flexiv/umi/arx5/ur/aloha/
umi_single/arx5_single) -- confirmed 2026-09-15 via direct HF metadata probes that all 7 have
tactile image features under the same `observation.images.<view>` naming convention.

Column-naming note: OpenNeoData uses `observation.images.<view>` (PLURAL "images") on every
platform, but this repo's canonical schema (n0vtla/policies/canonical_schema.py) was written
against `observation.image.<view>` (SINGULAR -- an earlier/internal NeoData naming convention).
This script renames every such column (in both the data and episode-metadata parquet files) and
the corresponding `videos/` subdirectory names, so the downloaded slice matches what the
existing configs already expect with zero changes to them.

State/action representation note (PLATFORM-DEPENDENT, verified 2026-09-15 -- do not assume this
uniformly across platforms): some platforms (flexiv, arx5, ur, aloha, arx5_single) ship BOTH a
raw joint-space `observation.state`/`action` AND a separate `observation.eef_pose`/
`action.eef_pose` (Cartesian position + rot6d rotation + gripper) -- for these, the canonical
32-dim padded EEF layout this repo's Stage-1/Stage-2 configs expect under the plain
`observation.state`/`action` names needs the eef_pose columns PROMOTED into those names (this
script drops the raw joint columns and promotes eef_pose -> state/action, same as before).
OTHER platforms (umi, umi_single) have NO separate eef_pose columns at all -- their
`observation.state`/`action` IS ALREADY the EEF-pose representation (confirmed via HF
`meta/info.json` feature shapes: umi state=20 dims = 2 arms x (3 pos + 6 rot6d + 1 gripper), no
`eef_pose` key present). This script detects eef_pose-column presence per platform (via
info.json's `features` dict) and only promotes/drops when the columns actually exist --
treating their absence as "already canonical" rather than assuming every platform needs the
promotion.

Only chunk-000/file-000 is fetched (the first N episodes by episode_index, sorted) -- OpenNeoData
packs many episodes per data/video file (v3 "packed" format), so this is already enough data for
a smoke test without needing multiple chunk/file downloads (a separate real-scale download tool
handles pulling across multiple files). Requires HF access to NeoteAIEmbodied/OpenNeoData (gated
dataset; source env.sh first for HF_TOKEN, same convention as every other `hf`/HF Hub use in this
repo).

Usage (CPU-only, e.g. via sbatch):
    python scripts/download_openneodata_flexiv_smoke.py \\
        --platform flexiv \\
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
PLATFORMS = ("flexiv", "umi", "arx5", "ur", "aloha", "umi_single", "arx5_single")
OLD_PREFIX = "observation.images."
NEW_PREFIX = "observation.image."


def _rename_col(name: str) -> str:
    return name.replace(OLD_PREFIX, NEW_PREFIX) if OLD_PREFIX in name else name


def _eef_pose_helpers(has_eef_pose: bool):
    """Returns (is_raw_joint_column, promote_eef_pose) for this platform's schema.

    has_eef_pose=True (flexiv/arx5/ur/aloha/arx5_single): drop raw joint observation.state/
    action + their per-episode stats/*, promote observation.eef_pose/action.eef_pose (and their
    stats) to the canonical observation.state/action names.

    has_eef_pose=False (umi/umi_single): observation.state/action ARE ALREADY the canonical EEF
    representation -- drop nothing, rename nothing (both functions become identity/false).
    """
    if not has_eef_pose:
        return (lambda _name: False), (lambda name: name)

    drop_exact = {"observation.state", "action"}
    drop_stats_prefixes = ("stats/observation.state/", "stats/action/")
    eef_pose_rename = {"observation.eef_pose": "observation.state", "action.eef_pose": "action"}
    eef_pose_stats_prefixes = {
        "stats/observation.eef_pose/": "stats/observation.state/",
        "stats/action.eef_pose/": "stats/action/",
    }

    def is_raw_joint_column(name: str) -> bool:
        return name in drop_exact or name.startswith(drop_stats_prefixes)

    def promote_eef_pose(name: str) -> str:
        if name in eef_pose_rename:
            return eef_pose_rename[name]
        for old_prefix, new_prefix in eef_pose_stats_prefixes.items():
            if name.startswith(old_prefix):
                return new_prefix + name[len(old_prefix):]
        return name

    return is_raw_joint_column, promote_eef_pose


def _canonicalize_columns(table: pa.Table, is_raw_joint_column, promote_eef_pose) -> pa.Table:
    """Drop raw joint-space columns (if this platform has them), promote eef_pose columns to
    the canonical observation.state/action names (if present), then apply the image-prefix
    rename. See the state/action representation note in this module's docstring."""
    keep = [n for n in table.column_names if not is_raw_joint_column(n)]
    table = table.select(keep)
    new_names = [_rename_col(promote_eef_pose(n)) for n in table.column_names]
    return table.rename_columns(new_names)


def _download(filename: str, cache_dir: Path) -> Path:
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, cache_dir=str(cache_dir))
    return Path(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="flexiv", choices=PLATFORMS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=None,
                         help="HF hub download cache (default: HF_HOME env or ~/.cache/huggingface)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    platform = args.platform

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.output)
    (args.output / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (args.output / "data" / "chunk-000").mkdir(parents=True)

    import tempfile
    cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="openneodata_dl_"))

    logging.info(f"Downloading meta/episodes + info/tasks for {platform} from {REPO_ID}...")
    episodes_src = _download(f"{platform}/meta/episodes/chunk-000/file-000.parquet", cache_dir)
    info_src = _download(f"{platform}/meta/info.json", cache_dir)
    tasks_src = _download(f"{platform}/meta/tasks.parquet", cache_dir)
    stats_src = _download(f"{platform}/meta/stats.json", cache_dir)

    info = json.loads(info_src.read_text())
    has_eef_pose = "observation.eef_pose" in info.get("features", {})
    logging.info(f"Platform {platform}: has_eef_pose={has_eef_pose} "
                 f"({'promoting eef_pose to state/action' if has_eef_pose else 'state/action already canonical EEF'})")
    is_raw_joint_column, promote_eef_pose = _eef_pose_helpers(has_eef_pose)

    episodes_table = pq.read_table(episodes_src)
    episodes_all = sorted(episodes_table.to_pylist(), key=lambda row: row["episode_index"])
    selected = episodes_all[: args.num_episodes]
    if not selected:
        raise ValueError("No episodes found")
    data_chunk_files = {(row["data/chunk_index"], row["data/file_index"]) for row in selected}
    if data_chunk_files != {(0, 0)}:
        raise NotImplementedError(
            f"Selected episodes span data files {data_chunk_files}, expected only chunk 0/file 0 -- "
            "increase --num-episodes cautiously or use the real-scale multi-file download tool."
        )
    max_row = max(row["dataset_to_index"] for row in selected)
    logging.info(f"Selected {len(selected)} episodes (indices "
                 f"{[row['episode_index'] for row in selected]}), {max_row} total frames")

    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    logging.info(f"Video keys: {video_keys}")

    # Data parquet: slice to the selected episodes' rows, drop raw joint state/action + promote
    # eef_pose -> observation.state/action (platform-dependent), rename observation.images.*.
    logging.info(f"Downloading data/chunk-000/file-000.parquet...")
    data_src = _download(f"{platform}/data/chunk-000/file-000.parquet", cache_dir)
    data_table = pq.read_table(data_src).slice(0, max_row)
    data_table = _canonicalize_columns(data_table, is_raw_joint_column, promote_eef_pose)
    pq.write_table(data_table, args.output / "data" / "chunk-000" / "file-000.parquet")

    # Episodes metadata: keep only selected rows, same column drop/promote/rename as above
    # (including the per-episode stats/observation.state|action|eef_pose/* fields).
    episodes_table_trimmed = episodes_table.slice(0, len(selected))
    episodes_table_trimmed = _canonicalize_columns(episodes_table_trimmed, is_raw_joint_column, promote_eef_pose)
    pq.write_table(episodes_table_trimmed, args.output / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # Video files: one file-000.mp4 per camera/tactile key, renamed subdirectory.
    for key in video_keys:
        renamed_key = _rename_col(key)
        video_dst_dir = args.output / "videos" / renamed_key / "chunk-000"
        video_dst_dir.mkdir(parents=True, exist_ok=True)
        video_filename = f"{platform}/videos/{key}/chunk-000/file-000.mp4"
        logging.info(f"Downloading {video_filename}...")
        video_src = _download(video_filename, cache_dir)
        shutil.copy(video_src, video_dst_dir / "file-000.mp4")

    # info.json: drop raw joint state/action features (if present), promote eef_pose -> canonical
    # names (if present), rename image feature keys, patch episode/frame counts to match the slice.
    kept_features = {k: v for k, v in info["features"].items() if not is_raw_joint_column(k)}
    renamed_features = {_rename_col(promote_eef_pose(k)): v for k, v in kept_features.items()}
    info["features"] = renamed_features
    info["total_episodes"] = len(selected)
    info["total_frames"] = max_row
    info["total_tasks"] = info.get("total_tasks")  # left as-is; task_index values are unaffected
    info["splits"] = {"train": f"0:{len(selected)}"}
    (args.output / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    shutil.copy(tasks_src, args.output / "meta" / "tasks.parquet")
    # Dataset-wide stats.json: same drop/promote/rename as the data/episodes parquet columns,
    # applied to its top-level keys, so it doesn't keep describing dropped/renamed columns.
    stats = json.loads(stats_src.read_text())
    kept_stats = {k: v for k, v in stats.items() if not is_raw_joint_column(k)}
    renamed_stats = {_rename_col(promote_eef_pose(k)): v for k, v in kept_stats.items()}
    (args.output / "meta" / "stats.json").write_text(json.dumps(renamed_stats, indent=2))

    logging.info(f"Wrote local OpenNeoData {platform} smoke slice ({len(selected)} episodes, "
                 f"{max_row} frames) to {args.output}")


if __name__ == "__main__":
    main()
