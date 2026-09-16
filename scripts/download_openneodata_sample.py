#!/usr/bin/env python
"""Downloads a real-scale SAMPLE (a target percent of the corpus) of
NeoteAIEmbodied/OpenNeoData, spread proportionally across all 7 platforms, into
per-platform local LeRobot-v3 directories -- each spanning multiple chunk/file pairs
(OpenNeoData's own v3 "packed" multi-file layout), not a single merged file.

Reuses scripts/download_openneodata_flexiv_smoke.py's per-file transform routine
(column rename, eef_pose promotion, 32-dim canonical padding -- see that module's
docstring for the full rationale) for every source file this pulls; the only new
logic here is looping over MULTIPLE files per platform until a size target is hit,
instead of always just chunk-000/file-000.

Per-platform target bytes are proportional to that platform's OWN current total size
(queried live via the HF API at the start of each run, not hardcoded), so this stays
correct as the dataset grows. Files within a platform are pulled in file-index order
(0, 1, 2, ...) starting from file-000, so episode_index and each source file's own
chunk/file position both stay valid unchanged in the output -- no renumbering needed
(mirrors exactly what the single-file smoke script already assumes, just extended
across more files instead of slicing within one).

Usage:
    python scripts/download_openneodata_sample.py \\
        --output /DATA2/qianqian/openneodata_5pct --target-pct 5 --max-gb 470

    # narrower run, e.g. only 2 platforms:
    python scripts/download_openneodata_sample.py \\
        --output /DATA2/qianqian/openneodata_5pct --target-pct 5 \\
        --platforms flexiv,umi
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_openneodata_flexiv_smoke import (  # noqa: E402
    CANONICAL_ACTION_DIM,
    PLATFORMS,
    REPO_ID,
    _canonicalize_columns,
    _download,
    _drop_narrow_stats,
    _eef_pose_helpers,
    _pad_state_action_to_canonical,
    _rename_col,
)


def _platform_total_bytes(api: HfApi, platform: str) -> int:
    total = 0
    for item in api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=platform, recursive=True):
        total += getattr(item, "size", None) or 0
    return total


def _list_data_files(api: HfApi, platform: str) -> list[str]:
    """Sorted file-NNN basenames (no extension) under <platform>/data/chunk-000/."""
    names = []
    for item in api.list_repo_tree(
        REPO_ID, repo_type="dataset", path_in_repo=f"{platform}/data/chunk-000", recursive=False
    ):
        name = Path(item.path).stem  # "file-000"
        if name.startswith("file-"):
            names.append(name)
    return sorted(names)


def _process_platform(
    api: HfApi, platform: str, target_bytes: int, output_root: Path, cache_dir: Path
) -> int:
    """Downloads+transforms files for one platform until target_bytes is reached (or
    files run out). Returns bytes actually written (source-file-size proxy)."""
    out_dir = output_root / platform
    (out_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info_src = _download(f"{platform}/meta/info.json", cache_dir)
    tasks_src = _download(f"{platform}/meta/tasks.parquet", cache_dir)
    stats_src = _download(f"{platform}/meta/stats.json", cache_dir)
    info = json.loads(info_src.read_text())
    has_eef_pose = "observation.eef_pose" in info.get("features", {})
    is_raw_joint_column, promote_eef_pose = _eef_pose_helpers(has_eef_pose)
    native_width_key = "observation.eef_pose" if has_eef_pose else "observation.state"
    native_width = info["features"][native_width_key]["shape"][0]
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    logging.info(
        f"[{platform}] has_eef_pose={has_eef_pose} native_width={native_width} "
        f"video_keys={video_keys}"
    )

    file_names = _list_data_files(api, platform)
    logging.info(f"[{platform}] {len(file_names)} data files available, target={target_bytes / 1e9:.1f}GB")

    total_episodes = 0
    total_frames = 0
    bytes_done = 0
    files_written = 0

    for fname in file_names:
        if bytes_done >= target_bytes:
            break

        episodes_src = _download(f"{platform}/meta/episodes/chunk-000/{fname}.parquet", cache_dir)
        episodes_table = pq.read_table(episodes_src)
        n_episodes = episodes_table.num_rows
        if n_episodes == 0:
            continue

        data_src = _download(f"{platform}/data/chunk-000/{fname}.parquet", cache_dir)
        data_table = pq.read_table(data_src)
        n_frames = data_table.num_rows

        data_table = _canonicalize_columns(data_table, is_raw_joint_column, promote_eef_pose)
        data_table = _pad_state_action_to_canonical(data_table, native_width)
        pq.write_table(data_table, out_dir / "data" / "chunk-000" / f"{fname}.parquet")
        bytes_done += data_src.stat().st_size

        episodes_table = _canonicalize_columns(episodes_table, is_raw_joint_column, promote_eef_pose)
        episodes_table = episodes_table.select(_drop_narrow_stats(episodes_table.column_names))
        pq.write_table(episodes_table, out_dir / "meta" / "episodes" / "chunk-000" / f"{fname}.parquet")
        bytes_done += episodes_src.stat().st_size

        for key in video_keys:
            renamed_key = _rename_col(key)
            video_dst_dir = out_dir / "videos" / renamed_key / "chunk-000"
            video_dst_dir.mkdir(parents=True, exist_ok=True)
            video_filename = f"{platform}/videos/{key}/chunk-000/{fname}.mp4"
            video_src = _download(video_filename, cache_dir)
            (video_dst_dir / f"{fname}.mp4").write_bytes(video_src.read_bytes())
            bytes_done += video_src.stat().st_size

        total_episodes += n_episodes
        total_frames += n_frames
        files_written += 1
        logging.info(
            f"[{platform}] {fname}: +{n_episodes} episodes, +{n_frames} frames, "
            f"{bytes_done / 1e9:.1f}/{target_bytes / 1e9:.1f} GB"
        )

    if files_written == 0:
        logging.warning(f"[{platform}] nothing written (target too small or no files) -- skipping")
        return 0

    kept_features = {k: v for k, v in info["features"].items() if not is_raw_joint_column(k)}
    renamed_features = {_rename_col(promote_eef_pose(k)): v for k, v in kept_features.items()}
    for key in ("observation.state", "action"):
        if key in renamed_features:
            renamed_features[key] = {**renamed_features[key], "shape": [CANONICAL_ACTION_DIM]}
    renamed_features["action_mask"] = {"dtype": "bool", "shape": [CANONICAL_ACTION_DIM]}
    info["features"] = renamed_features
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"}
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    (out_dir / "meta" / "tasks.parquet").write_bytes(tasks_src.read_bytes())

    stats = json.loads(stats_src.read_text())
    kept_stats = {k: v for k, v in stats.items() if not is_raw_joint_column(k)}
    renamed_stats = {_rename_col(promote_eef_pose(k)): v for k, v in kept_stats.items()}
    renamed_stats = {k: v for k, v in renamed_stats.items() if k not in ("observation.state", "action")}
    (out_dir / "meta" / "stats.json").write_text(json.dumps(renamed_stats, indent=2))

    logging.info(
        f"[{platform}] done: {files_written} files, {total_episodes} episodes, "
        f"{total_frames} frames, {bytes_done / 1e9:.1f} GB -> {out_dir}"
    )
    return bytes_done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path, help="Root dir; one subdir per platform.")
    parser.add_argument("--target-pct", type=float, default=5.0, help="Target as %% of the full corpus.")
    parser.add_argument("--max-gb", type=float, default=None,
                         help="Hard cap across all platforms combined (safety guard; default: no cap "
                              "beyond --target-pct itself).")
    parser.add_argument("--platforms", default=",".join(PLATFORMS),
                         help=f"Comma-separated subset of {PLATFORMS} (default: all).")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    for p in platforms:
        if p not in PLATFORMS:
            raise ValueError(f"unknown platform {p!r}, must be one of {PLATFORMS}")

    import tempfile
    cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="openneodata_sample_dl_"))
    args.output.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    sizes = {p: _platform_total_bytes(api, p) for p in platforms}
    grand_total = sum(sizes.values())
    target_total = grand_total * args.target_pct / 100
    if args.max_gb is not None:
        target_total = min(target_total, args.max_gb * 1e9)
    logging.info(
        f"Corpus size (selected platforms): {grand_total / 1e9:.1f} GB; "
        f"target: {target_total / 1e9:.1f} GB ({args.target_pct}%, max_gb={args.max_gb})"
    )
    for p in platforms:
        logging.info(f"  {p}: {sizes[p] / 1e9:.1f} GB ({100 * sizes[p] / grand_total:.1f}% of selection)")

    grand_written = 0
    for p in platforms:
        if grand_written >= target_total:
            logging.info(f"global target reached before {p}; stopping")
            break
        platform_target = target_total * sizes[p] / grand_total
        remaining = target_total - grand_written
        platform_target = min(platform_target, remaining)
        grand_written += _process_platform(api, p, int(platform_target), args.output, cache_dir)

    logging.info(f"TOTAL: {grand_written / 1e9:.1f} GB written to {args.output}")


if __name__ == "__main__":
    main()
