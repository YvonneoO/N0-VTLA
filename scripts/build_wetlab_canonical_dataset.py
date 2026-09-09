"""Build one unified canonical wetlab dataset from all raw episodes in a split manifest.

Converts every raw episode independently with wetlab_smoke_adapter.convert() (using
a single shared tactile normalization -- see fit_wetlab_tactile_norm.py, and
ROBOT_POSTTRAIN_OPEN_ISSUES.md #3.2/#5.1/#5.2 for why a per-episode ad hoc fit isn't
enough beyond a one-episode smoke), then merges the resulting per-source mini
LeRobot datasets into one dataset with globally renumbered episode/frame indices.
Each merged episode is tagged with its source uuid, block, and split (train/val,
per the split manifest) so compute_canonical_norm.py and any later analysis can
filter by split without re-deriving it.

A raw episode that produces no qualifying contiguous window (contiguous_runs finds
nothing) is skipped and logged, not silently dropped -- see the printed/returned
skip list.

Usage:
  python scripts/build_wetlab_canonical_dataset.py \
    /DATA2/qianqian/n0vtla_robot_audit/cap_to_tray/smoke_test \
    scripts/wetlab_split_v1_tacwam_match.json \
    /DATA2/qianqian/n0vtla_robot_audit/wetlab_tactile_norm_v1.json \
    /DATA2/qianqian/n0vtla_robot_audit/canonical_wetlab_v1
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import wetlab_smoke_adapter as adapter
from itw_tactile_smoke_adapter import _info_json, _write_jsonl


def merge(per_source: list[tuple[str, Path]], split_manifest: dict, output: Path) -> dict:
    output_data = output / "data/chunk-000"
    output_meta = output / "meta"
    output_data.mkdir(parents=True)
    output_meta.mkdir(parents=True)

    block_of = split_manifest["block_of_episode"]
    val_set = set(split_manifest["val"])
    holdout_set = set(split_manifest["block_holdout_v1"]["holdout_episodes"])

    global_episode = 0
    global_index = 0
    episodes_out, stats_out, per_episode_meta = [], [], []
    data_bytes = video_bytes = 0
    video_keys: list[str] | None = None

    for uuid, source_output in per_source:
        local_episodes = [json.loads(l) for l in (source_output / "meta" / "episodes.jsonl").open()]
        local_stats = [json.loads(l) for l in (source_output / "meta" / "episodes_stats.jsonl").open()]
        local_info = json.loads((source_output / "meta" / "info.json").read_text())
        if video_keys is None:
            video_keys = [k for k in local_info["features"] if local_info["features"][k].get("dtype") == "video"]

        for local in local_episodes:
            local_idx = local["episode_index"]
            table = pq.read_table(source_output / "data/chunk-000" / f"episode_{local_idx:06d}.parquet")
            n = table.num_rows
            table = table.set_column(table.schema.get_field_index("episode_index"),
                                      "episode_index", pa.array([global_episode] * n, pa.int64()))
            table = table.set_column(table.schema.get_field_index("index"),
                                      "index", pa.array(range(global_index, global_index + n), pa.int64()))
            dst = output_data / f"episode_{global_episode:06d}.parquet"
            pq.write_table(table, dst, compression="zstd")
            data_bytes += dst.stat().st_size

            for key in video_keys:
                src_video = source_output / "videos/chunk-000" / key / f"episode_{local_idx:06d}.mp4"
                dst_dir = output / "videos/chunk-000" / key
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_video = dst_dir / f"episode_{global_episode:06d}.mp4"
                shutil.copy2(src_video, dst_video)
                video_bytes += dst_video.stat().st_size

            episodes_out.append(dict(episode_index=global_episode, tasks=local["tasks"], length=n,
                                      source_uuid=uuid, block=block_of.get(uuid, "unknown"),
                                      split="val" if uuid in val_set else "train",
                                      block_holdout=uuid in holdout_set))
            stat = local_stats[[s["episode_index"] for s in local_stats].index(local_idx)]
            stat["episode_index"] = global_episode
            stats_out.append(stat)
            global_episode += 1
            global_index += n

        per_episode_meta.append(dict(uuid=uuid, audit=json.loads((source_output / "meta" / "audit.json").read_text())))

    _write_jsonl(output_meta / "episodes.jsonl", episodes_out)
    _write_jsonl(output_meta / "episodes_stats.jsonl", stats_out)
    _write_jsonl(output_meta / "tasks.jsonl", [dict(task_index=0, task=adapter.TASK)])
    (output_meta / "per_source_audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in per_episode_meta))
    info = _info_json(global_episode, global_index, data_bytes, video_bytes, video_keys or [])
    info["robot_type"] = "xarm6_revo2"
    (output_meta / "info.json").write_text(json.dumps(info, indent=2))
    return dict(n_episodes=global_episode, n_frames=global_index,
                n_train=sum(1 for e in episodes_out if e["split"] == "train"),
                n_val=sum(1 for e in episodes_out if e["split"] == "val"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smoke_test_dir", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("norm_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scratch-dir", type=Path, default=None,
                         help="Where to write per-source intermediate conversions. "
                              "Defaults to a temp dir removed after merging.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = json.loads(args.split_manifest.read_text())
    all_uuids = sorted(manifest["train"] + manifest["val"])
    assert len(all_uuids) == len(set(all_uuids)) == len(manifest["train"]) + len(manifest["val"])

    scratch_ctx = tempfile.TemporaryDirectory() if args.scratch_dir is None else None
    scratch = args.scratch_dir if args.scratch_dir is not None else Path(scratch_ctx.name)
    scratch.mkdir(parents=True, exist_ok=True)

    per_source, skipped = [], []
    for uuid in all_uuids:
        source = args.smoke_test_dir / uuid
        source_output = scratch / uuid
        try:
            adapter.convert(source, source_output, allow_unverified_sync=True, norm_path=args.norm_path)
        except ValueError as exc:
            skipped.append(dict(uuid=uuid, reason=str(exc)))
            print(f"SKIP {uuid}: {exc}")
            continue
        per_source.append((uuid, source_output))
        print(f"converted {uuid} ({len(per_source)}/{len(all_uuids)} so far, {len(skipped)} skipped)")

    summary = merge(per_source, manifest, args.output)
    summary["skipped"] = skipped
    (args.output / "meta" / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if scratch_ctx is not None:
        scratch_ctx.cleanup()


if __name__ == "__main__":
    main()
