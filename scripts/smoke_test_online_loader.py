#!/usr/bin/env python3
"""Real (non-mocked) smoke test for ITWOnlineTactileDataset against real raw episodes,
using a real per-task-scale normalization file end to end -- unlike
tests/test_itw_online_dataset.py, which mocks _EpisodeCache.rgb_frame because
torchcodec isn't available locally. This exercises the FULL path: real torchcodec
video decode, real aligned_timeline, real per-task-scale lookup keyed by each
episode's actual task_info.json["name"].

Usage: python scripts/smoke_test_online_loader.py <raw_root> <normalization_json>
       [--dates itw07-23,itw07-24] [--max-episodes 20] [--samples-per-episode 3]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itw_pressure import load_normalization  # noqa: E402
from n0vtla.training.itw_online_dataset import (  # noqa: E402
    ITWOnlineTactileDataset,
    list_episode_dirs,
    read_task_name_for_norm,
    task_scale_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("normalization_json", type=Path)
    parser.add_argument("--dates", default=None, help="comma-separated date subdirs; default: scan raw_root")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dates = args.dates.split(",") if args.dates else None
    all_episodes = list_episode_dirs(args.raw_root, date_dirs=dates)
    print(f"found {len(all_episodes)} candidate episode dirs under {args.raw_root}"
          f"{' (dates=' + str(dates) + ')' if dates else ''}")
    if not all_episodes:
        print("SMOKE_TEST_FAIL: no episodes found")
        return 1

    rng = random.Random(args.seed)
    episodes = rng.sample(all_episodes, min(args.max_episodes, len(all_episodes)))
    print(f"sampling {len(episodes)} episodes for the smoke test")

    normalization = load_normalization(args.normalization_json)
    is_per_task = "task_scale" in normalization
    print(f"normalization schema={normalization.get('schema')} per_task_scale={is_per_task}")

    if is_per_task:
        coverage = task_scale_coverage(episodes, normalization)
        print(f"task_scale_coverage over sampled episodes: {coverage}")
        for ep in episodes[:5]:
            print(f"  sample task name for {ep.name}: {read_task_name_for_norm(ep)!r}")

    t0 = time.time()
    try:
        dataset = ITWOnlineTactileDataset(episodes, args.normalization_json)
    except Exception:
        print("SMOKE_TEST_FAIL: dataset construction (indexing) raised")
        traceback.print_exc()
        return 1
    t_index = time.time() - t0
    print(f"indexed dataset: {len(dataset)} total (episode, frame) samples in {t_index:.2f}s")

    n_ok, n_err = 0, 0
    idxs = rng.sample(range(len(dataset)), min(args.samples_per_episode * len(episodes), len(dataset)))
    t0 = time.time()
    for idx in idxs:
        try:
            sample = dataset[idx]
        except Exception:
            n_err += 1
            print(f"SMOKE_TEST_SAMPLE_ERROR at idx={idx}")
            traceback.print_exc()
            continue
        n_ok += 1
        if n_ok == 1:
            print("first sample keys and shapes:")
            for k, v in sample.items():
                print(f"  {k}: {getattr(v, 'shape', v)}")
    t_samples = time.time() - t0

    print(f"sampled {n_ok} OK, {n_err} errors, in {t_samples:.2f}s "
          f"({t_samples / max(n_ok, 1):.3f}s/sample avg, includes video decode)")
    print("SMOKE_TEST_OK" if n_err == 0 else "SMOKE_TEST_FAIL")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
