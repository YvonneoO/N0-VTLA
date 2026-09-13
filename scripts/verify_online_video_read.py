#!/usr/bin/env python3
"""One-off check: does torchcodec's indexed random access return the SAME raw decoded
frame as the offline path's cv2 sequential read, for the same target frame index?

The offline converter (itw_pressure.write_aligned_rgb) deliberately reads every video
strictly sequentially from frame 0 (`while current < index: cap.read()`) because cv2's
CAP_PROP_POS_FRAMES seeking is known-unreliable on H.264. The online dataset
(n0vtla/training/itw_online_dataset.py) needs random access instead (shuffled
(episode, frame) pairs), via torchcodec. This script isolates exactly that one
question -- comparing RAW decoded frames, before any letterbox resize or H.264
re-encode, so a mismatch can only mean "wrong frame", not "resize/encode noise".

Usage: python scripts/verify_online_video_read.py <episode_dir> [--views rgb_head,wrist_left,wrist_right] [--positions 0,10,50]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itw_pressure import aligned_timeline  # noqa: E402


def cv2_sequential_frame(video_path: Path, target_index: int) -> np.ndarray:
    """Verbatim copy of write_aligned_rgb's own read loop, stopped at one index,
    returning the raw BGR frame before any resize/re-encode."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    current, frame = -1, None
    try:
        while current < target_index:
            ok, frame = cap.read()
            current += 1
            if not ok:
                raise ValueError(f"video shorter than target index {target_index}: {video_path}")
    finally:
        cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def torchcodec_frame(video_path: Path, target_index: int) -> np.ndarray:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path))
    frame = decoder[target_index]  # (C, H, W) uint8
    return frame.permute(1, 2, 0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--views", default="rgb_head,wrist_left,wrist_right")
    parser.add_argument("--positions", default="0,10,50")
    args = parser.parse_args()

    mapping, _audit = aligned_timeline(args.episode_dir)
    n = len(mapping["master_timestamp_ns"])
    print(f"episode has {n} aligned frame positions")

    all_ok = True
    for view in args.views.split(","):
        video_path = args.episode_dir / f"{view}.mp4"
        for pos_str in args.positions.split(","):
            pos = int(pos_str)
            if pos >= n:
                print(f"  [{view} pos={pos}] SKIPPED (only {n} positions)")
                continue
            frame_idx = int(mapping[f"{view}_frame_index"][pos])
            cv2_frame = cv2_sequential_frame(video_path, frame_idx)
            tc_frame = torchcodec_frame(video_path, frame_idx)
            if cv2_frame.shape != tc_frame.shape:
                print(f"  [{view} pos={pos} frame_idx={frame_idx}] SHAPE MISMATCH: cv2={cv2_frame.shape} torchcodec={tc_frame.shape}")
                all_ok = False
                continue
            diff = np.abs(cv2_frame.astype(np.int16) - tc_frame.astype(np.int16))
            mean_abs_diff = float(diff.mean())
            max_abs_diff = int(diff.max())
            status = "OK" if max_abs_diff <= 2 else "MISMATCH"
            if status != "OK":
                all_ok = False
            print(f"  [{view} pos={pos} frame_idx={frame_idx}] mean_abs_diff={mean_abs_diff:.4f} max_abs_diff={max_abs_diff} -> {status}")

    print("ALL_OK" if all_ok else "SOME_MISMATCH")


if __name__ == "__main__":
    main()
