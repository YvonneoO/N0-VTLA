#!/usr/bin/env python3
"""Legacy pressure/shear comparison previews; not the current training encoding.

The current adapter uses itw_pressure.py and fixed pressure-only normalization.

Alignment follows tacWAM/tujian_v2.py and cosmos_tactile/pad_data.py: common
coverage of three cameras and both gloves, rational 30 Hz grid, nearest samples.
Canonical previews preserve pressure/shear channels. The pressure-only QC view
discards shear and adds pad outlines, so must not be substituted for training input.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from itw_tactile_smoke_adapter import (
    NEUTRAL_TAC_RGB, RGB_KEYS, TACTILE_SLOT_LAYOUT, _mirror_box,
    _pad_ids, _put_resized, _robust_limits, _slot_rgb,
)


def nearest_indices(source, query):
    if len(source) < 2 or np.any(np.diff(source) < 0):
        raise ValueError("Timestamps must be nondecreasing and contain at least two samples")
    right = np.clip(np.searchsorted(source, query), 1, len(source) - 1)
    left = right - 1
    index = np.where(query - source[left] <= source[right] - query, left, right)
    return index, np.abs(source[index] - query)


def load_hand(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files
                  if k == "timestamps" or k.startswith(("tactile_", "tf_tactile_"))}
    ids = _pad_ids(arrays)
    if set(ids) != set(TACTILE_SLOT_LAYOUT):
        raise ValueError(f"Unexpected tactile pad layout: {ids}")
    stride = max(1, len(arrays["timestamps"]) // 200)
    pressure = _robust_limits([arrays[f"tactile_{p}"][::stride] for p in ids])
    shear = [arrays[k][::stride].reshape(-1) for k in arrays if k.startswith("tf_tactile_")]
    magnitude = float(np.nanpercentile(np.abs(np.concatenate(shear)), 99)) if shear else 1.0
    return arrays, pressure, magnitude


def outline(canvas, box):
    cx, cy, w, h = box
    p0 = (round((cx - w / 2) * 224), round((cy - h / 2) * 224))
    p1 = (round((cx + w / 2) * 224), round((cy + h / 2) * 224))
    cv2.rectangle(canvas, p0, p1, (62, 62, 62), 1)


def render_hand(hand, index, mirror):
    arrays, pressure, magnitude = hand
    blue = np.broadcast_to(NEUTRAL_TAC_RGB, (224, 224, 3)).copy()
    black = np.zeros_like(blue)
    qc = np.zeros_like(blue)
    for pad, original_box in TACTILE_SLOT_LAYOUT.items():
        box = _mirror_box(original_box) if mirror else original_box
        rgb = _slot_rgb(arrays, pad, index, pressure, magnitude)
        if pad == "18":
            rgb = np.rot90(rgb)
        heat = cv2.cvtColor(cv2.applyColorMap(rgb[..., 0], cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        heat[rgb[..., 0] == 0] = 0
        _put_resized(blue, rgb, box)
        _put_resized(black, rgb, box)
        _put_resized(qc, heat, box)
        outline(qc, box)
    return blue, black, qc


def writer(path):
    return imageio.get_writer(str(path), fps=30, codec="libx264", pixelformat="yuv420p",
                             macro_block_size=1, ffmpeg_params=["-profile:v", "baseline",
                                                               "-crf", "18", "-movflags", "+faststart"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hands = {side: load_hand(args.episode / f"{side}_hand_data.npz") for side in ("left", "right")}
    streams = {side: np.rint(value[0]["timestamps"] * 1e9).astype(np.int64)
               for side, value in hands.items()}
    frame_ids = {}
    for filename in RGB_KEYS.values():
        name = Path(filename).stem
        with (args.episode / f"{name}.csv").open() as stream:
            reader = csv.DictReader(stream, skipinitialspace=True)
            rows = list(reader)
        frame_ids[name] = np.array([int(r["frame_index"]) for r in rows])
        streams[name] = np.rint(np.array([float(r["timestamp_s"]) for r in rows]) * 1e9).astype(np.int64)
    start, end = max(t[0] for t in streams.values()), min(t[-1] for t in streams.values())
    count = max(0, int(np.ceil((end - start) * 30 / 1e9)))
    master = start + np.rint(np.arange(count) * 1e9 / 30).astype(np.int64)
    master = master[master < end]
    if not len(master):
        raise ValueError("No common camera/glove time coverage")
    mapping, report = {"master_timestamp_ns": master}, {}
    for name, timestamps in streams.items():
        index, error = nearest_indices(timestamps, master)
        limit = 20_000_000 if name in hands else 17_500_000
        mapping[name + "_index"] = index
        mapping[name + "_error_ns"] = error
        if name in frame_ids:
            mapping[name + "_video_frame_index"] = frame_ids[name][index]
        report[name] = {"max_error_ms": float(error.max() / 1e6),
                        "invalid_frames": int((error > limit).sum())}
    np.savez_compressed(args.output / "source_mapping.npz", **mapping)
    metadata = {"episode": str(args.episode), "frames": len(master), "fps": 30,
                "alignment": report, "purpose": "visual inspection only",
                "normalization": {side: {"pressure_limits": value[1], "shear_magnitude": value[2]}
                                  for side, value in hands.items()}}
    (args.output / "alignment_audit.json").write_text(json.dumps(metadata, indent=2))
    names = ("canonical_neutral", "canonical_black", "pressure_qc")
    outputs = {name: writer(args.output / f"{name}.mp4") for name in (*names, "comparison")}
    best_score, best_frame = -1.0, None
    try:
        for t in range(len(master)):
            left = render_hand(hands["left"], mapping["left_index"][t], True)
            right = render_hand(hands["right"], mapping["right_index"][t], False)
            comparison = np.zeros((288, 1344, 3), np.uint8)
            for col, name in enumerate(names):
                pair = np.concatenate((left[col], right[col]), axis=1)
                outputs[name].append_data(pair)
                x = col * 448
                comparison[64:288, x:x + 448] = pair
                title = ("Canonical: neutral background", "Canonical: black background",
                         "QC: pressure only (no shear)")[col]
                cv2.putText(comparison, title, (x + 12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            .52, (230, 230, 230), 1, cv2.LINE_AA)
                for label, offset in (("LEFT", 64), ("RIGHT", 288)):
                    cv2.putText(comparison, label, (x + offset, 51), cv2.FONT_HERSHEY_SIMPLEX,
                                .42, (165, 165, 165), 1, cv2.LINE_AA)
            outputs["comparison"].append_data(comparison)
            score = float(left[1][..., 0].sum() + right[1][..., 0].sum())
            if score > best_score:
                best_score, best_frame = score, comparison.copy()
    finally:
        for output in outputs.values():
            output.close()
    imageio.imwrite(args.output / "comparison_contact.png", best_frame)
    print(json.dumps(metadata, indent=2))
    for name in (*names, "comparison"):
        path = args.output / f"{name}.mp4"
        print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
