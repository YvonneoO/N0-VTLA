"""Online Stage-1 (mid-train) dataset over the EgoTouch 3-view@224 HDF5 build (one <Scene>/<task>/<seq>.hdf5 per
sequence, built by Ego2Contact's scripts/core/convert_to_hdf5.py --img_size 224).

Produces the same raw sample dict as itw_online_dataset.ITWOnlineTactileDataset, so every transform after it
(Stage1ObservationOnly -> CanonicalTactileInputs -> model transforms) is unchanged:

  RGB      images/chest_color -> third_view, images/left_color -> left_wrist_view, images/right_color ->
           right_wrist_view. The build squashed each 640x480 (4:3) frame to 224x224; the frame is un-squashed to
           224x168 and letterboxed like every other N0-VTLA image (itw_online_dataset.letterbox_resize).
  tactile  pressure/{left,right}_pressure_grid (T, 21, 21), already baseline-corrected and divided by the
           sequence's own maximum (values 0..1, NaN = no sensor). The value is used directly as the normalized
           pressure x of itw_pressure.normalize_pressure (x = 1 at the sequence's peak), no normalization file.
           Each anatomical region of the 21x21 glove is cut out and drawn in the slot of the corresponding Tujian
           pad on the same 224x224 hand canvas the robot glove uses (itw_pressure.rasterize_pressure_frame), so
           mid-train and post-train see one tactile image layout. The pinky base region has no Tujian pad (17 is
           absent) and is dropped. Stack = [baseline = frame 0, current, current + future_offset], like TouchScale.

Region map (rows / cols inclusive, in the RIGHT grid's orientation: thumb at cols 1-3, fingers up; the left grid
is the horizontal mirror, c -> 20 - c). Tip / mid-base rows follow Ego2Contact's build_egotouch_partmap; its merged
"mid-base" block of the four fingers (rows 7-11) is split 7-8 = middle phalanx, 9-11 = proximal (Tujian mid pads
have 3 rows, base pads 4). Thumb base = the thumb-side end of the palm band (partmap's thumb_base).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from n0vtla.training.itw_online_dataset import _frames, letterbox_resize, rasterize_pressure_frame

RGB_VIEWS = {
    "observation.image.third_view": "chest_color",
    "observation.image.left_wrist_view": "left_color",
    "observation.image.right_wrist_view": "right_color",
}
TACTILE_KEYS = {
    "observation.image.left_wrist_left_tactile": "left",
    "observation.image.right_wrist_right_tactile": "right",
}
SOURCE_ASPECT = 4 / 3          # native EgoTouch camera frames are 640x480

# Tujian pad id -> (row0, row1, col0, col1) in the right grid. Pad ids/anatomy: memory tacwam-glove-spec.
REGIONS_RIGHT = {
    "0": (6, 9, 1, 3),      # thumb tip
    "1": (11, 12, 1, 3),    # thumb mid
    "2": (13, 17, 1, 4),    # thumb base (palm band, thumb side)
    "3": (2, 5, 5, 7),      # index tip
    "4": (7, 8, 5, 7),      # index mid
    "5": (9, 11, 5, 7),     # index base
    "7": (2, 5, 9, 11),     # middle tip
    "8": (7, 8, 9, 11),
    "9": (9, 11, 9, 11),
    "11": (2, 5, 13, 15),   # ring tip
    "12": (7, 8, 13, 15),
    "13": (9, 11, 13, 15),
    "15": (2, 5, 17, 19),   # pinky tip
    "16": (7, 8, 17, 19),   # pinky mid; pinky base rows 9-11 cols 17-19 dropped (no Tujian pad 17)
    "18": (13, 17, 5, 19),  # palm
}


def region(hand: str, pad: str) -> tuple[slice, slice]:
    r0, r1, c0, c1 = REGIONS_RIGHT[pad]
    if hand == "left":
        c0, c1 = 20 - c1, 20 - c0
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def grid_to_pad_arrays(grid: np.ndarray, hand: str) -> dict[str, np.ndarray]:
    """(T, 21, 21) EgoTouch grid -> {pad id: (T, rows, cols) normalized pressure} for rasterize_pressure_frame.

    NaN (no sensor) -> 0. rasterize_pressure_frame rotates pad 18 by 90 degrees (Tujian palm is stored portrait);
    the palm block is landscape already, so it is passed pre-rotated the other way and comes out as-is.
    """
    g = np.nan_to_num(np.asarray(grid, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out = {}
    for pad in REGIONS_RIGHT:
        rs, cs = region(hand, pad)
        block = g[:, rs, cs]
        out[pad] = np.ascontiguousarray(np.rot90(block, k=-1, axes=(1, 2)) if pad == "18" else block)
    return out


def unsquash(frame: np.ndarray, aspect: float = SOURCE_ASPECT) -> np.ndarray:
    """224x224 squashed frame -> letterboxed 224x224 with the source aspect restored."""
    import cv2

    h, w = frame.shape[:2]
    restored = cv2.resize(frame, (w, max(1, round(w / aspect))), interpolation=cv2.INTER_AREA)
    return letterbox_resize(restored)


def task_prompt(task_name: str) -> str:
    return task_name.replace("_", " ").strip() or "Perform the task."


def sequence_frames(path: Path) -> int:
    import h5py

    with h5py.File(path, "r") as f:
        lengths = [f["images"][v].shape[0] for v in RGB_VIEWS.values()]
        lengths += [f["pressure"][f"{h}_pressure_grid"].shape[0] for h in ("left", "right")]
    return int(min(lengths))


def index_sequences(paths: list[Path], *, min_frames: int = 51, cache_path: str | Path | None = None):
    """(kept paths, sample_seq int32, sample_t int32); same contract and cache format as
    itw_online_dataset.index_episodes ({"episodes": [[path, n_frames], ...]})."""
    if cache_path is not None and Path(cache_path).exists():
        rows = json.loads(Path(cache_path).read_text())["episodes"]
        want = [str(p) for p in paths]
        have = [r[0] for r in rows]
        if not set(have) <= set(want) or [w for w in want if w in set(have)] != have:
            raise ValueError(f"index cache {cache_path} does not match the requested sequence list")
        kept = [Path(r[0]) for r in rows]
        counts = np.array([int(r[1]) for r in rows], np.int64)
    else:
        kept, counts = [], []
        for p in paths:
            try:
                n = sequence_frames(p)
            except (OSError, KeyError) as exc:
                logging.warning("egotouch_online_dataset: skipping %s (%s)", p, exc)
                continue
            if n < min_frames:
                logging.warning("egotouch_online_dataset: skipping %s (only %d frames)", p, n)
                continue
            kept.append(p)
            counts.append(n)
        counts = np.array(counts, np.int64)
    logging.info("egotouch_online_dataset: %d sequences, %d samples", len(kept), int(counts.sum()))
    return kept, np.repeat(np.arange(len(kept), dtype=np.int32), counts), _frames(counts)


class _SequenceCache:
    """Open HDF5 handle of ONE sequence (opened lazily inside each worker). Only the frames a sample needs are read:
    shuffled training switches sequence on almost every sample, so loading whole grids would dominate the cost."""

    def __init__(self):
        self.path: Path | None = None
        self.file = None
        self.task = ""

    def load(self, path: Path) -> None:
        if self.path == path:
            return
        import h5py

        if self.file is not None:
            self.file.close()
        self.file = h5py.File(path, "r")
        self.path = path
        self.task = str(self.file["metadata"].attrs.get("task_name", ""))

    def rgb(self, view: str, t: int) -> np.ndarray:
        return self.file["images"][view][t]

    def pads(self, hand: str, frames: list[int]) -> tuple[dict[str, np.ndarray], list[int]]:
        """Pad arrays over the unique sorted `frames` + the position of each requested frame in them."""
        uniq = sorted(set(frames))
        grid = self.file["pressure"][f"{hand}_pressure_grid"][uniq]
        return grid_to_pad_arrays(grid, hand), [uniq.index(t) for t in frames]


class EgoTouchOnlineTactileDataset(torch.utils.data.Dataset):
    """One sample = one (sequence, frame); see the module docstring for what each key holds."""

    def __init__(self, sequences: list[Path], *, future_frame_offset: int = 50, index_cache=None,
                 source_aspect: float = SOURCE_ASPECT):
        self.future_frame_offset = int(future_frame_offset)
        self.source_aspect = float(source_aspect)
        self.episodes, self.sample_ep, self.sample_t = index_sequences(sequences, cache_path=index_cache)
        if len(self.sample_ep) == 0:
            raise ValueError("no valid EgoTouch sequences indexed")
        self.lengths = np.bincount(self.sample_ep, minlength=len(self.episodes))
        self._cache = _SequenceCache()

    def __len__(self) -> int:
        return len(self.sample_ep)

    def __getitem__(self, idx: int) -> dict:
        path, t = self.episodes[int(self.sample_ep[idx])], int(self.sample_t[idx])
        c = self._cache
        c.load(path)
        h = self.future_frame_offset
        t_future = min(t + h, int(self.lengths[int(self.sample_ep[idx])]) - 1)
        future_is_pad = t_future != t + h
        sample: dict = {"task": task_prompt(c.task)}
        for out_key, view in RGB_VIEWS.items():
            sample[out_key] = unsquash(c.rgb(view, t), self.source_aspect)
        for out_key, hand in TACTILE_KEYS.items():
            pads, pos = c.pads(hand, [0, t, t_future])
            sample[out_key] = np.stack([rasterize_pressure_frame(pads, i, hand=hand) for i in pos], axis=0)
            sample[f"{out_key}_is_pad"] = np.array([False, False, future_is_pad], dtype=bool)
        return sample
