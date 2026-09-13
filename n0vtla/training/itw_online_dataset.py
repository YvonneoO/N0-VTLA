"""Online (no-materialization) ITW/tacWAM dataset for Stage-1 training.

Reads episodes directly from a raw tacWAM-format tree (rgb_head.mp4, wrist_left.mp4,
wrist_right.mp4, left_hand_data.npz, right_hand_data.npz, csvs -- the exact layout
scripts/itw_pressure.py and scripts/itw_tactile_smoke_adapter.py already parse) and
aligns/rasterizes each sample on read, so training over a corpus far too large to
duplicate as a materialized canonical dataset (2.5TB raw vs ~1.1TB free project quota
on VISION, 2026-09-13) never needs a converted copy on disk at all.

Reuses scripts/itw_pressure.py's alignment (`aligned_timeline`) and pressure
normalization/rasterization (`load_hand_pressure_arrays`, `rasterize_pressure_frame`)
UNCHANGED, so a sample rasterized here is pixel-identical to what the offline converter
would have produced for the same (episode, frame) -- there is no second numerical path
to drift out of sync with the fixed train-only statistics.

KNOWN-UNVERIFIED RISK (flagging explicitly, not papering over it): RGB frames are read
via `torchcodec.decoders.VideoDecoder`'s indexed access for frame-ACCURATE random seeks.
The offline converter (`itw_pressure.write_aligned_rgb`) deliberately avoids
`cv2`'s `CAP_PROP_POS_FRAMES` for exactly this reason -- on H.264, OpenCV's seek often
lands on the nearest keyframe rather than the requested frame, and unlike the offline
path (which reads every video strictly sequentially from frame 0), an online loader
fundamentally needs random access into shuffled (episode, frame) pairs, so the
"just read sequentially" workaround the offline path uses isn't available here.
torchcodec is built for accurate ML-training random access and is already a pinned
dependency, but this module has NOT been validated against a real episode yet (no
sample tacWAM data was available locally while writing it) -- before trusting it for a
real training run, decode the SAME frame both through this module and through
`itw_pressure.write_aligned_rgb` on one real episode and diff the two images.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from itw_pressure import (  # noqa: E402
    aligned_timeline,
    load_hand_pressure_arrays,
    load_normalization,
    rasterize_pressure_frame,
)

RGB_KEYS = {
    "observation.image.third_view": "rgb_head",
    "observation.image.left_wrist_view": "wrist_left",
    "observation.image.right_wrist_view": "wrist_right",
}
TACTILE_KEYS = {
    "observation.image.left_wrist_left_tactile": ("left", "left_hand_data.npz"),
    "observation.image.right_wrist_right_tactile": ("right", "right_hand_data.npz"),
}
DEFAULT_TASK = "Perform the task."


def read_task(ep_dir: Path) -> str:
    """Same convention as scripts/itw_tactile_smoke_adapter.py::_read_task."""
    path = ep_dir / "task_info.json"
    if not path.exists():
        return DEFAULT_TASK
    info = json.loads(path.read_text(encoding="utf-8"))
    steps = info.get("steps") or []
    if steps:
        return str(steps[0])
    return str(info.get("name") or DEFAULT_TASK)


def read_task_name_for_norm(ep_dir: Path) -> str | None:
    """The task-identity key a per-task-scale normalization file's `task_scale` table
    is keyed by -- DELIBERATELY the SAME field (`task_info.json["name"]`) tacWAM's own
    code uses for this purpose (tacwam/tujian_v2.py:766, tacwam/cosmos_tactile/
    pad_data.py:407: both `task.get("name", "")`), and DELIBERATELY DIFFERENT from
    `read_task`'s "steps[0]" convention used for the language-instruction prompt.

    Confirmed via a real tujian_v5_recent episode (VISION job 534565, 2026-09-13) that
    these two fields genuinely differ in the same file: `name` is a short canonical
    task label (e.g. "把水杯挪出备菜区", matching the per-task-scale table's key
    style), `steps[0]` is a long multi-sentence instruction paragraph -- using
    `read_task`'s value here would silently miss every table entry (long freeform
    text never matches a canonical-name key) and fall back to `default_scale` for
    every sample, quietly reproducing the crushed-signal problem per-task-scale
    exists to fix, without ever raising an error.

    Returns None (not "") when absent, so an empty-string task never accidentally
    collides with an empty key in the table -- normalize_pressure's None handling
    (see scripts/itw_pressure.py) then correctly falls back to `default_scale`.
    """
    path = ep_dir / "task_info.json"
    if not path.exists():
        return None
    info = json.loads(path.read_text(encoding="utf-8"))
    name = info.get("name")
    return str(name) if name else None


def letterbox_resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Same aspect-preserving letterbox as scripts/itw_pressure.py::write_aligned_rgb,
    applied to a single already-decoded frame instead of a whole video."""
    import cv2

    h, w = img.shape[:2]
    ratio = size / max(h, w)
    resized = cv2.resize(img, (max(1, round(w * ratio)), max(1, round(h * ratio))))
    canvas = np.zeros((size, size, 3), np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def index_episodes(episode_dirs: list[Path], *, min_frames: int = 51) -> list[tuple[Path, int]]:
    """Flat (episode_dir, frame_position) sample index.

    Every frame position in an episode's aligned 30Hz timeline is indexed as a valid
    "current" sample; future-frame tail-clamping is a per-sample mask applied in
    `ITWOnlineTactileDataset.__getitem__` (matching
    n0vtla/policies/canonical_tactile_policy.py::CanonicalTactileInputs), not an
    exclusion here. Episodes that fail `aligned_timeline`'s own red-line checks (bad
    clocks, missing files, too-short overlap -- the same gates
    scripts/itw_pressure.py enforces at offline-conversion time) are skipped with a
    warning instead of aborting the whole index build; one bad recording should not
    block training on the other tens of thousands.
    """
    index: list[tuple[Path, int]] = []
    n_skipped = 0
    for ep_dir in episode_dirs:
        try:
            mapping, _audit = aligned_timeline(ep_dir)
        except (ValueError, OSError, FileNotFoundError) as exc:
            logging.warning("itw_online_dataset: skipping %s (%s)", ep_dir, exc)
            n_skipped += 1
            continue
        n = len(mapping["master_timestamp_ns"])
        if n < min_frames:
            logging.warning("itw_online_dataset: skipping %s (only %d aligned frames)", ep_dir, n)
            n_skipped += 1
            continue
        index.extend((ep_dir, t) for t in range(n))
    logging.info(
        "itw_online_dataset: indexed %d episodes (%d skipped), %d total samples",
        len(episode_dirs) - n_skipped, n_skipped, len(index),
    )
    return index


def task_scale_coverage(episode_dirs: list[Path], normalization: dict) -> dict:
    """Fraction of `episode_dirs` whose task name resolves to a real `task_scale` table
    entry, vs. falling back to `default_scale`. Run this once against a sample of the
    real corpus before trusting per-task-scale on a real training run: normalize_
    pressure intentionally never raises on a name/table mismatch (matching tacWAM's
    own scale_for), so a systematic wiring mistake (e.g. reading the wrong
    task_info.json field) would otherwise silently degrade every sample to the
    shared/global-scale behavior tacWAM's own audit already rejected, with no error
    anywhere to catch it.

    No-op (`match_rate=None`) for a non-per-task normalization file.
    """
    if "task_scale" not in normalization:
        return {"matched": 0, "total": len(episode_dirs), "match_rate": None}
    table = normalization["task_scale"]
    total = len(episode_dirs)
    matched = sum(1 for ep in episode_dirs if read_task_name_for_norm(ep) in table)
    return {"matched": matched, "total": total, "match_rate": (matched / total) if total else None}


class _EpisodeCache:
    """Alignment + tactile arrays + open video decoders for ONE episode.

    Size-1: random shuffling means consecutive __getitem__ calls rarely share an
    episode, but within one shuffled epoch the SAME episode is still visited many
    times (once per its own frame count), just not consecutively -- an LRU of a few
    episodes would cut re-alignment/re-load work further; not implemented yet since
    the current bottleneck (video decode) dominates either way.
    """

    def __init__(self):
        self.episode_dir: Path | None = None
        self.mapping: dict | None = None
        self.pressure: dict[str, dict[str, np.ndarray]] = {}
        self.decoders: dict[str, "VideoDecoder"] = {}  # noqa: F821 -- lazily imported, see rgb_frame

    def load(self, episode_dir: Path, normalization: dict) -> None:
        if self.episode_dir == episode_dir:
            return
        self.episode_dir = episode_dir
        self.mapping, _audit = aligned_timeline(episode_dir)
        # Resolved once per episode (task identity is episode-level, not per-frame) --
        # ignored by load_hand_pressure_arrays/normalize_pressure unless `normalization`
        # is a per-task-scale file. See read_task_name_for_norm's docstring for why
        # this is deliberately NOT the same field as the language-instruction prompt.
        task_name = read_task_name_for_norm(episode_dir)
        self.pressure = {
            hand: load_hand_pressure_arrays(episode_dir / npz_name, normalization, hand=hand, task_name=task_name)
            for hand, npz_name in (("left", "left_hand_data.npz"), ("right", "right_hand_data.npz"))
        }
        self.decoders = {}  # lazy: only open the views actually requested

    def rgb_frame(self, view: str, frame_index: int) -> np.ndarray:
        decoder = self.decoders.get(view)
        if decoder is None:
            from torchcodec.decoders import VideoDecoder  # lazy: only needed when actually decoding

            decoder = VideoDecoder(str(self.episode_dir / f"{view}.mp4"))
            self.decoders[view] = decoder
        frame = decoder[frame_index]  # (C, H, W) uint8, frame-accurate per torchcodec's docs
        return frame.permute(1, 2, 0).numpy()  # -> (H, W, C)


class ITWOnlineTactileDataset(torch.utils.data.Dataset):
    """One sample = one (episode, frame) position, aligned + rasterized on read.

    Produces the same flat-dict shape
    n0vtla/policies/canonical_tactile_policy.py::CanonicalTactileInputs expects as
    input, MINUS `observation.state`/`action` (n0vtla.policies.canonical_tactile_policy
    .Stage1ObservationOnly overwrites both with zero placeholders regardless -- see
    docs/PRETRAIN_IMPLEMENTATION.md Sec 2.3) and `prompt` (n0vtla.training.config
    .ModelTransformFactory's InjectDefaultPrompt already injects a default when the key
    is absent). Callers must still apply Stage1ObservationOnly -> CanonicalTactileInputs
    -> model_transforms afterward, exactly as the offline (materialized-dataset) path
    does -- this class's only job is producing the raw canonical-shaped sample without
    a materialized copy on disk, not a fully model-ready tensor.
    """

    def __init__(
        self,
        episode_dirs: list[Path],
        normalization_path: str | Path,
        *,
        future_frame_offset: int = 50,
    ):
        self.normalization = load_normalization(normalization_path)
        self.future_frame_offset = int(future_frame_offset)
        self.index = index_episodes(episode_dirs)
        if not self.index:
            raise ValueError("no valid episodes indexed -- see the warnings logged above")
        self._cache = _EpisodeCache()
        self._task_cache: dict[Path, str] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _task(self, ep_dir: Path) -> str:
        if ep_dir not in self._task_cache:
            self._task_cache[ep_dir] = read_task(ep_dir)
        return self._task_cache[ep_dir]

    def __getitem__(self, idx: int) -> dict:
        ep_dir, t = self.index[idx]
        self._cache.load(ep_dir, self.normalization)
        mapping = self._cache.mapping
        n = len(mapping["master_timestamp_ns"])
        h = self.future_frame_offset
        t_future = min(t + h, n - 1)
        future_is_pad = t_future != t + h  # tail-clamped -> not a valid InfoNCE target

        sample: dict = {"task": self._task(ep_dir)}

        for out_key, view in RGB_KEYS.items():
            frame_idx = int(mapping[f"{view}_frame_index"][t])
            sample[out_key] = letterbox_resize(self._cache.rgb_frame(view, frame_idx))

        for out_key, (hand, _npz_name) in TACTILE_KEYS.items():
            arrays = self._cache.pressure[hand]
            hand_index = mapping[f"{hand}_index"]
            i_baseline, i_current, i_future = (
                int(hand_index[0]), int(hand_index[t]), int(hand_index[t_future])
            )
            baseline = rasterize_pressure_frame(arrays, i_baseline, hand=hand)
            current = rasterize_pressure_frame(arrays, i_current, hand=hand)
            future = rasterize_pressure_frame(arrays, i_future, hand=hand)
            sample[out_key] = np.stack([baseline, current, future], axis=0)
            sample[f"{out_key}_is_pad"] = np.array([False, False, future_is_pad], dtype=bool)

        return sample


def list_episode_dirs(raw_root: str | Path, date_dirs: list[str] | None = None) -> list[Path]:
    """Enumerate `<raw_root>/<date>/<episode_uuid>` directories.

    `date_dirs`: restrict to these date subfolder names (e.g. a train-split subset of
    tujian_v5_recent's 38 dates); None scans every date under `raw_root`. Only lists
    directories (cheap top-level `iterdir`, not a recursive scan -- see @TAMU/CLAUDE.md's
    no-broad-find rule for the shared VISION filesystem) -- alignment validity is
    checked later, per-episode, in `index_episodes`.
    """
    raw_root = Path(raw_root)
    dates = date_dirs if date_dirs is not None else sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    episodes = []
    for date in dates:
        date_dir = raw_root / date
        if not date_dir.is_dir():
            logging.warning("itw_online_dataset: date dir missing, skipping: %s", date_dir)
            continue
        episodes.extend(sorted(p for p in date_dir.iterdir() if p.is_dir()))
    return episodes
