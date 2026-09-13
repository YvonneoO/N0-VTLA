import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from itw_pressure import HANDS, PAD_IDS, SCHEMA  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n0vtla.training.itw_online_dataset import (  # noqa: E402
    ITWOnlineTactileDataset,
    _EpisodeCache,
    index_episodes,
    letterbox_resize,
    list_episode_dirs,
    read_task,
)


def _write_camera_csv(root: Path, name: str, n: int = 90, start_frame: int = 2) -> None:
    with (root / f"{name}.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp_s"])
        for i in range(n):
            writer.writerow([i + start_frame, 1000 + i / 30])


def _write_hand_npz(root: Path, hand: str, n: int = 120) -> None:
    """Each pad's pressure at index i is `(i % 6) * (pad_slot + 1) * 0.1`: bounded well
    inside normalize_pressure's [-1, 8] clip range (an earlier unbounded ramp saturated
    every index past ~8 to the same clipped max, making current/future indistinguishable
    by construction -- a test-fixture bug, not a module bug) and non-monotonic, so two
    different frame indices reliably render different pixel values."""
    data = {"timestamps": 1000 + np.arange(n) / 40}
    for slot, pad in enumerate(PAD_IDS):
        ramp = (np.arange(n, dtype=np.float32) % 6) * (slot + 1) * 0.1
        data[f"tactile_{pad}"] = ramp.reshape(n, 1, 1)
    np.savez(root / f"{hand}_hand_data.npz", **data)


def _identity_normalization() -> dict:
    return {
        "schema": SCHEMA,
        "fit_split": "train",
        "normal_baseline": [0.0] * 30,
        "normal_scale": [1.0] * 30,
        "contact_threshold": [0.5] * 30,
    }


def _build_episode(root: Path, *, n_frames: int = 120, task: str = "insert the plug") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for view in ("rgb_head", "wrist_left", "wrist_right"):
        _write_camera_csv(root, view, n=n_frames)
    for hand in HANDS:
        _write_hand_npz(root, hand, n=n_frames)
    (root / "task_info.json").write_text(json.dumps({"steps": [task]}))
    return root


class IndexEpisodesTests(unittest.TestCase):
    def test_indexes_every_frame_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = _build_episode(Path(tmp) / "ep0")
            index = index_episodes([ep])
            # aligned_timeline's master grid length depends on the common overlap; just
            # check every position 0..n-1 is present exactly once, for exactly this episode.
            self.assertTrue(all(e == ep for e, _t in index))
            positions = sorted(t for _e, t in index)
            self.assertEqual(positions, list(range(len(positions))))
            self.assertGreater(len(index), 50)

    def test_skips_bad_episode_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = _build_episode(Path(tmp) / "good")
            bad = Path(tmp) / "bad"
            bad.mkdir()
            _write_camera_csv(bad, "rgb_head", n=90)
            # wrist_left/wrist_right csvs missing entirely -> aligned_timeline raises,
            # index_episodes must skip it rather than propagate.
            index = index_episodes([good, bad])
            self.assertTrue(all(e == good for e, _t in index))

    def test_skips_too_short_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            short = _build_episode(Path(tmp) / "short", n_frames=10)
            index = index_episodes([short], min_frames=51)
            self.assertEqual(index, [])


class ReadTaskAndResizeTests(unittest.TestCase):
    def test_read_task_from_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = _build_episode(Path(tmp) / "ep", task="close the drawer")
            self.assertEqual(read_task(ep), "close the drawer")

    def test_read_task_default_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = Path(tmp) / "ep"
            ep.mkdir()
            self.assertEqual(read_task(ep), "Perform the task.")

    def test_letterbox_preserves_aspect_and_pads(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)  # 2:1 landscape
        out = letterbox_resize(img, size=224)
        self.assertEqual(out.shape, (224, 224, 3))
        # Top rows should be the zero-padded letterbox bars, not resized content.
        np.testing.assert_array_equal(out[0], 0)


class OnlineDatasetGetItemTests(unittest.TestCase):
    """Exercises the FULL __getitem__ (mask/index math for baseline/current/future),
    mocking only the video-decode step (`_EpisodeCache.rgb_frame`) since torchcodec
    isn't available in this environment -- see the module's KNOWN-UNVERIFIED-RISK note
    for why the video-decode path itself still needs an on-cluster check against real
    tacWAM data before being trusted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ep = _build_episode(self.root / "ep0", n_frames=120)
        self.norm_path = self.root / "norm.json"
        self.norm_path.write_text(json.dumps(_identity_normalization()))

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, future_frame_offset=50):
        with mock.patch.object(_EpisodeCache, "rgb_frame", return_value=np.zeros((224, 224, 3), np.uint8)):
            ds = ITWOnlineTactileDataset(
                [self.ep], self.norm_path, future_frame_offset=future_frame_offset
            )
        return ds

    def test_sample_shapes_and_keys(self):
        ds = self._dataset()
        with mock.patch.object(_EpisodeCache, "rgb_frame", return_value=np.zeros((224, 224, 3), np.uint8)):
            sample = ds[10]
        for key in (
            "observation.image.third_view",
            "observation.image.left_wrist_view",
            "observation.image.right_wrist_view",
        ):
            self.assertEqual(sample[key].shape, (224, 224, 3))
        for key in (
            "observation.image.left_wrist_left_tactile",
            "observation.image.right_wrist_right_tactile",
        ):
            self.assertEqual(sample[key].shape, (3, 224, 224, 3))  # [baseline, current, future]
            mask = sample[f"{key}_is_pad"]
            np.testing.assert_array_equal(mask, [False, False, False])  # not tail-clamped at t=10
        self.assertEqual(sample["task"], "insert the plug")

    def test_baseline_current_future_are_distinct_frames(self):
        # With the per-pad value ramp in _write_hand_npz, baseline (t=0) must render
        # DARKER than a mid-episode current/future frame -- catches an accidental
        # baseline==current indexing bug (e.g. reusing `t` for both).
        # Small offset, deliberately far from the tail: isolates "are the three frames
        # distinct" from tail-clamp behavior, which test_future_tail_clamp_sets_pad_mask
        # covers separately.
        ds = self._dataset(future_frame_offset=5)
        with mock.patch.object(_EpisodeCache, "rgb_frame", return_value=np.zeros((224, 224, 3), np.uint8)):
            sample = ds[10]
        stack = sample["observation.image.left_wrist_left_tactile"]
        baseline, current, future = stack[0], stack[1], stack[2]
        # Canvas background is black (0); zero pressure inside a sensor pad renders gray
        # 28 (see itw_pressure.pressure_rgb) -- baseline (frame 0, zero pressure by
        # construction) must contain ONLY those two values, nothing brighter.
        self.assertTrue(set(np.unique(baseline).tolist()) <= {0, 28})
        self.assertGreater(int(current.max()), 28)  # ramped pressure at t=10 is brighter
        self.assertFalse((current == baseline).all())
        self.assertFalse((future == current).all())

    def test_future_tail_clamp_sets_pad_mask(self):
        # future_frame_offset larger than remaining frames from t -> clamped AND masked.
        ds = self._dataset(future_frame_offset=1000)
        with mock.patch.object(_EpisodeCache, "rgb_frame", return_value=np.zeros((224, 224, 3), np.uint8)):
            sample = ds[0]
        mask = sample["observation.image.left_wrist_left_tactile_is_pad"]
        self.assertEqual(mask.tolist(), [False, False, True])


class ListEpisodeDirsTests(unittest.TestCase):
    def test_lists_only_directories_under_requested_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "date_a" / "ep1").mkdir(parents=True)
            (root / "date_a" / "ep2").mkdir(parents=True)
            (root / "date_b" / "ep3").mkdir(parents=True)
            (root / "date_a" / "not_a_dir.txt").write_text("x")
            episodes = list_episode_dirs(root, date_dirs=["date_a"])
            self.assertEqual(sorted(p.name for p in episodes), ["ep1", "ep2"])


if __name__ == "__main__":
    unittest.main()
