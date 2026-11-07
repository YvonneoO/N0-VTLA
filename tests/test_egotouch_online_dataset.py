"""EgoTouchOnlineTactileDataset on a synthetic HDF5 with the real 21x21 sensor masks of the EgoTouch build."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

from n0vtla.training import egotouch_online_dataset as E

# Active-cell masks measured on all 1929 sequences of the build (row strings, right grid has the thumb at cols 1-3).
_RIGHT = ["0" * 21, "0" * 21] + ["000001110111011101110"] * 4 + ["011100000000000000000"] + \
         ["011101110111011101110"] * 3 + ["000001110111011101110", "011101110111011101110", "011100000000000000000",
                                         "000001111111111111110"] + ["011111111111111111110"] * 4 + ["0" * 21] * 3
RIGHT_MASK = np.array([[c == "1" for c in r] for r in _RIGHT])
LEFT_MASK = RIGHT_MASK[:, ::-1]


def _write(path: Path, n: int = 80, value: float = 0.5) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_group("metadata").attrs["task_name"] = "arrange_pillow"
        g = f.create_group("images")
        for v in E.RGB_VIEWS.values():
            g.create_dataset(v, data=rng.integers(1, 255, (n, 224, 224, 3), dtype=np.uint8))
        p = f.create_group("pressure")
        for hand, mask in (("left", LEFT_MASK), ("right", RIGHT_MASK)):
            grid = np.where(mask, value, np.nan).astype(np.float32)
            p.create_dataset(f"{hand}_pressure_grid", data=np.repeat(grid[None], n, 0))


@unittest.skipIf(h5py is None, "h5py not installed")
class EgoTouchDatasetTests(unittest.TestCase):
    def test_regions_cover_every_sensor_but_the_pinky_base(self):
        for hand, mask in (("left", LEFT_MASK), ("right", RIGHT_MASK)):
            mapped = np.zeros((21, 21), bool)
            for pad in E.REGIONS_RIGHT:
                rs, cs = E.region(hand, pad)
                self.assertFalse(mapped[rs, cs].any(), f"{hand} pad {pad} overlaps another pad")
                mapped[rs, cs] = True
            unmapped = mask & ~mapped
            pinky = (slice(9, 12), slice(17, 20) if hand == "right" else slice(1, 4))
            self.assertEqual(int(unmapped.sum()), int(mask[pinky].sum()))
            self.assertTrue(unmapped[pinky].all())

    def test_sample_shapes_baseline_and_future_pad(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.hdf5"
            _write(path)
            ds = E.EgoTouchOnlineTactileDataset([path], future_frame_offset=50)
            self.assertEqual(len(ds), 80)
            s = ds[10]
            self.assertEqual(s["task"], "arrange pillow")
            for key in E.RGB_VIEWS:
                self.assertEqual(s[key].shape, (224, 224, 3))
                self.assertTrue((s[key][:28] == 0).all() and (s[key][-28:] == 0).all())   # 4:3 letterbox bars
            for key in E.TACTILE_KEYS:
                self.assertEqual(s[key].shape, (3, 224, 224, 3))
                self.assertEqual(s[key].dtype, np.uint8)
                self.assertEqual(s[f"{key}_is_pad"].tolist(), [False, False, False])
                # x = 0.5 -> gray round(1.5 * 255 / 9) = 42 on sensor cells, 28 (x = 0) on the few no-sensor cells
                # inside a region block, 0 outside the pad slots
                vals, counts = np.unique(s[key][1], return_counts=True)
                self.assertEqual(set(vals.tolist()), {0, 28, 42})
                self.assertGreater(counts[vals == 42][0], 10 * counts[vals == 28][0])
            self.assertTrue(ds[60]["observation.image.left_wrist_left_tactile_is_pad"][2])   # 60 + 50 > 79

    def test_left_canvas_mirrors_right(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.hdf5"
            _write(path)
            s = E.EgoTouchOnlineTactileDataset([path])[0]
            left = s["observation.image.left_wrist_left_tactile"][1] > 0
            right = s["observation.image.right_wrist_right_tactile"][1] > 0
            self.assertGreater((left == right[:, ::-1]).mean(), 0.99)

    def test_jpeg_repack_reads_like_the_source(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import repack_egotouch_jpeg as R

        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "src" / "Home" / "t", Path(tmp) / "dst"
            src.mkdir(parents=True)
            _write(src / "seq.hdf5")
            self.assertEqual(R.repack_one((src / "seq.hdf5", dst / "seq.hdf5", 95))[1], "ok")
            a = E.EgoTouchOnlineTactileDataset([src / "seq.hdf5"])[30]
            b = E.EgoTouchOnlineTactileDataset([dst / "seq.hdf5"])[30]
            for key in E.TACTILE_KEYS:
                np.testing.assert_array_equal(a[key], b[key])
            for key in E.RGB_VIEWS:
                self.assertEqual(b[key].shape, (224, 224, 3))
            self.assertEqual(a["task"], b["task"])

    def test_short_sequence_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, short = Path(tmp) / "a.hdf5", Path(tmp) / "b.hdf5"
            _write(good)
            _write(short, n=40)
            kept, sample_ep, _ = E.index_sequences([good, short])
            self.assertEqual(kept, [good])
            self.assertEqual(len(sample_ep), 80)


if __name__ == "__main__":
    unittest.main()
