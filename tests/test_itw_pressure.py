import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from itw_pressure import HANDS, PAD_IDS, aligned_timeline, fit_normalization, normalize_pressure, pressure_rgb


class PressureTests(unittest.TestCase):
    def test_fit_quantiles_and_corpus_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for h, hand in enumerate(HANDS):
                data = {"timestamps": np.arange(4, dtype=float)}
                for slot, pad in enumerate(PAD_IDS):
                    p = h * 15 + slot
                    scale = 0 if p == 0 else p + 1
                    data[f"tactile_{pad}"] = np.tile(np.arange(4, dtype=np.float32) * scale, (4, 1, 1))
                np.savez(root / f"{hand}_hand_data.npz", **data)
            norm = fit_normalization([root])
            np.testing.assert_array_equal(norm["normal_baseline"], np.zeros(30))
            np.testing.assert_allclose(norm["normal_scale"][1:], np.arange(2, 31) * 3)
            self.assertAlmostEqual(norm["normal_scale"][0], 2.4)
            self.assertEqual(norm, fit_normalization([root]))

    def test_fixed_per_pad_mapping_and_clipping(self):
        norm = {"normal_baseline": [1.] * 30, "normal_scale": [2.] * 30}
        norm["normal_scale"][15] = 4.
        raw = np.array([-100., 1., 3., 100.], dtype=np.float32)
        np.testing.assert_array_equal(normalize_pressure(raw, norm, "left", 0), [-1, 0, 1, 8])
        self.assertEqual(normalize_pressure(raw, norm, "right", 0)[2], .5)
        # No per-episode rescaling: extra extreme observations do not change earlier values.
        np.testing.assert_array_equal(normalize_pressure(raw[:3], norm, "left", 0), [-1, 0, 1])
        with self.assertRaises(ValueError):
            normalize_pressure(np.array([np.nan]), norm, "left", 0)

    def test_pressure_transport_has_no_shear_or_color(self):
        rgb = pressure_rgb(np.array([-1., 0., 8.]))
        np.testing.assert_array_equal(rgb[:, 0], [0, 28, 255])
        np.testing.assert_array_equal(rgb[:, 0], rgb[:, 1])
        np.testing.assert_array_equal(rgb[:, 1], rgb[:, 2])

    def test_common_timeline_and_gap_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def camera(name, drop=False):
                with (root / f"{name}.csv").open("w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["frame_index", "timestamp_s"])
                    for i in range(90):
                        if not drop or i not in range(35, 45):
                            writer.writerow([i + 2, 1000 + i / 30])
            for view in ("rgb_head", "wrist_left", "wrist_right"):
                camera(view)
            for hand in HANDS:
                np.savez(root / f"{hand}_hand_data.npz", timestamps=1000 + np.arange(120) / 40)
            mapping, audit = aligned_timeline(root)
            self.assertGreater(len(mapping["master_timestamp_ns"]), 50)
            self.assertEqual(mapping["rgb_head_frame_index"][0], 2)
            self.assertTrue(all(v["invalid_frames"] == 0 for v in audit.values()))
            camera("wrist_left", drop=True)
            with self.assertRaisesRegex(ValueError, "alignment exceeds"):
                aligned_timeline(root)


if __name__ == "__main__":
    unittest.main()
