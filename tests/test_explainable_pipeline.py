"""Small, deterministic checks for the explainable custom path."""
import unittest

import cv2
import numpy as np

from explainable_pipeline import explain_canny, hough_circle_trace


class PipelineTests(unittest.TestCase):
    def test_blank_image_no_edge_no_circle(self):
        image = np.full((128, 160), 30, dtype=np.uint8)
        trace = explain_canny(image)
        circles, votes, profile = hough_circle_trace(trace, min_radius=12, max_radius=35)
        self.assertEqual(int(trace.edges.sum()), 0)
        self.assertEqual(circles, [])
        self.assertEqual(float(votes.sum()), 0)
        self.assertEqual(float(profile.sum()), 0)

    def test_known_five_circles_clean_and_noisy(self):
        truth = [(72, 66, 24), (192, 68, 32), (315, 75, 22),
                 (112, 215, 29), (275, 210, 34)]
        clean = np.full((303, 384), 44, dtype=np.uint8)
        for (x, y, r), intensity in zip(truth, (230, 190, 215, 205, 245)):
            cv2.circle(clean, (x, y), r, intensity, -1)
        rng = np.random.default_rng(2026)
        noisy = np.rint(np.clip(clean + rng.normal(0, 20, clean.shape), 0, 255)).astype(np.uint8)
        for image in (clean, noisy):
            trace = explain_canny(image)
            circles, _, _ = hough_circle_trace(trace)
            self.assertEqual(len(circles), len(truth))
            for x, y, r in truth:
                self.assertTrue(
                    any(np.hypot(c["x"] - x, c["y"] - y) <= 3 and abs(c["r"] - r) <= 3
                        for c in circles),
                    f"Missing circle {(x, y, r)}",
                )

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            explain_canny(np.zeros((0, 0), dtype=np.uint8))
        with self.assertRaises(ValueError):
            explain_canny(np.zeros((16, 16), dtype=np.uint8), low=120, high=50)
        with self.assertRaises(ValueError):
            hough_circle_trace(explain_canny(np.zeros((32, 32), dtype=np.uint8)),
                               min_radius=20, max_radius=12)


if __name__ == "__main__":
    unittest.main()
