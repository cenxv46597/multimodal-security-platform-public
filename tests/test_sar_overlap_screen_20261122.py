import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sar_overlap_screen_20261122 import screen


class OverlapScreenTests(unittest.TestCase):
    def test_shifted_crop(self):
        rng = np.random.default_rng(42)
        scene = rng.integers(0, 256, size=(512 + 176, 512), dtype=np.uint8)
        left = scene[:512]
        right = scene[176:]
        match = screen(left, right)
        self.assertIsNotNone(match)
        self.assertEqual((match['dy'], match['dx']), (176, 0))
        self.assertGreater(match['correlation'], 0.99)

    def test_unrelated_and_constant_images(self):
        rng = np.random.default_rng(23)
        left = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
        right = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
        self.assertIsNone(screen(left, right))
        self.assertIsNone(screen(np.zeros_like(left), np.zeros_like(left)))

    def test_rotated_same_scene(self):
        rng = np.random.default_rng(78)
        scene = rng.integers(0, 256, size=(256 + 64, 256), dtype=np.uint8)
        left, right = scene[:256], np.rot90(scene[64:])
        match = screen(left, right)
        self.assertIsNotNone(match)
        self.assertGreater(match['correlation'], 0.99)

    def test_non_multiple_size(self):
        rng = np.random.default_rng(17)
        image = rng.integers(0, 256, size=(224, 224), dtype=np.uint8)
        match = screen(image, image)
        self.assertIsNotNone(match)
        self.assertEqual((match['dy'], match['dx']), (0, 0))

    def test_rejects_unsupported_geometry(self):
        with self.assertRaises(ValueError):
            screen(np.zeros((128, 256), dtype=np.uint8), np.zeros((128, 256), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
