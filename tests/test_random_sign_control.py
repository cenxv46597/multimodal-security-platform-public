import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from random_sign_control import random_sign_control


class RandomSignControlTests(unittest.TestCase):
    def test_matches_every_pixel_including_boundaries(self):
        original = np.array([[0, 255, 80], [1, 254, 120]], dtype=np.uint8)
        attack = np.array([[8, 247, 82], [5, 250, 120]], dtype=np.uint8)
        control, constrained = random_sign_control(original, attack, 12)
        expected = np.abs(attack.astype(np.int16) - original.astype(np.int16))
        np.testing.assert_array_equal(np.abs(control.astype(np.int16) - original.astype(np.int16)), expected)
        self.assertGreaterEqual(constrained, 0)
        np.testing.assert_array_equal(control, random_sign_control(original, attack, 12)[0])

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            random_sign_control(np.zeros((2, 2)), np.zeros((2, 2)), 1)
        with self.assertRaises(ValueError):
            random_sign_control(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 3), dtype=np.uint8), 1)


if __name__ == "__main__":
    unittest.main()
