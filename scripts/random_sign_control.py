"""Build per-pixel magnitude-matched random-sign controls on grayscale arrays."""

import numpy as np


def random_sign_control(original, attack, seed):
    original = np.asarray(original)
    attack = np.asarray(attack)
    if original.dtype != np.uint8 or attack.dtype != np.uint8:
        raise ValueError("Inputs must be 8-bit unsigned arrays")
    if original.ndim != 2 or original.shape != attack.shape or original.size == 0:
        raise ValueError("Inputs must be nonempty grayscale arrays with matching shapes")
    base = original.astype(np.int16)
    magnitude = np.abs(attack.astype(np.int16) - base)
    rng = np.random.default_rng(seed)
    sign = np.where(rng.integers(0, 2, size=base.shape) == 0, -1, 1)
    proposal = base + sign * magnitude
    constrained = (proposal < 0) | (proposal > 255)
    sign[constrained] *= -1
    result = (base + sign * magnitude).astype(np.uint8)
    return result, int(np.count_nonzero(constrained))
