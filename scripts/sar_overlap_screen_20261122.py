"""Propose same-size SAR image overlap candidates without model outputs.

This is a screening aid. Matches require human review and nonmatches do not
establish scene independence. See the associated research protocol.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


THUMB_SIZE = 128
TOP_PEAKS = 16
MIN_OVERLAP = 0.25
MIN_STANDARD_DEVIATION = 4.0
THUMB_THRESHOLD = 0.90
FULL_THRESHOLD = 0.95


def overlap(a, b, dy, dx):
    height, width = a.shape
    ya, xa = max(0, dy), max(0, dx)
    yb, xb = max(0, -dy), max(0, -dx)
    h, w = height - abs(dy), width - abs(dx)
    if h <= 0 or w <= 0:
        return None
    return a[ya:ya + h, xa:xa + w], b[yb:yb + h, xb:xb + w]


def correlation(a, b, dy, dx):
    region = overlap(a, b, dy, dx)
    if region is None:
        return None
    left, right = region
    fraction = left.size / a.size
    if fraction < MIN_OVERLAP:
        return None
    x, y = left.astype(np.float64), right.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    std_x, std_y = np.sqrt(np.mean(x * x)), np.sqrt(np.mean(y * y))
    if min(std_x, std_y) < MIN_STANDARD_DEVIATION:
        return None
    return float(np.mean(x * y) / (std_x * std_y)), float(fraction)


def phase_candidates(a, b):
    height, width = a.shape
    shape = (height * 2, width * 2)
    product = np.fft.rfft2(a - a.mean(), s=shape) * np.conj(np.fft.rfft2(b - b.mean(), s=shape))
    product /= np.maximum(np.abs(product), 1e-12)
    phase = np.fft.irfft2(product, s=shape)
    count = min(TOP_PEAKS, phase.size)
    flat = np.argpartition(phase.ravel(), -count)[-count:]
    positions = sorted(flat, key=lambda i: (-phase.ravel()[i], int(i)))
    return [(int(y if y < height else y - 2 * height),
             int(x if x < width else x - 2 * width))
            for y, x in (np.unravel_index(i, phase.shape) for i in positions)]


def screen(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError('Only same-size square grayscale images are supported')
    if a.shape[0] < THUMB_SIZE:
        raise ValueError('Image side must be at least 128')
    scale = a.shape[0] / THUMB_SIZE
    radius = math.ceil(scale)
    left = np.asarray(Image.fromarray(a).resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.BILINEAR), dtype=np.float64)
    matches = []
    for flip in (False, True):
        for turns in range(4):
            oriented = np.rot90(np.fliplr(b) if flip else b, turns)
            thumb = np.asarray(Image.fromarray(oriented).resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.BILINEAR), dtype=np.float64)
            for dy, dx in phase_candidates(left, thumb):
                coarse = correlation(left, thumb, dy, dx)
                if coarse is None or coarse[0] < THUMB_THRESHOLD:
                    continue
                center_y, center_x = round(dy * scale), round(dx * scale)
                for fy in range(center_y - radius, center_y + radius + 1):
                    for fx in range(center_x - radius, center_x + radius + 1):
                        score = correlation(a, oriented, fy, fx)
                        if score is not None and score[0] >= FULL_THRESHOLD:
                            matches.append({'correlation': score[0], 'overlap_fraction': score[1],
                                            'dy': fy, 'dx': fx, 'rotation_ccw_quarters': turns,
                                            'horizontal_flip_before_rotation': flip})
    matches.sort(key=lambda x: (-x['correlation'], -x['overlap_fraction'], x['dy'], x['dx']))
    return matches[0] if matches else None


def load(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('L')
        return np.asarray(image, dtype=np.uint8)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args()
    result = {'left_sha256': sha(args.left), 'right_sha256': sha(args.right),
              'match': screen(load(args.left), load(args.right)),
              'scope': 'pairwise screening only; no automatic exclusion'}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
