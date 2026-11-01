"""Synthetic scanning / preparation artefacts with known ground truth.

There is no public, pixel-labelled artefact set for CAMELYON16, so - as is
standard in focus-quality and QC literature (e.g. FocusLiteNN, DeepFocus,
HistoQC robustness studies) - we inject controlled artefacts into *real*
clean H&E tiles. This gives exact labels and lets us report per-artefact
recall. Each generator is deterministic given an `np.random.Generator`.
"""
from __future__ import annotations

from typing import Callable, Dict

import cv2
import numpy as np


def gaussian_blur(img, rng):
    """Out-of-focus scan: defocus PSF ~ Gaussian. sigma in px at 20x."""
    s = rng.uniform(1.5, 4.0)
    return cv2.GaussianBlur(img, (0, 0), s)


def motion_blur(img, rng):
    """Stage moved during line-scan exposure: 1-D box PSF at random angle."""
    L = int(rng.integers(9, 25))
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), rng.uniform(0, 180), 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    k /= k.sum() + 1e-8
    return cv2.filter2D(img, -1, k)


def partial_blur(img, rng):
    """Focus gradient: only part of the tile is blurred (hardest case)."""
    out = img.copy()
    blurred = cv2.GaussianBlur(img, (0, 0), rng.uniform(2.0, 4.0))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    if rng.random() < 0.5:
        mask[:, : int(w * rng.uniform(0.4, 0.7))] = 1
    else:
        mask[: int(h * rng.uniform(0.4, 0.7)), :] = 1
    mask = cv2.GaussianBlur(mask, (0, 0), 8)[..., None]
    return (out * (1 - mask) + blurred * mask).astype(np.uint8)


_INKS = np.array([[20, 120, 40], [30, 60, 160], [25, 25, 25], [20, 140, 150]], np.float32)


def pen_mark(img, rng):
    """Pathologist marker ink (green/blue/black/teal) drawn over tissue."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pts = np.cumsum(rng.normal(0, 40, size=(6, 2)), axis=0) + rng.uniform(0, [w, h])
    cv2.polylines(mask, [pts.astype(np.int32)], False, 255, int(rng.integers(18, 45)))
    mask = cv2.GaussianBlur(mask, (0, 0), 2).astype(np.float32)[..., None] / 255
    ink = _INKS[rng.integers(len(_INKS))]
    alpha = rng.uniform(0.6, 0.9)
    out = img.astype(np.float32) * (1 - alpha * mask) + ink * alpha * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def tissue_fold(img, rng):
    """Folded section = double thickness: darker, more saturated band."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    c = rng.uniform(0, [w, h]).astype(int)
    ax = (int(rng.integers(w // 2, w)), int(rng.integers(h // 8, h // 3)))
    cv2.ellipse(mask, tuple(c), ax, rng.uniform(0, 180), 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 6).astype(np.float32)[..., None] / 255
    od = -np.log((img.astype(np.float32) + 1) / 256)
    folded = 256 * np.exp(-od * rng.uniform(2.0, 3.0)) - 1   # Beer-Lambert: 2-3x thickness
    out = img * (1 - mask) + folded * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def washout(img, rng):
    """Faded / under-stained or over-exposed tile."""
    a = rng.uniform(0.55, 0.8)
    return np.clip(img * (1 - a) + 255 * a, 0, 255).astype(np.uint8)


def dark_exposure(img, rng):
    g = rng.uniform(1.8, 3.0)
    return np.clip(255 * (img / 255.0) ** g * rng.uniform(0.5, 0.8), 0, 255).astype(np.uint8)


def sensor_noise(img, rng):
    n = rng.normal(0, rng.uniform(15, 30), img.shape)
    return np.clip(img + n, 0, 255).astype(np.uint8)


def jpeg_artifact(img, rng):
    q = int(rng.integers(3, 10))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def bubble(img, rng):
    """Air bubble under the coverslip: bright washed disc with dark rim."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    c = tuple(rng.uniform(0, [w, h]).astype(int))
    r = int(rng.integers(w // 4, w // 2))
    inner = np.zeros((h, w), np.uint8)
    cv2.circle(inner, c, r, 255, -1)
    rim = np.zeros((h, w), np.uint8)
    cv2.circle(rim, c, r, 255, int(rng.integers(4, 9)))
    inner = cv2.GaussianBlur(inner, (0, 0), 3).astype(np.float32)[..., None] / 255
    rim = cv2.GaussianBlur(rim, (0, 0), 1.5).astype(np.float32)[..., None] / 255
    out = out * (1 - 0.6 * inner) + 245 * 0.6 * inner
    out = out * (1 - 0.8 * rim) + 40 * 0.8 * rim
    return np.clip(out, 0, 255).astype(np.uint8)


ARTIFACTS: Dict[str, Callable] = {
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "partial_blur": partial_blur,
    "pen_mark": pen_mark,
    "tissue_fold": tissue_fold,
    "bubble": bubble,
    "washout": washout,
    "dark_exposure": dark_exposure,
    "sensor_noise": sensor_noise,
    "jpeg": jpeg_artifact,
}
