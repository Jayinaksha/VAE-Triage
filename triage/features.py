"""Hand-crafted, edge-friendly tile descriptors for WSI quality triage.

Every feature is a closed-form image statistic computable with a few OpenCV /
NumPy passes over a 256x256 tile (~1 ms on one CPU core). Each one maps to a
physical failure mode of slide scanning:

| group      | feature(s)                         | catches                              |
|------------|------------------------------------|--------------------------------------|
| tissue     | tissue_frac, od_mean, white_frac   | glass background, faded stain        |
| stain      | H/E concentration (Ruifrok-Johnston colour deconvolution), sat stats | pen ink, colour casts, over/under stain |
| focus      | Laplacian variance, Tenengrad, Brenner, FFT high-freq ratio | Gaussian (defocus) blur, motion blur |
| structure  | Canny edge density, gray entropy   | blur, flat regions, folds            |
| noise      | residual vs. median filter         | sensor noise, JPEG artefacts         |
| darkness   | dark_frac, val_mean                | tissue folds, under-exposure, black ink |

Heavy-tailed features (variances, energies) are log-transformed so a Gaussian
(mixture) model of the feature vector is a reasonable approximation.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

# Ruifrok & Johnston (2001) stain vectors (rows: Hematoxylin, Eosin, DAB),
# the same matrix used by skimage.color.rgb2hed.
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]], dtype=np.float32)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)

FEATURE_NAMES: List[str] = [
    "tissue_frac", "white_frac", "dark_frac",
    "od_mean", "od_std",
    "h_mean", "e_mean", "h_std", "e_std",
    "sat_mean", "sat_std", "val_mean", "val_std",
    "non_he_hue_frac", "colorfulness",
    "log_lap_var", "log_tenengrad", "log_brenner", "lap_norm", "hf_ratio",
    "edge_density", "gray_entropy", "log_noise",
]


def _fft_highfreq_ratio(gray: np.ndarray, radius_frac: float = 0.25) -> float:
    """Fraction of spectral energy outside a centred low-frequency disc.
    Defocus (Gaussian blur) is a low-pass filter, so this ratio collapses."""
    g = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = spec.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    total = spec.sum() + 1e-8
    return float(spec[r > radius_frac * min(h, w)].sum() / total)


def tile_features(rgb: np.ndarray) -> np.ndarray:
    """Compute the feature vector (len(FEATURE_NAMES),) for one RGB uint8 tile."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)

    # Beer-Lambert optical density: OD = -log10(I / I0)
    od = -np.log10((rgb.astype(np.float32) + 1.0) / 256.0)
    od_sum = od.sum(axis=2)
    tissue = (sat > 20) & (val < 235) | (od_sum > 0.45)
    tissue_frac = float(tissue.mean())
    white_frac = float(((val > 220) & (sat < 20)).mean())
    dark_frac = float((val < 60).mean())

    hed = od.reshape(-1, 3) @ _HED_FROM_RGB
    h_c, e_c = hed[:, 0], hed[:, 1]

    # Pen ink is typically green/blue/black - hues that H&E never produce.
    # OpenCV hue in [0,180): H&E pinks/purples live in ~[125,180] U [0,15].
    non_he = (sat > 60) & (hue > 25) & (hue < 115)
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    gf = gray.astype(np.float32)
    lap = cv2.Laplacian(gf, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float((gx ** 2 + gy ** 2).mean())
    brenner = float(((gf[:, 2:] - gf[:, :-2]) ** 2).mean())
    lap_norm = float(np.log1p(lap_var) - np.log1p(gf.var()))  # contrast-invariant focus

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    gray_entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    noise = float(np.abs(gf - cv2.medianBlur(gray, 3).astype(np.float32)).mean())

    return np.array([
        tissue_frac, white_frac, dark_frac,
        float(od_sum.mean()), float(od_sum.std()),
        float(h_c.mean()), float(e_c.mean()), float(h_c.std()), float(e_c.std()),
        float(sat.mean()), float(sat.std()), float(val.mean()), float(val.std()),
        float(non_he.mean()), colorfulness,
        float(np.log1p(lap_var)), float(np.log1p(tenengrad)), float(np.log1p(brenner)),
        lap_norm, _fft_highfreq_ratio(gray),
        edge_density, gray_entropy, float(np.log1p(noise)),
    ], dtype=np.float32)


def batch_features(tiles) -> np.ndarray:
    return np.stack([tile_features(t) for t in tiles]) if len(tiles) else np.zeros((0, len(FEATURE_NAMES)), np.float32)


def as_dict(vec: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(FEATURE_NAMES, vec)}
