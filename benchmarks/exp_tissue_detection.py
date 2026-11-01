"""Stage-0 benchmark: slide-level tissue vs. glass segmentation on thumbnails.

Ground truth: CAMELYON16 `background_tissue` masks (produced by the CNN of
Bandi et al., PeerJ 2019, at 2 um/px). We compare classical detectors that
run on a ~4 um/px thumbnail in milliseconds - this is the first and cheapest
gate of the pipeline, it decides which tiles are ever read at 20x.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from benchmarks.common import FIGS, TEST_SLIDES, TRAIN_SLIDES, save_json
from data.wsi import WSIReader

ROOT = "/home/user/data/camelyon16"


def clean(mask, k=5, min_area=150):
    """Opening removes thin scanner stripes / dust, closing fills nuclei-sized
    holes, then connected components smaller than `min_area` px are dropped."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def otsu(ch):
    t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def methods(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    s = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)
    od = np.clip((-np.log((rgb.astype(np.float32) + 1) / 256)).sum(2), 0, 3.0)  # clip ink/black outliers
    # avoid black slide-scanner padding being counted as tissue
    valid = gray > 15
    out = {}
    out["Fixed threshold (gray < 220)"] = (gray < 220) & valid
    out["Otsu on grayscale"] = (gray < otsu(gray)) & valid
    out["Otsu on HSV saturation"] = (s > otsu(s)) & valid
    out["Otsu on saturation + morphology"] = clean((s > otsu(s)) & valid)
    feats = np.stack([s.ravel() / 255.0, od.ravel()], 1)
    sub = feats[np.random.default_rng(0).choice(len(feats), min(len(feats), 60000), replace=False)]
    km = KMeans(2, n_init=3, random_state=0).fit(sub)
    lab = km.predict(feats).reshape(s.shape)
    tissue_c = int(np.argmax(km.cluster_centers_.sum(1)))
    out["K-means (k=2) on (S, OD)"] = clean((lab == tissue_c) & valid)
    g = GaussianMixture(2, covariance_type="full", random_state=0).fit(sub)
    post = g.predict_proba(feats)[:, int(np.argmax(g.means_[:, 0]))].reshape(s.shape)
    out["GMM (K=2) on (S, OD) + morphology [ours]"] = clean((post > 0.5) & valid)
    return out


def main():
    rows = {}
    per_slide = {}
    for s in TRAIN_SLIDES + TEST_SLIDES:
        w = WSIReader(f"{ROOT}/images/{s}.tif")
        lvl = w.best_level_for_downsample(16)
        while w.levels[lvl].width > 6000:  # keep thumbnails <= ~6k px (RAM)
            lvl += 1
        rgb = w.read_level(lvl)
        gt_r = WSIReader(f"{ROOT}/tissue/{s}_tissue.tif")
        gl = int(np.argmin([abs(lv.width - rgb.shape[1]) for lv in gt_r.levels]))
        gt = gt_r.read_level(gl)
        gt = cv2.resize(gt.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        per_slide[s] = {}
        t = time.perf_counter()
        preds = methods(rgb)
        dt = (time.perf_counter() - t) * 1000 / len(preds)
        for name, pred in preds.items():
            tp = (pred & gt).sum()
            fp = (pred & ~gt).sum()
            fn = (~pred & gt).sum()
            r = dict(dice=2 * tp / (2 * tp + fp + fn + 1e-9), iou=tp / (tp + fp + fn + 1e-9),
                     precision=tp / (tp + fp + 1e-9), recall=tp / (tp + fn + 1e-9))
            per_slide[s][name] = {k: float(v) for k, v in r.items()}
            rows.setdefault(name, []).append(r)
        per_slide[s]["thumbnail_shape"] = list(rgb.shape)
        per_slide[s]["gt_tissue_fraction"] = float(gt.mean())
        per_slide[s]["avg_ms_per_method"] = dt
        if s == "tumor_091":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
            ax[0].imshow(rgb)
            ax[0].set_title(f"{s} thumbnail (ds={w.levels[lvl].downsample:.0f})")
            ax[1].imshow(gt, cmap="gray")
            ax[1].set_title("GT tissue (CAMELYON16 mask)")
            ax[2].imshow(preds["Otsu on grayscale"], cmap="gray")
            ax[2].set_title("Otsu on grayscale")
            ax[3].imshow(preds["GMM (K=2) on (S, OD) + morphology [ours]"], cmap="gray")
            ax[3].set_title("GMM K=2 on (S, OD)")
            for a in ax:
                a.axis("off")
            plt.tight_layout()
            plt.savefig(f"{FIGS}/tissue_detection_{s}.png", dpi=90)
            plt.close()
        print(s, {k: round(v["dice"], 3) for k, v in per_slide[s].items() if isinstance(v, dict)}, flush=True)
    summary = {k: {m: float(np.mean([r[m] for r in v])) for m in ["dice", "iou", "precision", "recall"]} for k, v in rows.items()}
    for k, v in summary.items():
        print(f"{k:45s} Dice={v['dice']:.3f} IoU={v['iou']:.3f} P={v['precision']:.3f} R={v['recall']:.3f}")
    save_json(dict(summary=summary, per_slide=per_slide), "exp0_tissue_detection.json")


if __name__ == "__main__":
    main()
