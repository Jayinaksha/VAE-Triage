"""Shared loading, splitting and metric helpers for the benchmarks."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TILE_DIR = os.environ.get("C16_TILES", "/home/user/data/c16_tiles")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "docs", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# Slide-level split (never mix tiles of one slide across train/test -> no leakage)
TRAIN_SLIDES = ["normal_108", "normal_150", "tumor_082"]
TEST_SLIDES = ["tumor_075", "tumor_084", "tumor_091", "tumor_079"]


def load_slides(names):
    out = {}
    for s in names:
        d = np.load(f"{TILE_DIR}/{s}.npz")
        out[s] = {k: d[k] for k in d.files}
    return out


def stack(slides: Dict, classes, key="tiles"):
    xs, ys, ss = [], [], []
    for s, d in slides.items():
        m = np.isin(d["labels"], classes)
        xs.append(d[key][m])
        ys.append(d["labels"][m])
        ss += [s] * int(m.sum())
    return np.concatenate(xs), np.concatenate(ys), np.array(ss)


def bootstrap_auc(y, s, n=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def binary_report(y_true, score, pred=None, name=""):
    """AUROC / AUPRC are threshold-free; P/R/F1 are at the given decision."""
    y_true = np.asarray(y_true).astype(int)
    r = dict(method=name, n=int(len(y_true)), pos=int(y_true.sum()),
             auroc=float(roc_auc_score(y_true, score)), auprc=float(average_precision_score(y_true, score)))
    r["auroc_ci95"] = bootstrap_auc(y_true, np.asarray(score))
    if pred is not None:
        p, rc, f, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        r.update(precision=float(p), recall=float(rc), f1=float(f),
                 specificity=float(tn / max(tn + fp, 1)), accuracy=float((pred == y_true).mean()))
    return r


def save_json(obj, name):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t
