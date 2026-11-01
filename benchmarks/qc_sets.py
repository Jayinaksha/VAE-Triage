"""Deterministic construction of the QC (artefact) and tumour-triage sets.

Cached to <TILE_DIR>/qc_sets.npz so the classical and deep scripts score
exactly the same images.
"""
from __future__ import annotations

import os

import numpy as np

from benchmarks.common import TEST_SLIDES, TILE_DIR, TRAIN_SLIDES, load_slides, stack
from triage.artifacts import ARTIFACTS

PER_ARTIFACT = 150


def _make_artifacts(clean, rng, per_artifact):
    imgs, kinds, src = [], [], []
    for name, fn in ARTIFACTS.items():
        idx = rng.choice(len(clean), size=per_artifact, replace=False)
        for i in idx:
            imgs.append(fn(clean[i].copy(), rng))
            kinds.append(name)
            src.append(i)
    return np.stack(imgs), np.array(kinds), np.array(src)


def build(force=False):
    path = f"{TILE_DIR}/qc_sets.npz"
    if os.path.exists(path) and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    tr, te = load_slides(TRAIN_SLIDES), load_slides(TEST_SLIDES)
    rng_tr, rng_te = np.random.default_rng(1), np.random.default_rng(2)

    # ---- train side: clean tissue (normal+tumour) and, for supervised
    # baselines only, artefacts + background generated with a *different* seed
    tr_clean, tr_clean_lab, tr_clean_slide = stack(tr, ["normal", "tumor"])
    tr_bg, _, _ = stack(tr, ["background"])
    tr_art, tr_art_kind, tr_src = _make_artifacts(tr_clean, rng_tr, PER_ARTIFACT // 2)

    # ---- test side
    te_clean, te_clean_lab, te_clean_slide = stack(te, ["normal", "tumor"])
    te_bg, _, te_bg_slide = stack(te, ["background"])
    tr_bg, _, tr_bg_slide = stack(tr, ["background"])
    te_border, _, _ = stack(te, ["border"])
    te_art, te_art_kind, te_src = _make_artifacts(te_clean, rng_te, PER_ARTIFACT)

    out = dict(tr_clean=tr_clean, tr_clean_lab=tr_clean_lab, tr_clean_slide=tr_clean_slide,
               tr_bg=tr_bg, tr_bg_slide=tr_bg_slide, tr_art=tr_art, tr_art_kind=tr_art_kind, tr_art_slide=tr_clean_slide[tr_src],
               te_clean=te_clean, te_clean_lab=te_clean_lab, te_clean_slide=te_clean_slide,
               te_bg=te_bg, te_bg_slide=te_bg_slide, te_border=te_border, te_art=te_art, te_art_kind=te_art_kind,
               te_art_slide=te_clean_slide[te_src])
    np.savez(path, **out)
    return out


if __name__ == "__main__":
    d = build(force=True)
    for k, v in d.items():
        print(k, v.shape, v.dtype)
