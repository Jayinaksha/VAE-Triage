"""Two ablations that matter for the edge / 'why unsupervised' argument.

A) Feature cost: compute the 23 features on a 2x-downsampled tile (128 px) -
   how much faster, how much AUROC is lost?
B) Unseen artefact types: supervised classifiers are trained WITHOUT
   {bubble, pen_mark, tissue_fold, jpeg}; the GMM never sees any artefact.
   Recall on the held-out types at each model's operating point.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.common import save_json
from benchmarks.qc_sets import build
from triage.features import tile_features
from triage.gmm_triage import GMMTriage

HELD_OUT = ["bubble", "pen_mark", "tissue_fold", "jpeg"]


def loso_thr(X, sl, k, q=95):
    th = []
    for s in np.unique(sl):
        m = GMMTriage(k_range=(k,)).fit(X[sl != s])
        th.append(np.percentile(m.score(X[sl == s]), q))
    return float(np.mean(th))


def main():
    d = build()
    F = dict(np.load("/home/user/data/c16_tiles/qc_features.npz"))
    sl = d["tr_clean_slide"]
    kind = np.concatenate([["clean"] * len(F["te_clean"]), ["background"] * len(F["te_bg"]), d["te_art_kind"]])
    y = (kind != "clean").astype(int)
    out = {}

    # ---------------- A) downsampled features
    def feats_ds(tiles):
        return np.stack([tile_features(cv2.resize(t, (128, 128), interpolation=cv2.INTER_AREA)) for t in tiles])
    t = time.perf_counter()
    Ftr = feats_ds(d["tr_clean"])
    ms = 1000 * (time.perf_counter() - t) / len(Ftr)
    Fte = np.concatenate([feats_ds(d["te_clean"]), feats_ds(d["te_bg"]), feats_ds(d["te_art"])])
    for k in (1, 6):
        g = GMMTriage(k_range=(k,)).fit(Ftr)
        s = g.score(Fte)
        thr = loso_thr(Ftr, sl, k)
        p = s > thr
        out[f"downsampled128_K{k}"] = dict(auroc=float(roc_auc_score(y, s)), recall=float(p[y == 1].mean()),
                                           specificity=float(1 - p[y == 0].mean()), feature_ms_per_tile=ms)
    print(out, flush=True)

    # ---------------- B) leave-artefact-type-out
    Xte = np.concatenate([F["te_clean"], F["te_bg"], F["te_art"]])
    tr_kind = d["tr_art_kind"]
    keep = ~np.isin(tr_kind, HELD_OUT)
    Xs = np.concatenate([F["tr_clean"], F["tr_bg"], F["tr_art"][keep]])
    ys = np.r_[np.zeros(len(F["tr_clean"])), np.ones(len(F["tr_bg"]) + keep.sum())]
    res = {}
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xs, ys)
    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(Xs, ys)
    for name, p in [("Logistic regression (supervised, 6 artefact types seen)", lr.predict_proba(Xte)[:, 1] > 0.5),
                    ("Random forest (supervised, 6 artefact types seen)", rf.predict_proba(Xte)[:, 1] > 0.5)]:
        res[name] = {k: float(p[kind == k].mean()) for k in HELD_OUT} | {"clean_false_reject": float(p[kind == "clean"].mean())}
    for k in (1, 6):
        g = GMMTriage(k_range=(k,)).fit(F["tr_clean"])
        p = g.score(Xte) > loso_thr(F["tr_clean"], sl, k)
        res[f"GMM K={k} (no artefact ever seen)"] = {a: float(p[kind == a].mean()) for a in HELD_OUT} | {"clean_false_reject": float(p[kind == "clean"].mean())}
    for k_, v in res.items():
        print(k_, {a: round(b, 3) for a, b in v.items()}, flush=True)
    out["unseen_artifact_recall"] = res
    save_json(out, "exp4_ablations.json")


if __name__ == "__main__":
    main()
