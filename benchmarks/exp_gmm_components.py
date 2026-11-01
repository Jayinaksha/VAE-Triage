"""Experiment 3: GMM tissue-component (incl. fluid) segmentation.

No pixel-level fluid annotation exists for CAMELYON16, so we evaluate the
component maps (a) qualitatively (figure), (b) by stability (agreement of
two GMMs fitted on disjoint slides), (c) against k-means on the same
descriptor, and (d) by downstream usefulness: do the component fractions
(e.g. nuclear fraction, fluid fraction) separate tumour from normal tiles?
"""
from __future__ import annotations

import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import adjusted_rand_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from benchmarks.common import FIGS, save_json  # noqa: E402
from benchmarks.qc_sets import build  # noqa: E402
from triage.gmm_components import COMPONENTS, GMMTissueComponents, colorize, pixel_descriptor  # noqa: E402


def main():
    d = build()
    rng = np.random.default_rng(0)
    tr, trs, trl = d["tr_clean"], d["tr_clean_slide"], d["tr_clean_lab"]
    fit_tiles = tr[rng.choice(len(tr), 120, replace=False)]
    t = time.perf_counter()
    gc = GMMTissueComponents(k=7).fit(fit_tiles)
    fit_s = time.perf_counter() - t
    names = [COMPONENTS[c] for c in gc.comp2class]
    print("component naming:", names)

    # (b) stability: GMMs fitted on disjoint slides agree?
    a = GMMTissueComponents(k=7, seed=1).fit(tr[trs == "normal_108"][:60], n_pixels=80_000)
    b = GMMTissueComponents(k=7, seed=2).fit(tr[trs == "tumor_082"][:60], n_pixels=80_000)
    te = d["te_clean"]
    probe = te[rng.choice(len(te), 30, replace=False)]
    ari = [adjusted_rand_score(a.predict(x).ravel(), b.predict(x).ravel()) for x in probe]
    agree = [float((a.predict(x) == b.predict(x)).mean()) for x in probe]

    # (c) k-means on the same descriptor, mapped with the same rules
    Xs = np.concatenate([pixel_descriptor(x)[::40] for x in fit_tiles])
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    km = KMeans(7, n_init=3, random_state=0).fit((Xs - mu) / sd)
    km_names = GMMTissueComponents._name_components(km.cluster_centers_ * sd + mu)
    km_ari = [adjusted_rand_score(gc.predict(x).ravel(),
                                  km_names[km.predict((pixel_descriptor(x) - mu) / sd)]) for x in probe]
    km_a = GMMTissueComponents  # noqa: F841

    # (d) component fractions -> tumour vs normal (test slides)
    t = time.perf_counter()
    Fte = np.stack([gc.fractions(x) for x in te])
    ms_tile = 1000 * (time.perf_counter() - t) / len(te)
    Ftr = np.stack([gc.fractions(x) for x in tr])
    y = (d["te_clean_lab"] == "tumor").astype(int)
    per_comp_auc = {c: float(roc_auc_score(y, Fte[:, i])) for i, c in enumerate(COMPONENTS)}
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(
        Ftr, (trl == "tumor").astype(int))
    auc_lr = float(roc_auc_score(y, lr.predict_proba(Fte)[:, 1]))
    mean_frac = {lab: dict(zip(COMPONENTS, Fte[d["te_clean_lab"] == lab].mean(0).round(4).tolist()))
                 for lab in ["normal", "tumor"]}
    out = dict(component_names=names, fit_seconds=fit_s, ms_per_tile_256=ms_tile,
               stability_ARI_disjoint_slides=float(np.mean(ari)), stability_pixel_agreement=float(np.mean(agree)), gmm_vs_kmeans_ARI=float(np.mean(km_ari)),
               kmeans_component_names=[COMPONENTS[c] for c in km_names],
               tumour_auc_single_fraction=per_comp_auc, tumour_auc_fraction_vector_logreg=auc_lr,
               mean_fractions=mean_frac)
    print(out)
    save_json(out, "exp3_gmm_components.json")

    # figure
    picks = [te[d["te_clean_lab"] == "normal"][i] for i in (3, 40, 77)] + [te[d["te_clean_lab"] == "tumor"][i] for i in (5, 60)]
    fig, ax = plt.subplots(3, 5, figsize=(16, 10))
    for j, x in enumerate(picks):
        lab, P = gc.predict(x, return_proba=True)
        km_lab = km_names[km.predict((pixel_descriptor(x) - mu) / sd)].reshape(lab.shape)
        ax[0, j].imshow(x)
        ax[0, j].set_title("normal" if j < 3 else "tumour")
        ax[1, j].imshow(colorize(lab))
        ax[1, j].set_title("GMM components")
        ax[2, j].imshow(colorize(km_lab))
        ax[2, j].set_title("k-means (same features)")
    for a_ in ax.flat:
        a_.axis("off")
    from matplotlib.patches import Patch
    from triage.gmm_components import COLORS
    fig.legend([Patch(color=c / 255) for c in COLORS], COMPONENTS, loc="lower center", ncol=5, fontsize=11)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(f"{FIGS}/gmm_tissue_components.png", dpi=80)


if __name__ == "__main__":
    main()
