"""Experiments 1 & 2 with classical / statistical methods.

Exp-1  Tile QC triage : clean tissue (normal+tumour)  vs  {glass background,
                        10 synthetic artefacts}. Unsupervised methods are
                        fitted on clean TRAIN-slide tiles only and the
                        rejection threshold is the 95th percentile of the
                        score on held-out clean train tiles (=> ~5 % false
                        rejection of good tissue by construction).
Exp-2  Tumour triage  : normal vs metastasis tiles on TEST slides. GMM etc.
                        fitted on *normal* train tissue only (novelty
                        detection, as in Shvetsova et al. 2021).

Also dumps the statistics used in the report: feature correlation /
covariance, KL & JS divergences clean-vs-artefact, BIC curve, stain JS
divergence between slides, latency.
"""
from __future__ import annotations

import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402
from sklearn.ensemble import IsolationForest, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

from benchmarks.common import FIGS, RESULTS, TILE_DIR, binary_report, save_json  # noqa: E402
from benchmarks.qc_sets import build  # noqa: E402
from triage.features import FEATURE_NAMES, batch_features  # noqa: E402
from triage.gmm_triage import EdgeGMMScorer, GMMTriage  # noqa: E402

FI = {n: i for i, n in enumerate(FEATURE_NAMES)}


def feats(d):
    cache = f"{TILE_DIR}/qc_features.npz"
    if os.path.exists(cache):
        c = np.load(cache)
        return {k: c[k] for k in c.files}
    keys = ["tr_clean", "tr_bg", "tr_art", "te_clean", "te_bg", "te_border", "te_art"]
    F = {}
    for k in keys:
        t = time.perf_counter()
        F[k] = batch_features(d[k])
        print(f"features {k}: {len(d[k])} tiles {1000 * (time.perf_counter() - t) / len(d[k]):.2f} ms/tile", flush=True)
    np.savez(cache, **F)
    return F


STAIN_FEATS = ["od_mean", "od_std", "h_mean", "e_mean", "h_std", "e_std", "sat_mean", "sat_std",
               "val_mean", "val_std", "colorfulness"]


def slide_normalize(F, slides, ref_F, ref_slides, global_ref, cols=None):
    """Per-slide location normalisation (a feature-space analogue of
    Reinhard stain normalisation): x' = x - median_slide + median_global.
    median_slide is taken over a random sample of that slide's tissue tiles,
    which a deployed system gets for free on the first pass over the slide."""
    cols = np.arange(F.shape[1]) if cols is None else np.asarray(cols)
    out = F.copy()
    for s in np.unique(slides):
        off = np.median(ref_F[ref_slides == s][:, cols], axis=0) - global_ref[cols]
        m = slides == s
        out[np.ix_(m, cols)] -= off
    return out


def thresh_at(score_clean_val, q=95):
    return float(np.percentile(score_clean_val, q))


def histoqc_rules(F_tr):
    """HistoQC-style hand-tuned rule set (Janowczyk et al. 2019): each rule
    is a 1st/99th-percentile bound learnt from clean training tiles; the
    score is the number of violated rules (+ a tiny tie-breaker)."""
    lo = np.percentile(F_tr, 1, axis=0)
    hi = np.percentile(F_tr, 99, axis=0)
    rules = [("tissue_frac", "lo"), ("log_lap_var", "lo"), ("lap_norm", "lo"), ("hf_ratio", "lo"),
             ("non_he_hue_frac", "hi"), ("dark_frac", "hi"), ("white_frac", "hi"), ("log_noise", "hi"),
             ("od_mean", "lo"), ("od_mean", "hi"), ("sat_mean", "lo")]

    def score(F):
        s = np.zeros(len(F))
        for n, side in rules:
            i = FI[n]
            s += (F[:, i] < lo[i]) if side == "lo" else (F[:, i] > hi[i])
        return s + 1e-3 * np.random.default_rng(0).random(len(F))
    return score


def kl_js_divergence(a, b, bins=40):
    lo, hi = np.percentile(np.r_[a, b], [0.5, 99.5])
    if hi <= lo:
        return 0.0, 0.0
    p, _ = np.histogram(a, bins=bins, range=(lo, hi))
    q, _ = np.histogram(b, bins=bins, range=(lo, hi))
    p = (p + 1e-3) / (p + 1e-3).sum()
    q = (q + 1e-3) / (q + 1e-3).sum()
    kl = float((p * np.log(p / q)).sum())
    js = float(jensenshannon(p, q, base=2) ** 2)
    return kl, js


def main():
    d = build()
    F = feats(d)
    rng = np.random.default_rng(0)

    # ---------------- split clean TRAIN tiles: fit (80%) / calibration (20%)
    n = len(F["tr_clean"])
    perm = rng.permutation(n)
    fit_idx, cal_idx = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    Xfit, Xcal = F["tr_clean"][fit_idx], F["tr_clean"][cal_idx]

    # ================================================================ EXP-1
    Xte = np.concatenate([F["te_clean"], F["te_bg"], F["te_art"]])
    kind = np.concatenate([np.array(["clean"] * len(F["te_clean"])), np.array(["background"] * len(F["te_bg"])),
                           d["te_art_kind"]])
    clean_lab = np.concatenate([d["te_clean_lab"], np.array([""] * (len(Xte) - len(F["te_clean"])))])
    y = (kind != "clean").astype(int)
    Xtr, tr_sl = F["tr_clean"], d["tr_clean_slide"]
    results, scores = [], {}
    QC_FEATS = [FI[c] for c in ["tissue_frac", "white_frac", "dark_frac", "non_he_hue_frac", "log_lap_var",
                                "log_tenengrad", "log_brenner", "lap_norm", "hf_ratio", "edge_density",
                                "log_noise", "od_mean", "sat_mean", "val_mean"]]

    def loso_threshold(fit, score, q=95):
        """Calibrate the reject threshold on UNSEEN slides: fit on 2 train
        slides, take the q-th percentile of scores on the 3rd, average."""
        th = []
        for s_ in np.unique(tr_sl):
            m = fit(Xtr[tr_sl != s_])
            th.append(np.percentile(score(m, Xtr[tr_sl == s_]), q))
        return float(np.mean(th))

    def add(name, fit, score, latency_ms=None, supervised_pred=None, thr=None):
        model = fit(Xtr)
        s_te = score(model, Xte)
        if supervised_pred is not None:
            p = supervised_pred(model, Xte)
        else:
            thr = loso_threshold(fit, score) if thr is None else thr
            p = (s_te > thr).astype(int)
        r = binary_report(y, s_te, p, name)
        r["per_kind_recall"] = {k: float(p[kind == k].mean()) for k in sorted(set(kind)) if k != "clean"}
        r["reject_rate_clean_normal"] = float(p[(kind == "clean") & (clean_lab == "normal")].mean())
        r["reject_rate_clean_tumor"] = float(p[(kind == "clean") & (clean_lab == "tumor")].mean())
        r["latency_ms_per_tile_model_only"] = latency_ms
        results.append(r)
        scores[name] = s_te
        print(f"{name:42s} AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} spec={r['specificity']:.3f} tumour-rej={r['reject_rate_clean_tumor']:.3f}", flush=True)
        return model

    ident = lambda X: None  # noqa: E731
    neg_feat = lambda f: (lambda m, X: -X[:, FI[f]])  # noqa: E731
    add("Tissue fraction (Otsu-like)", ident, neg_feat("tissue_frac"))
    add("Laplacian variance (focus)", ident, neg_feat("log_lap_var"))
    add("Tenengrad (Sobel energy)", ident, neg_feat("log_tenengrad"))
    add("HistoQC-style rule set", histoqc_rules, lambda m, X: m(X))
    add("Single Gaussian (Mahalanobis, K=1)", lambda X: GMMTriage(k_range=(1,)).fit(X), lambda m, X: m.score(X))

    gmm_fit = lambda X: GMMTriage().fit(X)  # noqa: E731
    gmm = gmm_fit(Xtr)
    export_path = os.path.join(RESULTS, "gmm_qc_model.json")
    gmm.threshold_ = loso_threshold(gmm_fit, lambda m, X: m.score(X))
    gmm.export(export_path)
    edge = EdgeGMMScorer(export_path)
    Xbig = np.repeat(Xte, 5, axis=0)
    t = time.perf_counter()
    edge.score(Xbig)
    gmm_ms = 1000 * (time.perf_counter() - t) / len(Xbig)
    assert np.allclose(edge.score(Xte[:50]), gmm.score(Xte[:50]), rtol=1e-3, atol=1e-2)
    # pitfall row: threshold from held-out tiles of the SAME slides
    n = len(Xtr)
    perm = rng.permutation(n)
    g_same = GMMTriage().fit(Xtr[perm[: int(.8 * n)]], Xtr[perm[int(.8 * n):]])
    add(f"GMM (K={g_same.gmm_.n_components}) same-slide calibration [pitfall]", lambda X: g_same,
        lambda m, X: m.score(X), thr=g_same.threshold_)
    add(f"GMM (K={gmm.gmm_.n_components}, BIC) LOSO calibration [ours]", lambda X: gmm if len(X) == n else gmm_fit(X),
        lambda m, X: m.score(X), latency_ms=gmm_ms)

    class Sub:  # GMM on the artefact-specific feature subset
        def __init__(self, X):
            self.g = GMMTriage().fit(X[:, QC_FEATS])

        def score(self, X):
            return self.g.score(X[:, QC_FEATS])
    sel = GMMTriage()
    k_star = sel.select_k_by_group(Xtr, tr_sl)
    print("held-out-slide log-lik per K:", sel.heldout_ll_, "-> K*", k_star)
    gmm_cv_fit = lambda X: GMMTriage(k_range=(k_star,)).fit(X)  # noqa: E731
    gmm_cv = add(f"GMM (K={k_star}, slide-CV) LOSO calibration [ours, final]", gmm_cv_fit, lambda m, X: m.score(X),
                 latency_ms=gmm_ms)
    gmm_cv.threshold_ = loso_threshold(gmm_cv_fit, lambda m, X: m.score(X))
    gmm_cv.export(os.path.join(RESULTS, "gmm_qc_model_final.json"))
    add("GMM on QC-only feature subset [ours, tumour-safe]", Sub, lambda m, X: m.score(X), latency_ms=gmm_ms)

    def iso_fit(X):
        sc = StandardScaler().fit(X)
        return sc, IsolationForest(n_estimators=200, random_state=0).fit(sc.transform(X))
    add("Isolation Forest", iso_fit, lambda m, X: -m[1].score_samples(m[0].transform(X)))

    def oc_fit(X):
        sc = StandardScaler().fit(X)
        return sc, OneClassSVM(gamma="scale", nu=0.05).fit(sc.transform(X))
    add("One-class SVM (RBF)", oc_fit, lambda m, X: -m[1].score_samples(m[0].transform(X)))

    # SUPERVISED upper bounds on the same features (need artefact labels!)
    Xs = np.concatenate([F["tr_clean"], F["tr_bg"], F["tr_art"]])
    ys = np.r_[np.zeros(len(F["tr_clean"])), np.ones(len(F["tr_bg"]) + len(F["tr_art"]))]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xs, ys)
    add("Logistic regression (supervised)", lambda X: lr, lambda m, X: m.predict_proba(X)[:, 1],
        supervised_pred=lambda m, X: (m.predict_proba(X)[:, 1] > 0.5).astype(int))
    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(Xs, ys)
    add("Random forest (supervised)", lambda X: rf, lambda m, X: m.predict_proba(X)[:, 1],
        supervised_pred=lambda m, X: (m.predict_proba(X)[:, 1] > 0.5).astype(int))
    Xfit = Xtr

    save_json(dict(results=results, bic=gmm.bic_, heldout_ll=sel.heldout_ll_, k_slide_cv=k_star, k=gmm.gmm_.n_components, threshold=gmm.threshold_,
                   gmm_param_count=int(gmm.gmm_._n_parameters()), gmm_json_bytes=os.path.getsize(export_path)),
              "exp1_qc_classical.json")
    np.savez(os.path.join(RESULTS, "exp1_scores_classical.npz"), y=y, kind=kind, clean_lab=clean_lab,
             **{k: v for k, v in scores.items()})

    expl = {}
    for k in ["gaussian_blur", "motion_blur", "pen_mark", "tissue_fold", "washout", "jpeg", "background"]:
        i = np.where(kind == k)[0][0]
        expl[k] = gmm.explain(Xte[i], FEATURE_NAMES)
    save_json(expl, "exp1_gmm_explanations.json")
    save_json({"gmm_reject_rate_border_tiles": float(gmm.predict_reject(F["te_border"]).mean()),
               "gmm_reject_rate_test_clean": float(gmm.predict_reject(F["te_clean"]).mean()),
               "gmm_reject_rate_test_background": float(gmm.predict_reject(F["te_bg"]).mean())},
              "exp1_gmm_rates.json")

    # ================================================================ EXP-2
    tr_norm = F["tr_clean"][d["tr_clean_lab"] == "normal"]
    te_lab = d["te_clean_lab"]
    y2 = (te_lab == "tumor").astype(int)
    X2 = F["te_clean"]
    res2 = []
    gn = GMMTriage(reject_percentile=95).fit(tr_norm)
    res2.append(binary_report(y2, gn.score(X2), gn.predict_reject(X2).astype(int), f"GMM novelty (K={gn.gmm_.n_components}) on features"))
    sc2 = StandardScaler().fit(tr_norm)
    iso2 = IsolationForest(n_estimators=200, random_state=0).fit(sc2.transform(tr_norm))
    res2.append(binary_report(y2, -iso2.score_samples(sc2.transform(X2)), None, "Isolation Forest on features"))
    # stain-only cue: haematoxylin density (tumour nuclei are large & dense)
    res2.append(binary_report(y2, X2[:, FI["h_mean"]], None, "Hematoxylin mean (single feature)"))
    # supervised (uses tumour labels from train slide tumor_082)
    Xtr2 = F["tr_clean"]
    ytr2 = (d["tr_clean_lab"] == "tumor").astype(int)
    lr2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(Xtr2, ytr2)
    res2.append(binary_report(y2, lr2.predict_proba(X2)[:, 1], (lr2.predict_proba(X2)[:, 1] > 0.5).astype(int), "Logistic regression (supervised)"))
    rf2 = RandomForestClassifier(300, random_state=0, n_jobs=-1, class_weight="balanced").fit(Xtr2, ytr2)
    res2.append(binary_report(y2, rf2.predict_proba(X2)[:, 1], (rf2.predict_proba(X2)[:, 1] > 0.5).astype(int), "Random forest (supervised)"))
    for r in res2:
        print(f"[tumour] {r['method']:45s} AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f}", flush=True)
    save_json(res2, "exp2_tumor_classical.json")

    # ================================================================ STATISTICS
    Z = (Xfit - Xfit.mean(0)) / (Xfit.std(0) + 1e-8)
    corr = np.corrcoef(Z.T)
    cov = np.cov(Xfit.T)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, fontsize=7)
    plt.colorbar(im, fraction=0.046)
    ax.set_title("Pearson correlation of QC features (clean train tiles)")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/feature_correlation.png", dpi=110)
    plt.close()
    eig = np.linalg.eigvalsh(np.corrcoef(Z.T))[::-1]
    divs = {}
    for k in sorted(set(kind)):
        if k == "clean":
            continue
        Xa = Xte[kind == k]
        divs[k] = {fn: dict(zip(["kl", "js"], kl_js_divergence(F["te_clean"][:, i], Xa[:, i]))) for i, fn in enumerate(FEATURE_NAMES)}
    top = {k: sorted(((fn, v["js"]) for fn, v in dd.items()), key=lambda t: -t[1])[:3] for k, dd in divs.items()}
    # stain variability between slides (two hospitals in CAMELYON16)
    slides = sorted(set(d["te_clean_slide"]) | set(d["tr_clean_slide"]))
    allc = np.concatenate([d["tr_clean"], d["te_clean"]])
    alls = np.concatenate([d["tr_clean_slide"], d["te_clean_slide"]])
    hists = {}
    for s in slides:
        px = allc[alls == s][:150].reshape(-1, 3)[::17]
        h, _ = np.histogramdd(px, bins=(8, 8, 8), range=((0, 256),) * 3)
        hists[s] = (h.ravel() + 1e-6) / (h.sum() + 1e-6 * h.size)
    jsm = np.array([[jensenshannon(hists[a], hists[b], base=2) ** 2 for b in slides] for a in slides])
    strong = [(FEATURE_NAMES[i], FEATURE_NAMES[j], float(corr[i, j])) for i in range(len(corr)) for j in range(i + 1, len(corr)) if abs(corr[i, j]) > 0.85]
    save_json(dict(strong_correlations=strong, corr_eigenvalues=eig.tolist(),
                   effective_rank_90pct=int(np.searchsorted(np.cumsum(eig) / eig.sum(), 0.9) + 1),
                   cov_condition_number=float(np.linalg.cond(cov)),
                   top_js_features_per_artifact=top, divergences=divs,
                   slide_color_js=dict(slides=slides, matrix=jsm.tolist())), "stats.json")

    # BIC curve + score distributions
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ks = sorted(gmm.bic_)
    ax[0].plot(ks, [gmm.bic_[k] for k in ks], "o-")
    ax[0].set_xlabel("K (components)")
    ax[0].set_ylabel("BIC (lower is better)")
    ax[0].set_title("GMM model selection")
    s = gmm.score(Xte)
    lo, hi = np.percentile(s, [0.5, 99])
    for k, c in [("clean", "tab:green"), ("background", "tab:gray"), ("gaussian_blur", "tab:blue"),
                 ("pen_mark", "tab:purple"), ("tissue_fold", "tab:red"), ("partial_blur", "tab:orange")]:
        ax[1].hist(np.clip(s[kind == k], lo, hi), bins=60, alpha=0.5, label=k, color=c, density=True)
    ax[1].axvline(gmm.threshold_, color="k", ls="--", label="reject threshold (95% clean pass)")
    ax[1].set_xlabel("-log p(x)  (GMM anomaly score)")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIGS}/gmm_bic_and_scores.png", dpi=110)
    plt.close()
    print("done")


if __name__ == "__main__":
    main()
