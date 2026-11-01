"""Gaussian-Mixture-Model tile triage (unsupervised, edge-deployable).

Model
-----
Let x in R^d be the hand-crafted descriptor of a tile (triage/features.py).
We fit, on *clean tissue tiles only*,

    p(x) = sum_k pi_k N(x | mu_k, Sigma_k)          (full covariances)

with EM, choosing K by the Bayesian Information Criterion
BIC = -2 log L + p log n. The anomaly score of a new tile is its negative
log-likelihood  s(x) = -log p(x). A tile is rejected when s(x) > tau, where
tau is the q-th percentile of s on held-out clean tiles - i.e. we fix the
false-rejection rate on good tissue (e.g. q = 95 -> ~5% of good tiles lost)
and every metric is reported at that operating point. No artefact labels
are needed to train or to set tau.

Why a GMM for edge triage
-------------------------
* Parameters: K*(d + d(d+1)/2 + 1) floats -> ~ 2-10 KB. A VQ-VAE is ~MBs.
* Inference: K Mahalanobis distances with pre-factorised Cholesky matrices,
  i.e. O(K d^2) FLOPs (~10^4) - microseconds; the features themselves
  (~1 ms/tile) dominate. No GPU, no deep-learning runtime required.
* Interpretable: per-component means are "tissue types"; the score
  decomposes into per-feature z-scores, so a QC operator sees *why*
  a tile was rejected (e.g. log_lap_var 5 sigma low -> out of focus).
* Calibrated: -log p(x) is a proper density score; tau has a direct
  false-rejection-rate meaning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class GMMTriage:
    k_range: tuple = (1, 2, 3, 4, 6, 8)
    covariance_type: str = "full"
    reject_percentile: float = 95.0
    reg_covar: float = 1e-3
    seed: int = 0
    # fitted
    center_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    gmm_: Optional[GaussianMixture] = None
    threshold_: Optional[float] = None
    bic_: dict = field(default_factory=dict)

    # robust standardisation (median / IQR) - artefact-free train data still has outliers
    def _std(self, X):
        return (X - self.center_) / self.scale_

    def select_k_by_group(self, X: np.ndarray, groups: np.ndarray) -> int:
        """Pick K by mean held-out log-likelihood with leave-one-group-out
        (group = slide). BIC assumes i.i.d. tiles; tiles of one slide are
        strongly correlated, so BIC over-estimates K. Held-out *slides* measure
        what we actually need: generalisation to a new slide / scanner run."""
        ll = {}
        for k in self.k_range:
            vals = []
            for g in np.unique(groups):
                m = GMMTriage(k_range=(k,), covariance_type=self.covariance_type,
                              reg_covar=self.reg_covar, seed=self.seed).fit(X[groups != g])
                vals.append(-m.score(X[groups == g]).mean())
            ll[k] = float(np.mean(vals))
        self.heldout_ll_ = ll
        self.k_range = (max(ll, key=ll.get),)
        return self.k_range[0]

    def fit(self, X_clean: np.ndarray, X_clean_val: Optional[np.ndarray] = None):
        self.center_ = np.median(X_clean, axis=0)
        q75, q25 = np.percentile(X_clean, [75, 25], axis=0)
        self.scale_ = np.where((q75 - q25) > 1e-6, (q75 - q25) / 1.349, X_clean.std(0) + 1e-6)
        Z = self._std(X_clean)
        best = None
        for k in self.k_range:
            if k > len(Z) // 10:
                continue
            g = GaussianMixture(k, covariance_type=self.covariance_type, reg_covar=self.reg_covar,
                                random_state=self.seed, n_init=2, max_iter=300).fit(Z)
            b = g.bic(Z)
            self.bic_[k] = float(b)
            if best is None or b < best[0]:
                best = (b, g)
        self.gmm_ = best[1]
        ref = X_clean_val if X_clean_val is not None else X_clean
        self.threshold_ = float(np.percentile(self.score(ref), self.reject_percentile))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Negative log-likelihood (higher = more anomalous)."""
        return -self.gmm_.score_samples(self._std(X))

    def predict_reject(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold_

    def explain(self, x: np.ndarray, names: List[str], top: int = 3):
        """Per-feature z-scores w.r.t. the most responsible component."""
        z = self._std(x[None])[0]
        k = int(self.gmm_.predict(z[None])[0])
        d = (z - self.gmm_.means_[k]) / np.sqrt(np.diag(self.gmm_.covariances_[k]))
        idx = np.argsort(-np.abs(d))[:top]
        return [(names[i], float(d[i])) for i in idx]

    # ---------------------------------------------------------------- export
    def export(self, path: str):
        """Dump to JSON with pre-computed Cholesky factors of the precisions,
        so the edge scorer (EdgeGMMScorer) needs only NumPy."""
        g = self.gmm_
        payload = dict(
            center=self.center_.tolist(), scale=self.scale_.tolist(),
            weights=g.weights_.tolist(), means=g.means_.tolist(),
            prec_chol=g.precisions_cholesky_.tolist(), threshold=self.threshold_,
        )
        with open(path, "w") as f:
            json.dump(payload, f)
        return path


class EdgeGMMScorer:
    """Dependency-free (NumPy) GMM scorer for edge devices / scanner PCs."""

    def __init__(self, path: str):
        p = json.load(open(path))
        self.center = np.asarray(p["center"], np.float32)
        self.scale = np.asarray(p["scale"], np.float32)
        self.logw = np.log(np.asarray(p["weights"], np.float32))
        self.means = np.asarray(p["means"], np.float32)
        self.L = np.asarray(p["prec_chol"], np.float32)          # (K, d, d)
        self.threshold = p["threshold"]
        d = self.means.shape[1]
        self.log_det = np.log(np.diagonal(self.L, axis1=1, axis2=2)).sum(1)  # log|Sigma^-1|^(1/2)
        self.const = -0.5 * d * np.log(2 * np.pi)

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.center) / self.scale                        # (n, d)
        # y_k = (z - mu_k) @ L_k  ->  squared Mahalanobis distance
        Y = np.einsum("nd,kde->nke", Z, self.L) - np.einsum("kd,kde->ke", self.means, self.L)[None]
        log_comp = self.logw + self.const + self.log_det - 0.5 * (Y ** 2).sum(-1)
        m = log_comp.max(1, keepdims=True)
        return -(m[:, 0] + np.log(np.exp(log_comp - m).sum(1)))

    def reject(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold
