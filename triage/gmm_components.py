"""Unsupervised H&E tissue-component segmentation with a pixel-level GMM
(nuclei / stroma-cytoplasm / FLUID / red blood cells / background-fat).

Physics
-------
Beer-Lambert: optical density OD_c = -log(I_c / I_0) is *linear* in stain
amount, so in OD space every pixel is (approximately) a non-negative mix of
the haematoxylin and eosin stain vectors. Colour deconvolution
(Ruifrok & Johnston 2001) gives per-pixel stain concentrations (H, E).

Pixel descriptor  x = [H, E, OD_total, a* (Lab red-green), local std of OD]
  * H high                     -> nuclei (chromatin)
  * E high, textured           -> stroma / cytoplasm (collagen fibres)
  * E low-moderate, SMOOTH,    -> FLUID: proteinaceous / serous fluid,
    pale pink, low OD             plasma, oedema, lymph in sinuses/vessels
  * a* very high (red)         -> red blood cells / haemorrhage
  * OD ~ 0                     -> glass, fat vacuoles, lumen

Why a GMM (and not k-means / Otsu) for this
-------------------------------------------
* Clusters in OD space are elongated along the stain vectors and have very
  different sizes/variances (background is a tight blob, stroma is broad).
  k-means = GMM with equal, spherical, shared covariance and hard
  assignment -> Voronoi cells that cut big clusters in half.
  A full-covariance GMM fits ellipsoids with their own priors pi_k.
* Soft posteriors p(k | x) give a per-pixel *confidence*; fluid vs. pale
  stroma is genuinely ambiguous and the posterior exposes that.
* The model is ~K*(d + d^2) numbers: trivially deployable on an edge device,
  and fitted without any annotation.
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from triage.features import _HED_FROM_RGB

COMPONENTS = ["background/fat", "fluid", "stroma/cytoplasm", "nuclei", "blood (RBC)"]
COLORS = np.array([[245, 245, 245], [120, 200, 255], [255, 150, 200], [70, 30, 160], [220, 20, 20]], np.uint8)


def pixel_descriptor(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    od = -np.log10((f + 1.0) / 256.0)
    hed = od.reshape(-1, 3) @ _HED_FROM_RGB
    od_sum = od.sum(2)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_star = (lab[..., 1] - 128.0) / 128.0
    mu = cv2.GaussianBlur(od_sum, (0, 0), 3)
    loc_std = np.sqrt(np.maximum(cv2.GaussianBlur(od_sum ** 2, (0, 0), 3) - mu ** 2, 0))
    return np.column_stack([hed[:, 0], hed[:, 1], od_sum.ravel(), a_star.ravel(), loc_std.ravel()]).astype(np.float32)


class GMMTissueComponents:
    def __init__(self, k: int = 7, seed: int = 0):
        self.k, self.seed = k, seed

    def fit(self, tiles, n_pixels: int = 200_000):
        rng = np.random.default_rng(self.seed)
        X = np.concatenate([pixel_descriptor(t)[rng.choice(t.shape[0] * t.shape[1], n_pixels // len(tiles))]
                            for t in tiles])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.gmm = GaussianMixture(self.k, covariance_type="full", random_state=self.seed, n_init=2).fit((X - self.mu) / self.sd)
        self.comp2class = self._name_components(self.gmm.means_ * self.sd + self.mu)
        return self

    @staticmethod
    def _name_components(m):
        """Name the K Gaussians from their mean descriptor [H, E, OD, a*, std]
        (OD = summed log10 optical density, so glass ~0.1, dense nuclei ~1.5+):
          OD < 0.20                          -> background / fat vacuole
          H / (H + E) > 0.6                  -> nuclei (haematoxylin-dominant)
          a* > 0.45 and E > H                -> red blood cells
          0.20 <= OD < 0.45 and E >= H       -> fluid: pale eosinophilic
                                                (proteinaceous fluid, plasma,
                                                 oedema, lymph in sinuses)
          otherwise                          -> stroma / cytoplasm
        """
        H, E, OD, A, S = m.T
        names = np.full(len(m), 2)
        ratio = H / np.maximum(H + E, 1e-6)
        names[(OD >= 0.20) & (OD < 0.45) & (E >= H)] = 1
        names[(A > 0.45) & (E > H) & (OD >= 0.20)] = 4
        names[(ratio > 0.6) & (OD >= 0.45)] = 3
        names[OD < 0.20] = 0
        return names

    def predict(self, rgb: np.ndarray, return_proba: bool = False):
        X = (pixel_descriptor(rgb) - self.mu) / self.sd
        post = self.gmm.predict_proba(X)                      # (N, K) soft responsibilities
        cls_post = np.zeros((len(X), len(COMPONENTS)), np.float32)
        for k, c in enumerate(self.comp2class):
            cls_post[:, c] += post[:, k]
        lab = cls_post.argmax(1).reshape(rgb.shape[:2])
        if return_proba:
            return lab, cls_post.reshape(rgb.shape[:2] + (len(COMPONENTS),))
        return lab

    def fractions(self, rgb: np.ndarray) -> np.ndarray:
        lab = self.predict(rgb)
        return np.bincount(lab.ravel(), minlength=len(COMPONENTS)) / lab.size


def colorize(lab):
    return COLORS[lab]
