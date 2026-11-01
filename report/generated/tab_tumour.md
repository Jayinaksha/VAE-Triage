**Tumour (metastasis) vs. normal tiles on test slides (490 tumour / 1800 normal). Literature rows use different splits and are indicative only.**

| Method | AUROC | AUPRC | Type |
|---|---|---|---|
| GMM novelty (K=4) on features | 0.740 | 0.318 | unsup. |
| Isolation Forest on features | 0.661 | 0.257 | unsup. |
| Hematoxylin mean (single feature) | 0.711 | 0.290 | unsup. |
| Logistic regression (supervised) | 0.890 | 0.586 | sup. |
| Random forest (supervised) | 0.945 | 0.797 | sup. |
| VQ-VAE recon-MSE novelty (trained on normal) | 0.749 | 0.317 | unsup. |
| ResNet18 feats + GMM novelty (trained on normal) | 0.671 | 0.272 | unsup. |
| MobileNetV3-Small supervised (1 tumour slide) | 0.557 | 0.436 | sup. |
| GMM tissue-component fractions + LR [ours] | 0.931 | -- | unsup. feats + sup. LR |
| Lit: L1 autoencoder, unsup. (Shvetsova'21) | 0.211 | -- | literature |
| Lit: AE (cross-organ, AnoPILaD'25) | 0.658 | -- | literature |
| Lit: AnoDDPM diffusion (AnoPILaD'25) | 0.686 | -- | literature |
| Lit: Deep-IF ImageNet feats (Shvetsova'21) | 0.906 | -- | literature |
| Lit: Deep perceptual AE (Shvetsova'21) | 0.934 | -- | literature |
