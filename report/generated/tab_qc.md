**Stage-1 tile QC on 4 unseen test slides: 2290 clean tissue tiles vs. 600 glass + 1500 artefact tiles. Unsupervised thresholds calibrated on unseen training slides (95\% clean pass).**

| Method | AUROC | AUPRC | Prec. | Recall | F1 | Spec. | Tum. rej. |
|---|---|---|---|---|---|---|---|
| Tissue fraction (Otsu-like) | 0.609 | 0.690 | 0.833 | 0.348 | 0.491 | 0.936 | 0.004 |
| Laplacian variance (focus) | 0.759 | 0.794 | 0.874 | 0.537 | 0.665 | 0.929 | 0.004 |
| Tenengrad (Sobel energy) | 0.764 | 0.792 | 0.847 | 0.494 | 0.624 | 0.918 | 0.004 |
| HistoQC-style rule set | 0.857 | 0.861 | 0.936 | 0.463 | 0.620 | 0.971 | 0.004 |
| Single Gaussian (Mahalanobis, K=1) | 0.954 | 0.943 | 0.923 | 0.890 | 0.906 | 0.932 | 0.006 |
| GMM (K=4) same-slide calibration [pitfall] | 0.942 | 0.936 | 0.735 | 0.948 | 0.828 | 0.687 | 0.431 |
| GMM (K=6, BIC) LOSO calibration [ours] | 0.939 | 0.933 | 0.894 | 0.892 | 0.893 | 0.903 | 0.084 |
| GMM (K=6, slide-CV) LOSO calibration [ours, final] | 0.939 | 0.933 | 0.918 | 0.876 | 0.897 | 0.928 | 0.035 |
| GMM on QC-only feature subset [ours, tumour-safe] | 0.937 | 0.928 | 0.944 | 0.721 | 0.818 | 0.961 | 0.002 |
| Isolation Forest | 0.913 | 0.876 | 0.899 | 0.690 | 0.780 | 0.929 | 0.004 |
| One-class SVM (RBF) | 0.905 | 0.904 | 0.938 | 0.658 | 0.773 | 0.960 | 0.006 |
| Logistic regression (supervised) | 0.966 | 0.949 | 0.925 | 0.927 | 0.926 | 0.931 | 0.004 |
| Random forest (supervised) | 0.960 | 0.955 | 0.863 | 0.930 | 0.895 | 0.865 | 0.020 |
| VQ-VAE recon-MSE (LOSO-calibrated threshold) | 0.376 | 0.517 | 0.736 | 0.178 | 0.286 | 0.941 | 0.020 |
| VQ-VAE recon-MSE (repo default threshold 0.05) | 0.376 | 0.517 | 0.582 | 0.207 | 0.305 | 0.863 | 0.129 |
| ResNet18-ImageNet feats + PCA32 + GMM(K=4) | 0.769 | 0.708 | 0.714 | 0.197 | 0.309 | 0.928 | 0.043 |
| MobileNetV3-Small supervised QC | 0.976 | 0.958 | 0.876 | 0.982 | 0.926 | 0.873 | 0.051 |
