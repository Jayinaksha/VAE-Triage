**ISPRS Potsdam (eroded boundaries, clutter excluded from the mean). Ours: 12 fixed 1024x1024 crops of test tiles 2\_13, 3\_14, 5\_15. Literature: full 14-tile test set, GPU training for 100+ epochs.**

| Method | mIoU5 | mF1-5 | OA | Imp. | Bldg | LowVeg | Tree | Car |
|---|---|---|---|---|---|---|---|---|
| K-means (k=6, unsupervised, Hungarian-mapped) | 19.8 | 32.5 | 36.1 | 24.8 | 10.9 | 25.6 | 25.0 | 12.6 |
| GMM-Bayes (4 comps/class) | 43.1 | 60.0 | 60.0 | 51.6 | 41.7 | 46.6 | 38.7 | 36.8 |
| GMM-Bayes + posterior smoothing | 48.5 | 65.2 | 64.8 | 55.1 | 45.2 | 52.1 | 45.7 | 44.5 |
| Random Forest (RGB features) | 47.1 | 63.9 | 62.2 | 54.5 | 46.6 | 46.6 | 40.9 | 46.7 |
| Random Forest (RGB + NIR/NDVI) | 50.4 | 66.8 | 65.9 | 62.2 | 51.9 | 48.1 | 44.0 | 45.9 |
| Residual U-Net (repo), no TTA (30 min CPU) | 58.2 | 73.4 | 72.7 | 62.9 | 57.5 | 61.3 | 48.4 | 61.0 |
| Residual U-Net (repo), 50% overlap stitching (30 min CPU) | 60.0 | 74.8 | 73.5 | 63.2 | 57.9 | 62.9 | 49.8 | 66.3 |
| Residual U-Net (repo), 8x TTA (30 min CPU) | 61.3 | 75.8 | 74.2 | 64.4 | 60.2 | 63.0 | 49.7 | 69.0 |
| Lit: ERFNet (UNetFormer T9) | 76.2 | -- | -- | -- | -- | -- | -- | -- |
| Lit: BiSeNet R18 | 81.7 | -- | -- | -- | -- | -- | -- | -- |
| Lit: DANet R18 | 80.3 | -- | -- | -- | -- | -- | -- | -- |
| Lit: ABCNet R18 | 86.5 | -- | -- | -- | -- | -- | -- | -- |
| Lit: SwinUperNet Swin-T | 85.8 | -- | -- | -- | -- | -- | -- | -- |
| Lit: UNetFormer R18 | 86.8 | -- | -- | -- | -- | -- | -- | -- |
| Lit: FT-UNetFormer Swin-B | 87.5 | -- | -- | -- | -- | -- | -- | -- |
