**ISPRS Potsdam (eroded boundaries, clutter excluded from the mean). Ours: 12 fixed 1024x1024 crops of test tiles 2\_13, 3\_14, 5\_15. Literature: full 14-tile test set, GPU training for 100+ epochs.**

| Method | mIoU5 | mF1-5 | OA | Imp. | Bldg | LowVeg | Tree | Car |
|---|---|---|---|---|---|---|---|---|
| K-means (k=6, unsupervised, Hungarian-mapped) | 15.7 | 25.9 | 31.9 | 25.1 | 16.6 | 25.8 | 0.6 | 10.4 |
| GMM-Bayes (4 comps/class) | 40.5 | 57.5 | 56.6 | 47.5 | 44.0 | 43.4 | 34.2 | 33.6 |
| GMM-Bayes + posterior smoothing | 45.9 | 62.7 | 61.7 | 51.2 | 48.3 | 49.6 | 40.9 | 39.3 |
| Random Forest (RGB features) | 45.7 | 62.5 | 60.7 | 54.0 | 47.3 | 46.8 | 37.7 | 42.6 |
| Random Forest (RGB + NIR/NDVI) | 48.4 | 64.9 | 64.0 | 60.7 | 50.9 | 48.6 | 41.0 | 40.8 |
| Lit: ERFNet (UNetFormer T9) | 76.2 | -- | -- | -- | -- | -- | -- | -- |
| Lit: BiSeNet R18 | 81.7 | -- | -- | -- | -- | -- | -- | -- |
| Lit: DANet R18 | 80.3 | -- | -- | -- | -- | -- | -- | -- |
| Lit: ABCNet R18 | 86.5 | -- | -- | -- | -- | -- | -- | -- |
| Lit: SwinUperNet Swin-T | 85.8 | -- | -- | -- | -- | -- | -- | -- |
| Lit: UNetFormer R18 | 86.8 | -- | -- | -- | -- | -- | -- | -- |
| Lit: FT-UNetFormer Swin-B | 87.5 | -- | -- | -- | -- | -- | -- | -- |
