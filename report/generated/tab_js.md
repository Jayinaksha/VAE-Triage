**Which features move most under each artefact (Jensen-Shannon divergence, base 2, in [0,1]).**

| Artefact | Top-3 features by JS divergence (clean vs artefact) |
|---|---|
| background | colorfulness (0.95), tissue_frac (0.94), sat_std (0.93) |
| bubble | hf_ratio (0.35), h_std (0.31), lap_norm (0.29) |
| dark_exposure | val_mean (0.94), tissue_frac (0.80), h_mean (0.56) |
| gaussian_blur | lap_norm (0.94), log_lap_var (0.91), log_noise (0.90) |
| jpeg | dark_frac (0.31), log_noise (0.27), edge_density (0.23) |
| motion_blur | log_lap_var (0.82), lap_norm (0.79), hf_ratio (0.66) |
| partial_blur | log_noise (0.37), edge_density (0.36), log_lap_var (0.32) |
| pen_mark | non_he_hue_frac (0.31), h_mean (0.22), h_std (0.22) |
| sensor_noise | non_he_hue_frac (0.94), log_noise (0.76), colorfulness (0.43) |
| tissue_fold | dark_frac (0.46), od_std (0.36), val_std (0.35) |
| washout | e_std (0.78), sat_std (0.76), od_std (0.69) |
