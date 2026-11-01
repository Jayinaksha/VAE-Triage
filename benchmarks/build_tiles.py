"""Build the CAMELYON16 tile benchmark used by every experiment.

For each slide we
 1. lay a non-overlapping 256x256 grid at pyramid level 1 (20x, ~0.49 um/px);
 2. label every grid cell from the *low-resolution* masks only (cheap):
      tissue_frac  <- CAMELYON16 background/tissue mask (Bandi et al. 2019)
      tumor_frac   <- CAMELYON16 pixel tumour mask (expert polygons)
 3. sample tiles per class and read only those tiles from the compressed
    pyramid (lazy zarr access - the 0.5-1 GB slide is never decoded fully).

Classes
  normal     : tissue_frac >= 0.6 and tumor_frac == 0
  tumor      : tumor_frac >= 0.6                       (tumour slides only)
  background : tissue_frac <= 0.02
  border     : 0.1 <= tissue_frac <= 0.5 (partial tissue)

Output: <out>/<slide>.npz with tiles (N,256,256,3) uint8 + labels/coords,
and <out>/grid_stats.json (whole-slide tile census used for throughput math).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.wsi import WSIReader  # noqa: E402

LEVEL = 1
TILE = 256


def mask_fraction_grid(mask_reader: WSIReader, slide_w0: int, slide_h0: int, tile_l0: int, values):
    """Fraction of each level-0 tile footprint whose mask value is in `values`,
    computed on the coarsest mask level that still has >= 4 px per tile."""
    lvl = 0
    for lv in mask_reader.levels:
        scale = lv.width / slide_w0
        if tile_l0 * scale >= 4:
            lvl = lv.level
    m = mask_reader.read_level(lvl)
    if m.ndim == 3:
        m = m[..., 0]
    hit = np.isin(m, values).astype(np.float32)
    scale = m.shape[1] / slide_w0
    step = tile_l0 * scale
    ny, nx = slide_h0 // tile_l0, slide_w0 // tile_l0
    out = np.zeros((ny, nx), np.float32)
    # integral image -> exact box averages
    ii = np.pad(hit.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.clip(np.round(np.arange(ny + 1) * step).astype(int), 0, m.shape[0])
    xs = np.clip(np.round(np.arange(nx + 1) * step).astype(int), 0, m.shape[1])
    for i in range(ny):
        y0, y1 = ys[i], max(ys[i + 1], ys[i] + 1)
        a = ii[y1, xs[1:]] - ii[y0, xs[1:]] - ii[y1, xs[:-1]] + ii[y0, xs[:-1]]
        area = (y1 - y0) * np.maximum(xs[1:] - xs[:-1], 1)
        out[i] = a / area
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/data/camelyon16")
    ap.add_argument("--out", default="/home/user/data/c16_tiles")
    ap.add_argument("--n_normal", type=int, default=450)
    ap.add_argument("--n_tumor", type=int, default=450)
    ap.add_argument("--n_background", type=int, default=150)
    ap.add_argument("--n_border", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    slides = sorted(f[:-4] for f in os.listdir(f"{args.root}/images") if f.endswith(".tif"))
    stats = {}
    for s in slides:
        wsi = WSIReader(f"{args.root}/images/{s}.tif")
        W0, H0 = wsi.dimensions(0)
        ds = wsi.levels[LEVEL].downsample
        tile_l0 = int(round(TILE * ds))
        tissue = mask_fraction_grid(WSIReader(f"{args.root}/tissue/{s}_tissue.tif"), W0, H0, tile_l0, [1, 255])
        tumor = mask_fraction_grid(WSIReader(f"{args.root}/masks/{s}_mask.tif"), W0, H0, tile_l0, [2])
        # tumour mask also marks tissue -> union for robustness
        tissue = np.maximum(tissue, tumor)
        groups = {
            "normal": np.argwhere((tissue >= 0.6) & (tumor == 0)),
            "tumor": np.argwhere(tumor >= 0.6),
            "background": np.argwhere(tissue <= 0.02),
            "border": np.argwhere((tissue >= 0.1) & (tissue <= 0.5)),
        }
        quota = dict(normal=args.n_normal, tumor=args.n_tumor, background=args.n_background, border=args.n_border)
        n_grid = tissue.size
        stats[s] = dict(width0=W0, height0=H0, grid_tiles=int(n_grid),
                        tissue_tiles=int((tissue >= 0.1).sum()), tumor_tiles=int((tumor >= 0.1).sum()),
                        background_tiles=int((tissue < 0.1).sum()),
                        **{f"pool_{k}": int(len(v)) for k, v in groups.items()})
        tiles, labels, coords, tfr, tuf = [], [], [], [], []
        for name, idx in groups.items():
            if len(idx) == 0:
                continue
            pick = idx[rng.choice(len(idx), size=min(quota[name], len(idx)), replace=False)]
            for (i, j) in pick:
                x, y = int(j * TILE), int(i * TILE)  # level-1 coords
                t = wsi.read_region(x, y, TILE, TILE, level=LEVEL)
                tiles.append(t)
                labels.append(name)
                coords.append((x, y))
                tfr.append(tissue[i, j])
                tuf.append(tumor[i, j])
        np.savez_compressed(f"{args.out}/{s}.npz", tiles=np.stack(tiles), labels=np.array(labels),
                            coords=np.array(coords), tissue_frac=np.array(tfr), tumor_frac=np.array(tuf))
        print(s, stats[s], {k: labels.count(k) for k in quota}, flush=True)
        wsi.close()
    json.dump(stats, open(f"{args.out}/grid_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
