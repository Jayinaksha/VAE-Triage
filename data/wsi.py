"""Lazy, pyramid-aware reader for gigapixel whole-slide images (WSIs).

Why not `tifffile.memmap`?
    Real WSIs (CAMELYON16, TCGA, scanner output) are *tiled, JPEG-compressed,
    multi-resolution* (pyramidal) TIFFs. `tifffile.memmap` only works for
    uncompressed, contiguous TIFFs - it raises on every real slide. Instead we
    open each pyramid level as a zarr store (`tifffile.aszarr`), which decodes
    only the compressed TIFF tiles that intersect the requested region.
    RAM usage is therefore O(tile), not O(slide).

Coordinates: all public methods take (x, y) in the pixel grid of the
requested `level` (level 0 = full resolution, each level ~2x downsampled).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import tifffile
import zarr


@dataclass
class LevelInfo:
    level: int
    height: int
    width: int
    downsample: float


class WSIReader:
    def __init__(self, path: str):
        self.path = path
        self._tif = tifffile.TiffFile(path)
        series = self._tif.series[0]
        self._levels = series.levels
        self._store = series.aszarr()
        self._root = zarr.open(self._store, mode="r")
        h0, w0 = self._levels[0].shape[:2]
        self.levels: List[LevelInfo] = []
        for i, lv in enumerate(self._levels):
            h, w = lv.shape[:2]
            self.levels.append(LevelInfo(i, h, w, w0 / w))

    # ------------------------------------------------------------------ utils
    def _arr(self, level: int):
        if isinstance(self._root, zarr.Array):  # single-level file
            return self._root
        return self._root[str(level)]

    @property
    def level_count(self) -> int:
        return len(self.levels)

    def dimensions(self, level: int = 0) -> Tuple[int, int]:
        lv = self.levels[level]
        return lv.width, lv.height

    def best_level_for_downsample(self, ds: float) -> int:
        best = 0
        for lv in self.levels:
            if lv.downsample <= ds + 1e-6:
                best = lv.level
        return best

    # ------------------------------------------------------------------- read
    def read_region(self, x: int, y: int, w: int, h: int, level: int = 0) -> np.ndarray:
        """Return an (h, w, C) uint8 array; out-of-bounds is padded white."""
        arr = self._arr(level)
        H, W = arr.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        out_shape = (h, w) + tuple(arr.shape[2:])
        out = np.full(out_shape, 255, dtype=arr.dtype)
        if x1 > x0 and y1 > y0:
            out[y0 - y:y1 - y, x0 - x:x1 - x] = arr[y0:y1, x0:x1]
        if out.ndim == 3 and out.shape[2] == 4:  # RGBA -> RGB
            out = out[..., :3]
        return out

    def read_level(self, level: int) -> np.ndarray:
        """Read a whole (low-resolution) level, e.g. for a thumbnail."""
        arr = np.asarray(self._arr(level)[:])
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        return arr

    def thumbnail(self, max_side: int = 2048) -> Tuple[np.ndarray, int]:
        """Smallest-level image whose longer side is >= max_side."""
        chosen = self.level_count - 1
        for lv in self.levels:
            if max(lv.height, lv.width) >= max_side:
                chosen = lv.level
        return self.read_level(chosen), chosen

    def iter_tile_coords(self, tile: int, stride: Optional[int] = None,
                         level: int = 0) -> Iterator[Tuple[int, int]]:
        stride = stride or tile
        W, H = self.dimensions(level)
        for y in range(0, H - tile + 1, stride):
            for x in range(0, W - tile + 1, stride):
                yield x, y

    def close(self):
        self._store.close()
        self._tif.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
