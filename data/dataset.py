import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.wsi import WSIReader


class GigapixelTileDataset(Dataset):
    """Tiles a gigapixel WSI lazily.

    Fixes vs. the first version:
      * real WSIs are tiled + JPEG-compressed pyramids; `tifffile.memmap`
        raises on them. We use `WSIReader` (zarr-backed, decodes only the
        TIFF tiles that are touched) and can tile any pyramid `level`.
      * the docstring promised background filtering but the code kept every
        tile. We now compute a tissue mask on a low-resolution thumbnail
        (Otsu on HSV saturation) and keep only grid cells whose tissue
        fraction >= `min_tissue` - typically 60-90 % of a slide is glass, so
        this is the single biggest compute saving in the pipeline.
    """

    def __init__(self, image_path, tile_size=256, stride=256, transform=None,
                 level=0, min_tissue=0.25):
        self.wsi = WSIReader(image_path)
        self.tile_size, self.stride, self.level = tile_size, stride, level
        self.transform = transform
        self.min_tissue = min_tissue
        self.tiles = self._compute_tiles()

    def _compute_tiles(self):
        import cv2
        thumb, tl = self.wsi.thumbnail(max_side=2048)
        sat = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)[..., 1]
        _, mask = cv2.threshold(cv2.GaussianBlur(sat, (5, 5), 0), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scale = self.wsi.levels[self.level].downsample / self.wsi.levels[tl].downsample
        valid = []
        for x, y in self.wsi.iter_tile_coords(self.tile_size, self.stride, self.level):
            x0, y0 = int(x * scale), int(y * scale)
            s = max(1, int(self.tile_size * scale))
            cell = mask[y0:y0 + s, x0:x0 + s]
            if cell.size and cell.mean() >= self.min_tissue:
                valid.append((y, x))
        return valid

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        y, x = self.tiles[idx]
        patch = self.wsi.read_region(x, y, self.tile_size, self.tile_size, level=self.level)
        image = Image.fromarray(patch)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([y, x])


class NpzTileDataset(Dataset):
    """Pre-extracted tiles (benchmarks/build_tiles.py output)."""

    def __init__(self, tiles: np.ndarray, transform=None):
        self.tiles, self.transform = tiles, transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img = Image.fromarray(self.tiles[idx])
        return (self.transform(img) if self.transform else img), torch.tensor([idx, 0])
