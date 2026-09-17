import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
from PIL import Image

class GigapixelTileDataset(Dataset):
    """
    Dataset to handle large pathology/aerial images.
    It reads a large TIF file using memory mapping, and extracts tiles.
    """
    def __init__(self, image_path, tile_size=256, stride=256, transform=None, background_threshold=0.9):
        self.image_path = image_path
        self.tile_size = tile_size
        self.stride = stride
        self.transform = transform
        self.background_threshold = background_threshold
        
        # Open as memmap using tifffile to avoid loading whole image into RAM
        # Assuming typical RGB image shape (H, W, 3) or (3, H, W).
        # tifffile returns (H, W, 3) for standard RGB tif
        self.memmap = tifffile.memmap(image_path)
        
        if len(self.memmap.shape) == 3 and self.memmap.shape[2] == 3:
            self.H, self.W, self.C = self.memmap.shape
        elif len(self.memmap.shape) == 3 and self.memmap.shape[0] == 3:
            # Handle (C, H, W)
            self.C, self.H, self.W = self.memmap.shape
            # For simplicity, we'll transpose on the fly if needed, or assume HWC for reading
        else:
            raise ValueError(f"Unexpected image shape: {self.memmap.shape}")

        self.tiles = self._compute_tiles()

    def _compute_tiles(self):
        """
        Pre-compute coordinates for all valid tiles.
        We skip tiles that are almost entirely background (white).
        """
        valid_tiles = []
        for y in range(0, self.H - self.tile_size + 1, self.stride):
            for x in range(0, self.W - self.tile_size + 1, self.stride):
                valid_tiles.append((y, x))
        return valid_tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        y, x = self.tiles[idx]
        
        if self.memmap.shape[-1] == 3:
            patch = self.memmap[y:y+self.tile_size, x:x+self.tile_size, :]
        else:
            patch = self.memmap[:, y:y+self.tile_size, x:x+self.tile_size]
            patch = np.transpose(patch, (1, 2, 0)) # Convert to HWC for PIL
            
        # Convert to PIL Image for standard torchvision transforms
        image = Image.fromarray(patch)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([y, x])
