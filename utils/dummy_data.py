import numpy as np
import tifffile
import os

def generate_dummy_wsi(output_path, size=5000):
    """
    Generates a synthetic high-resolution image to simulate a Whole Slide Image (WSI).
    It includes some 'normal' tissue-like patterns and some artificial 'artifacts'
    to test the anomaly detection script.
    """
    print(f"Generating synthetic WSI of size {size}x{size}...")
    # Base background (whitish, common in pathology)
    img = np.ones((size, size, 3), dtype=np.uint8) * 240
    
    # Add random 'cellular' noise
    noise = np.random.normal(loc=0, scale=20, size=(size, size, 3))
    img = np.clip(img - np.abs(noise), 0, 255).astype(np.uint8)
    
    # Introduce 'artifacts' (e.g., black sensor dead zones, or bright colored blobs)
    # Artifact 1: Black square
    img[1000:1500, 1000:1500] = 0
    # Artifact 2: Solid blue blob (out of distribution for normal pink/purple H&E)
    img[3000:3500, 3000:3500] = [0, 0, 255]
    
    tifffile.imwrite(output_path, img, photometric='rgb')
    print(f"Saved synthetic WSI to {output_path}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/../", exist_ok=True)
    generate_dummy_wsi("dummy_wsi.tif")
