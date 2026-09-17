import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vq_vae import VQVAE
from data.dataset import GigapixelTileDataset
from data.transforms import get_inference_transforms

def triage_image(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = VQVAE(num_hiddens=128, num_residual_hiddens=32, num_residual_layers=2,
                  embedding_dim=64, num_embeddings=512, commitment_cost=0.25).to(device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Loaded checkpoint successfully.")
    else:
        print("WARNING: No checkpoint found, using untrained model for demonstration.")
    
    model.eval()

    # Load Data
    transform = get_inference_transforms()
    dataset = GigapixelTileDataset(args.image_path, tile_size=args.tile_size, stride=args.tile_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = []
    anomalies = []

    print("Running triage...")
    with torch.no_grad():
        for data, coords in tqdm(dataloader):
            data = data.to(device)
            
            # Forward pass
            vq_loss, data_recon, _ = model(data)
            
            # Calculate per-sample reconstruction error
            # Shape: (B, C, H, W) -> (B,)
            mse = F.mse_loss(data_recon, data, reduction='none')
            mse_per_sample = mse.mean(dim=[1,2,3]).cpu().numpy()
            
            for i in range(len(mse_per_sample)):
                y, x = coords[i].numpy()
                error = float(mse_per_sample[i])
                
                tile_info = {"y": int(y), "x": int(x), "mse_error": error}
                results.append(tile_info)
                
                if error > args.anomaly_threshold:
                    anomalies.append(tile_info)

    print(f"Processed {len(results)} tiles.")
    print(f"Found {len(anomalies)} anomalous tiles (Error > {args.anomaly_threshold}).")
    
    # Save valid tiles (triage) to be passed to segmentation model
    valid_tiles = [t for t in results if t['mse_error'] <= args.anomaly_threshold]
    
    output_dict = {
        "total_tiles": len(results),
        "anomalous_tiles_count": len(anomalies),
        "valid_tiles": valid_tiles,
        "anomalies": anomalies
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_dict, f, indent=4)
    print(f"Triage results saved to {args.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to high-res TIF")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/vq_vae_epoch_5.pt")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--anomaly_threshold", type=float, default=0.05, help="MSE threshold to flag artifacts")
    parser.add_argument("--output_json", type=str, default="triage_results.json")
    args = parser.parse_args()
    
    triage_image(args)
