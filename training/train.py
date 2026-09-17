import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vq_vae import VQVAE
from data.dataset import GigapixelTileDataset
from data.transforms import get_train_transforms
from utils.dummy_data import generate_dummy_wsi

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate dummy data if not provided
    image_path = args.image_path
    if image_path is None or not os.path.exists(image_path):
        print("No valid image provided. Generating synthetic 10000x10000 WSI...")
        image_path = "dummy_wsi.tif"
        if not os.path.exists(image_path):
            generate_dummy_wsi(image_path, size=10000)

    transform = get_train_transforms(args.tile_size)
    dataset = GigapixelTileDataset(image_path, tile_size=args.tile_size, stride=args.tile_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = VQVAE(num_hiddens=128, num_residual_hiddens=32, num_residual_layers=2,
                  embedding_dim=64, num_embeddings=512, commitment_cost=0.25).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(args.save_dir, exist_ok=True)
    
    model.train()
    for epoch in range(args.epochs):
        train_res_recon_error = []
        train_res_perplexity = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for data, _ in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                vq_loss, data_recon, perplexity = model(data)
                recon_error = F.mse_loss(data_recon, data)
                loss = recon_error + vq_loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())
            
            pbar.set_postfix({'Recon': f"{recon_error.item():.4f}", 'PPL': f"{perplexity.item():.2f}"})

        print(f"Epoch {epoch+1} | Recon Error: {sum(train_res_recon_error)/len(train_res_recon_error):.4f} | PPL: {sum(train_res_perplexity)/len(train_res_perplexity):.2f}")
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"vq_vae_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=None, help="Path to high-res TIF")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    train(args)
