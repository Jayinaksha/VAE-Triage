import torch
import os
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vq_vae import VQVAE

def export_to_onnx(args):
    print("Loading model...")
    model = VQVAE(num_hiddens=128, num_residual_hiddens=32, num_residual_layers=2,
                  embedding_dim=64, num_embeddings=512, commitment_cost=0.25)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("WARNING: Checkpoint not found. Exporting untrained model.")
        
    model.eval()

    # Create dummy input for tracing
    # Shape: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, args.tile_size, args.tile_size, device="cpu")

    print(f"Exporting to {args.output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        args.output_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_images'],
        output_names=['reconstruction', 'vq_loss', 'perplexity'],
        dynamic_axes={'input_images': {0: 'batch_size'},
                      'reconstruction': {0: 'batch_size'}}
    )
    
    print("ONNX export complete. This model can now be optimized with TensorRT or OpenVINO for fast deployment.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/vq_vae_epoch_5.pt")
    parser.add_argument("--output_path", type=str, default="vq_vae_triage.onnx")
    parser.add_argument("--tile_size", type=int, default=256)
    args = parser.parse_args()
    
    export_to_onnx(args)
