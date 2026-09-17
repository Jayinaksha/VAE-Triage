# Digital Pathology VQ-VAE Triage and QC Pipeline

This repository implements a Gigapixel Pathology Image Triage & QC Pipeline using PyTorch. It is designed to act as a preprocessing gate for high-resolution (~50K x 50K) digital pathology images before they are passed to heavy segmentation models.

## Purpose

Processing gigapixel images (Whole Slide Images - WSIs) is computationally expensive. A significant portion of these images consists of blank background, blur, or sensor artifacts. 

Instead of running a heavy U-Net or DeepLabv3+ segmentation model on every single tile, this pipeline:
1. **Tiles** the gigapixel image efficiently using memory mapping to prevent out-of-memory errors.
2. Uses a **Vector Quantized Variational Autoencoder (VQ-VAE)** trained on normal tissue patches to compute a reconstruction error for each tile.
3. **Triages** tiles: Tiles with high reconstruction error (anomalies, blur, artifacts, or extreme out-of-distribution regions) are flagged and dropped.
4. **Compresses** valid tiles into a latent space representation which can be used for fast downstream tasks.

## Architecture

- `data/`: Custom PyTorch `GigapixelTileDataset` for loading WSIs via memory mapping and extracting overlapping patches.
- `models/`: PyTorch implementation of the VQ-VAE (Encoder, Learnable Codebook Quantizer, Decoder).
- `training/`: Training loop with Automatic Mixed Precision (AMP) and VQ commitment loss.
- `inference/`: Scripts for triage (anomaly detection via thresholding) and ONNX export for efficient INT8/FP16 deployment (e.g., on Jetson or CPU).

## Usage

### 1. Installation
Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data (If needed)
If you do not have a .tif Whole Slide Image on hand, you can generate a synthetic 5000x5000 test image with simulated artifacts:
```bash
python utils/dummy_data.py
```

### 3. Training
Train the VQ-VAE on the image tiles:
```bash
python training/train.py --image_path dummy_wsi.tif --epochs 5 --batch_size 32
```

### 4. Inference & Triage
Run the anomaly detection script to flag bad tiles:
```bash
python inference/triage.py --image_path dummy_wsi.tif --checkpoint checkpoints/vq_vae_epoch_5.pt
```

### 5. Deployment Export
Export the trained model to ONNX for production optimization:
```bash
python inference/export.py --checkpoint checkpoints/vq_vae_epoch_5.pt --output_path vq_vae_triage.onnx
```

## Features for the Resume
- Custom memory-mapped tiling for high-resolution images.
- Unsupervised anomaly detection via VQ-VAE reconstruction loss.
- Scalable deployment pipeline (ONNX).
