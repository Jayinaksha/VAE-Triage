"""Pipeline diagrams (PNG) shared by the LaTeX report, Markdown and slides."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=9.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.45, body, ha="center", va="top", fontsize=7.6, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, label=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.1, label, ha="center", fontsize=7.2, color="#444")


def pathology():
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    W, H = 2.6, 2.3
    y = 3.4
    box(ax, 0.1, y, W, H, "0. WSI reader", "pyramidal JPEG-TIFF\n(~100k x 200k px)\nzarr lazy access:\ndecode only touched\n512px TIFF tiles", "#eef3fb")
    box(ax, 3.0, y, W, H, "1. Thumbnail tissue mask", "level ds16-32 (~4-8 um/px)\npixel GMM K=2 on\n(saturation, optical density)\n+ morphology\nDice 0.84 (Otsu 0.53)", "#e7f5ee")
    box(ax, 5.9, y, W, H, "2. Tile grid @20x", "256x256, keep cells with\ntissue >= 25%\n~70% of grid is glass\n-> skipped for free", "#e7f5ee")
    box(ax, 8.8, y, W, H, "3. QC triage (edge)", "23 hand-crafted features\n(focus, stain, ink, folds,\nnoise) ~5 ms/tile\nGMM -log p(x): 0.01 ms\nAUROC 0.94, 45 KB model", "#e7f5ee")
    box(ax, 11.7, y, W + 0.6, H, "4. Tissue components", "pixel GMM in OD/HED space\nnuclei | stroma | FLUID |\nRBC | background/fat\nfractions -> tumour AUROC 0.93\n(no pixel labels needed)", "#e7f5ee")
    for x in (2.7, 5.6, 8.5, 11.4):
        arrow(ax, x, y + H / 2, x + 0.3, y + H / 2)
    box(ax, 3.0, 0.3, 3.6, 2.3, "Rejected tiles -> QC report", "glass: dropped silently\nartefacts: flagged with reason\n(e.g. 'log_lap_var -10 sigma'\n= out of focus) -> rescan\nNEVER silently drop tissue", "#fdf0e6")
    box(ax, 7.2, 0.3, 3.6, 2.3, "Optional: VQ-VAE / deep", "VQ-VAE recon error,\nlatent codes for retrieval /\ncompression; ONNX Runtime\nFP32/INT8 on server or Jetson", "#f3eef9")
    box(ax, 11.4, 0.3, 3.4, 2.3, "5. Heavy model (GPU/cloud)", "U-Net / DeepLab / ViT-MIL\nonly on kept tiles\n(~25-35% of the grid)\n-> 3-4x less GPU time", "#eef3fb")
    arrow(ax, 10.1, y, 5.2, 2.65, "reject")
    arrow(ax, 10.1, y, 9.0, 2.65, "")
    arrow(ax, 13.3, y, 13.1, 2.65, "keep + component stats")
    ax.text(7.5, 6.05, "Edge-first WSI triage pipeline (VAE-Triage, improved)", ha="center", fontsize=12, weight="bold")
    ax.text(7.5, 5.8, "green = runs on CPU / scanner-side edge box with NumPy+OpenCV only", ha="center", fontsize=8.5, color="#1b7f5b")
    plt.savefig(f"{OUT}/pipeline_pathology.png", dpi=150, bbox_inches="tight")
    plt.close()


def satellite():
    fig, ax = plt.subplots(figsize=(15, 3.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    W, H, y = 2.7, 2.4, 0.4
    box(ax, 0.1, y, W, H, "6000x6000 tile", "RGB(+NIR), 5 cm GSD\nlabels: 6 classes\n(eroded boundaries)", "#eef3fb")
    box(ax, 3.1, y, W, H, "Patch extraction", "256x256, stride 256\n(or 128 = 50% overlap)\n+ border-aligned last\npatch (bug fix)", "#e7f5ee")
    box(ax, 6.1, y, W, H, "Augmentation", "rot90 x flip (D4 group)\nbrightness/contrast\nmulti-scale (rates 1-3)\n(stain/colour jitter)", "#e7f5ee")
    box(ax, 9.1, y, W, H, "Residual U-Net", "4 enc/dec blocks\nconcat skips\ndilated bottleneck\n(rates 1,2,4,8)\nsoftmax 6 classes", "#f3eef9")
    box(ax, 12.1, y, W + 0.2, H, "Stitch + TTA", "average overlapping\nsoftmax maps, 8x D4 TTA\nargmax -> 6000x6000 map\nmetrics: IoU / F1 / OA", "#eef3fb")
    for x in (2.8, 5.8, 8.8, 11.8):
        arrow(ax, x, y + H / 2, x + 0.3, y + H / 2)
    ax.text(7.5, 3.25, "Satellite / aerial high-resolution segmentation pipeline (same tiling logic as a WSI)", ha="center", fontsize=11, weight="bold")
    plt.savefig(f"{OUT}/pipeline_satellite.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    pathology()
    satellite()
