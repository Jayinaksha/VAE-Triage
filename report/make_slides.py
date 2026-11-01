"""Interview slide deck (PPTX) built from the same results/figures."""
import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
R = os.path.join(os.path.dirname(HERE), "results")
SAT_R = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Satellite_based_tile_segmentation", "results")
GREEN, DARK, GREY = RGBColor(0x1B, 0x7F, 0x5B), RGBColor(0x22, 0x22, 0x22), RGBColor(0x66, 0x66, 0x66)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def title(slide, text, sub=None):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.9)).text_frame
    tb.text = text
    p = tb.paragraphs[0]
    p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, DARK
    if sub:
        q = tb.add_paragraph()
        q.text = sub
        q.font.size, q.font.color.rgb = Pt(15), GREEN
    bar = slide.shapes.add_shape(1, Inches(0.5), Inches(1.25), Inches(1.4), Emu(45000))
    bar.fill.solid()
    bar.fill.fore_color.rgb = GREEN
    bar.line.fill.background()


def bullets(slide, items, x=0.5, y=1.5, w=12.3, h=5.6, size=18):
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        lvl = 1 if it.startswith("  ") else 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "– ") + it.strip()
        p.level = lvl
        p.font.size = Pt(size - 3 * lvl)
        p.font.color.rgb = DARK if lvl == 0 else GREY
        p.space_after = Pt(6)


def image(slide, name, x=0.5, y=1.5, w=None, h=None):
    path = os.path.join(FIG, name)
    if os.path.exists(path):
        slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)


def table(slide, header, rows, x=0.5, y=1.5, w=12.3, size=12):
    t = slide.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(0.4 * (len(rows) + 1))).table
    for j, h in enumerate(header):
        c = t.cell(0, j)
        c.text = h
        c.text_frame.paragraphs[0].font.size = Pt(size)
        c.text_frame.paragraphs[0].font.bold = True
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i + 1, j)
            c.text = str(v)
            c.text_frame.paragraphs[0].font.size = Pt(size)


def slide():
    return prs.slides.add_slide(BLANK)


e1, ed, e3, ab = load(f"{R}/exp1_qc_classical.json"), load(f"{R}/exp_deep.json"), load(f"{R}/exp3_gmm_components.json"), load(f"{R}/exp4_ablations.json")

s = slide()
title(s, "Edge-First Triage & Segmentation for Gigapixel Pathology",
      "VAE-Triage (WSI QC + GMM tissue components)  ·  High-resolution U-Net segmentation")
bullets(s, ["Real data: 7 CAMELYON16 breast lymph-node WSIs (203k tiles @20x) + ISPRS Potsdam",
            "Everything measured on a 4-core CPU – no GPU – i.e. edge conditions",
            "GMM-based, label-free, explainable gate in front of heavy deep models"], y=2.2, size=22)

s = slide()
title(s, "Problem", "A WSI is ~10^10 pixels; most of it should never reach the GPU")
bullets(s, ["Pyramidal JPEG-TIFF, 100k × 200k px, 1–4 GB per slide",
            "69.6% of 203,168 tiles in our 7 slides are empty glass (30–78% per slide)",
            "Some tissue tiles are unusable: blur, pen ink, folds, bubbles, exposure, JPEG",
            "Heavy models (U-Net / DeepLab / ViT-MIL) cost 10–100 ms per tile on GPU",
            "Goal: decide cheaply, per tile, at the edge: tissue? usable? what's in it?"])

s = slide()
title(s, "Pipeline")
image(s, "pipeline_pathology.png", x=0.4, y=1.45, w=12.5)

s = slide()
title(s, "Stage 0 – tissue detection on the thumbnail", "Pixel GMM (K=2) on saturation + optical density")
image(s, "chart_tissue.png", x=0.4, y=1.5, w=6.3)
image(s, "tissue_detection_tumor_091.png", x=6.8, y=1.7, w=6.2)
bullets(s, ["GMM Dice 0.84  vs  Otsu 0.53  vs  k-means 0.14",
            "Glass = tight cluster, tissue = broad cluster → needs unequal covariances"], y=5.3, size=16)

s = slide()
title(s, "Stage 1 – tile QC with a GMM density", "23 physics features → −log p(x), threshold = 95% clean pass on UNSEEN slides")
image(s, "chart_qc.png", x=0.3, y=1.45, w=7.4)
bullets(s, ["GMM: AUROC 0.939, F1 0.897, spec 0.928",
            "Single Gaussian 0.954 (few slides → simplest density wins)",
            "Laplacian 0.76 · HistoQC-style 0.86 · IsoForest 0.91",
            "Supervised MobileNetV3 0.976 (needs labels)",
            "45 KB model · 0.01 ms score · 2.1 ms features (128 px)"], x=7.9, y=1.6, w=5.2, size=16)

s = slide()
title(s, "Why unsupervised: artefacts never seen in training")
if ab:
    rows = [[k.replace(" (supervised, 6 artefact types seen)", " (sup.)").replace(" (no artefact ever seen)", " (unsup.)")]
            + [f"{v[a]:.2f}" for a in ["bubble", "pen_mark", "tissue_fold", "jpeg"]] for k, v in ab["unseen_artifact_recall"].items()]
    table(s, ["Model", "bubble", "pen ink", "fold", "JPEG"], rows, y=1.6, size=15)
bullets(s, ["Production scanners create failure modes nobody annotated → novelty detection",
            "Best design: GMM gate for the unknown + small supervised head for known artefacts"], y=4.6, size=18)

s = slide()
title(s, "Per-artefact recall")
image(s, "chart_qc_per_artifact.png", x=1.0, y=1.4, h=5.9)

s = slide()
title(s, "Two traps we found (and fixed)")
bullets(s, ["Calibration trap: threshold from tiles of the SAME slides",
            "  specificity 0.69 and 43% of tumour tiles rejected on new slides",
            "  leave-one-slide-out calibration → specificity 0.93, tumour rejected 3.5%",
            "VQ-VAE reconstruction error as QC score: AUROC 0.376 (worse than chance)",
            "  blurred / empty tiles are smooth → EASIER to reconstruct → look 'normal'",
            "  keep VQ-VAE as compressor/tokeniser + tumour-novelty signal (0.749)",
            "BIC over-estimates K: tiles of one slide are not i.i.d. → select K on held-out slides"])

s = slide()
title(s, "Stage 2 – GMM tissue components incl. FLUID", "Beer–Lambert → OD → colour deconvolution (H, E) → pixel GMM")
image(s, "gmm_tissue_components.png", x=0.3, y=1.45, w=8.0)
bullets(s, ["Components: nuclei · stroma/cytoplasm · fluid · blood · fat/glass",
            "Fluid = pale, eosin-dominant, low OD (plasma, oedema, proteinaceous fluid)",
            "k-means on same features: no fluid cluster",
            f"Cross-slide stability: {100 * e3['stability_pixel_agreement']:.0f}% pixel agreement" if e3 else "",
            f"5 fractions → tumour AUROC {e3['tumour_auc_fraction_vector_logreg']:.2f}" if e3 else ""],
        x=8.5, y=1.6, w=4.6, size=15)

s = slide()
title(s, "Tumour vs normal tiles", "Unsupervised scores vs supervised vs literature")
image(s, "chart_tumour.png", x=0.4, y=1.45, h=5.8)

s = slide()
title(s, "Edge cost (measured, 4-vCPU, no GPU)")
if ed:
    L = ed["latency"]
    rows = [["Features (23-D) + GMM", "2.1 + 0.01 ms", "45 KB", "no labels, explainable"],
            ["VQ-VAE ONNX Runtime FP32 / INT8", f"{L['vqvae_ort_fp32_ms_per_tile']:.1f} / {L['vqvae_ort_int8_ms_per_tile']:.1f} ms", f"{L['vqvae_onnx_fp32_MB']:.1f} / {L['vqvae_onnx_int8_MB']:.1f} MB", "INT8: smaller, not faster"],
            ["MobileNetV3-S ORT FP32 / INT8", f"{L['mobilenet_ort_fp32_ms_per_tile']:.1f} / {L['mobilenet_ort_int8_ms_per_tile']:.1f} ms", f"{L['mobilenet_onnx_fp32_MB']:.1f} / {L['mobilenet_onnx_int8_MB']:.1f} MB", "needs labels per artefact"],
            ["ResNet-18 features (PyTorch)", f"{L['resnet18_feature_ms_per_tile_torch']:.1f} ms", "45 MB", "generic, weak for QC"]]
    table(s, ["Component", "ms / tile", "Size", "Note"], rows, y=1.6, size=14)
bullets(s, ["Slide with 29k grid tiles → ~8.8k tissue tiles → QC in ~5 s on 4 cores",
            "Only ~30% of the grid reaches the GPU model → 3–4× less compute",
            "Always measure on target hardware (INT8 was slower on this CPU)"], y=4.4, size=17)

s = slide()
title(s, "Project 2 – high-resolution segmentation", "Residual U-Net + dilated bottleneck, patch stitching, 8× TTA")
image(s, "pipeline_satellite.png", x=0.4, y=1.5, w=12.5)
bullets(s, ["Same problem as WSI segmentation: image ≫ network input",
            "Bugs fixed: last 112 px of each 6000-px tile never predicted (3.7% of pixels); num_classes fell back to 5",
            "Terminology: 'residual' blocks are concatenations (DenseNet-style)"], y=4.6, size=16)

s = slide()
title(s, "Potsdam results vs literature", "Classical pixel classifiers (minutes on CPU) vs published deep models")
image(s, "chart_potsdam.png", x=0.4, y=1.45, h=5.8)
bullets(s, ["k-means 19.8 → GMM-Bayes 43.1 → + smoothing 48.5",
            "Random forest 47.1 → + NIR/NDVI 50.4",
            "Repo U-Net, 30 min CPU: 58.2 → 61.3 with 8× TTA",
            "Deep SOTA 86.5–87.5: shape & context need a CNN/transformer",
            "→ GMMs for colour/density problems, U-Net for semantics"], x=7.6, y=1.8, w=5.5, size=15)

s = slide()
title(s, "What I changed and what's next")
bullets(s, ["Done: fixed crashes, real WSI reader, tissue mask, GMM QC + export, LOSO calibration,",
            "  GMM tissue components (fluid), explainable rejections, benchmarks on real slides",
            "Next: stain augmentation + many slides for supervised heads (1-slide CNN → 0.56 AUROC)",
            "Next: feature-space (perceptual) anomaly score instead of pixel MSE (lit. 0.66 → 0.93)",
            "Next: U-Net with pretrained encoder, 50% overlap + Gaussian blending, Dice+CE loss",
            "Next: export GMM to C/ONNX for scanner software; GMM likelihood as drift monitor"])

s = slide()
title(s, "60-second summary")
bullets(s, ["70% of a slide is glass: a K=2 GMM on the thumbnail removes it (Dice 0.84)",
            "A GMM density on 23 physics features flags bad tiles: AUROC 0.94, 45 KB, explainable",
            "It catches artefact types it never saw (64–99% vs 7–64% for supervised)",
            "A pixel GMM in optical-density space separates nuclei, stroma, fluid, blood, fat",
            "Only good tissue reaches the U-Net – cheap decisions at the edge, expensive ones on the GPU"], size=20)

out = os.path.join(HERE, "Interview_Slides.pptx")
prs.save(out)
print("saved", out, len(prs.slides), "slides")
