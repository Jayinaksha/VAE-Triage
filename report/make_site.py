"""Static GitHub Pages site (docs/index.html) for both projects.

Writes the same page into VAE-Triage/docs and Satellite_based_tile_segmentation/docs,
copying the videos, figures, report PDF and slides into docs/assets.
GitHub: Settings -> Pages -> Deploy from branch -> <branch> /docs
"""
from __future__ import annotations

import html
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VT = os.path.dirname(HERE)
SAT = os.path.join(os.path.dirname(VT), "Satellite_based_tile_segmentation")
FIG = os.path.join(HERE, "figures")
GH = "https://github.com/Jayinaksha"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def table(header, rows, hl=None):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in header)
    body = ""
    for r in rows:
        cls = ' class="hl"' if hl and hl(r) else ""
        body += f"<tr{cls}>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def fig(name, cap):
    return f'<figure><img src="assets/{name}" alt="{html.escape(cap)}" loading="lazy"><figcaption>{html.escape(cap)}</figcaption></figure>'


def page():
    R = os.path.join(VT, "results")
    e1, ed, e3 = load(f"{R}/exp1_qc_classical.json"), load(f"{R}/exp_deep.json"), load(f"{R}/exp3_gmm_components.json")
    ab, e0 = load(f"{R}/exp4_ablations.json"), load(f"{R}/exp0_tissue_detection.json")
    cs, un, yo = load(f"{SAT}/results/classical_seg.json"), load(f"{SAT}/results/unet_f16.json"), load(f"{SAT}/results/yolo_car.json")

    qc = (e1["results"] if e1 else []) + (ed["qc"] if ed else [])
    keep = ["Laplacian", "HistoQC", "Isolation", "final", "Single Gaussian", "same-slide", "VQ-VAE recon-MSE (LOSO", "ResNet18", "Random forest", "MobileNet"]
    qrows = [[r["method"], f"{r['auroc']:.3f}", f"{r.get('f1', 0):.3f}", f"{r.get('specificity', 0):.3f}", f"{100 * r.get('reject_rate_clean_tumor', 0):.1f}%"]
             for r in qc if any(k in r["method"] for k in keep)]
    trows = [[k, f"{v['dice']:.3f}", f"{v['iou']:.3f}"] for k, v in (e0["summary"].items() if e0 else [])]
    urows = [[k] + [f"{100 * v[a]:.0f}%" for a in ["bubble", "pen_mark", "tissue_fold", "jpeg"]] for k, v in (ab["unseen_artifact_recall"].items() if ab else [])]
    prow = []
    for r in (cs["results"] if cs else []):
        prow.append([r["method"], f"{100 * r['mIoU_5_no_clutter']:.1f}", f"{100 * r['mF1_5_no_clutter']:.1f}", f"{100 * r['OA']:.1f}"])
    if un:
        for k in ["no_tta", "overlap50", "tta8"]:
            if k in un:
                r = un[k]
                prow.append([f"{r['method']} ({un['minutes']:.0f} min CPU)", f"{100 * r['mIoU_5_no_clutter']:.1f}", f"{100 * r['mF1_5_no_clutter']:.1f}", f"{100 * r['OA']:.1f}"])
    prow += [["UNetFormer R18 (literature, GPU)", "86.8", "92.8", "91.3"], ["ABCNet R18 (literature, GPU)", "86.5", "92.7", "91.3"]]
    final = next((r for r in qc if "final" in r["method"]), None)
    lat = ed["latency"] if ed else {}
    yolo_html = ""
    if yo:
        yolo_html = (f"<p><b>YOLOv8n-seg for cars</b> ({yo['train_minutes']:.0f} min CPU fine-tune): mask mAP50 {yo['mask_mAP50']:.2f}, "
                     f"car pixel IoU {100 * yo['car_pixel_IoU_test_crops']:.1f}% on the shared test crops.</p>")

    stats = [("69.6%", "of 203k slide tiles are empty glass"), ("0.84", "tissue Dice, GMM (Otsu 0.53)"),
             (f"{final['auroc']:.2f}" if final else "0.94", "QC AUROC, no labels"), ("45 KB", "edge QC model, NumPy only"),
             ("64–99%", "recall on never-seen artefacts"), (f"{e3['tumour_auc_fraction_vector_logreg']:.2f}" if e3 else "0.93", "tumour AUROC from GMM tissue fractions")]
    stat_html = "".join(f'<div class="stat"><div class="n">{n}</div><div class="l">{html.escape(l)}</div></div>' for n, l in stats)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge-First Pathology Triage & High-Res Segmentation</title>
<meta name="description" content="GMM-based edge triage for gigapixel H&E slides and high-resolution U-Net segmentation: pipelines, benchmarks, videos and report.">
<style>
:root{{--bg:#fafaf8;--fg:#1e1e1e;--mut:#5d6168;--card:#fff;--line:#e3e5e8;--acc:#1b7f5b;--acc2:#e7f5ee;--hl:#eef8f2}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121414;--fg:#e9ecea;--mut:#a6aca9;--card:#1b1e1e;--line:#2c3130;--acc:#4cc095;--acc2:#16302a;--hl:#18302a}}}}
*{{box-sizing:border-box}}body{{margin:0;font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}}
a{{color:var(--acc)}}header{{padding:56px 16px 36px;text-align:center;border-bottom:1px solid var(--line)}}
h1{{font-size:clamp(1.7rem,4vw,2.6rem);margin:0 0 10px;line-height:1.2}}h2{{font-size:1.6rem;margin:48px 0 8px}}h3{{margin:28px 0 6px}}
.sub{{color:var(--mut);max-width:760px;margin:0 auto 22px}}.btns a{{display:inline-block;margin:4px;padding:9px 16px;border-radius:8px;border:1px solid var(--acc);text-decoration:none;font-weight:600}}
.btns a.p{{background:var(--acc);color:#fff}}main{{max-width:1080px;margin:0 auto;padding:0 16px 80px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:28px 0}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}.stat .n{{font-size:1.7rem;font-weight:800;color:var(--acc)}}.stat .l{{color:var(--mut);font-size:.9rem}}
.vids{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}}video{{width:100%;border-radius:10px;border:1px solid var(--line);background:#000}}
figure{{margin:18px 0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}}figure img{{width:100%;height:auto;border-radius:6px;background:#fff}}
figcaption{{color:var(--mut);font-size:.9rem;margin-top:6px}}.tw{{overflow-x:auto;margin:14px 0}}table{{border-collapse:collapse;width:100%;font-size:.92rem;background:var(--card)}}
th,td{{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left}}th{{background:var(--acc2)}}tr.hl td{{background:var(--hl);font-weight:600}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}.note{{border-left:4px solid var(--acc);background:var(--card);padding:10px 14px;border-radius:6px}}
.warn{{border-left-color:#c0563b}}code{{background:var(--acc2);padding:1px 5px;border-radius:4px}}footer{{text-align:center;color:var(--mut);padding:30px 16px;border-top:1px solid var(--line)}}
nav{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;text-align:center;z-index:2;overflow-x:auto;white-space:nowrap}}nav a{{margin:0 10px;text-decoration:none;font-weight:600}}
</style></head><body>
<header><h1>Edge-First Image Analysis for Digital Pathology</h1>
<p class="sub">Two projects: <b>VAE-Triage</b> — Gaussian-mixture-model triage, quality control and fluid separation for gigapixel H&amp;E slides —
and <b>High-resolution segmentation</b> with a residual U-Net. Benchmarked on real CAMELYON16 breast lymph-node slides and ISPRS Potsdam, CPU only.</p>
<div class="btns"><a class="p" href="assets/Technical_Report.pdf">📑 Technical report (PDF)</a><a href="assets/Interview_Slides.pdf">🎤 Slides (PDF)</a>
<a href="assets/Interview_Slides.pptx">Slides (PPTX)</a><a href="{GH}/VAE-Triage">VAE-Triage repo</a><a href="{GH}/Satellite_based_tile_segmentation">Segmentation repo</a></div></header>
<nav><a href="#videos">Videos</a><a href="#triage">VAE-Triage</a><a href="#fluid">Fluid / components</a><a href="#edge">Edge</a><a href="#seg">Segmentation</a><a href="#fixes">Fixes</a><a href="#repro">Reproduce</a></nav>
<main>
<div class="stats">{stat_html}</div>

<h2 id="videos">🎬 Explainer videos</h2>
<div class="vids"><div><h3>VAE-Triage (3.6 min)</h3><video controls preload="metadata" poster="assets/poster_vae.jpg"><source src="assets/VAE-Triage_explainer.mp4" type="video/mp4"></video></div>
<div><h3>High-resolution segmentation</h3><video controls preload="metadata" poster="assets/poster_sat.jpg"><source src="assets/Satellite_segmentation_explainer.mp4" type="video/mp4"></video></div></div>

<h2 id="triage">🔬 VAE-Triage — edge-first WSI triage</h2>
<p>A whole-slide image is ~10<sup>10</sup> pixels and ~70% of it is glass. The gate reads the pyramid lazily, removes glass on a thumbnail, scores every
20× tile with a GMM density over 23 physical features (focus, stain, ink, folds, noise), separates tissue components, and sends only good tissue to the heavy model.</p>
{fig("pipeline_pathology.png", "Pipeline: lazy WSI reader → thumbnail GMM tissue mask → tile grid → GMM QC → GMM tissue components → heavy model")}
<h3>Stage 1 — tissue detection</h3>
<div class="grid2">{fig("tissue_detection_tumor_091.png", "Otsu misses faint tissue; the GMM fits the broad tissue cluster")}{fig("chart_tissue.png", "Dice vs CAMELYON16 tissue masks, 7 slides")}</div>
{table(["Method", "Dice", "IoU"], trows, hl=lambda r: "GMM" in r[0])}
<h3>Stage 2 — tile quality control</h3>
<p>Test: 2290 clean tiles vs 600 glass + 1500 artefact tiles on 4 unseen slides; thresholds calibrated on unseen training slides.</p>
{table(["Method", "AUROC", "F1", "Specificity", "Tumour rejected"], qrows, hl=lambda r: "final" in r[0])}
<div class="grid2">{fig("chart_qc.png", "QC methods: AUROC and F1")}{fig("c16_artifacts.png", "10 artefact simulators applied to real tiles")}</div>
{fig("chart_qc_per_artifact.png", "Recall per artefact at the operating point")}
<h3>Why unsupervised: artefacts never seen in training</h3>
{table(["Model", "bubble", "pen ink", "fold", "JPEG"], urows, hl=lambda r: "GMM" in r[0])}
<div class="note warn"><b>Two traps.</b> Calibrating the threshold on tiles of the <i>same</i> slides rejects 43% of tumour tiles on new slides (leave-one-slide-out: 3.5%).
The original VQ-VAE reconstruction error scores <b>0.376 AUROC</b> for QC — blurred and empty tiles are smooth, so they are <i>easier</i> to reconstruct.</div>

<h2 id="fluid">💧 GMM tissue components &amp; fluid separation</h2>
<p>Beer–Lambert optical density → colour deconvolution into haematoxylin and eosin → pixel GMM on [H, E, OD, a*, local std]. The pale, eosin-dominant, low-OD component is
<b>fluid</b> (proteinaceous fluid, plasma, oedema); k-means on the same features cannot isolate it. Two GMMs fitted on disjoint slides agree on
{100 * e3['stability_pixel_agreement']:.0f}% of pixels; the five component fractions alone separate tumour from normal tiles with AUROC {e3['tumour_auc_fraction_vector_logreg']:.2f}.</p>
{fig("gmm_tissue_components.png", "GMM components vs k-means: nuclei, stroma/cytoplasm, fluid, blood, background/fat")}
{fig("chart_tumour.png", "Tumour vs normal tiles: unsupervised, supervised and literature (orange)")}

<h2 id="edge">⚡ Edge deployment</h2>
{table(["Component", "ms / 256² tile (4-vCPU, no GPU)", "Size"], [["23 features at 128 px", "2.1", "–"], ["GMM scorer (NumPy)", "0.009", "45 KB JSON"],
       ["VQ-VAE ONNX Runtime FP32 / INT8", f"{lat.get('vqvae_ort_fp32_ms_per_tile', 0):.1f} / {lat.get('vqvae_ort_int8_ms_per_tile', 0):.1f}", f"{lat.get('vqvae_onnx_fp32_MB', 0):.1f} / {lat.get('vqvae_onnx_int8_MB', 0):.1f} MB"],
       ["MobileNetV3-S ONNX Runtime FP32 / INT8", f"{lat.get('mobilenet_ort_fp32_ms_per_tile', 0):.1f} / {lat.get('mobilenet_ort_int8_ms_per_tile', 0):.1f}", f"{lat.get('mobilenet_onnx_fp32_MB', 0):.1f} / {lat.get('mobilenet_onnx_int8_MB', 0):.1f} MB"]])}
<div class="grid2">{fig("chart_edge_tradeoff.png", "Accuracy vs CPU cost")}{fig("chart_slide_js.png", "Stain/scanner shift between slides (Jensen–Shannon divergence)")}</div>
<p class="note">Edge = cheap, label-free, high-recall decisions next to the scanner; expensive, label-hungry decisions on the GPU. INT8 made models 3.5–3.7× smaller but not faster on this CPU — always measure on the target device.</p>

<h2 id="seg">🛰️ High-resolution segmentation (ISPRS Potsdam)</h2>
<p>6000×6000 orthophotos → 256 px patches → D4 + photometric augmentation → U-Net with dilated bottleneck → overlap-averaged softmax stitching → 8× TTA.
The same tiling/stitching logic is used for whole-slide segmentation.</p>
{fig("pipeline_satellite.png", "Segmentation pipeline")}
{table(["Method", "mIoU (5 cls)", "mF1", "OA"], prow, hl=lambda r: "U-Net" in r[0])}
{yolo_html}
<div class="grid2">{fig("chart_potsdam.png", "Potsdam mIoU: ours vs literature")}{fig("potsdam_classical_examples.png", "Test crop: RGB, ground truth, k-means, GMM-Bayes, random forest + NIR")}</div>

<h2 id="fixes">🐛 Fixes made to the original code</h2>
<div class="grid2"><div class="note"><b>VAE-Triage</b><ul><li>Training/triage crashed: model outputs unpacked in the wrong order</li>
<li><code>tifffile.memmap</code> cannot open real compressed WSIs → lazy zarr reader</li><li>Documented background filtering was not implemented</li>
<li>No GMM in the code → GMM QC, tissue mask and components added</li><li>Arbitrary MSE threshold → calibrated on unseen slides</li></ul></div>
<div class="note"><b>Segmentation</b><ul><li>Last 112 px of every 6000-px tile never predicted (3.7% of pixels)</li><li>Nested config → <code>num_classes</code> silently 5</li>
<li>"Residual" blocks are concatenations; stitching is a loop, not <code>scatter_nd</code></li><li>README IoUs not produced by any script → re-measured</li></ul></div></div>

<h2 id="repro">🔁 Reproduce</h2>
<p>CAMELYON16 is free on AWS Open Data; Potsdam via the Hugging Face mirror <code>aamante/potsdam</code>. Every experiment is a script in <code>benchmarks/</code>;
raw numbers are in <code>results/*.json</code>; the report, slides, videos and this page are regenerated by <code>report/make_*.py</code>. See chapter 7 of the
<a href="assets/Technical_Report.pdf">technical report</a>.</p>
</main>
<footer>© 2026 Jayinaksha · Data: CAMELYON16 (Radboud UMC, UMC Utrecht), ISPRS Potsdam</footer>
</body></html>"""


def main():
    vids = os.path.join(HERE, "videos")
    for repo in (VT, SAT):
        docs = os.path.join(repo, "docs")
        assets = os.path.join(docs, "assets")
        os.makedirs(assets, exist_ok=True)
        for f in os.listdir(FIG):
            shutil.copy(os.path.join(FIG, f), assets)
        for f in ["Technical_Report.pdf", "Interview_Slides.pdf", "Interview_Slides.pptx"]:
            shutil.copy(os.path.join(HERE, f), assets)
        for v, poster in [("VAE-Triage_explainer.mp4", "poster_vae.jpg"), ("Satellite_segmentation_explainer.mp4", "poster_sat.jpg")]:
            shutil.copy(os.path.join(vids, v), assets)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "2", "-i", os.path.join(vids, v), "-frames:v", "1",
                            "-q:v", "4", os.path.join(assets, poster)], check=True)
        open(os.path.join(docs, "index.html"), "w").write(page())
        open(os.path.join(docs, ".nojekyll"), "w").write("")
        print("site ->", docs)


if __name__ == "__main__":
    main()
