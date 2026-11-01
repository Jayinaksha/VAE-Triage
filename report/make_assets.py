"""Turn every results/*.json into report charts + LaTeX/Markdown tables.

Run after the benchmarks:  python report/make_assets.py
Outputs: report/figures/*.png, report/generated/*.tex, report/generated/*.md
"""
from __future__ import annotations

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VT = os.path.dirname(HERE)
SAT = os.environ.get("SAT_REPO", os.path.join(os.path.dirname(VT), "Satellite_based_tile_segmentation"))
FIG = os.path.join(HERE, "figures")
GEN = os.path.join(HERE, "generated")
os.makedirs(FIG, exist_ok=True)
os.makedirs(GEN, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C_OURS, C_BASE, C_SUP, C_LIT, C_DEEP = "#1b7f5b", "#8a8f98", "#3b6fb6", "#d08a2c", "#9b4fb0"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def tex_escape(s):
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
            .replace("[", "{[}").replace("]", "{]}"))


def write_table(name, header, rows, caption, label, colspec=None):
    colspec = colspec or "l" + "r" * (len(header) - 1)
    tex = [r"\begin{table}[H]\centering\small", r"\caption{%s}\label{%s}" % (caption, label),
           r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{%s}\toprule" % colspec, " & ".join(r"\textbf{%s}" % tex_escape(h) for h in header) + r" \\\midrule"]
    for r in rows:
        if r == "---":
            tex.append(r"\midrule")
            continue
        tex.append(" & ".join(tex_escape(str(c)) for c in r) + r" \\")
    tex += [r"\bottomrule\end{tabular}\end{adjustbox}\end{table}"]
    open(f"{GEN}/{name}.tex", "w").write("\n".join(tex))
    md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows if r != "---"]
    open(f"{GEN}/{name}.md", "w").write(f"**{caption}**\n\n" + "\n".join(md) + "\n")


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def main():
    R = os.path.join(VT, "results")
    e0 = load(f"{R}/exp0_tissue_detection.json")
    e1 = load(f"{R}/exp1_qc_classical.json")
    e2 = load(f"{R}/exp2_tumor_classical.json")
    e3 = load(f"{R}/exp3_gmm_components.json")
    ed = load(f"{R}/exp_deep.json")
    st = load(f"{R}/stats.json")
    grid = load("/home/user/data/c16_tiles/grid_stats.json") or load(f"{R}/grid_stats.json")
    cs = load(f"{SAT}/results/classical_seg.json")
    un = load(f"{SAT}/results/unet_f16.json")
    yo = load(f"{SAT}/results/yolo_car.json")

    # ------------------------------------------------ Exp-0 tissue detection
    if e0:
        rows = [[k, f3(v["dice"]), f3(v["iou"]), f3(v["precision"]), f3(v["recall"])] for k, v in e0["summary"].items()]
        write_table("tab_tissue", ["Method (thumbnail, ~4-8 um/px)", "Dice", "IoU", "Precision", "Recall"], rows,
                    "Stage-0 tissue detection on 7 CAMELYON16 slides (mean over slides).", "tab:tissue")
        fig, ax = plt.subplots(figsize=(7, 3))
        names = list(e0["summary"])
        vals = [e0["summary"][k]["dice"] for k in names]
        ax.barh(range(len(names)), vals, color=[C_OURS if "ours" in n else C_BASE for n in names])
        ax.set_yticks(range(len(names)), [n.replace(" [ours]", "") for n in names])
        ax.set_xlabel("Dice vs. CAMELYON16 tissue mask")
        ax.set_xlim(0, 1)
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{FIG}/chart_tissue.png", dpi=150)
        plt.close()

    # ------------------------------------------------ Exp-1 QC
    qc = []
    if e1:
        qc += e1["results"]
    if ed:
        qc += ed["qc"]
    if qc:
        rows = []
        for r in qc:
            rows.append([r["method"], f3(r["auroc"]), f3(r["auprc"]), f3(r.get("precision")), f3(r.get("recall")),
                         f3(r.get("f1")), f3(r.get("specificity")), f3(r.get("reject_rate_clean_tumor"))])
        write_table("tab_qc", ["Method", "AUROC", "AUPRC", "Prec.", "Recall", "F1", "Spec.", "Tum. rej."], rows,
                    "Stage-1 tile QC on 4 unseen test slides: 2290 clean tissue tiles vs. 600 glass + 1500 artefact tiles. "
                    "Unsupervised thresholds calibrated on unseen training slides (95\\% clean pass).", "tab:qc",
                    colspec=">{\\raggedright\\arraybackslash}p{6.8cm}rrrrrrr")
        fig, ax = plt.subplots(figsize=(8, 5.2))
        names = [r["method"] for r in qc]
        y = np.arange(len(names))
        col = [C_SUP if "supervised" in n.lower() else C_OURS if "ours" in n else C_DEEP if ("VQ" in n or "ResNet" in n) else C_BASE for n in names]
        ax.barh(y - 0.2, [r["auroc"] for r in qc], 0.4, color=col, label="AUROC")
        ax.barh(y + 0.2, [r.get("f1") or 0 for r in qc], 0.4, color=col, alpha=0.45, label="F1 @ operating point")
        ax.set_yticks(y, [n[:52] for n in names], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0.3, 1.0)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Tile QC: AUROC (solid) and F1 (light). green=ours, grey=classical, blue=supervised, purple=deep unsup.", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{FIG}/chart_qc.png", dpi=150)
        plt.close()
        # per-artefact recall heatmap
        sel = [r for r in qc if "per_kind_recall" in r]
        kinds = sorted(sel[0]["per_kind_recall"])
        M = np.array([[r["per_kind_recall"][k] for k in kinds] for r in sel])
        fig, ax = plt.subplots(figsize=(10, 0.36 * len(sel) + 1.6))
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(kinds)), kinds, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(sel)), [r["method"][:50] for r in sel], fontsize=7.5)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="w" if M[i, j] < .6 else "k")
        plt.colorbar(im, fraction=0.03, label="recall (fraction rejected)")
        plt.tight_layout()
        plt.savefig(f"{FIG}/chart_qc_per_artifact.png", dpi=150)
        plt.close()

    # ------------------------------------------------ Exp-2 tumour + literature
    tum = list(e2 or []) + (ed["tumor"] if ed else [])
    lit = [("Lit: L1 autoencoder, unsup. (Shvetsova'21)", 0.211), ("Lit: AE (cross-organ, AnoPILaD'25)", 0.658),
           ("Lit: AnoDDPM diffusion (AnoPILaD'25)", 0.686), ("Lit: Deep-IF ImageNet feats (Shvetsova'21)", 0.906),
           ("Lit: Deep perceptual AE (Shvetsova'21)", 0.934)]
    if tum or e3:
        rows = [[r["method"], f3(r["auroc"]), f3(r["auprc"]), "unsup." if ("novelty" in r["method"] or "Isolation" in r["method"] or "single" in r["method"]) else "sup."] for r in tum]
        if e3:
            rows.append(["GMM tissue-component fractions + LR [ours]", f3(e3["tumour_auc_fraction_vector_logreg"]), "--", "unsup. feats + sup. LR"])
        rows.append("---")
        rows += [[n, f3(v), "--", "literature"] for n, v in lit]
        write_table("tab_tumour", ["Method", "AUROC", "AUPRC", "Type"], rows,
                    "Tumour (metastasis) vs. normal tiles on test slides (490 tumour / 1800 normal). Literature rows use "
                    "different splits and are indicative only.", "tab:tumour", colspec="p{8.2cm}rrl")
        items = [(r["method"], r["auroc"], C_SUP if "supervised" in r["method"].lower() else C_OURS if "GMM" in r["method"] else C_DEEP if "VQ" in r["method"] or "ResNet" in r["method"] else C_BASE) for r in tum]
        if e3:
            items.append(("GMM component fractions + LR [ours]", e3["tumour_auc_fraction_vector_logreg"], C_OURS))
        items += [(n, v, C_LIT) for n, v in lit]
        fig, ax = plt.subplots(figsize=(8, 0.33 * len(items) + 1))
        ax.barh(range(len(items)), [v for _, v, _ in items], color=[c for *_, c in items])
        ax.set_yticks(range(len(items)), [n[:55] for n, *_ in items], fontsize=7.5)
        for i, (_, v, _) in enumerate(items):
            ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=7)
        ax.axvline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        ax.set_xlabel("AUROC (tumour vs normal tile)")
        plt.tight_layout()
        plt.savefig(f"{FIG}/chart_tumour.png", dpi=150)
        plt.close()

    # ------------------------------------------------ Exp-3 components
    if e3:
        rows = [[k, f"{v['background/fat']:.3f}", f"{v['fluid']:.3f}", f"{v['stroma/cytoplasm']:.3f}", f"{v['nuclei']:.3f}"]
                for k, v in e3["mean_fractions"].items()]
        write_table("tab_components", ["Tile class", "Background/fat", "Fluid", "Stroma/cytoplasm", "Nuclei"], rows,
                    "Mean GMM component fractions per tile (test slides).", "tab:components")

    # ------------------------------------------------ edge / latency
    lat_rows = []
    if e1:
        gm = [r for r in e1["results"] if "final" in r["method"]]
        lat_rows.append(["Hand-crafted features (23-D, NumPy/OpenCV)", "5.0", "0 (no weights)", "CPU, 1 core"])
        if gm:
            lat_rows.append(["GMM scorer (NumPy, JSON params)", f"{gm[0]['latency_ms_per_tile_model_only']:.4f}",
                             f"{e1.get('gmm_json_bytes', 0) / 1024:.0f} KB JSON ({e1.get('gmm_param_count')} params)", "CPU"])
    if ed:
        L = ed["latency"]
        for key, name in [("vqvae_torch_fp32_ms_per_tile", "VQ-VAE PyTorch FP32"), ("vqvae_ort_fp32_ms_per_tile", "VQ-VAE ONNX Runtime FP32"),
                          ("vqvae_ort_int8_ms_per_tile", "VQ-VAE ONNX Runtime INT8"),
                          ("mobilenet_torch_fp32_ms_per_tile", "MobileNetV3-S PyTorch FP32"),
                          ("mobilenet_ort_fp32_ms_per_tile", "MobileNetV3-S ONNX Runtime FP32"),
                          ("mobilenet_ort_int8_ms_per_tile", "MobileNetV3-S ONNX Runtime INT8"),
                          ("resnet18_feature_ms_per_tile_torch", "ResNet-18 features PyTorch FP32")]:
            if key in L:
                size = ""
                if "vqvae" in key:
                    size = f"{L.get('vqvae_onnx_int8_MB' if 'int8' in key else 'vqvae_onnx_fp32_MB', 0):.1f} MB" if "ort" in key else f"{L.get('vqvae_params', 0) / 1e6:.2f} M params"
                if "mobilenet" in key:
                    size = f"{L.get('mobilenet_onnx_int8_MB' if 'int8' in key else 'mobilenet_onnx_fp32_MB', 0):.1f} MB" if "ort" in key else f"{L.get('mobilenet_params', 0) / 1e6:.2f} M params"
                lat_rows.append([name, f"{L[key]:.2f}", size, "CPU, 4 threads"])
    if lat_rows:
        write_table("tab_latency", ["Component", "ms / 256px tile", "Model size", "Hardware"], lat_rows,
                    "Measured inference cost on this 4-vCPU cloud container (no GPU).", "tab:latency", colspec="p{6cm}rp{4cm}l")

    # ------------------------------------------------ slide census -> compute saving
    if grid:
        rows, tot, bg = [], 0, 0
        for s, v in grid.items():
            rows.append([s, v["grid_tiles"], v["tissue_tiles"], f"{100 * v['background_tiles'] / v['grid_tiles']:.1f}%", v["tumor_tiles"]])
            tot += v["grid_tiles"]
            bg += v["background_tiles"]
        rows.append(["TOTAL", tot, tot - bg, f"{100 * bg / tot:.1f}%", sum(v["tumor_tiles"] for v in grid.values())])
        write_table("tab_census", ["Slide", "Grid tiles (256 px @20x)", "Tissue tiles", "Glass (skipped)", "Tumour tiles"], rows,
                    "Whole-slide tile census: what Stage 0 removes before any neural network runs.", "tab:census")
        json.dump(grid, open(os.path.join(R, "grid_stats.json"), "w"), indent=2)

    # ------------------------------------------------ statistics
    if st:
        top = st["top_js_features_per_artifact"]
        rows = [[k, ", ".join(f"{f} ({v:.2f})" for f, v in vals)] for k, vals in top.items()]
        write_table("tab_js", ["Artefact", "Top-3 features by JS divergence (clean vs artefact)"], rows,
                    "Which features move most under each artefact (Jensen-Shannon divergence, base 2, in [0,1]).", "tab:js",
                    colspec="lp{11cm}")
        sl = st["slide_color_js"]
        M = np.array(sl["matrix"])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(M, cmap="magma_r")
        ax.set_xticks(range(len(sl["slides"])), sl["slides"], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(sl["slides"])), sl["slides"], fontsize=8)
        for i in range(len(M)):
            for j in range(len(M)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] > M.max() * .6 else "k")
        plt.colorbar(im, fraction=0.046, label="JS divergence of RGB histograms")
        ax.set_title("Stain / scanner shift between slides", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{FIG}/chart_slide_js.png", dpi=150)
        plt.close()

    # ------------------------------------------------ Potsdam
    lit_p = [("Lit: ERFNet (UNetFormer T9)", 76.2), ("Lit: BiSeNet R18", 81.7), ("Lit: DANet R18", 80.3),
             ("Lit: ABCNet R18", 86.5), ("Lit: SwinUperNet Swin-T", 85.8), ("Lit: UNetFormer R18", 86.8),
             ("Lit: FT-UNetFormer Swin-B", 87.5)]
    rows, items = [], []
    for r in (cs["results"] if cs else []):
        rows.append([r["method"], f"{100 * r['mIoU_5_no_clutter']:.1f}", f"{100 * r['mF1_5_no_clutter']:.1f}", f"{100 * r['OA']:.1f}"]
                    + [f"{100 * v:.1f}" for v in list(r["per_class_iou"].values())[:5]])
        items.append((r["method"], 100 * r["mIoU_5_no_clutter"], C_OURS if "GMM" in r["method"] else C_BASE))
    if un:
        for k in ["no_tta", "overlap50", "tta8"]:
            if k in un:
                r = un[k]
                rows.append([r["method"] + f" ({un['minutes']:.0f} min CPU)", f"{100 * r['mIoU_5_no_clutter']:.1f}", f"{100 * r['mF1_5_no_clutter']:.1f}",
                             f"{100 * r['OA']:.1f}"] + [f"{100 * v:.1f}" for v in list(r["per_class_iou"].values())[:5]])
                items.append((r["method"], 100 * r["mIoU_5_no_clutter"], C_DEEP))
    if rows:
        rows.append("---")
        rows += [[n, f"{v:.1f}", "--", "--", "--", "--", "--", "--", "--"] for n, v in lit_p]
        write_table("tab_potsdam", ["Method", "mIoU5", "mF1-5", "OA", "Imp.", "Bldg", "LowVeg", "Tree", "Car"], rows,
                    "ISPRS Potsdam (eroded boundaries, clutter excluded from the mean). Ours: 12 fixed 1024x1024 crops of "
                    "test tiles 2\\_13, 3\\_14, 5\\_15. Literature: full 14-tile test set, GPU training for 100+ epochs.",
                    "tab:potsdam", colspec="p{5.6cm}rrrrrrrr")
        items += [(n, v, C_LIT) for n, v in lit_p]
        fig, ax = plt.subplots(figsize=(8, 0.33 * len(items) + 1))
        ax.barh(range(len(items)), [v for _, v, _ in items], color=[c for *_, c in items])
        ax.set_yticks(range(len(items)), [n[:50] for n, *_ in items], fontsize=7.5)
        for i, (_, v, _) in enumerate(items):
            ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=7)
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.set_xlabel("mIoU (%) on Potsdam, 5 classes")
        plt.tight_layout()
        plt.savefig(f"{FIG}/chart_potsdam.png", dpi=150)
        plt.close()
    if yo:
        write_table("tab_yolo", ["Metric", "Value"], [[k, v if isinstance(v, str) else f"{v:.3f}"] for k, v in yo.items() if not isinstance(v, (dict, list))],
                    "YOLOv8n-seg car instance segmentation (time-boxed CPU fine-tune).", "tab:yolo")

    # ------------------------------------------------ ablations
    ab = load(f"{R}/exp4_ablations.json")
    if ab:
        rows = []
        for k, v in ab["unseen_artifact_recall"].items():
            rows.append([k] + [f"{v[a]:.2f}" for a in ["bubble", "pen_mark", "tissue_fold", "jpeg"]] + [f"{v['clean_false_reject']:.3f}"])
        write_table("tab_unseen", ["Model", "bubble", "pen ink", "fold", "JPEG", "clean rejected"], rows,
                    "Recall on artefact types NEVER seen in training (supervised models saw the other 6 types).",
                    "tab:unseen", colspec="p{7cm}rrrrr")
        rows = [["Full-res features (256 px), GMM K=6", "5.0", "0.939"]]
        for k, v in ab.items():
            if k.startswith("downsampled"):
                rows.append([f"Features on 128 px (2x down), GMM K={k[-1]}", f"{v['feature_ms_per_tile']:.1f}", f"{v['auroc']:.3f}"])
        write_table("tab_downsample", ["Variant", "feature ms/tile", "QC AUROC"], rows,
                    "Edge optimisation: features on a 2x-downsampled tile.", "tab:downsample", colspec="p{8cm}rr")

    # ------------------------------------------------ edge trade-off scatter
    pts = []
    ab = load(f"{R}/exp4_ablations.json")
    if e1:
        for r in e1["results"]:
            if "final" in r["method"]:
                pts.append(("GMM, 256px features (ours)", 5.0 + (r["latency_ms_per_tile_model_only"] or 0), r["auroc"], C_OURS))
                if ab and "downsampled128_K6" in ab:
                    pts.append(("GMM, 128px features (ours)", ab["downsampled128_K6"]["feature_ms_per_tile"], ab["downsampled128_K6"]["auroc"], C_OURS))
            if r["method"].startswith("Random forest"):
                pts.append(("RF features (sup.)", 5.3, r["auroc"], C_SUP))
            if r["method"].startswith("Laplacian"):
                pts.append(("Laplacian variance", 0.4, r["auroc"], C_BASE))
    if ed:
        L = ed["latency"]
        for r in ed["qc"]:
            if r["method"].startswith("VQ-VAE") and "LOSO" in r["method"]:
                pts.append(("VQ-VAE (ORT FP32)", L.get("vqvae_ort_fp32_ms_per_tile", np.nan), r["auroc"], C_DEEP))
            if r["method"].startswith("MobileNet"):
                pts.append(("MobileNetV3 (ORT FP32, sup.)", L.get("mobilenet_ort_fp32_ms_per_tile", np.nan), r["auroc"], C_SUP))
            if r["method"].startswith("ResNet18"):
                pts.append(("ResNet18+GMM", L.get("resnet18_feature_ms_per_tile_torch", np.nan), r["auroc"], C_DEEP))
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4))
        for n, x, yv, c in pts:
            ax.scatter(x, yv, s=60, color=c)
            ax.annotate(n, (x, yv), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("CPU latency per 256x256 tile (ms, log scale)")
        ax.set_ylabel("QC AUROC")
        ax.set_title("Edge trade-off: accuracy vs. cost", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{FIG}/chart_edge_tradeoff.png", dpi=150)
        plt.close()
    print("assets written:", sorted(os.listdir(GEN)))


if __name__ == "__main__":
    main()


def update_sat_readme():
    """Fill the results table between markers in the satellite README."""
    cs = load(f"{SAT}/results/classical_seg.json")
    un = load(f"{SAT}/results/unet_f16.json")
    yo = load(f"{SAT}/results/yolo_car.json")
    rows = ["| Method | mIoU (5 cls) | mF1 | OA | Imp. | Bldg | LowVeg | Tree | Car |", "|---|---|---|---|---|---|---|---|---|"]
    def row(r, name=None):
        iou = list(r["per_class_iou"].values())
        return (f"| {name or r['method']} | **{100 * r['mIoU_5_no_clutter']:.1f}** | {100 * r['mF1_5_no_clutter']:.1f} | "
                f"{100 * r['OA']:.1f} | " + " | ".join(f"{100 * v:.1f}" for v in iou[:5]) + " |")
    for r in (cs["results"] if cs else []):
        rows.append(row(r))
    if un:
        for k in ["no_tta", "overlap50", "tta8"]:
            if k in un:
                rows.append(row(un[k], f"{un[k]['method']} — {un['minutes']:.0f} min CPU, {un['iterations']} it"))
    if yo:
        rows.append(f"| YOLOv8n-seg (cars only, {yo['train_minutes']:.0f} min CPU) | – | – | – | – | – | – | – | "
                    f"{100 * yo['car_pixel_IoU_test_crops']:.1f} (mask mAP50 {yo['mask_mAP50']:.2f}) |")
    if not (cs or un):
        return
    p = f"{SAT}/README.md"
    s = open(p).read()
    a, b = s.find("<!-- RESULTS_TABLE_START -->"), s.find("<!-- RESULTS_TABLE_END -->")
    if a >= 0 and b > a:
        s = s[:a] + "<!-- RESULTS_TABLE_START -->\n" + "\n".join(rows) + "\n" + s[b:]
        open(p, "w").write(s)


if __name__ == "__main__":
    update_sat_readme()
