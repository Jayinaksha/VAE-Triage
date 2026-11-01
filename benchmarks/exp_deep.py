"""Deep-learning baselines on exactly the same tiles as exp_classical.py.

 * VQ-VAE (the repository's model, bugs fixed) - reconstruction-MSE anomaly
   score. Trained on clean tiles of normal_108 + normal_150, threshold
   calibrated on the unseen slide tumor_082 (single leave-one-slide-out fold).
 * MobileNetV3-Small (ImageNet-pretrained, fine-tuned) - SUPERVISED upper
   bound for QC and for tumour detection.
 * ResNet-18 ImageNet features + GMM (a "Deep-IF / PaDiM-style" density on
   deep features) - unsupervised, heavier than hand-crafted features.
 * ONNX Runtime latency (FP32 and INT8 static quantisation) for edge claims.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from benchmarks.common import RESULTS, binary_report, save_json
from benchmarks.qc_sets import build
from models.vq_vae import VQVAE
from triage.gmm_triage import GMMTriage

torch.set_num_threads(4)
torch.manual_seed(0)
DEV = torch.device("cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_t(x, imagenet=False):
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN) / STD if imagenet else t * 2 - 1   # VQ-VAE uses [-1, 1]


def batched(fn, X, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(fn(X[i:i + bs]))
    return np.concatenate(out)


def aug(x):
    if np.random.rand() < 0.5:
        x = x.flip(3)
    if np.random.rand() < 0.5:
        x = x.flip(2)
    return x


# ---------------------------------------------------------------- VQ-VAE
def train_vqvae(X, epochs=12, bs=16, lr=1e-3):
    m = VQVAE().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(X) // bs))
    m.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(X))
        tot = []
        for i in range(0, len(X) - bs + 1, bs):
            x = aug(to_t(X[perm[i:i + bs]]))
            recon, vq_loss, ppl = m(x)
            loss = F.mse_loss(recon, x) + vq_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot.append(loss.item())
        print(f"  vqvae epoch {ep + 1}/{epochs} loss={np.mean(tot):.4f} ppl={ppl.item():.1f}", flush=True)
    m.eval()
    return m


def vq_score(m, X):
    def f(xb):
        x = to_t(xb)
        r, _, _ = m(x)
        return ((r - x) ** 2).mean(dim=(1, 2, 3)).numpy()
    return batched(f, X, 32)


# ---------------------------------------------------------------- MobileNet
def mobilenet(n_out=1):
    m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_out)
    return m


def train_cls(X, y, epochs=4, bs=32, lr=3e-4):
    m = mobilenet()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    pos_w = torch.tensor([(1 - y.mean()) / max(y.mean(), 1e-3)])
    m.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(X))
        tot = []
        for i in range(0, len(X) - bs + 1, bs):
            idx = perm[i:i + bs]
            x = aug(to_t(X[idx], True))
            logit = m(x)[:, 0]
            loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[idx]).float(), pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot.append(loss.item())
        print(f"  mobilenet epoch {ep + 1}/{epochs} loss={np.mean(tot):.4f}", flush=True)
    m.eval()
    return m


def cls_prob(m, X):
    return batched(lambda xb: torch.sigmoid(m(to_t(xb, True))[:, 0]).numpy(), X, 64)


def resnet_features(X):
    r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    r.fc = nn.Identity()
    r.eval()
    return batched(lambda xb: r(to_t(xb, True)).numpy(), X, 64)


def ort_latency(onnx_path, shape, n=30):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(*shape).astype(np.float32)
    name = s.get_inputs()[0].name
    for _ in range(3):
        s.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(n):
        s.run(None, {name: x})
    return 1000 * (time.perf_counter() - t) / n / shape[0]


def main():
    d = build()
    Xte = np.concatenate([d["te_clean"], d["te_bg"], d["te_art"]])
    kind = np.concatenate([np.array(["clean"] * len(d["te_clean"])), np.array(["background"] * len(d["te_bg"])),
                           d["te_art_kind"]])
    clean_lab = np.concatenate([d["te_clean_lab"], np.array([""] * (len(Xte) - len(d["te_clean"])))])
    y = (kind != "clean").astype(int)
    res1, res2, lat = [], [], {}

    def report1(name, s, p):
        r = binary_report(y, s, p, name)
        r["per_kind_recall"] = {k: float(p[kind == k].mean()) for k in sorted(set(kind)) if k != "clean"}
        r["reject_rate_clean_normal"] = float(p[(kind == "clean") & (clean_lab == "normal")].mean())
        r["reject_rate_clean_tumor"] = float(p[(kind == "clean") & (clean_lab == "tumor")].mean())
        res1.append(r)
        print(f"[QC] {name:45s} AUROC={r['auroc']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"spec={r['specificity']:.3f} tumour-rej={r['reject_rate_clean_tumor']:.3f}", flush=True)

    y2 = (d["te_clean_lab"] == "tumor").astype(int)
    sl = d["tr_clean_slide"]
    fit_m = np.isin(sl, ["normal_108", "normal_150"])
    cal_m = sl == "tumor_082"

    # ---------------- VQ-VAE
    t0 = time.perf_counter()
    vq = train_vqvae(d["tr_clean"][fit_m])
    train_min = (time.perf_counter() - t0) / 60
    s_cal = vq_score(vq, d["tr_clean"][cal_m])
    s_te = vq_score(vq, Xte)
    thr = np.percentile(s_cal, 95)
    report1("VQ-VAE recon-MSE (LOSO-calibrated threshold)", s_te, (s_te > thr).astype(int))
    report1("VQ-VAE recon-MSE (repo default threshold 0.05)", s_te, (s_te > 0.05).astype(int))
    s2 = s_te[: len(d["te_clean"])]
    res2.append(binary_report(y2, s2, None, "VQ-VAE recon-MSE novelty (trained on normal)"))
    torch.save(vq.state_dict(), "/home/user/data/vqvae_c16.pt")

    # ---------------- ResNet18 features + GMM (unsupervised)
    t0 = time.perf_counter()
    Ftr = resnet_features(d["tr_clean"])
    Fte = resnet_features(Xte)
    rn_ms = 1000 * (time.perf_counter() - t0) / (len(Ftr) + len(Fte))
    from sklearn.decomposition import PCA
    pca = PCA(32, random_state=0).fit(Ftr[fit_m])
    g = GMMTriage(k_range=(1, 2, 4)).fit(pca.transform(Ftr[fit_m]), pca.transform(Ftr[cal_m]))
    sg = g.score(pca.transform(Fte))
    report1(f"ResNet18-ImageNet feats + PCA32 + GMM(K={g.gmm_.n_components})", sg, (sg > g.threshold_).astype(int))
    res2.append(binary_report(y2, sg[: len(d["te_clean"])], None, "ResNet18 feats + GMM novelty (trained on normal)"))

    # ---------------- MobileNetV3 supervised QC (upper bound)
    Xs = np.concatenate([d["tr_clean"], d["tr_bg"], d["tr_art"]])
    ys = np.r_[np.zeros(len(d["tr_clean"])), np.ones(len(d["tr_bg"]) + len(d["tr_art"]))].astype(np.float32)
    mq = train_cls(Xs, ys)
    pq = cls_prob(mq, Xte)
    report1("MobileNetV3-Small supervised QC", pq, (pq > 0.5).astype(int))

    # ---------------- MobileNetV3 supervised tumour (labels from tumor_082)
    yt = (d["tr_clean_lab"] == "tumor").astype(np.float32)
    mt = train_cls(d["tr_clean"], yt)
    pt = cls_prob(mt, d["te_clean"])
    res2.append(binary_report(y2, pt, (pt > 0.5).astype(int), "MobileNetV3-Small supervised (1 tumour slide)"))
    for r in res2:
        print(f"[tumour] {r['method']:50s} AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f}", flush=True)

    # ---------------- latency: PyTorch & ONNX Runtime (FP32 / INT8)
    x1 = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        for name, mdl in [("vqvae", vq), ("mobilenet_v3_small", mq)]:
            xb = torch.randn(16, 3, 256, 256)
            mdl(xb)
            t = time.perf_counter()
            for _ in range(3):
                mdl(xb)
            lat[f"{name}_torch_fp32_ms_per_tile"] = 1000 * (time.perf_counter() - t) / 48
    os.makedirs("/home/user/data/onnx", exist_ok=True)
    vq_onnx = "/home/user/data/onnx/vqvae.onnx"
    torch.onnx.export(vq, x1, vq_onnx, input_names=["x"], output_names=["recon", "vq_loss", "ppl"],
                      dynamic_axes={"x": {0: "b"}, "recon": {0: "b"}}, opset_version=17, dynamo=False)
    mb_onnx = "/home/user/data/onnx/mobilenet.onnx"
    torch.onnx.export(mq, x1, mb_onnx, input_names=["x"], output_names=["logit"],
                      dynamic_axes={"x": {0: "b"}, "logit": {0: "b"}}, opset_version=17, dynamo=False)
    lat["vqvae_ort_fp32_ms_per_tile"] = ort_latency(vq_onnx, (16, 3, 256, 256))
    lat["mobilenet_ort_fp32_ms_per_tile"] = ort_latency(mb_onnx, (16, 3, 256, 256))
    try:
        from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process

        class Reader(CalibrationDataReader):
            def __init__(self, X, imagenet):
                self.it = iter([{"x": to_t(X[i:i + 8], imagenet).numpy()} for i in range(0, 64, 8)])

            def get_next(self):
                return next(self.it, None)
        for name, path, imn in [("vqvae", vq_onnx, False), ("mobilenet", mb_onnx, True)]:
            pre = path.replace(".onnx", "_pre.onnx")
            quant_pre_process(path, pre)
            q = path.replace(".onnx", "_int8.onnx")
            quantize_static(pre, q, Reader(d["tr_clean"], imn), weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
            lat[f"{name}_ort_int8_ms_per_tile"] = ort_latency(q, (16, 3, 256, 256))
            lat[f"{name}_onnx_fp32_MB"] = os.path.getsize(path) / 1e6
            lat[f"{name}_onnx_int8_MB"] = os.path.getsize(q) / 1e6
        # INT8 fidelity for the VQ-VAE score
        import onnxruntime as ort
        sq = ort.InferenceSession(vq_onnx.replace(".onnx", "_int8.onnx"), providers=["CPUExecutionProvider"])
        xs = to_t(Xte[:256]).numpy()
        rq = np.concatenate([sq.run(None, {"x": xs[i:i + 32]})[0] for i in range(0, 256, 32)])
        sq_score = ((rq - xs) ** 2).mean(axis=(1, 2, 3))
        lat["vqvae_int8_vs_fp32_score_spearman"] = float(
            np.corrcoef(np.argsort(np.argsort(sq_score)), np.argsort(np.argsort(s_te[:256])))[0, 1])
    except Exception as e:  # pragma: no cover
        lat["int8_error"] = repr(e)
    lat["resnet18_feature_ms_per_tile_torch"] = rn_ms
    lat["vqvae_train_minutes_cpu"] = train_min
    lat["vqvae_params"] = int(sum(p.numel() for p in vq.parameters()))
    lat["mobilenet_params"] = int(sum(p.numel() for p in mq.parameters()))
    print(lat)
    save_json(dict(qc=res1, tumor=res2, latency=lat), "exp_deep.json")


if __name__ == "__main__":
    main()
