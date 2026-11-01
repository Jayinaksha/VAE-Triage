"""Explainer videos (MP4, 1280x720) for both projects: figure / text scenes with
burned-in captions and an espeak-ng voice-over, assembled with ffmpeg.

    python report/make_videos.py            # -> report/videos/*.mp4
Re-run after new results: numbers are read from results/*.json.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
import wave

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
VT = os.path.dirname(HERE)
SAT = os.path.join(os.path.dirname(VT), "Satellite_based_tile_segmentation")
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "videos")
TMP = "/tmp/claude-0/video_build"
W, H = 1280, 720
GREEN, DARK, GREY, BG = (27, 127, 91), (30, 30, 30), (90, 90, 90), (250, 250, 248)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


CACHE = "/home/user/data/tts_cache"
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Charon")


def _gemini_models():
    import requests
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": GEMINI_KEY, "pageSize": 200}, timeout=30)
    r.raise_for_status()
    names = [m["name"].split("/")[-1] for m in r.json().get("models", []) if "tts" in m["name"].lower()]
    pref = [os.environ.get("GEMINI_TTS_MODEL", ""), "gemini-2.5-flash-preview-tts", "gemini-2.5-flash-tts", "gemini-2.5-pro-preview-tts"]
    return [m for m in pref if m in names] + [m for m in names if m not in pref]


def tts(text, wav):
    """Gemini TTS (24 kHz PCM) if GEMINI_API_KEY is set, else espeak-ng. Cached by text."""
    import hashlib
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(f"{GEMINI_VOICE}|{text}".encode()).hexdigest()[:16]
    cached = os.path.join(CACHE, key + ".wav")
    if GEMINI_KEY and not os.path.exists(cached):
        import base64
        import time as _t
        import requests
        prompt = "Read this aloud as a calm, clear, confident technical presenter, at a natural pace: " + text
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["AUDIO"],
                                     "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": GEMINI_VOICE}}}}}
        for model in tts.models:
            for attempt in range(6):
                r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                  params={"key": GEMINI_KEY}, json=body, timeout=120)
                if r.status_code == 429 or r.status_code >= 500:
                    _t.sleep(min(60, 8 * 2 ** attempt))
                    continue
                break
            if r.ok:
                pcm = base64.b64decode(r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
                with wave.open(cached, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(24000)
                    w.writeframes(pcm)
                break
            print("gemini tts failed", model, r.status_code, r.text[:200])
    if os.path.exists(cached):
        shutil.copy(cached, wav)
    else:
        subprocess.run(["espeak-ng", "-v", "en-us+m3", "-s", "150", "-p", "45", "-w", wav, text], check=True)


tts.models = []


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def gradient():
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)], fill=(int(12 + 20 * t), int(60 + 50 * t), int(48 + 30 * t)))
    return im


def frame(title, image=None, caption="", bullets=None, big=None, sub=None, progress=None):
    if big:
        im = gradient()
        d = ImageDraw.Draw(im)
        d.text((W // 2, 230), big, font=font(80, True), fill="white", anchor="mm")
        d.text((W // 2, 140), title.upper(), font=font(24, True), fill=(160, 230, 200), anchor="mm")
        for i, line in enumerate(textwrap.wrap(sub or "", 58)):
            d.text((W // 2, 340 + 44 * i), line, font=font(30), fill=(235, 245, 240), anchor="mm")
        if progress is not None:
            d.rectangle([0, H - 6, int(W * progress), H], fill=(120, 220, 170))
        return im
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 64], fill=GREEN)
    d.text((28, 14), title, font=font(30, True), fill="white")
    if progress is not None:
        d.rectangle([0, 64, int(W * progress), 68], fill=(120, 220, 170))
    top, bottom = 80, H - 110
    if bullets:
        y = top + 20
        for b in bullets:
            lines = textwrap.wrap(b, 70)
            for k, line in enumerate(lines):
                d.text((70 if k == 0 else 100, y), ("• " if k == 0 else "") + line, font=font(27), fill=DARK)
                y += 38
            y += 12
    if image:
        p = image if os.path.isabs(image) else os.path.join(FIG, image)
        pic = Image.open(p).convert("RGB")
        pic.thumbnail((W - 60, bottom - top - 10))
        im.paste(pic, ((W - pic.width) // 2, top + (bottom - top - pic.height) // 2))
    if caption:
        d.rectangle([0, H - 100, W, H], fill=(240, 244, 241))
        for i, line in enumerate(textwrap.wrap(caption, 95)[:3]):
            d.text((W // 2, H - 80 + 28 * i), line, font=font(22), fill=DARK, anchor="mm")
    return im


def build(name, scenes):
    if GEMINI_KEY and not tts.models:
        tts.models = _gemini_models()
        print("Gemini TTS models:", tts.models[:3])
    work = os.path.join(TMP, name)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    segs = []
    for i, sc in enumerate(scenes):
        png, wav, mp4 = f"{work}/{i:02d}.png", f"{work}/{i:02d}.wav", f"{work}/{i:02d}.mp4"
        frame(**sc["frame"], progress=(i + 1) / len(scenes)).save(png)
        tts(sc["say"], wav)
        with wave.open(wav) as w:
            dur = w.getnframes() / w.getframerate() + 0.8
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", png, "-i", wav, "-t", f"{dur:.2f}",
                        "-vf", (f"scale=2560:1440,zoompan=z='min(zoom+0.00035,1.05)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                                f":d={int(dur * 25) + 5}:s={W}x{H}:fps=25," if sc["frame"].get("image") else "")
                        + f"fade=t=in:st=0:d=0.4,fade=t=out:st={dur - 0.4:.2f}:d=0.4,format=yuv420p",
                        "-af", "apad", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-r", "25", "-c:a", "aac",
                        "-ar", "44100", "-shortest", mp4], check=True)
        segs.append(mp4)
    lst = f"{work}/list.txt"
    open(lst, "w").write("".join(f"file '{s}'\n" for s in segs))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    print("wrote", out)
    return out


def vae_triage():
    e1 = load(f"{VT}/results/exp1_qc_classical.json")
    final = [r for r in e1["results"] if "final" in r["method"]][0]
    S = []
    S.append(dict(frame=dict(title="VAE-Triage", big="Edge-first WSI triage",
                             sub="Gaussian mixture models for tissue detection, quality control and fluid separation in gigapixel H&E slides"),
                  say="This is VAE Triage: an edge first triage pipeline for gigapixel pathology slides, built on Gaussian mixture models."))
    S.append(dict(frame=dict(title="The problem", image="c16_tile_classes.png",
                             caption="A whole-slide image is ~10 billion pixels. Real CAMELYON16 tiles: normal, tumour, glass, partial tissue."),
                  say="A whole slide image of a breast lymph node has about ten billion pixels. Much of it is empty glass, and some tissue tiles are unusable. A heavy segmentation network should never see those tiles."))
    S.append(dict(frame=dict(title="How much can we skip?", big="69.6 %", sub="of 203,168 tiles in 7 real CAMELYON16 slides are empty glass (30-78 % per slide)"),
                  say="In our seven real slides, almost seventy percent of all tiles are empty glass. Removing them first saves three to four times the GPU compute."))
    S.append(dict(frame=dict(title="Pipeline", image="pipeline_pathology.png",
                             caption="Lazy WSI reader → thumbnail GMM tissue mask → 20x tile grid → GMM QC → GMM tissue components → heavy model"),
                  say="The pipeline reads the slide pyramid lazily, finds tissue on a thumbnail, grids the tissue at twenty x, scores every tile with a Gaussian mixture density, describes the tissue components, and only then calls the heavy model."))
    S.append(dict(frame=dict(title="Stage 1: tissue detection", image="tissue_detection_tumor_091.png",
                             caption="Pixel GMM (K=2) on saturation + optical density: Dice 0.84 vs Otsu 0.53 vs k-means 0.14"),
                  say="On the thumbnail, a two component mixture on saturation and optical density finds tissue with Dice zero point eight four. Otsu thresholding gets zero point five three and k means only zero point one four, because glass is a tight cluster and tissue a broad one."))
    S.append(dict(frame=dict(title="Artefacts the gate must catch", image="c16_artifacts.png",
                             caption="10 controlled artefacts injected into real tiles: Gaussian & motion blur, pen ink, folds, bubbles, exposure, noise, JPEG"),
                  say="We injected ten realistic artefacts into real tissue tiles: defocus and motion blur, pen ink, tissue folds, air bubbles, washout, dark exposure, sensor noise and JPEG compression."))
    S.append(dict(frame=dict(title="Stage 2: GMM quality control", image="chart_qc.png",
                             caption=f"23 physics features → GMM −log p(x). AUROC {final['auroc']:.3f}, F1 {final['f1']:.3f}, specificity {final['specificity']:.3f}. 45 KB model."),
                  say="Each tile becomes twenty three physical features: focus, stain, ink, folds and noise. A Gaussian mixture learned only from clean tissue gives an anomaly score. It reaches zero point nine four AUROC without any artefact labels, with a forty five kilobyte model."))
    S.append(dict(frame=dict(title="Per-artefact recall", image="chart_qc_per_artifact.png",
                             caption="Single cues catch one artefact family each; density models catch almost all."),
                  say="Single cues like Laplacian variance only catch blur. The density model catches nearly every artefact family."))
    S.append(dict(frame=dict(title="Why unsupervised?", bullets=[
        "Artefact types NEVER seen in training:",
        "Supervised random forest: bubble 7 %, fold 24 %, JPEG 22 %",
        "GMM (no artefact ever seen): bubble 64 %, fold 94 %, JPEG 99 %",
        "Supervised MobileNetV3 is best on KNOWN artefacts (AUROC 0.976)",
        "Design: GMM gate for the unknown + small supervised head for known"]),
                  say="Scanners produce failures nobody labelled. Supervised models trained without bubbles, folds or JPEG artefacts miss most of them. The Gaussian mixture, which never saw any artefact, catches sixty four to ninety nine percent."))
    S.append(dict(frame=dict(title="Two traps we found", bullets=[
        "Threshold calibrated on the SAME slides: 43 % of tumour tiles rejected",
        "Leave-one-slide-out calibration: 3.5 % tumour rejected, specificity 0.93",
        "VQ-VAE reconstruction error as QC score: AUROC 0.376 (worse than chance)",
        "Blurred / empty tiles are smooth, so they are EASIER to reconstruct"]),
                  say="Two traps. Calibrating the threshold on the same slides rejected forty three percent of tumour tiles; calibrating on unseen slides brings that to three and a half percent. And the original VQ VAE reconstruction error is worse than chance for quality control, because blurred and empty tiles are easier to reconstruct."))
    S.append(dict(frame=dict(title="Stage 3: tissue components & fluid", image="gmm_tissue_components.png",
                             caption="Beer–Lambert → optical density → H & E deconvolution → pixel GMM: nuclei · stroma · FLUID · blood · fat"),
                  say="Using the Beer Lambert law and colour deconvolution, a pixel level mixture separates nuclei, stroma, fluid, blood and fat. Fluid is the pale, eosin dominant, low density component. K means cannot isolate it. These five fractions alone separate tumour from normal tiles with AUROC zero point nine three."))
    S.append(dict(frame=dict(title="Tumour vs normal", image="chart_tumour.png",
                             caption="Unsupervised scores ≈ 0.67–0.75 (like autoencoders in the literature); component fractions 0.93; supervised RF 0.945."),
                  say="For tumour detection, unsupervised scores reach about zero point seven five, in line with autoencoders in the literature. Diagnosis needs semantics, so the heavy supervised model stays downstream."))
    S.append(dict(frame=dict(title="Edge cost", image="chart_edge_tradeoff.png",
                             caption="Features 2.1 ms + GMM 0.01 ms per tile on CPU · VQ-VAE 23 ms (ONNX) · INT8: 3.7× smaller, not faster on this CPU"),
                  say="On a four core CPU with no GPU, features take two milliseconds per tile and the mixture score takes ten microseconds. INT8 quantisation made the deep models smaller but not faster here, so always measure on the target device."))
    S.append(dict(frame=dict(title="Summary", bullets=[
        "70 % glass removed on the thumbnail (GMM, Dice 0.84)",
        "GMM QC: AUROC 0.94, 45 KB, explainable, catches unseen artefacts",
        "Pixel GMM separates nuclei, stroma, fluid, blood, fat",
        "Cheap decisions at the edge, expensive ones on the GPU"]),
                  say="In summary: remove glass on the thumbnail, gate tiles with an explainable mixture model at the edge, describe the tissue components including fluid, and spend GPU time only on good tissue."))
    return build("VAE-Triage_explainer", S)


def satellite():
    cs = load(f"{SAT}/results/classical_seg.json")
    un = load(f"{SAT}/results/unet_f16.json")
    yo = load(f"{SAT}/results/yolo_car.json")
    best = {r["method"]: 100 * r["mIoU_5_no_clutter"] for r in cs["results"]} if cs else {}
    S = []
    S.append(dict(frame=dict(title="High-resolution segmentation", big="6000 × 6000 px",
                             sub="Residual U-Net with a dilated bottleneck, patch stitching and test-time augmentation on ISPRS Potsdam"),
                  say="The second project segments six thousand by six thousand pixel aerial tiles with a U Net, using the same tiling logic as whole slide pathology."))
    S.append(dict(frame=dict(title="Why it matters for pathology", bullets=[
        "Image much larger than the network input → patches → stitch",
        "Class imbalance: cars ≈ 1 % of pixels ↔ small metastases",
        "Multi-scale objects: buildings vs cars ↔ glands vs nuclei",
        "Colour shift: season & sensor ↔ stain & scanner"]),
                  say="The engineering problems are the same as in pathology: images far larger than the network input, severe class imbalance, objects of very different sizes, and colour shift."))
    S.append(dict(frame=dict(title="Pipeline", image="pipeline_satellite.png",
                             caption="256 px patches → D4 + photometric augmentation → U-Net → overlap-averaged softmax stitching → 8× TTA"),
                  say="Patches of two hundred fifty six pixels are augmented with rotations, flips and brightness changes, segmented, and stitched back by averaging overlapping softmax maps, with eight fold test time augmentation."))
    S.append(dict(frame=dict(title="Architecture", image=os.path.join(SAT, "images", "model-arch.JPG"),
                             caption="Encoder–decoder with skip connections; dilated bottleneck (rates 1, 2, 4, 8) for multi-scale context"),
                  say="The encoder captures what is in the image, the decoder recovers where, and skip connections keep the boundaries sharp. Dilated convolutions in the bottleneck add multi scale context. Note the blocks concatenate rather than add, so they are dense rather than residual connections."))
    S.append(dict(frame=dict(title="Bugs found and fixed", bullets=[
        "Last 112 px of every 6000-px tile never predicted (3.7 % of pixels)",
        "Nested YAML config → num_classes silently 5 for a 6-class model",
        "'Loop-free scatter_nd' claim: stitching is actually a Python loop",
        "README IoUs not produced by any script → re-measured"]),
                  say="We found and fixed real bugs: the last hundred and twelve pixels of every tile were never predicted, and a nested configuration silently set the wrong number of classes."))
    if cs:
        S.append(dict(frame=dict(title="Classical baselines", image="potsdam_classical_examples.png",
                                 caption=f"k-means {best.get('K-means (k=6, unsupervised, Hungarian-mapped)', 0):.1f} · GMM-Bayes + smoothing {best.get('GMM-Bayes + posterior smoothing', 0):.1f} · RF + NIR {best.get('Random Forest (RGB + NIR/NDVI)', 0):.1f} mIoU"),
                      say="Classical pixel classifiers trained in minutes on a CPU reach about fifty percent mean IoU. A Gaussian mixture Bayes classifier with posterior smoothing matches a random forest, and near infrared adds physics."))
    msg = "Deep models: 86–88 mIoU in the literature (GPU, 100 epochs)."
    say = "Published deep models reach eighty six to eighty eight percent, because separating buildings from roads needs shape and context."
    if un and "no_tta" in un:
        u = un["no_tta"]
        msg = f"Repo U-Net, {un['minutes']:.0f} min CPU: {100 * u['mIoU_5_no_clutter']:.1f} mIoU (TTA {100 * un.get('tta8', u)['mIoU_5_no_clutter']:.1f}). Literature 86–88 with GPU training."
        say = (f"The repository's U Net trained for only {un['minutes']:.0f} minutes on a CPU reaches {100 * u['mIoU_5_no_clutter']:.0f} percent mean IoU. "
               "Published deep models trained for a hundred epochs on a GPU reach eighty six to eighty eight.")
    S.append(dict(frame=dict(title="Results vs literature", image="chart_potsdam.png", caption=msg), say=say))
    if yo:
        S.append(dict(frame=dict(title="YOLOv8-seg for cars", bullets=[
            f"Instances from ground truth → YOLOv8n-seg, {yo['train_minutes']:.0f} min CPU",
            f"Mask mAP50 {yo['mask_mAP50']:.2f} · car pixel IoU {100 * yo['car_pixel_IoU_test_crops']:.1f} %",
            "YOLO for countable objects (cars ↔ mitoses, nuclei)", "U-Net for every-pixel semantics"]),
                      say="YOLO handles countable objects such as cars, or mitoses and nuclei in pathology, while the U Net labels every pixel."))
    S.append(dict(frame=dict(title="Test-time augmentation", image=os.path.join(SAT, "images", "tile-pred-with-tsa.JPG"),
                             caption="8× D4 TTA: predict on 4 rotations × 2 flips, undo, average softmax (original author's figure)"),
                  say="Test time augmentation predicts on four rotations and two flips, undoes each, and averages the probabilities, usually adding a point or two of IoU."))
    S.append(dict(frame=dict(title="Summary", bullets=[
        "Same tiling / stitching / TTA logic as WSI segmentation",
        "Bugs fixed: border coverage, class-count config",
        "GMMs for colour & density problems; U-Net for semantics",
        "Next: pretrained encoder, Gaussian-blended overlap, Dice + CE loss"]),
                  say="In summary: the same tiling, stitching and augmentation logic transfers directly to pathology. Mixture models solve colour and density problems; the U Net solves semantics."))
    return build("Satellite_segmentation_explainer", S)


if __name__ == "__main__":
    vae_triage()
    satellite()
