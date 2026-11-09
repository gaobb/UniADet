"""UniADet demo: zero-shot and few-shot anomaly detection.

    pip install gradio
    bash scripts/demo.sh
"""

import os
import sys
import threading
from pathlib import Path

import albumentations as A
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import gaussian_blur

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "third_party"),
    str(ROOT / "third_party" / "perception_models"),
]


def _load_local_env():
    local = ROOT / "scripts" / "env.local.sh"
    if not local.is_file():
        return
    for line in local.read_text().splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_local_env()
os.environ.setdefault("UNIADET_CKPT", str(ROOT / "checkpoints"))
os.environ.setdefault("UNIADET_PRETRAINED", str(ROOT / "pretrained"))
os.environ.setdefault("UNIADET_DEVICE", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ["UNIADET_DEVICE"])

from models.backbone import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, build_backbone
from models.msdcs import MSDCS, multi_scale_nearest_neighbor
from utils.visualization import apply_ad_scoremap

FEATURES = [12, 15, 18, 21, 24]
WINDOW = [1]
SIGMA = 4
LAMDA_FS = 0.8
BACKBONES = {
    "CLIP ViT-L/14@336px": "ViT-L/14@336px",
    "DINOv2-Reg ViT-L-14": "ViT-L-14-DINOv2Reg",
    "DINOv3 ViT-L-16": "ViT-L-16-DINOv3",
}

_LOCK = threading.Lock()
_CACHE = {"key": None, "model": None, "msdcs": None, "meta": None}


def image_size_of(name):
    return 512 if "DINOv3" in name else 518


def default_checkpoint(name, train_on="visa"):
    root = Path(os.environ["UNIADET_CKPT"])
    size = image_size_of(name)
    # CLIP's model id contains '/', which is stored as '-' in checkpoint folders.
    tags = [name.replace("/", "-"), name]
    for tag in dict.fromkeys(tags):
        path = root / f"train_on_{train_on}_zs_{tag}_imagesize{size}_1015" / "epoch_15.pth"
        if path.is_file():
            return str(path)
    return str(root / f"train_on_{train_on}_zs_{tags[0]}_imagesize{size}_1015" / "epoch_15.pth")


def _mean_std(name):
    if name in ("ViT-L-16-DINOv3", "ViT-L-14-DINOv2Reg"):
        return IMAGENET_MEAN, IMAGENET_STD
    return CLIP_MEAN, CLIP_STD


def _preprocess(image, size, mean, std):
    rgb = np.array(image.convert("RGB"))
    resized = A.Resize(height=size, width=size)(image=rgb)["image"]
    tensor = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])(image=resized)["image"]
    return tensor, resized


def _load(backbone, checkpoint):
    checkpoint = str(Path(checkpoint).expanduser())
    key = (backbone, checkpoint)
    if _CACHE["key"] == key:
        return _CACHE["model"], _CACHE["msdcs"], _CACHE["meta"]
    if not Path(checkpoint).is_file():
        raise gr.Error(f"Checkpoint not found:\n{checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, embed_dim, patch_size = build_backbone(backbone, device)
    size = image_size_of(backbone)
    msdcs = MSDCS(size, patch_size, embed_dim=embed_dim, layers_num=len(FEATURES))
    state = torch.load(checkpoint, map_location="cpu")
    msdcs.load_state_dict(state["msdcs"])
    if hasattr(model, "transformer") and backbone == "ViT-L/14@336px":
        del model.transformer
    model.to(device).eval()
    msdcs.to(device).eval()
    meta = {"device": device, "image_size": size, "patch_size": patch_size, "backbone": backbone}
    _CACHE.update(key=key, model=model, msdcs=msdcs, meta=meta)
    return model, msdcs, meta


def _encode(model, tensor, device):
    with torch.no_grad():
        return model(tensor.unsqueeze(0).to(device), FEATURES)


def _score(query_feats, prompt_feats, meta, few_shot):
    msdcs = _CACHE["msdcs"]
    size = meta["image_size"]
    kernel = 2 * int(4.0 * SIGMA + 0.5) + 1
    lamda_pixel = 0.5 if few_shot else 0.1

    with torch.no_grad():
        image_logits, patch_scores = msdcs(query_feats, window_list=WINDOW)
        zs_patch = torch.concat([s[:, 1:] for s in patch_scores], dim=1).mean(dim=1)
        zs_image = torch.concat([g.softmax(-1)[:, 1:] for g in image_logits], dim=1).mean(dim=1)
        zs_patch = 1.0 / (1 + torch.exp(-15 * (zs_patch - 0.40)))
        anomaly_map = gaussian_blur(zs_patch, kernel_size=[kernel, kernel], sigma=[SIGMA, SIGMA])

        if few_shot:
            fs_patch = multi_scale_nearest_neighbor(
                query_feats, prompt_feats, size, meta["patch_size"], window_list=WINDOW,
            )
            fs_patch = gaussian_blur(fs_patch, kernel_size=[kernel, kernel], sigma=[SIGMA, SIGMA])
            fs_patch = 1.0 / (1 + torch.exp(-15 * (fs_patch - 0.30)))
            anomaly_map = gaussian_blur(
                (1 - LAMDA_FS) * zs_patch + LAMDA_FS * fs_patch,
                kernel_size=[kernel, kernel],
                sigma=[SIGMA, SIGMA],
            )

        anomaly_map = F.interpolate(anomaly_map[:, None], size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
        topk, _ = torch.topk(anomaly_map.view(1, -1), k=1, dim=1)
        image_score = (1 - lamda_pixel) * zs_image + lamda_pixel * topk.mean(dim=1)
        image_score = torch.nan_to_num(image_score, nan=0.0, posinf=0.0, neginf=0.0)
        anomaly_map = torch.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
        anomaly_map = F.interpolate(anomaly_map[:, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]

    return float(image_score.item()), anomaly_map.detach().cpu().numpy()


def detect(mode, backbone_label, checkpoint, query, ref1, ref2, ref3, ref4):
    if query is None:
        raise gr.Error("Upload a query image.")
    backbone = BACKBONES[backbone_label]
    few_shot = mode == "Few-shot"
    refs = [image for image in (ref1, ref2, ref3, ref4) if image is not None]
    if few_shot and not refs:
        raise gr.Error("Few-shot needs 1–4 normal images of the same object.")

    with _LOCK:
        model, _, meta = _load(backbone, checkpoint)
        mean, std = _mean_std(backbone)
        tensor, vis = _preprocess(query, meta["image_size"], mean, std)
        query_feats = _encode(model, tensor, meta["device"])
        prompt_feats = None
        if few_shot:
            shots = []
            for ref in refs[:4]:
                shot, _ = _preprocess(ref, meta["image_size"], mean, std)
                shots.append(_encode(model, shot, meta["device"]))
            prompt_feats = [torch.stack([shot[i] for shot in shots], dim=1) for i in range(len(FEATURES))]
        score, heatmap = _score(query_feats, prompt_feats, meta, few_shot)

    overlay = apply_ad_scoremap(vis, heatmap, alpha=0.5)
    return overlay, f"{score:.3f}"


def on_backbone(backbone_label):
    return default_checkpoint(BACKBONES[backbone_label])


def _clear():
    return None, ""


CSS = """
#submit-button {
  color: red !important;
  background-color: LightSalmon !important;
  border: none !important;
  padding: 10px 20px !important;
  border-radius: 5px !important;
  font-size: 16px !important;
}
#submit-button:hover { background-color: Coral !important; }
"""


def build_app():
    with gr.Blocks(title="UniADet", css=CSS) as demo:
        gr.HTML("<h1 style='text-align:center;'>UniADet: Language-Free Universal Anomaly Detection</h1>")
        gr.HTML("<h3 style='text-align:center;'>Zero-shot uses the query only. Few-shot adds 1–4 normal images.</h3>")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    query = gr.Image(type="pil", label="Query Image", height=480)
                with gr.Row(visible=False) as refs_box:
                    ref1 = gr.Image(type="pil", label="Normal 1")
                    ref2 = gr.Image(type="pil", label="Normal 2")
                    ref3 = gr.Image(type="pil", label="Normal 3")
                    ref4 = gr.Image(type="pil", label="Normal 4")
                mode = gr.Radio(["Zero-shot", "Few-shot"], value="Zero-shot", label="Mode")
                backbone = gr.Radio(list(BACKBONES), value="CLIP ViT-L/14@336px", label="Backbone")
                checkpoint = gr.Textbox(
                    label="MSDCS checkpoint",
                    value=default_checkpoint(BACKBONES["CLIP ViT-L/14@336px"]),
                )
            with gr.Column():
                anomaly_vis = gr.Image(type="numpy", label="Anomaly Results", height=480)
                anomaly_score = gr.Textbox(label="Anomaly Score")

        with gr.Row():
            submit_button = gr.Button("Submit", elem_id="submit-button")
            clear_button = gr.Button("Clear")

        mode.change(lambda m: gr.update(visible=m == "Few-shot"), mode, refs_box)
        backbone.change(on_backbone, backbone, checkpoint)
        submit_button.click(
            detect,
            [mode, backbone, checkpoint, query, ref1, ref2, ref3, ref4],
            [anomaly_vis, anomaly_score],
        )
        clear_button.click(_clear, outputs=[anomaly_vis, anomaly_score])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)
