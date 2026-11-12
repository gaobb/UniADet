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
os.environ.setdefault("UNIADET_DEVICE", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ["UNIADET_DEVICE"])

from models.backbone import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, build_backbone
from models.msdcs import MSDCS, multi_scale_nearest_neighbor
from utils.visualization import apply_ad_scoremap

FEATURES = [12, 15, 18, 21, 24]
SIGMA = 4
SCENARIOS = {
    "Industrial": {"window_list": [1], "lamda_fs": 0.8},
    "Medical": {"window_list": [1, 3, 5], "lamda_fs": 0.1},
}
BACKBONES = {
    "CLIP ViT-L-14@336px": "ViT-L-14@336px",
    "DINOv2-Reg ViT-L-14": "ViT-L-14-DINOv2Reg",
    "DINOv3 ViT-L-16": "ViT-L-16-DINOv3",
}

EXAMPLES = [
    ('Brain MRI · Medical', 'Medical',
     'medical-cls-BrainMRI-yes-Y10.jpg',
     None),
    ('Kvasir · Medical', 'Medical',
     'medical-seg-Kvasir-images-cju7dz5yy2i7z0801ausi7rna.jpg',
     None),
    ('ISIC · Medical', 'Medical',
     'medical-seg-ISIC-images-ISIC_0010498.jpg',
     None),
    ('Capsules · Industrial', 'Industrial',
     'visa-capsules-Data-Images-Anomaly-045.JPG',
     'visa-capsules-Data-Images-Normal-000.JPG'),
    ('PCB1 · Industrial', 'Industrial',
     'visa-pcb1-Data-Images-Anomaly-050.JPG',
     'visa-pcb1-Data-Images-Normal-0000.JPG'),
    ('Carpet · Industrial', 'Industrial',
     'mvtec-carpet-test-cut-012.png',
     'mvtec-carpet-train-good-000.png'),
]

_LOCK = threading.Lock()
_CACHE = {"key": None, "model": None, "msdcs": None, "meta": None}


def image_size_of(name):
    return 512 if "DINOv3" in name else 518


def default_checkpoint(name, train_on="visa"):
    root = Path(os.environ["UNIADET_CKPT"])
    size = image_size_of(name)
    return str(root / f"train_on_{train_on}_zs_{name}_imagesize{size}_1015" / "epoch_15.pth")


def _mean_std(name):
    if name in ("ViT-L-16-DINOv3", "ViT-L-14-DINOv2Reg"):
        return IMAGENET_MEAN, IMAGENET_STD
    return CLIP_MEAN, CLIP_STD


def _preprocess(image, size, mean, std):
    rgb = np.array(image.convert("RGB"))
    resized = A.Resize(height=size, width=size)(image=rgb)["image"]
    tensor = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])(image=resized)["image"]
    return tensor, resized


def _checkpoint_state(checkpoints):
    states = [torch.load(path, map_location="cpu")["msdcs"] for path in checkpoints]
    if len(states) == 1:
        return states[0]
    first, second = states
    if first.keys() != second.keys():
        raise gr.Error("Mix checkpoints have different parameter names.")
    averaged = {}
    for name, value in first.items():
        other = second[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise gr.Error(f"Mix checkpoint parameter mismatch: {name}")
        averaged[name] = (value + other) / 2
    return averaged


def _load(backbone, checkpoint):
    paths = checkpoint if isinstance(checkpoint, (list, tuple)) else [checkpoint]
    checkpoints = tuple(str(Path(path).expanduser()) for path in paths)
    if len(checkpoints) not in (1, 2):
        raise gr.Error("Select one checkpoint or a pair for Mix.")
    key = (backbone, checkpoints)
    if _CACHE["key"] == key:
        return _CACHE["model"], _CACHE["msdcs"], _CACHE["meta"]
    for path in checkpoints:
        if not Path(path).is_file():
            raise gr.Error(f"Checkpoint not found:\n{path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, embed_dim, patch_size = build_backbone(backbone, device)
    size = image_size_of(backbone)
    msdcs = MSDCS(size, patch_size, embed_dim=embed_dim, layers_num=len(FEATURES))
    msdcs.load_state_dict(_checkpoint_state(checkpoints))
    if hasattr(model, "transformer") and backbone == "ViT-L-14@336px":
        del model.transformer
    model.to(device).eval()
    msdcs.to(device).eval()
    meta = {"device": device, "image_size": size, "patch_size": patch_size, "backbone": backbone}
    _CACHE.update(key=key, model=model, msdcs=msdcs, meta=meta)
    return model, msdcs, meta


def _encode(model, tensor, device):
    with torch.no_grad():
        return model(tensor.unsqueeze(0).to(device), FEATURES)


def _score(query_feats, prompt_feats, meta, few_shot, scenario="Industrial"):
    msdcs = _CACHE["msdcs"]
    size = meta["image_size"]
    kernel = 2 * int(4.0 * SIGMA + 0.5) + 1
    config = SCENARIOS[scenario]
    medical = scenario == "Medical"
    window_list = config["window_list"]
    lamda_fs = config["lamda_fs"]
    lamda_pixel = 0.5 if medical or few_shot else 0.1

    with torch.no_grad():
        image_logits, patch_scores = msdcs(query_feats, window_list=window_list)
        zs_patch = torch.concat([s[:, 1:] for s in patch_scores], dim=1).mean(dim=1)
        zs_image = torch.concat([g.softmax(-1)[:, 1:] for g in image_logits], dim=1).mean(dim=1)
        if not medical:
            zs_patch = 1.0 / (1 + torch.exp(-15 * (zs_patch - 0.40)))
        anomaly_map = gaussian_blur(zs_patch, kernel_size=[kernel, kernel], sigma=[SIGMA, SIGMA])

        if few_shot:
            fs_patch = multi_scale_nearest_neighbor(
                query_feats, prompt_feats, size, meta["patch_size"], window_list=window_list,
            )
            fs_patch = gaussian_blur(fs_patch, kernel_size=[kernel, kernel], sigma=[SIGMA, SIGMA])
            if not medical:
                fs_patch = 1.0 / (1 + torch.exp(-15 * (fs_patch - 0.30)))
            anomaly_map = gaussian_blur(
                (1 - lamda_fs) * zs_patch + lamda_fs * fs_patch,
                kernel_size=[kernel, kernel],
                sigma=[SIGMA, SIGMA],
            )

        anomaly_map = F.interpolate(anomaly_map[:, None], size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
        # Match test.py: pool the top 2% after resizing for medical images.
        topk_count = int(anomaly_map.shape[-2] * anomaly_map.shape[-1] * 0.02) if medical else 1
        topk, _ = torch.topk(anomaly_map.view(1, -1), k=topk_count, dim=1)
        image_score = (1 - lamda_pixel) * zs_image + lamda_pixel * topk.mean(dim=1)
        image_score = torch.nan_to_num(image_score, nan=0.0, posinf=0.0, neginf=0.0)
        anomaly_map = torch.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
        anomaly_map = F.interpolate(anomaly_map[:, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]

    return float(image_score.item()), anomaly_map.detach().cpu().numpy()


def detect(mode, backbone_label, checkpoint, query, ref1, ref2, ref3, ref4, scenario="Industrial"):
    if query is None:
        raise gr.Error("Upload a query image.")
    if scenario not in SCENARIOS:
        raise gr.Error("Select Industrial or Medical.")
    backbone = BACKBONES[backbone_label]
    few_shot = mode == "Few-shot"
    refs = [image for image in (ref1, ref2, ref3, ref4) if image is not None]
    if few_shot and not refs:
        raise gr.Error("Few-shot needs 1–4 reference images of the same category.")

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
        score, heatmap = _score(query_feats, prompt_feats, meta, few_shot, scenario)

    overlay = apply_ad_scoremap(vis, heatmap, alpha=0.5)
    return overlay, f"{score:.3f}"


def on_backbone(backbone_label, train_on="mix"):
    backbone = BACKBONES[backbone_label]
    if train_on == "mix":
        return [default_checkpoint(backbone, source) for source in ("mvtec", "visa")]
    return default_checkpoint(backbone, train_on)


CHECKPOINT_SOURCES = {"MVTec": "mvtec", "VisA": "visa", "Mix": "mix"}


def on_checkpoint(backbone_label, source_label):
    return on_backbone(backbone_label, CHECKPOINT_SOURCES[source_label])


def examples_for_mode(mode):
    return [example for example in EXAMPLES if mode != "Few-shot" or example[3] is not None]


def example_gallery(mode):
    return [(str(ROOT / "assets" / "examples" / "query" / query_name), label)
            for label, _, query_name, _ in examples_for_mode(mode)]


def on_mode(mode):
    return gr.update(visible=mode == "Few-shot"), gr.update(value=example_gallery(mode), selected_index=None)


def on_example(mode, evt: gr.SelectData):
    _, domain, query_name, reference_name = examples_for_mode(mode)[evt.index]
    query_path = str(ROOT / "assets" / "examples" / "query" / query_name)
    reference_path = str(ROOT / "assets" / "examples" / "normal" / reference_name) if reference_name else None
    return (
        query_path, domain, reference_path, None, None, None,
        gr.update(visible=mode == "Few-shot"), None, "",
    )


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
#example-gallery .grid-container { grid-template-rows: 152px !important; grid-auto-rows: 152px !important; }
#example-gallery .thumbnail-item { height: 152px !important; aspect-ratio: auto !important; }
#example-gallery .thumbnail-lg > img { width: 100%; height: 100%; object-fit: contain; }
#refs-row > * { min-width: 0; flex: 1 1 0; }
#refs-row .image-container {
  width: 100% !important;
  height: auto !important;
  aspect-ratio: 1 / 1;
}
"""


def build_app():
    with gr.Blocks(title="UniADet", css=CSS) as demo:
        gr.HTML("<h1 style='text-align:center;'>UniADet: Language-Free Universal Anomaly Detection</h1>")
        gr.HTML("<h3 style='text-align:center;'>Zero-shot uses the query only. Few-shot adds 1–4 reference images.</h3>")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    query = gr.Image(type="pil", label="Query Image", height=480)
                with gr.Row(visible=False, elem_id="refs-row") as refs_box:
                    ref1 = gr.Image(type="pil", label="Reference 1")
                    ref2 = gr.Image(type="pil", label="Reference 2")
                    ref3 = gr.Image(type="pil", label="Reference 3")
                    ref4 = gr.Image(type="pil", label="Reference 4")
                mode = gr.Radio(["Zero-shot", "Few-shot"], value="Zero-shot", label="Mode")
                backbone = gr.Radio(list(BACKBONES), value="DINOv3 ViT-L-16", label="Backbone")
                checkpoint = gr.State(on_backbone("DINOv3 ViT-L-16", "mix"))
                with gr.Accordion("UniADet checkpoint (Pre-trained Auxiliary Dataset)", open=False):
                    checkpoint_source = gr.Radio(
                        list(CHECKPOINT_SOURCES), value="Mix", show_label=False,
                    )
                with gr.Accordion("Input Data Domain", open=False):
                    scenario = gr.Radio(list(SCENARIOS), value="Industrial", label="Image Domain")
                    gr.Markdown("Use normal reference images for industrial detection. For medical detection, use reference images from the same category.")
            with gr.Column():
                anomaly_vis = gr.Image(type="numpy", label="Anomaly Results", height=480)
                anomaly_score = gr.Textbox(label="Anomaly Score")

        with gr.Row():
            submit_button = gr.Button("Submit", elem_id="submit-button")
            clear_button = gr.Button("Clear")

        examples = gr.Gallery(
            value=example_gallery("Zero-shot"),
            label="Examples", columns=6, rows=1, height=200, elem_id="example-gallery",
            object_fit="contain", allow_preview=False, interactive=False,
        )
        gr.Markdown("Select an example to set its image domain automatically. Few-shot shows only examples with a matching normal reference.")
        examples.select(
            on_example, inputs=[mode],
            outputs=[query, scenario, ref1, ref2, ref3, ref4, refs_box, anomaly_vis, anomaly_score],
        )

        mode.change(on_mode, mode, [refs_box, examples])
        backbone.change(on_checkpoint, [backbone, checkpoint_source], checkpoint)
        checkpoint_source.change(on_checkpoint, [backbone, checkpoint_source], checkpoint)
        submit_button.click(
            detect,
            [mode, backbone, checkpoint, query, ref1, ref2, ref3, ref4, scenario],
            [anomaly_vis, anomaly_score],
        )
        clear_button.click(_clear, outputs=[anomaly_vis, anomaly_score])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0")
