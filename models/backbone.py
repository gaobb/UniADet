import os
import random
from pathlib import Path

import clip
import numpy as np
import perception_models.core.vision_encoder.modfied_pe as pe
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

ROOT = Path(__file__).resolve().parent.parent
# DINOv2 / DINOv3 checkpoints. CLIP and PE download their own weights.
# Point UNIADET_PRETRAINED at a local directory to skip the hub download.
PRETRAINED_DIR = Path(os.environ.get("UNIADET_PRETRAINED", ROOT / "pretrained"))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dino(name):
    dino_repo = {
        "dinov3_vit_large_16": ROOT / "third_party" / "dinov3",
        "dinov2reg_vit_large_14": ROOT / "third_party" / "dinov2",
    }
    dino_ckpt = {
        "dinov3_vit_large_16": PRETRAINED_DIR / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinov2reg_vit_large_14": PRETRAINED_DIR / "dinov2_vitl14_reg4_pretrain.pth",
    }
    if name not in dino_repo:
        raise ValueError(f"Unknown encoder: {name}")

    repo = str(dino_repo[name])
    ckpt = dino_ckpt[name]
    if name == "dinov3_vit_large_16":
        if ckpt.is_file():
            return torch.hub.load(repo, "dinov3_vitl16", source="local", weights=str(ckpt))
        return torch.hub.load(repo, "dinov3_vitl16", source="local")

    if ckpt.is_file():
        model = torch.hub.load(repo, "dinov2_vitl14_reg", source="local", pretrained=False)
        state_dict = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        return model
    return torch.hub.load(repo, "dinov2_vitl14_reg", source="local")

def build_backbone(model_name, device):
    if model_name == 'ViT-L-14@336px':
        model, _ = clip.load("ViT-L/14@336px", device=device)
        patch_size = 14
        embed_dim = 768
    if model_name == 'VITB16_PLUS_240':
        model, _ = clip.load(model_name, device=device)
        patch_size = 16
        embed_dim = 640

    if model_name == 'ViT-L-14-CLIPA-336':
        model, _ = clip.load(model_name, device=device)
        patch_size = 14
        embed_dim = 768

    if model_name == 'ViT-L-14-DINOv2Reg':
        encoder_name = 'dinov2reg_vit_large_14'
        model = load_dino(encoder_name)
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
            patch_size = 14
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
            patch_size = 14
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            patch_size = 14
        else:
            raise "Architecture not in small, base, large."

    if model_name == 'ViT-L-16-DINOv3':
        encoder_name = 'dinov3_vit_large_16'
        model = load_dino(encoder_name)
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
            patch_size = 14
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
            patch_size = 14
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            patch_size = 16
        else:
            raise "Architecture not in small, base, large."

    if model_name in ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']:
        model = pe.CLIP.from_config(model_name, pretrained=True)  # Downloads from HF
        model = model.visual.cuda()
        if model_name in ['PE-Core-L14-336']:
            embed_dim, num_heads = 1024, 16
            patch_size = 14
        if model_name in ['PE-Core-G14-448']:
            embed_dim, num_heads = 1280, 16
            patch_size = 14

    if model_name in ['PE-Spatial-G14-448']:
        model = pe.VisionTransformer.from_config("PE-Spatial-G14-448", pretrained=True)
        embed_dim, num_heads = 1536, 16
        patch_size = 14
    return model, embed_dim, patch_size
