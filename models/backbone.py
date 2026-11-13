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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _dinov3_rope_periods(embed_dim=1024, num_heads=16, base=100.0):
    # HF/ModelScope omit rope buffers; rebuild the official bf16-rounded periods.
    head_dim = embed_dim // num_heads
    quarter = head_dim // 4
    periods = base ** (2 * torch.arange(quarter, dtype=torch.float32) / (head_dim // 2))
    return periods.to(torch.bfloat16).to(torch.float32)


def _hf_dinov3_to_official(hf_state):
    import re

    official = {}
    qkv = {}
    for raw, value in hf_state.items():
        key = raw[6:] if raw.startswith("model.") else raw
        # HF drops inv_freq / bias_mask / k bias; rebuild them after the loop.
        if "inv_freq" in key or "bias_mask" in key or key.endswith("k_proj.bias"):
            continue
        key = key.replace("embeddings.cls_token", "cls_token")
        key = key.replace("embeddings.mask_token", "mask_token")
        key = key.replace("embeddings.register_tokens", "storage_tokens")
        key = key.replace("embeddings.patch_embeddings.projection", "patch_embed.proj")
        key = key.replace("embeddings.patch_embeddings", "patch_embed.proj")
        key = key.replace("rope_embeddings", "rope_embed")
        key = re.sub(r"layer\.(\d+)\.attention\.o_proj", r"blocks.\1.attn.proj", key)
        key = re.sub(r"layer\.(\d+)\.attention\.", r"blocks.\1.attn.", key)
        key = re.sub(r"layer\.(\d+)\.layer_scale(\d)\.lambda1", r"blocks.\1.ls\2.gamma", key)
        key = re.sub(r"layer\.(\d+)\.mlp\.up_proj", r"blocks.\1.mlp.fc1", key)
        key = re.sub(r"layer\.(\d+)\.mlp\.down_proj", r"blocks.\1.mlp.fc2", key)
        key = re.sub(r"layer\.(\d+)\.norm", r"blocks.\1.norm", key)
        if key == "mask_token" and value.ndim == 3:
            value = value.squeeze(1)
        match = re.match(r"blocks\.(\d+)\.attn\.(q|k|v)_proj\.(weight|bias)$", key)
        if match:
            qkv[(match.group(1), match.group(3), match.group(2))] = value
            continue
        official[key] = value

    layers = sorted({int(i) for i, _, _ in qkv})
    embed_dim = None
    for layer in layers:
        ref = qkv.get((str(layer), "weight", "q"))
        if ref is None:
            continue
        embed_dim = ref.shape[0]
        # Official mask_k_bias zeros the K bias; HF checkpoints omit k_proj.bias.
        weight = torch.cat([qkv[(str(layer), "weight", name)] for name in "qkv"], dim=0)
        q_bias = qkv.get((str(layer), "bias", "q"), torch.zeros(embed_dim))
        v_bias = qkv.get((str(layer), "bias", "v"), torch.zeros(embed_dim))
        bias = torch.cat([q_bias, torch.zeros(embed_dim), v_bias], dim=0)
        official[f"blocks.{layer}.attn.qkv.weight"] = weight
        official[f"blocks.{layer}.attn.qkv.bias"] = bias
        official[f"blocks.{layer}.attn.qkv.bias_mask"] = torch.cat(
            [torch.ones(embed_dim), torch.zeros(embed_dim), torch.ones(embed_dim)]
        )

    if embed_dim is None:
        raise ValueError("DINOv3 conversion found no attention layers")
    official["rope_embed.periods"] = _dinov3_rope_periods(embed_dim=embed_dim)
    return official


def _hf_dinov2_reg_to_official(hf_state, hidden_size=1024):
    import re

    official = {}
    qkv = {}
    for raw, value in hf_state.items():
        key = raw
        for prefix in ("dinov2_with_registers.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith("classifier.") or "inv_freq" in key:
            continue
        key = key.replace("embeddings.cls_token", "cls_token")
        key = key.replace("embeddings.mask_token", "mask_token")
        key = key.replace("embeddings.position_embeddings", "pos_embed")
        key = key.replace("embeddings.register_tokens", "register_tokens")
        key = key.replace("embeddings.patch_embeddings.projection", "patch_embed.proj")
        key = re.sub(r"encoder\.layer\.(\d+)\.norm1", r"blocks.\1.norm1", key)
        key = re.sub(r"encoder\.layer\.(\d+)\.norm2", r"blocks.\1.norm2", key)
        key = re.sub(
            r"encoder\.layer\.(\d+)\.attention\.output\.dense",
            r"blocks.\1.attn.proj",
            key,
        )
        key = re.sub(
            r"encoder\.layer\.(\d+)\.layer_scale(\d)\.lambda1",
            r"blocks.\1.ls\2.gamma",
            key,
        )
        key = re.sub(r"encoder\.layer\.(\d+)\.mlp\.fc1", r"blocks.\1.mlp.fc1", key)
        key = re.sub(r"encoder\.layer\.(\d+)\.mlp\.fc2", r"blocks.\1.mlp.fc2", key)
        key = key.replace("layernorm.", "norm.")
        match = re.match(
            r"encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)\.(weight|bias)$",
            key,
        )
        if match:
            name = {"query": "q", "key": "k", "value": "v"}[match.group(2)]
            qkv[(match.group(1), match.group(3), name)] = value
            continue
        if key == "mask_token" and value.ndim == 3:
            value = value.squeeze(0)
        official[key] = value

    for layer, part in {(i, p) for i, p, _ in qkv}:
        pieces = [qkv[(layer, part, name)] for name in "qkv"]
        official[f"blocks.{layer}.attn.qkv.{part}"] = torch.cat(pieces, dim=0)
    return official


def _modelscope_snapshot(repo):
    from modelscope.hub.snapshot_download import snapshot_download

    # Keep ModelScope caches under HF_HOME so Space restarts reuse downloads.
    root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "modelscope"
    root.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(repo, cache_dir=str(root)))


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _dinov3_converted_ok(path):
    if not path.is_file():
        return False
    state = _torch_load(path)
    return "rope_embed.periods" in state and "blocks.0.attn.qkv.bias_mask" in state


def _dinov3_weights_from_modelscope():
    repo = os.environ.get("UNIADET_DINOV3_MODELSCOPE", "facebook/dinov3-vitl16-pretrain-lvd1689m")
    cache = _modelscope_snapshot(repo)
    # Official dinov3_vitl16() only accepts local paths named like "*-XXXXXXXX.pth".
    converted = cache / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    if _dinov3_converted_ok(converted):
        return converted
    if converted.is_file():
        converted.unlink()
    safetensors = cache / "model.safetensors"
    if not safetensors.is_file():
        # ModelScope may nest the snapshot one level deeper depending on version.
        matches = list(cache.rglob("model.safetensors"))
        if not matches:
            raise FileNotFoundError(f"No DINOv3 weights under {cache}")
        safetensors = matches[0]
        converted = safetensors.parent / converted.name
        if _dinov3_converted_ok(converted):
            return converted
    from safetensors.torch import load_file

    torch.save(_hf_dinov3_to_official(load_file(str(safetensors))), converted)
    return converted


def _dinov2_reg_weights_from_modelscope():
    repo = os.environ.get("UNIADET_DINOV2_MODELSCOPE", "facebook/dinov2-with-registers-large")
    cache = _modelscope_snapshot(repo)
    converted = cache / "dinov2_vitl14_reg4_pretrain.pth"
    if converted.is_file():
        return converted
    safetensors = cache / "model.safetensors"
    if not safetensors.is_file():
        matches = list(cache.rglob("model.safetensors"))
        if not matches:
            raise FileNotFoundError(f"No DINOv2 weights under {cache}")
        safetensors = matches[0]
        converted = safetensors.parent / converted.name
        if converted.is_file():
            return converted
    from safetensors.torch import load_file

    torch.save(_hf_dinov2_reg_to_official(load_file(str(safetensors))), converted)
    return converted


def load_dino(name):
    if name == "dinov2reg_vit_large_14":
        from dinov2.hub.backbones import dinov2_vitl14_reg
        source = os.environ.get("UNIADET_DINOV2_SOURCE", "official")
        if source == "modelscope":
            # dinov2 hub rejects local paths in weights=; load manually.
            model = dinov2_vitl14_reg(pretrained=False)
            state = _torch_load(_dinov2_reg_weights_from_modelscope())
            model.load_state_dict(state, strict=True)
            return model
        return dinov2_vitl14_reg(pretrained=True)
    if name == "dinov3_vit_large_16":
        from dinov3.hub.backbones import dinov3_vitl16
        source = os.environ.get("UNIADET_DINOV3_SOURCE", "official")
        if source == "modelscope":
            # Load manually so converted Hub weights do not go through URL hash checks.
            model = dinov3_vitl16(pretrained=False)
            model.load_state_dict(_torch_load(_dinov3_weights_from_modelscope()), strict=True)
            return model
        return dinov3_vitl16(pretrained=True)
    raise ValueError(f"Unknown encoder: {name}")

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
