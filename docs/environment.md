# Environment

Python ≥ 3.10 and a CUDA 12.8 PyTorch build. Edit only your local path file; do not commit machine-specific paths.

Next: [Data](data.md) · [Training and Evaluation](train_eval.md)

## Installation

From the repo root:

```bash
pip install -r requirements.txt
```

`third_party/` (CLIP, DINOv2, DINOv3, Perception Encoder) is put on `PYTHONPATH` by `scripts/env.sh`.

## Local paths

Copy the template and point it at your disks:

```bash
cp scripts/env.local.sh.example scripts/env.local.sh
```

```bash
export UNIADET_DATA=/path/to/datasets
export UNIADET_PRETRAINED=/path/to/pre-trained-models
```

Optional: `UNIADET_CKPT`, `UNIADET_RESULTS`, `UNIADET_DEVICE`.  
Every `scripts/*.sh` sources `scripts/env.sh`, which loads `env.local.sh` if present.

Unset variables default to `./data`, `./pretrained`, `./checkpoints`, `./results`, and GPU `0`.

## Frozen backbones

Put frozen backbone weights in `UNIADET_PRETRAINED` (DINOv2-Reg / DINOv3). CLIP (`ViT-L-14@336px`) is fetched by the official CLIP loader.

```
dinov2_vitl14_reg4_pretrain.pth
dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Learnable MSDCS checkpoints go in `UNIADET_CKPT`; see [Training and Evaluation](train_eval.md).
