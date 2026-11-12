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
```

Optional: `UNIADET_CKPT`, `UNIADET_RESULTS`, `UNIADET_DEVICE`.  
Every `scripts/*.sh` sources `scripts/env.sh`, which loads `env.local.sh` if present.

Unset variables default to `./data`, `./checkpoints`, `./results`, and GPU `0`.

## Frozen backbones

CLIP, DINOv2-Reg, and DINOv3 are loaded by their official loaders (`clip.load`, `dinov2_vitl14_reg`, `dinov3_vitl16`). The first run downloads the weights into that loader's own cache. UniADet does not take a local backbone directory.

Learnable MSDCS checkpoints go in `UNIADET_CKPT`; see [Training and Evaluation](train_eval.md).
