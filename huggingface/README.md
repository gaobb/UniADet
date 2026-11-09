---
license: other
license_name: uniadet-weights
license_link: https://neurips.cc/virtual/2026/poster/149035
language:
  - en
pipeline_tag: image-segmentation
tags:
  - anomaly-detection
  - zero-shot
  - few-shot
  - vision
  - pytorch
  - clip
  - dinov2
  - dinov3
  - industrial-inspection
  - medical-imaging
gated: auto
extra_gated_heading: "Request access to UniADet weights"
extra_gated_description: "Access is granted after you submit the form below. Please use an institutional email when possible."
extra_gated_button_content: "Agree and request access"
extra_gated_prompt: >
  UniADet MSDCS weights are released for research and evaluation.
  By requesting access you agree that: (1) you will cite the UniADet paper if you use these weights;
  (2) you will not redistribute the weights as a standalone package without this model card;
  (3) the authors may keep a record of this access request (Hugging Face username, name, affiliation, email, country, intended use, and request time)
  to measure academic and industrial adoption. This is an access-request log, not a per-file download log.
  Hugging Face also counts anonymous repository downloads automatically.
extra_gated_fields:
  Full name: text
  Affiliation: text
  Official email: text
  Position:
    type: select
    options:
      - Faculty
      - Researcher
      - PhD student
      - Master student
      - Undergraduate
      - Engineer
      - Other
  Country: country
  I want to use this model for:
    type: select
    options:
      - Academic research
      - Teaching
      - Industrial R&D
      - Benchmark reproduction
      - label: Other
        value: other
  Brief description of intended use: text
  I agree to cite UniADet and not redistribute the weights without this model card: checkbox
---

# UniADet

> Official MSDCS weights of [One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection](https://neurips.cc/virtual/2026/poster/149035), NeurIPS 2026.

This Hugging Face repository stores **learnable MSDCS checkpoints only**. Training and evaluation **code** is in a separate GitHub repository: [gaobb/UniADet](https://github.com/gaobb/UniADet). Frozen CLIP / DINOv2 / DINOv3 backbones are **not** included here.

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-blue.svg)](https://neurips.cc/virtual/2026/poster/149035)
[![GitHub](https://img.shields.io/badge/GitHub-gaobb%2FUniADet-black)](https://github.com/gaobb/UniADet)
[![HF downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fcsgaobb%2FUniADet&query=downloads&label=HF%20downloads)](https://huggingface.co/csgaobb/UniADet)

- [Paper (NeurIPS 2026 poster)](https://neurips.cc/virtual/2026/poster/149035)
- [Code](https://github.com/gaobb/UniADet)
- [SNU DSBA Lab talk (Korean)](https://www.youtube.com/watch?v=y_QrUOyW2c4)

## News

- **[2026-11-09]**: Pre-trained MSDCS weights are released in this repository (gated: request access, then `hf download`).
- **[2026-09-25]**: UniADet has been accepted to **NeurIPS 2026**. Official page: [Poster](https://neurips.cc/virtual/2026/poster/149035).

## Introduction

UniADet is a **language-free** universal (zero- and few-shot) visual anomaly detection framework. It outperforms language-dependent zero-/few-shot AD models while remaining simple and efficient: only **0.015M or 0.020M** learnable parameters on top of a frozen foundation model.

<div align="center">
  <img src="assets/LangDepVSLangFree.svg" width="80%" alt="Language-dependent vs language-free anomaly detection">
</div>

- Language prompts and text encoders are unnecessary. The same recipe works with **CLIP**, **DINOv2-Reg**, and **DINOv3**.
- Global anomaly classification and local anomaly segmentation are **fully decoupled** across multi-scale hierarchical features (MSDCS: layer-wise cls/seg weights).
- Few-shot UniADet is the first to outperform full-shot state-of-the-art on several industrial benchmarks.

<div align="center">
  <img src="assets/UniADet-Framework-v3.svg" width="80%" alt="UniADet framework">
</div>

## Checkpoints in this repo

MSDCS uses 5 layers, dual cls/seg embeddings, and 2 tokens (normal / anomaly): \(5 \times 2 \times 2 \times D\). Files are fp32 `.pth` with a small pickle header. Each file stores `{"msdcs": state_dict}` only.

| Backbone | \(D\) | Learnable params | File size | Frozen backbone (not in this repo) |
| :--- | :---: | ---: | ---: | ---: |
| CLIP `ViT-L-14@336px` | 768 | \(1.5\times10^{-2}\) M | 64.5 KB | 342.9 M |
| DINOv2-Reg `ViT-L-14` | 1024 | \(2.0\times10^{-2}\) M | 84.5 KB | 304.4 M |
| DINOv3 `ViT-L-16` | 1024 | \(2.0\times10^{-2}\) M | 84.5 KB | 303.2 M |

Layout after download (same as `$UNIADET_CKPT` in the code repo):

```
train_on_<visa|mvtec>_zs_<backbone>_imagesize<H>_1015/epoch_15.pth
```

| Path | Backbone | Trained on | Input size |
| :--- | :--- | :---: | :---: |
| `train_on_visa_zs_ViT-L-14@336px_imagesize518_1015/epoch_15.pth` | CLIP ViT-L-14@336px | VisA | 518 |
| `train_on_mvtec_zs_ViT-L-14@336px_imagesize518_1015/epoch_15.pth` | CLIP ViT-L-14@336px | MVTec | 518 |
| `train_on_visa_zs_ViT-L-14-DINOv2Reg_imagesize518_1015/epoch_15.pth` | DINOv2-Reg ViT-L/14 | VisA | 518 |
| `train_on_mvtec_zs_ViT-L-14-DINOv2Reg_imagesize518_1015/epoch_15.pth` | DINOv2-Reg ViT-L/14 | MVTec | 518 |
| `train_on_visa_zs_ViT-L-16-DINOv3_imagesize512_1015/epoch_15.pth` | DINOv3 ViT-L/16 | VisA | 512 |
| `train_on_mvtec_zs_ViT-L-16-DINOv3_imagesize512_1015/epoch_15.pth` | DINOv3 ViT-L/16 | MVTec | 512 |

Paper protocol: train on VisA, test on the target set, unless you specifically need the MVTec-trained weights.

## How to download

This repo is **gated**. Open https://huggingface.co/csgaobb/UniADet while logged in, fill the form, then:

```bash
pip install -U huggingface_hub
hf auth login
hf download csgaobb/UniADet --local-dir "$UNIADET_CKPT"
```

Please use `hf download` / `snapshot_download` (whole repo). That fetches `config.json`, which is the file Hugging Face uses to **count downloads**. Directly clicking a single `.pth` in the file browser may not increment the public download counter.

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="csgaobb/UniADet", local_dir="checkpoints")
```

Then follow the GitHub docs: [environment](https://github.com/gaobb/UniADet/blob/main/docs/environment.md), [data](https://github.com/gaobb/UniADet/blob/main/docs/data.md), [training / evaluation](https://github.com/gaobb/UniADet/blob/main/docs/train_eval.md).

## Results (Image-AUROC / Pixel-AUPR)

| Backbone | Shot | MVTec-AD | VisA | Real-IAD |
| :--- | :---: | :---: | :---: | :---: |
| **CLIP** (ViT-L-14@336px) | **0** | 92.4 / 42.8 | 88.0 / 28.0 | 78.6 / 33.6 |
| **CLIP** (ViT-L-14@336px) | **4** | 97.7 / 58.8 | 93.3 / 36.7 | 84.3 / 37.2 |
| **DINOv2** (Register ViT-L/14) | **0** | 93.5 / 50.9 | 91.3 / **32.7** | **82.5 / 43.1** |
| **DINOv2** (Register ViT-L/14) | **4** | **98.7** / 65.4 | 96.9 / 45.2 | **90.3** / 48.5 |
| **DINOv3** (ViT-L/16) | **0** | **94.0 / 52.7** | **91.9** / 32.5 | 81.2 / 41.6 |
| **DINOv3** (ViT-L/16) | **4** | 98.2 / **69.0** | **97.1 / 45.5** | 88.5 / **49.8** |

Full comparisons, ablations, and efficiency tables: [GitHub README](https://github.com/gaobb/UniADet).

## Intended use

- **Intended:** academic research and reproduction of UniADet zero-shot / few-shot visual anomaly detection on public industrial and medical benchmarks.
- **Out of scope:** clinical diagnosis, safety-critical deployment without additional validation, or treating these tiny MSDCS files as a standalone detector without the frozen backbone and official code.

## Bias, risks, limitations

- Weights were trained on auxiliary industrial data (VisA or MVTec). Cross-domain performance depends on the frozen backbone and the evaluation protocol in the paper.
- Checkpoints contain MSDCS embeddings only. You still need the corresponding official CLIP / DINOv2-Reg / DINOv3 weights.
- Gated access records who *requested* the repo, not every subsequent `GET` of a weight file.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{uniadet,
  title={One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection},
  author={Gao, Bin-Bin},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
  url={https://neurips.cc/virtual/2026/poster/149035}
}
```
