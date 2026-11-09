# Training and Evaluation

Requires [Environment](environment.md) and [Data](data.md). Frozen backbones stay in `UNIADET_PRETRAINED`; learnable MSDCS weights go in `UNIADET_CKPT`.

## MSDCS weights

Download the learnable MSDCS weights (layer-wise cls/seg weights \(W_{\mathrm{cls}}^{l}, W_{\mathrm{seg}}^{l}\)) from [Hugging Face](https://huggingface.co/csgaobb/UniADet) into `UNIADET_CKPT`. The Hub repo is **gated**: request access on the model page, then:

```bash
hf auth login
hf download csgaobb/UniADet --local-dir "$UNIADET_CKPT"
```

Weights and code are hosted separately. This GitHub repo does not vendor the `.pth` files.

MSDCS uses 5 layers, dual cls/seg embeddings, and 2 tokens (normal / anomaly): \(5 \times 2 \times 2 \times D\). Checkpoints are fp32; file size includes a small pickle header.

| Backbone | \(D\) | Learnable params | Checkpoint (`.pth`) | Frozen backbone |
| :--- | :---: | ---: | ---: | ---: |
| CLIP `ViT-L-14@336px` | 768 | \(1.5\times10^{-2}\) M | 64.5 KB | 342.9 M |
| DINOv2-Reg `ViT-L-14` | 1024 | \(2.0\times10^{-2}\) M | 84.5 KB | 304.4 M |
| DINOv3 `ViT-L-16` | 1024 | \(2.0\times10^{-2}\) M | 84.5 KB | 303.2 M |

VisA-trained and MVTec-trained MSDCS checkpoints have the same size. The frozen backbone is **not** stored in the `.pth`; only MSDCS weights are.

To evaluate the paper checkpoints, skip training and place the Hugging Face MSDCS weights under the same `UNIADET_CKPT` layout as below.

## Training

```bash
bash scripts/train.sh
```

Trains learnable MSDCS weights on VisA and MVTec for 15 epochs, with the backbone frozen (CLIP / DINOv2-Reg / DINOv3):

```
$UNIADET_CKPT/train_on_<dataset>_zs_<backbone>_imagesize<H>_1015/epoch_15.pth
```

Input size: 518 for `ViT-L-14@336px` and `ViT-L-14-DINOv2Reg`; 512 for `ViT-L-16-DINOv3`.

## Evaluation

Zero-shot (`k_shot=0`) and few-shot (`k_shot=1,2,4`, seeds 10/20/30). Default protocol: train on VisA, test on the target set.

```bash
bash scripts/test_industry.sh           # mvtec, DTD-Synthetic, SDD, BTAD
bash scripts/test_medical.sh            # medical-cls, medical-seg
bash scripts/test_realiad.sh
bash scripts/test_realiad_variety.sh
```

Metrics: Image AUROC / AP / F1-max and Pixel AUROC / AP / F1-max / AUPRO. Outputs go to `UNIADET_RESULTS`.

## Demo

```bash
pip install gradio
bash scripts/demo.sh
```

Zero-shot takes a query image. Few-shot also takes 1–4 normal images of the same object. Defaults follow the paper protocol (`lamda_pixel` 0.1 / 0.5, `lamda_fs` 0.8). The app opens at `http://0.0.0.0:7860`.
