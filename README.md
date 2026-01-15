# UniADet

> Official PyTorch Implementation of [One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection](https://arxiv.org/abs/2601.05552), 2026.


## Introduction 
UniADet, a **language-free** framework that demonstrates superior performance while achieving remarkable simplicity and efficiency.



<div align="center">
    <img src="./assets/Compare-UniADet.jpg" width="80%">
</div>

- We rethink vision-language ADs and find that language prompts and encoders are unnecessary. This insight leads to an embarrassingly **simple**, **parameter-efficient**, **general**, and **effective** framework for universal anomaly detection.
- We fully **decouple global anomaly classification and local anomaly segmentation** across multi-scale hierarchical features, effectively mitigating the learning conflict between different feature manifolds and substantially improving AD performance.
- Comprehensive experiments conclusively validate that our approach achieves **state-of-the-art zero-shot and few-shot performance**. Notably, our few-shot UniADet is **the first** to outperform full-shot state-of-the-art.

## UniADet Framework

<div align="center">
    <img src="./assets/UniADet-Framework.jpeg" width="80%">
</div>


## Comparison with State-of-the-Arts

| Methods | Venue | Shots | MVTec | VisA  | Real-IAD |
| :--- | :---: | :--- | :--- | :--- | :--- |
| UniADet$^{‡}$ | ours| 0 | 93.5  / 50.9 | 91.3 / 32.7 | 82.5 / 43.1 |
| [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) | ICLR 24 | 0 | 91.6 / 34.5 | 82.0 / 21.3 | 69.5 / 26.7 |
| [Bayes-PFL](https://github.com/xiaozhen228/Bayes-PFL) | CVPR 25| 0 |92.5 / 48.3 | 87.0 / 29.8 | 70.0 / 27.6| 
| UniADet$^{‡}$ | ours| 1 | 97.6 / 63.1 | 95.2 / 42.1 | 88.7 / 48.4 |
| UniADet$^{‡}$ | ours| 2 | 98.0 / 64.1 | 96.1 / 44.2 | 89.0 / 46.7 |
| UniADet$^{‡}$ | ours| 4 | **98.7 / 65.4** | **96.9 / 45.2** | **90.3 / 48.5** |
| [UniVAD](https://github.com/FantasticGNU/UniVAD) |CVPR 25  | 1 | 97.8 / 55.6 | 93.5 / 42.8 | 85.1 / 37.6 |
| [AdaptCLIP](https://github.com/gaobb/AdaptCLIP) |AAAI 26  | 1 | 94.5 / 53.7 | 90.5 / 38.9 | 81.8 / 36.6 |
|  [AdaptCLIP](https://github.com/gaobb/AdaptCLIP) |AAAI 26  | 2 | 95.7 / 55.1 | 92.2 / 40.7 | 82.9 / 37.8 |
|  [AdaptCLIP](https://github.com/gaobb/AdaptCLIP) |AAAI 26  | 4 | 96.6 / 57.2 | 93.1 / 41.8 | 83.9 / 39.1 |
| [MetaUAS](https://github.com/gaobb/MetaUAS) |NeurIPS 24 | 1 | 90.7 / 59.3 | 81.2 / 42.7 | 80.0 / 36.6 |
| [Dinomaly](https://github.com/guojiajeremy/Dinomaly) |CVPR 25  | full | 99.6 / 69.3 | 98.7 / 53.2 | 89.3 / 42.8 |
| [UniAD](https://github.com/zhiyuanyou/UniAD) |NeurIPS 24| full | 96.5 / 44.7 | 90.8 / 33.6 | 83.0 / 21.1 |

Note: The performance is mesured by Image-AUROC / Pixel-AUPR.

## Complexity and Efficiency Comparisons

| Shots | Methods | Models | Input Size | # Params (M) | Inf. Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | AdaCLIP | CLIP ViT-L/14@336px | 518×518 | 428.8 + 1.1e+1 | 107.4 |
| **0** | AnomalyCLIP | CLIP ViT-L/14@336px | 518×518 | 427.9 + 5.6e+0 | 70.7 |
| **0** | Bayes-PFL | CLIP ViT-L/14@336px | 518×518 | 427.9 + 2.7e+1 | 154.9 |
| **0** | AdaptCLIP-Zero | CLIP ViT-L/14@336px | 518×518 | 427.9 + 6.0e-1 | 57.5 |
| **0** | **UniADet$^{†}$** | CLIP ViT-L/14@336px | 518×518 | **342.9 + 1.5e-2** | **15.7** |
| **0** | **UniADet$^{‡}$** | DINOv2 ViT-L/14 | 518×518 | **303.2 + 2.0e-2** | **41.9** |
| **1** | InCtrl | CLIP ViT-B-16+240 | 240×240 | 208.4 + 3.0e-1 | 59.0 |
| **1** | AnomalyCLIP+ | CLIP ViT-L/14@336px | 518×518 | 427.9 + 5.6e+0 | 76.2 |
| **1** | AdaptCLIP | CLIP ViT-L/14@336px | 518×518 | 342.9 + 1.8e+0 | 58.7 |
| **1** | **UniADet$^†$** | CLIP ViT-L/14@336px | 518×518 | **342.9 + 1.5e-2** | **22.4** |
| **1** | **UniADet$^{‡}$** | DINOv2 ViT-L/14 | 518×518 | **303.2 + 2.0e-2** | **48.4** |

## Ablation Studies
Ablation studies about different components.
| No | DCS| DHF| CAA | Shot | MVTec  | VisA  |
| :-- | :--: | :--: | :--: | :--: | :-- | :-- |
| 0 | ✗ | ✗ | ✗ |  0 | 85.4 / 36.4 | 77.9 / 26.1 |
| 1 | ✓ | ✗ | ✗ |  0 | 91.8 / 38.3 | 85.9 / 27.2 |
| 2 | ✓ | ✓ | ✗ |  0 | 92.2 / 40.7 | 86.0 / 27.6 |
| 3 | ✓ | ✓ | ✓ |  0 | 92.4 / 42.8 | 88.0 / 28.0 |
| 4 | ✓ | ✓ | random |  0 | 91.3 / 41.5 | 87.5 / 26.6 |
| 5 | ✓ | ✓ | ✓ |  1 | **95.9 / 54.6** | **91.3 / 32.5** |

Note: The ablation studies are conducted by UniADet$^†$ (i.e., using CLIP ViT-L/14@336px).


## ToDo List
- [ ] release pre-trained [UniADet models]()
- [ ] deploy [online UniADet Demo]() on huggingface
- [ ] open training and testing code


## Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{uniadet,
  title={One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection},
  author={Gao, Bin-Bin and Wang, Chengjie},
  booktitle={arXiv:2601.05552},
  year={2026}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gaobb/UniADet&type=timeline&legend=top-left)](https://www.star-history.com/#gaobb/UniADet&type=timeline&legend=top-left)
