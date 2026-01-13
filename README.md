# UniADet

> Official PyTorch Implementation of [One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection](https://arxiv.org/abs/2601.05552), 2026.


## Introduction 
UniADet, a **language-free** framework that demonstrates superior performance while achieving remarkable simplicity and efficiency.



<div align="center">
    <img src="https://arxiv.org/html/2601.05552v1/x1.png" width="80%">
</div>

- We rethink vision-language ADs and find that language prompts and encoders are unnecessary. This insight leads to an embarrassingly **simple, parameter-efficient, and highly general language-free** framework for universal anomaly detection.
- We fully **decouple global anomaly classification and local anomaly segmentation** across multi-scale hierarchical features, effectively mitigating the learning conflict between different feature manifolds and substantially improving AD performance.
- Comprehensive experiments conclusively validate that our approach achieves **state-of-the-art zero-shot and few-shot performance**. Notably, our few-shot UniADet is **the first** to outperform full-shot state-of-the-art.

## UniADet Framework

<div align="center">
    <img src="https://arxiv.org/html/2601.05552v1/x2.png" width="80%">
</div>

## Ablation Studies
Ablation studies about different components.
| No | DC | SD | HF | CA | Shot | MVTec  | VisA  |
| :-- | :--: | :--: | :--: | :--: | :--: | :-- | :-- |
| 0 | ✗ | ✗ | ✗ | - | 0 | 85.4 / 36.4 | 77.9 / 26.1 |
| 1 | ✓ | ✗ | ✗ | - | 0 | 91.8 / 38.3 | 85.9 / 27.2 |
| 2 | ✓ | ✓ | ✗ | - | 0 | 92.2 / 40.7 | 86.0 / 27.6 |
| 3 | ✓ | ✓ | ✓ | - | 0 | 92.4 / 42.8 | 88.0 / 28.0 |
| 4 | ✓ | ✓ | random | - | 0 | 91.3 / 41.5 | 87.5 / 26.6 |
| 5 | ✓ | ✓ | ✓ | - | 1 | **95.9 / 54.6** | **91.3 / 32.5** |


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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gaobb/UniADet&type=timeline&legend=top-left)](https://www.star-history.com/#gaobb/UniADet&type=timeline&legend=top-left)
