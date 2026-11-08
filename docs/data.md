# Data

Set `UNIADET_DATA` in [Environment](environment.md) first. Paths inside `meta.json` are relative to each dataset folder.

Next: [Training and Evaluation](train_eval.md)

## Layout

`$UNIADET_DATA/<dataset>/` with images **and** a `meta.json`.

```
$UNIADET_DATA/
  visa/{meta.json, ...}
  mvtec/{meta.json, ...}
  ...
```

Auxiliary training uses VisA or MVTec. Evaluation covers industrial benchmarks, Real-IAD / Real-IAD-Variety, and medical (`medical-cls`, `medical-seg`).

## Option A — generate `meta.json`

After the images are in place:

```bash
source scripts/env.sh
python datasets/generate_json/visa.py
python datasets/generate_json/mvtec.py
python datasets/generate_json/DTD.py                 # DTD-Synthetic
python datasets/generate_json/medical_ano_cls.py
python datasets/generate_json/medical_ano_seg.py
python datasets/generate_json/realiad.py             # RealIAD
```

Other industrial sets (BTAD, MPDD, SDD, MVTec3D, …) have matching scripts in `datasets/generate_json/`.

## Option B — download prepared listings

Pre-generated `meta.json` files are on [Google Drive](https://drive.google.com/drive/folders/TODO). Unpack each file as `$UNIADET_DATA/<dataset>/meta.json` next to the images. Images are **not** included; obtain each dataset from its official source.
