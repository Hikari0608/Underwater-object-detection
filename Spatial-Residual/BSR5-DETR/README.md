# BSR5-DETR

This directory contains BSR5-DETR for TPAMI-2025 Spatial Residual, built by combining RT-DETR Decoder with the proposed method.


## Supported Models

| Model | Role | Notes |
| --- | --- | --- |
| `bsr5detr_1024_ruod` | Ours | Main RT-DETR-based underwater model |
| `bsr5detr_512_ruod` | Ours | Smaller-input variant for efficiency comparison |

## How to Use This Branch

### Install

```bash
pip install -r requirements.txt # for 4090
```

### Train

Use the training entry point in this directory with the Spatial Residual configs.

```bash
bash run.sh
```

## Ultralytics Branch Note

The `BSR5-DETR-ultralytics` branch is kept for compatibility and lightweight inference use cases. Training on that branch is unstable at the moment, so the native RT-DETR-based branch in this directory should be used for training and mainline experimentation.