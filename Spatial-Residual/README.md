# Spatial Residual for Underwater Object Detection

**\[TPAMI 2025]**

This repository contains code and models for the paper **"Spatial Residual for Underwater Object Detection"**.

---

## 🔔 Updates

* ✔️ **TensorRT weights** (used in Table 8) are now available.
* ✔️ **BSR-DETR training code** has been released.
* 😢 **BSR5-YOLO training code** will be released later due to a tight schedule.

> ✅ TensorRT weights are compatible with Volta (e.g. V100) and later architectures (e.g. Ampere). You can use them for inference directly.

---

## 🚀 Quick Start

### 1. BSR-DETR Training

#### 🛠 Installation

```bash
# Recommended environment
python=3.10
pytorch==2.2
torchvision==0.17
cudatoolkit=11.8

cd Spatial-Residual/BSR5-DETR/ultralytics
pip install -v -e .
```

#### ▶️ Start Training

```bash
cd Spatial-Residual/BSR5-DETR
yolo train model=models/bsr5-detr.yaml data=../../datas/ruod.yaml epochs=300 imgsz=640 device=0
```

---

### 2. BSR-YOLO Training *(Coming Soon)*

#### 🛠 Installation

```bash
# Create conda environment
conda create -n mmyolo python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate mmyolo

# Install dependencies
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<4.0.0"

# Go to MMYOLO directory
cd Underwater-object-detection/Spatial-Residual/BSR5-YOLO/mmyolo

# Install albumentations
pip install -r requirements/albu.txt

# Install MMYOLO
mim install -v -e .
```

#### ▶️ Start Training

```bash
cd Spatial-Residual/BSR-YOLO/mmyolo
python tools/train.py configs/bsr5-yolo/ours-l.py
```

---

## 📌 Citation

*Coming soon with the final release.*