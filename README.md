# DeKAHT: Data-Efficient Kolmogorov-Arnold Hierarchical Transformer

Official PyTorch implementation of **DeKAHT**, a hierarchical vision transformer that improves data efficiency by replacing standard feed-forward networks with **Grouped Nonlinear KAN (GN-KAN)** modules.

This repository accompanies the paper:

**DeKAHT: Data-Efficient Kolmogorov-Arnold Hierarchical Transformer**
(Under review at IEEE ICIP 2026)

---

# ⚠️ Important Naming Clarification

This codebase was developed using earlier internal naming conventions.

| Paper Name | Code Name |
| ---------- | --------- |
| **DeKAHT** | **SwKAT** |
| **GN-KAN** | **GRKAN** |

Therefore:

* `swkat.py` implements the **DeKAHT architecture**
* `GRKAN` modules inside the code correspond to **GN-KAN modules described in the paper**

These naming differences are purely internal and **do not change the method described in the paper**.

---

# Repository Structure

This repository contains two implementations:

## GELU Variant

Located in:

gelu_variant/

Implements:

GN-KAN-GELU architecture.

---

## KANAct Variant

Located in:

KAN_Act_varient/

Implements:

GN-KAN-KANAct architecture with spline-based activations.

Contains:

* DeKAHT architecture (named SwKAT internally)
* GN-KAN implementation (named GRKAN internally)
* Model definitions for Tiny, Small, and Base variants

This file corresponds to:

**Section 3 — Proposed Methodology** in the paper.

---

### train.py

Contains:

* Training pipeline
* Dataset loading logic
* Model initialization
* Training and validation loops

This file was used for experiments on:

* ImageNet-10
* ImageNet-100
* Medical datasets

---

# ⚙️ Installation

Create environment:

```bash
conda create -n dekaht python=3.10
conda activate dekaht
```

Install required packages:

```bash
pip install torch torchvision timm numpy scipy matplotlib scikit-learn tqdm
```

---

# 🚀 Training

Example training command:

```bash
torchrun --nproc_per_node=2 \
  /home/gurramht_iitp/DEKAHT/train.py \
  --data /scratch/gurramht_iitp/datasets/imagenet10 \
  --model-variant swkat-tiny \
  --img-size 224 \
  --epochs 300 \
  --batch-size 196 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --warmup-epochs 5 \
  --opt adamw \
  --num-classes 10 \
  --workers 8 \
  --output-dir /scratch/gurramht_iitp/outputs/DEKAHT/imagenet10/tiny \
  --use-custom-kan \
  --grad-clip 1.0 \
  --no-amp \
  --mixup-alpha 0.0 \
  --cutmix-alpha 0.0 \
  --label-smoothing 0.0 \
  --ramp-epochs 0

```

Available model sizes:

* tiny
* small
* base
* large
---

# 📊 Datasets

Experiments were conducted using:

* ImageNet-10
* ImageNet-100
* Brain Tumor MRI datasets (4-class and 44-class)

Medical datasets were aggregated from publicly available sources, including:

* Figshare Brain Tumor Dataset
* SARTAJ Brain MRI Dataset
* Br35H Dataset

Dataset preparation scripts and links will be released after paper acceptance.

---

# 🧠 Method Overview

DeKAHT modifies the hierarchical transformer block by replacing the standard feed-forward network (FFN) with a **Grouped Nonlinear KAN (GN-KAN)** module.

The GN-KAN module includes:

* Channel expansion
* Nonlinear activation
* Channel grouping
* Depthwise convolution
* Gated recomposition
* Projection layer

This design improves nonlinear diversity while maintaining a comparable computational cost to standard Swin Transformer models.

---

# 🔬 Reproducibility Notes

All experiments:

* trained from scratch
* used identical preprocessing
* followed consistent hyperparameters across baseline and proposed models

Multi-seed evaluation and extended ablation results will be added in future updates.

---

NOTE:
GPU-enabled PyTorch should be installed according to the user's CUDA version.
Refer to:
https://pytorch.org/get-started/locally/

# 📦 Model Weights

Pretrained model weights will be released after paper acceptance.

---

# 📬 Contact

Gurram Harshamanya Thilak
Indian Institute of Technology Patna

For research-related questions, please contact via institutional email.

---

# ⚠️ Disclaimer

This repository is released for academic and research purposes only.
