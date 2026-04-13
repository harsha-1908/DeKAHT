# DeKAHT: Data-Efficient Kolmogorov-Arnold Hierarchical Transformer

Official PyTorch implementation of **DeKAHT**, a hierarchical vision transformer designed to improve **data efficiency** by replacing standard feed-forward networks with **Grouped Nonlinear KAN (GN-KAN)** modules.

This repository accompanies the paper:

**DeKAHT: Data-Efficient Kolmogorov-Arnold Hierarchical Transformer**  
(Under review at IEEE ICIP 2026)

---

# ⚠️ Important Naming Clarification

This codebase was developed using earlier internal naming conventions.

| Paper Name | Code Name |
|------------|-----------|
| **DeKAHT** | **swkat** |
| **GN-KAN** | **GRKAN** |

Therefore:

- `dekaht.py` implements the **DeKAHT architecture**
- `GRKAN` modules inside the code correspond to **GN-KAN modules described in the paper**

These naming differences are purely internal and **do not affect the methodology described in the manuscript**.

---

# 📂 Repository Structure

```
.
├── KAN_Act_variant/
│   ├── dekaht.py
│   └── train.py
│
├── gelu_variant/
│   ├── dekaht.py
│   └── train.py
│
├── hybrid_variants_for_ablation/
│   ├── swmsa_mlp_wmsa_kan_hybrid_DEKAHT/
│   │   ├── dekaht_hybrid.py
│   │   └── train_dekaht_hybrid.py
│   │
│   ├── wmsa_mlp_swmsa_kan_hybrid_DEKAHT/
│   │   ├── dekaht_hybrid_wmsa_kan.py
│   │   └── train_dekaht_hybrid.py
│
├── ablation_results.md
├── example_run.sh
├── requirements.txt
└── README.md
```

---

# 🧠 Implemented Variants

## 1️⃣ GELU Variant

Location:

```
gelu_variant/
```

Implements:

**GN-KAN-GELU**

Used as the **stable baseline variant**, especially for larger datasets.

---

## 2️⃣ KANAct Variant

Location:

```
KAN_Act_variant/
```

Implements:

**GN-KAN-KANAct**

Uses spline-based nonlinear activations inspired by Kolmogorov–Arnold Networks.

---

## 3️⃣ Hybrid Variants (Ablation Study)

Location:

```
hybrid_variants_for_ablation/
```

These variants evaluate the effect of GN-KAN placement.

### Hybrid Variant A

```
swmsa_mlp_wmsa_kan_hybrid_DEKAHT/
```

Configuration:

```
SW-MSA → MLP  
W-MSA → GN-KAN
```

---

### Hybrid Variant B

```
wmsa_mlp_swmsa_kan_hybrid_DEKAHT/
```

Configuration:

```
W-MSA → MLP  
SW-MSA → GN-KAN
```

All hybrid results are documented in:

```
ablation_results.md
```

---

# ⚙️ Installation

Create environment:

```bash
conda create -n dekaht python=3.10
conda activate dekaht
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install GPU-enabled PyTorch according to your CUDA version:

https://pytorch.org/get-started/locally/

---

# 🚀 Training Examples

## GELU Variant Training

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
---

## KANAct Variant Training

Use same training recipe as gelu variant, but suggested to use smaller batch sizes between 8 to 128 as KAN activation is very unstable.

---

## Hybrid Variant Training (Example)

Same as gelu varient.

---

# 📊 Ablation Results

All experimental results are summarized in:

```
ablation_results.md
```

This includes:

- GELU vs KANAct comparison  
- Group size analysis (G sweep)  
- Hybrid placement ablations  
- DeiT baseline comparison  
- Multi-dataset evaluations  
- Stability analysis  

This file is continuously updated with new results.

---

# 📂 Datasets Used

Experiments were conducted using:

- ImageNet-10  
- ImageNet-100  
- Brain Tumor MRI (4-class)  
- Brain Tumor MRI (44-class)

Medical datasets were aggregated from publicly available sources including:

- Figshare Brain Tumor Dataset  
- SARTAJ Brain MRI Dataset  
- Br35H Dataset  

Dataset preparation scripts and links will be released after paper acceptance.

---

# 🧠 Method Overview

DeKAHT replaces the standard feed-forward network (FFN) in hierarchical transformers with a **Grouped Nonlinear KAN (GN-KAN)** module.

The GN-KAN module consists of:

- Channel expansion  
- Nonlinear transformation  
- Channel grouping  
- Depthwise spatial convolution  
- Gated recomposition  
- Projection layer  

This improves:

- Nonlinear diversity  
- Feature separability  
- Data efficiency  

while maintaining:

- Comparable parameter count  
- Comparable computational cost  

relative to baseline transformer architectures.

---

# 🔬 Reproducibility

All experiments:

- Trained from scratch  
- Used consistent preprocessing  
- Used identical hyperparameters  

Additional:

- Hybrid placement studies included  
- Group size ablations included  
- DeiT comparisons included  

All experimental summaries are documented in:

```
ablation_results.md
```

---

# 📦 Model Weights

Pretrained model weights will be released after paper acceptance.

---

# 📬 Contact

**Gurram Harshamanya Thilak**  
Indian Institute of Technology Patna  

For research-related inquiries, please contact via institutional email.

---

# ⚠️ Disclaimer

This repository is released strictly for **academic and research purposes**.
