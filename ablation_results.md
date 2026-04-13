# Ablation Results and Additional Experiments

This file contains additional experimental results requested during the review process.  
All values correspond to **maximum validation performance** achieved during training.

Metrics reported:

- Accuracy
- F1-score
- Precision
- Recall

---

# 1️⃣ DeiT Baseline Comparison

This experiment evaluates the performance of standard DeiT models under identical training settings used for DeKAHT.

---

## ImageNet-10

| Model | Accuracy | F1 | Precision | Recall |
|------|----------|----|-----------|--------|
| DeiT-Tiny | 81.00 | 0.808 | 0.814 | 0.810 |
| DeiT-Small | 87.20 | 0.872 | 0.875 | 0.872 |
| DeiT-Base | 86.60 | 0.865 | 0.869 | 0.866 |

---

## ImageNet-100

| Model | Accuracy | F1 | Precision | Recall |
|------|----------|----|-----------|--------|
| DeiT-Tiny | 76.96 | 0.766 | 0.769 | 0.769 |
| DeiT-Small | 78.40 | 0.782 | 0.786 | 0.784 |
| DeiT-Base | 79.00 | 0.787 | 0.791 | 0.790 |

---

## Brain Tumor (4-class)

| Model | Accuracy | F1 | Precision | Recall |
|------|----------|----|-----------|--------|
| DeiT-Tiny | 87.03 | 0.863 | 0.865 | 0.864 |
| DeiT-Small | 89.93 | 0.894 | 0.895 | 0.893 |
| DeiT-Base | 91.30 | 0.908 | 0.909 | 0.908 |

---

## Brain Tumor (44-class)

| Model | Accuracy | F1 | Precision | Recall |
|------|----------|----|-----------|--------|
| DeiT-Tiny | 66.89 | 0.542 | 0.686 | 0.510 |
| DeiT-Small | 79.78 | 0.740 | 0.796 | 0.715 |
| DeiT-Base | 85.36 | 0.825 | 0.863 | 0.808 |

---

# 2️⃣ Group Size Ablation (G Sweep)

This experiment evaluates the impact of **channel grouping size (G)** on model performance.

Dataset:

```
ImageNet-10
```

Variant:

```
DeKAHT GN-KAN + GELU
```

---

## Tiny Model

| Group Size | Accuracy | F1 | Precision | Recall |
|-------------|----------|----|-----------|--------|
| G=4 | 89.00 | 0.891 | 0.893 | 0.892 |
| G=16 | 89.00 | 0.891 | 0.893 | 0.892 |
| G=32 | 89.00 | 0.889 | 0.892 | 0.890 |

---

## Small Model

| Group Size | Accuracy | F1 | Precision | Recall |
|-------------|----------|----|-----------|--------|
| G=4 | 90.00 | 0.895 | 0.896 | 0.896 |
| G=16 | 90.00 | 0.900 | 0.901 | 0.902 |
| G=32 | 91.00 | 0.911 | 0.912 | 0.912 |

---

## Base Model

| Group Size | Accuracy | F1 | Precision | Recall |
|-------------|----------|----|-----------|--------|
| G=4 | 92.00 | 0.918 | 0.920 | 0.920 |
| G=16 | 81.00 | 0.809 | 0.820 | 0.808 |
| G=32 | 92.00 | 0.919 | 0.921 | 0.920 |

---

# 3️⃣ Notes on Experimental Protocol

All additional experiments:

- Used identical training pipelines
- Used consistent preprocessing
- Used identical hyperparameters
- Were trained from scratch
- Used identical augmentation policies

Performance values correspond to:

```
Maximum validation performance
(best epoch)
```

This matches the reporting methodology used in the main manuscript.

---

# 4️⃣ Additional Experiments

The following additional experiments are currently included or planned:

- Hybrid placement ablation (W-MSA vs SW-MSA GN-KAN placement)
- GELU vs KANAct stability comparison
- Multi-seed evaluation analysis
- Extended cross-dataset validation

These experiments further support the robustness of the proposed architecture.

---
