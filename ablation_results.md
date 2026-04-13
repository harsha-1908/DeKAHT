# Ablation Results and Extended Comparisons

This file contains additional experimental results requested during the review process.  
All values correspond to **maximum validation performance** achieved during training.

Metrics reported:

- Accuracy
- F1-score
- Precision
- Recall

---

# Comprehensive Performance Comparison (Extended with DeiT)

This table extends the original manuscript results by including **DeiT baseline comparisons** under identical training settings.

---

## ImageNet-10

| Dataset | Model | DeKAHT_KANAct |  |  |  | DeKAHT_GELU |  |  |  | DeiT |  |  |  | Swin |  |  |  |
|---------|-------|----------------|----|----|----|--------------|----|----|----|------|----|----|----|------|----|----|----|
|         |       | Acc | F1 | Prec | Rec | Acc | F1 | Prec | Rec | Acc | F1 | Prec | Rec | Acc | F1 | Prec | Rec |

| ImageNet-10 | Tiny | **90.80** | **0.907** | **0.910** | **0.908** | 88.60 | 0.885 | 0.888 | 0.886 | 81.00 | 0.808 | 0.814 | 0.810 | 85.25 | 0.852 | 0.855 | 0.852 |
| ImageNet-10 | Small | **90.60** | **0.905** | **0.906** | **0.906** | 84.80 | 0.850 | 0.856 | 0.848 | 87.20 | 0.872 | 0.875 | 0.872 | 69.80 | 0.689 | 0.691 | 0.698 |
| ImageNet-10 | Base | **90.00** | **0.898** | **0.903** | **0.900** | 84.00 | 0.839 | 0.843 | 0.840 | 86.60 | 0.865 | 0.869 | 0.866 | 60.85 | 0.594 | 0.598 | 0.608 |

---

## Brain Tumor (4-class)

| Dataset | Model | DeKAHT_KANAct |  |  |  | DeKAHT_GELU |  |  |  | DeiT |  |  |  | Swin |  |  |  |
|---------|-------|----------------|----|----|----|--------------|----|----|----|------|----|----|----|------|----|----|----|
|         |       | Acc | F1 | Prec | Rec | Acc | F1 | Prec | Rec | Acc | F1 | Prec | Rec | Acc | F1 | Prec | Rec |

| Brain Tumor (4) | Tiny | 99.39 | 0.994 | 0.994 | 0.993 | **99.85** | **0.998** | **0.998** | **0.998** | 87.03 | 0.863 | 0.865 | 0.864 | 81.39 | 0.800 | 0.810 | 0.805 |
| Brain Tumor (4) | Small | 99.24 | 0.992 | 0.992 | 0.992 | **99.85** | **0.999** | **0.999** | **0.998** | 89.93 | 0.894 | 0.895 | 0.893 | 63.84 | 0.609 | 0.625 | 0.617 |
| Brain Tumor (4) | Base | 81.62 | 0.798 | 0.816 | 0.806 | **100.00** | **1.000** | **1.000** | **1.000** | 91.30 | 0.908 | 0.909 | 0.908 | 51.33 | 0.436 | 0.400 | 0.500 |

---

## Brain Tumor (44-class)

| Dataset | Model | DeKAHT_KANAct |  |  |  | DeKAHT_GELU |  |  |  | DeiT |  |  |  | Swin |  |  |  |
|---------|-------|----------------|----|----|----|--------------|----|----|----|------|----|----|----|------|----|----|----|
|         |       | Acc | F1 | Prec | Rec | Acc | F1 | Prec | Rec | Acc | F1 | Prec | Rec | Acc | F1 | Prec | Rec |

| Brain Tumor (44) | Tiny | 92.02 | 0.907 | 0.935 | 0.895 | **97.21** | **0.946** | **0.940** | **0.962** | 66.89 | 0.542 | 0.686 | 0.510 | 65.46 | 0.520 | 0.580 | 0.507 |
| Brain Tumor (44) | Small | 87.10 | 0.830 | 0.876 | 0.812 | **95.63** | **0.933** | **0.956** | **0.927** | 79.78 | 0.740 | 0.796 | 0.715 | 40.77 | 0.245 | 0.310 | 0.250 |
| Brain Tumor (44) | Base | 83.61 | 0.778 | 0.812 | 0.778 | **96.07** | **0.940** | **0.952** | **0.937** | 85.36 | 0.825 | 0.863 | 0.808 | 12.02 | 0.012 | 0.007 | 0.038 |

---

## ImageNet-100 (DeiT Baseline)

| Dataset | Model | DeiT |  |  |  |
|---------|-------|------|----|----|----|
|         |       | Acc | F1 | Prec | Rec |

| ImageNet-100 | Tiny | 76.96 | 0.766 | 0.769 | 0.769 |
| ImageNet-100 | Small | 78.40 | 0.782 | 0.786 | 0.784 |
| ImageNet-100 | Base | 79.00 | 0.787 | 0.791 | 0.790 |

---

# Notes

- All values correspond to **maximum validation performance**
- Training conditions were kept identical across models
- DeiT models were trained using identical pipelines
- Results directly support fair comparison and reproducibility
