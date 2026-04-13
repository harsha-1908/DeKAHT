# Ablation Results and Extended Experimental Analysis

This document provides additional experimental results requested during the review process.

All reported values correspond to:

**Maximum validation performance (best epoch).**

Metrics reported:

- Accuracy  
- F1-score  
- Precision  
- Recall  

All models were trained under identical experimental conditions.

---

# 1️⃣ Comprehensive Performance Comparison Across Datasets

(Extended with DeiT baseline)

---

# ImageNet-10

<table>
<thead>

<tr>
<th rowspan="2">Model</th>

<th colspan="4">DeKAHT_GN-KAN-KANAct (ours)</th>
<th colspan="4">DeKAHT_GN-KAN+GELU (ours)</th>
<th colspan="4">DeiT</th>
<th colspan="4">Swin</th>

</tr>

<tr>

<th>Acc</th><th>F1</th><th>Prec</th><th>Rec</th>
<th>Acc</th><th>F1</th><th>Prec</th><th>Rec</th>
<th>Acc</th><th>F1</th><th>Prec</th><th>Rec</th>
<th>Acc</th><th>F1</th><th>Prec</th><th>Rec</th>

</tr>

</thead>

<tbody>

<tr>
<td>Tiny</td>
<td><b>90.80</b></td><td><b>0.907</b></td><td><b>0.910</b></td><td><b>0.908</b></td>
<td>88.60</td><td>0.885</td><td>0.888</td><td>0.886</td>
<td>81.00</td><td>0.808</td><td>0.814</td><td>0.810</td>
<td>85.25</td><td>0.852</td><td>0.855</td><td>0.852</td>
</tr>

<tr>
<td>Small</td>
<td><b>90.60</b></td><td><b>0.905</b></td><td><b>0.906</b></td><td><b>0.906</b></td>
<td>84.80</td><td>0.850</td><td>0.856</td><td>0.848</td>
<td>87.20</td><td>0.872</td><td>0.875</td><td>0.872</td>
<td>69.80</td><td>0.689</td><td>0.691</td><td>0.698</td>
</tr>

<tr>
<td>Base</td>
<td><b>90.00</b></td><td><b>0.898</b></td><td><b>0.903</b></td><td><b>0.900</b></td>
<td>84.00</td><td>0.839</td><td>0.843</td><td>0.840</td>
<td>86.60</td><td>0.865</td><td>0.869</td><td>0.866</td>
<td>60.85</td><td>0.594</td><td>0.598</td><td>0.608</td>
</tr>

<tr>
<td>Large</td>
<td><b>88.80</b></td><td><b>0.888</b></td><td><b>0.893</b></td><td><b>0.888</b></td>
<td>84.00</td><td>0.841</td><td>0.845</td><td>0.840</td>
<td>—</td><td>—</td><td>—</td><td>—</td>
<td>67.05</td><td>0.673</td><td>0.678</td><td>0.680</td>
</tr>

</tbody>
</table>

---

# ImageNet-100

<table>
<thead>

<tr>

<th rowspan="2">Model</th>

<th colspan="4">KANAct</th>
<th colspan="4">GELU</th>
<th colspan="4">Swin</th>

</tr>

<tr>

<th>Acc</th><th>F1</th><th>Prec</th><th>Rec</th>
<th>Acc</th><th>F1</th><th>Prec</th><th>Rec</th>
<th>Acc</th><th>F1</th><th>Prec</th><th>Rec</th>

</tr>

</thead>

<tbody>

<tr>
<td>Tiny</td>
<td>81.28</td><td>0.811</td><td>0.817</td><td>0.813</td>
<td><b>82.40</b></td><td><b>0.820</b></td><td><b>0.824</b></td><td><b>0.824</b></td>
<td>80.20</td><td>0.798</td><td>0.803</td><td>0.802</td>
</tr>

<tr>
<td>Small</td>
<td>81.40</td><td>0.812</td><td>0.817</td><td>0.814</td>
<td>83.60</td><td>0.829</td><td>0.837</td><td>0.834</td>
<td><b>84.94</b></td><td><b>0.846</b></td><td><b>0.851</b></td><td><b>0.850</b></td>
</tr>

<tr>
<td>Base</td>
<td>50.66</td><td>0.499</td><td>0.534</td><td>0.507</td>
<td><b>84.74</b></td><td><b>0.846</b></td><td><b>0.851</b></td><td><b>0.848</b></td>
<td>84.69</td><td>0.845</td><td>0.851</td><td>0.847</td>
</tr>

</tbody>
</table>

---

# 2️⃣ Hybrid GN-KAN Placement Ablation

(W-MSA vs SW-MSA)

---

# ImageNet-100

<table>

<thead>

<tr>

<th rowspan="2">Model</th>
<th colspan="4">W-MSA-KAN</th>
<th colspan="4">SW-MSA-KAN</th>

</tr>

<tr>

<th>Acc</th><th>F1</th><th>Prec</th><th>Rec</th>
<th>Acc</th><th>F1</th><th>Prec</th><th>Rec</th>

</tr>

</thead>

<tbody>

<tr>
<td>Tiny</td>
<td><b>81.32</b></td><td><b>0.810</b></td><td><b>0.817</b></td><td><b>0.813</b></td>
<td>80.36</td><td>0.799</td><td>0.804</td><td>0.804</td>
</tr>

<tr>
<td>Small</td>
<td><b>83.04</b></td><td><b>0.828</b></td><td><b>0.835</b></td><td><b>0.830</b></td>
<td>81.88</td><td>0.815</td><td>0.822</td><td>0.819</td>
</tr>

<tr>
<td>Base</td>
<td><b>83.12</b></td><td><b>0.828</b></td><td><b>0.837</b></td><td><b>0.831</b></td>
<td>82.52</td><td>0.821</td><td>0.825</td><td>0.825</td>
</tr>

</tbody>

</table>

---

# 3️⃣ Group Size Ablation (G)

Dataset: **ImageNet-10**

Default configuration:

```
G = 8
```

---

## Tiny

<table>

<thead>

<tr>
<th rowspan="2">Groups</th>
<th colspan="4">Performance</th>
</tr>

<tr>
<th>Acc</th>
<th>F1</th>
<th>Precision</th>
<th>Recall</th>
</tr>

</thead>

<tbody>

<tr>
<td>G=4</td>
<td>89.00</td>
<td>0.891</td>
<td>0.893</td>
<td>0.892</td>
</tr>

<tr>
<td><b>G=8 (default)</b></td>
<td><b>90.80</b></td>
<td><b>0.907</b></td>
<td><b>0.910</b></td>
<td><b>0.908</b></td>
</tr>

<tr>
<td>G=16</td>
<td>89.00</td>
<td>0.891</td>
<td>0.893</td>
<td>0.892</td>
</tr>

<tr>
<td>G=32</td>
<td>89.00</td>
<td>0.889</td>
<td>0.892</td>
<td>0.890</td>
</tr>

</tbody>

</table>

---

# Experimental Notes

All experiments:

- Used identical training pipelines  
- Used identical hyperparameters  
- Used identical augmentations  
- Were trained from scratch  
- Used consistent evaluation metrics  

Performance corresponds to:

**Best validation epoch**

---
