#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_imagenet100.py
# Enhanced training script: schedules, EMA, SAM, SWA, resolution ramping, robust AMP handling.
# Enforces that SwkatBlockOpt uses GR_KAN_Conv (100% KAN) before training begins when requested.

import argparse
import os
import time
from pathlib import Path
import math
import random
import copy
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, datasets

from torch.optim.lr_scheduler import LambdaLR

import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, top_k_accuracy_score, precision_score, recall_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# AMP
from torch.cuda.amp import GradScaler, autocast

# optional extras
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# import model entrypoint and types
from dekaht_hybrid_wmsa_kan import SwkatGRKAN_Opt, set_prefer_kat, GR_KAN_Conv, SwkatBlockOpt

# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="root dir containing train/ and val/ subfolders")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--model-variant", type=str, default="swkat-tiny",
                   choices=["swkat-tiny","swkat-small","swkat-base","swkat-large","custom"],
                   help="Model variant to instantiate. Use 'custom' to keep explicit embed_dim/depths args.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64, help="Per-process (per-GPU) batch size when using DDP")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-dir", default="out")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--save-attention-samples", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument('--opt', type=str, default='adamw', choices=['adamw','sgd'], help='optimizer: adamw or sgd')
    p.add_argument('--warmup-epochs', type=int, default=5)
    # scheduling & reg
    p.add_argument('--mixup-alpha', type=float, default=0.8)
    p.add_argument('--mixup-start', type=int, default=0, help='epoch to start mixup')
    p.add_argument('--mixup-end', type=int, default=150, help='epoch to fade mixup to zero (linear)')
    p.add_argument('--cutmix-alpha', type=float, default=1.0)
    p.add_argument('--cutmix-start', type=int, default=0)
    p.add_argument('--cutmix-end', type=int, default=150)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--ls-start', type=int, default=0)
    p.add_argument('--ls-end', type=int, default=150)
    p.add_argument('--rand-augment', type=str, default="N2 M9")
    p.add_argument('--ra-start', type=int, default=0)
    p.add_argument('--ra-end', type=int, default=120)
    # EMA, SWA, SAM
    p.add_argument('--use-ema', action='store_true')
    p.add_argument('--ema-decay', type=float, default=0.9999)
    p.add_argument('--use-swa', action='store_true')
    p.add_argument('--swa-start', type=int, default=280)
    p.add_argument('--swa-lr', type=float, default=1e-5)
    p.add_argument('--use-sam', action='store_true', help='Use SAM optimizer (recommended with adamw/sgd)')
    # resolution ramp-up
    p.add_argument('--ramp-epochs', type=int, default=30, help='resolution ramp-up epochs (crop from small to img-size)')
    p.add_argument('--min-res', type=int, default=160, help='minimum crop size at epoch 0')
    # distributed flag (script-level)
    p.add_argument('--distributed', action='store_true', help='Set if running under torch.distributed (auto when using torchrun)')
    # patch size (used when resizing pos-embeddings)
    p.add_argument('--patch-size', type=int, default=4, help='patch size used in the model (needed for pos-embed resizing)')
    # model custom params (only used if model-variant == custom)
    p.add_argument('--embed-dim', type=int, default=96)
    p.add_argument('--depths', type=str, default="2,2,6,2", help='comma separated depths for each stage when custom')
    p.add_argument('--num-heads', type=str, default="3,6,12,24", help='comma separated num_heads for each stage when custom')
    # KAT toggle
    p.add_argument('--use-kat', action='store_true', help='If set, attempt to enable compiled KAT operator (best-effort).')
    # <-- ADDED: single "checkbox" to require the custom pure-PyTorch KAN (GR_KAN_Conv)
    p.add_argument('--use-custom-kan', action='store_true', help='Require that SwkatBlockOpt uses custom pure-PyTorch GR_KAN_Conv (100% KAN). If set, script will fail early if any block does not use GR_KAN_Conv.')
    # gradient clipping
    p.add_argument('--grad-clip', type=float, default=0.0, help='max grad norm (0 = disabled)')
    # model statistics/throughput measurement
    p.add_argument('--measure-stats', action='store_true', help='Measure params / FLOPs / throughput before training')
    p.add_argument('--throughput-batch', type=int, default=16, help='Batch size for throughput measurement (synthetic data)')
    p.add_argument('--throughput-iters', type=int, default=30, help='Iters for throughput measurement (warmup excluded)')
    # logging / niceties
    p.add_argument('--log-interval', type=int, default=200, help='Batches between progress logs')
    p.add_argument('--log-to-file', action='store_true', help='Also write logs to a file in output-dir')
    # small conveniences
    p.add_argument('--no-amp', action='store_true', help='Disable automatic mixed precision (AMP)')
    p.add_argument('--local_rank', type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help='Local rank passed by torchrun')
    # validation policy: model, ema, or both
    p.add_argument('--validate-with', type=str, default='model', choices=['model','ema','both'],
                   help="Which weights to run validation with: 'model' (live model), 'ema' (ema model), or 'both' (run both and log)")
    return p.parse_args()

# -----------------------------
def make_train_transform(img_size, rand_augment_spec=None):
    tlist = [transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip()]
    if rand_augment_spec:
        try:
            from torchvision.transforms import RandAugment
            toks = rand_augment_spec.strip().split()
            if len(toks) == 2 and toks[0].startswith("N") and toks[1].startswith("M"):
                n = int(toks[0][1:]); m = int(toks[1][1:])
                tlist.append(RandAugment(num_ops=n, magnitude=m))
            else:
                tlist.append(RandAugment())
        except Exception:
            pass
    tlist += [transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    return transforms.Compose(tlist)

def make_val_transform(img_size):
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def make_datasets(data_root, train_img_size, val_img_size, rand_augment_spec=None):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise RuntimeError(f"Expected train/ and val/ under {data_root}. Found: {os.listdir(data_root) if os.path.isdir(data_root) else 'data root missing'}")
    train_ds = datasets.ImageFolder(train_dir, transform=make_train_transform(train_img_size, rand_augment_spec))
    val_ds = datasets.ImageFolder(val_dir, transform=make_val_transform(val_img_size))
    return train_ds, val_ds

def make_loaders(train_ds, val_ds, batch_size, workers, is_distributed=False):
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=workers,
                                  pin_memory=True,
                                  persistent_workers=(workers>0),
                                  prefetch_factor=2 if workers>0 else 2)
        val_loader = DataLoader(val_ds,
                                batch_size=batch_size,
                                sampler=val_sampler,
                                num_workers=max(1, workers // 2),
                                pin_memory=True,
                                persistent_workers=(workers>0),
                                prefetch_factor=2 if workers>0 else 2)
    else:
        if workers > 0:
            train_loader = DataLoader(train_ds,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=workers,
                                      pin_memory=True,
                                      persistent_workers=True,
                                      prefetch_factor=2)
            val_loader = DataLoader(val_ds,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=max(1, workers // 2),
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=2)
        else:
            train_loader = DataLoader(train_ds,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=True)
            val_loader = DataLoader(val_ds,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True)
    return train_loader, val_loader

# -----------------------------
def save_checkpoint(state, outdir, name="checkpoint_last.pth", is_main=True):
    if not is_main:
        return None
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fp = os.path.join(outdir, name)
    torch.save(state, fp)
    return fp

def compute_confusion_and_per_class_stats(y_true, y_pred, class_names, out_dir, epoch):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class_acc = {}
    for i, cname in enumerate(class_names):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        per_class_acc[cname] = {"class_index": i, "correct": correct, "total": total,
                                "accuracy": float(correct / total) if total > 0 else float("nan")}
    per_class_df = pd.DataFrame.from_dict(per_class_acc, orient="index")
    per_class_csv = os.path.join(out_dir, f"per_class_accuracy_epoch{epoch:03d}.csv")
    per_class_df.to_csv(per_class_csv)

    cm_path = os.path.join(out_dir, "confusion_matrices"); os.makedirs(cm_path, exist_ok=True)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_pct = np.divide(cm, row_sums, where=row_sums!=0) * 100.0

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cm_pct, annot=False, cmap="viridis", ax=ax, vmin=0, vmax=100)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (percent) - epoch {epoch:03d}")
    n_labels = len(class_names)
    for i in range(n_labels):
        for j in range(n_labels):
            count = int(cm[i, j]); pct = cm_pct[i, j]
            txt = f"{count}\n{pct:4.1f}%"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                    color="white" if pct > 40 else "black", fontsize=7)
    ax.set_xticks(np.arange(n_labels)+0.5); ax.set_yticks(np.arange(n_labels)+0.5)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6); ax.set_yticklabels(class_names, rotation=0, fontsize=7)
    fig.tight_layout()
    out_png = os.path.join(cm_path, f"confusion_epoch{epoch:03d}.png"); fig.savefig(out_png, dpi=150); plt.close(fig)
    np.save(os.path.join(cm_path, f"confusion_epoch{epoch:03d}.npy"), cm)
    return per_class_csv, out_png

def save_attention_maps(model, samples, out_dir, epoch, device):
    attn_maps = []; handles = []
    def make_hook(name):
        def hook(module, input, output):
            try:
                cand = output[0] if isinstance(output, tuple) else output
                arr = cand.detach().cpu().numpy()
                attn_maps.append((name, arr))
            except Exception:
                pass
        return hook
    for nm, mod in model.named_modules():
        if "attn" in nm.lower() or "attention" in nm.lower():
            try: handles.append(mod.register_forward_hook(make_hook(nm.replace(" ", "_"))))
            except Exception: pass

    model.eval()
    with torch.no_grad():
        if len(samples) == 0: pass
        else:
            inputs = torch.stack([s[0] for s in samples]).to(device)
            _ = model(inputs)
    att_out_dir = os.path.join(out_dir, "attention_maps"); os.makedirs(att_out_dir, exist_ok=True)
    saved = []
    for (name, att) in attn_maps[:32]:
        avg = None
        try:
            if att.ndim >= 4: avg = att.mean(axis=0).mean(axis=0)
            elif att.ndim == 3: avg = att.mean(axis=0)
            elif att.ndim == 2: avg = att
            else: avg = att.reshape(att.shape[0], -1).mean(axis=0)
        except Exception: continue
        try:
            fig, ax = plt.subplots(figsize=(4,4)); sns.heatmap(avg, ax=ax, cbar=False); ax.set_title(name[:40])
            fname = f"att_{epoch:03d}_{name.replace('/', '_')[:80]}.png"; fp = os.path.join(att_out_dir, fname)
            fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig); saved.append(fp)
        except Exception: continue
    for h in handles:
        try: h.remove()
        except Exception: pass
    return saved

# -----------------------------
def rand_bbox(size, lam):
    H = size[2]; W = size[3]
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return y1, x1, y2, x2

def mixup_data(x, y, alpha=1.0, device=None):
    if alpha <= 0: return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    if device is None:
        device = x.device
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam

def cutmix_data(x, y, alpha=1.0, device=None):
    if alpha <= 0: return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    if device is None:
        device = x.device
    rand_index = torch.randperm(batch_size).to(device)
    y_a = y; y_b = y[rand_index]
    y1, x1, y2, x2 = rand_bbox(x.size(), lam)
    if y2 > y1 and x2 > x1:
        x[:, :, y1:y2, x1:x2] = x[rand_index, :, y1:y2, x1:x2]
    area = (x2 - x1) * (y2 - y1)
    total = float(x.size(-1) * x.size(-2))
    lam = 1 - (area / total) if total > 0 else lam
    return x, (y_a, y_b), lam

def soft_cross_entropy(pred, soft_targets):
    logp = F.log_softmax(pred, dim=1)
    loss = - (soft_targets * logp).sum(dim=1).mean()
    return loss

def make_smoothed_targets(labels, n_classes, smoothing=0.0, device=None):
    assert 0.0 <= smoothing < 1.0
    with torch.no_grad():
        B = labels.size(0)
        if device is None:
            device = labels.device
        off_value = smoothing / (n_classes - 1) if n_classes > 1 else 0.0
        one_hot = torch.full((B, n_classes), off_value, device=device, dtype=torch.float32)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    return one_hot

# -----------------------------
def create_ema_model(model):
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters(): p.requires_grad_(False)
    return ema_model

def update_ema(ema_model, model, decay):
    with torch.no_grad():
        msd = model.state_dict()
        esd = ema_model.state_dict()
        for k in esd.keys():
            if k not in msd:
                continue
            tgt = esd[k]
            src = msd[k]
            if not torch.is_floating_point(tgt):
                continue
            try:
                src_cast = src.to(tgt.dtype).to(tgt.device)
            except Exception:
                continue
            try:
                tgt.mul_(decay).add_(src_cast, alpha=(1.0 - decay))
            except Exception:
                try:
                    new = tgt * float(decay) + src_cast * float(1.0 - decay)
                    esd[k] = new.to(tgt.dtype).to(tgt.device)
                except Exception:
                    continue

# -----------------------------
def sam_first_step(model, rho=0.05, eps=1e-12):
    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        try:
            gn = float(g.norm().item())
            grad_norm_sq += gn * gn
        except Exception:
            try:
                grad_norm_sq += float((g ** 2).sum().item())
            except Exception:
                pass
    grad_norm = math.sqrt(max(grad_norm_sq, eps))
    scale = rho / grad_norm
    for p in model.parameters():
        if p.grad is None:
            p._sam_e_w = None
            continue
        e_w = (p.grad.detach() * scale)
        try:
            e_w = e_w.to(p.data.dtype).to(p.data.device)
        except Exception:
            try:
                e_w = e_w.to(p.data.device)
            except Exception:
                pass
        try:
            p._sam_e_w = e_w.clone()
        except Exception:
            p._sam_e_w = e_w
        with torch.no_grad():
            p.data = p.data + p._sam_e_w

def sam_second_step(model):
    for p in model.parameters():
        e_w = getattr(p, "_sam_e_w", None)
        if e_w is None:
            continue
        with torch.no_grad():
            try:
                p.data = p.data - e_w
            except Exception:
                try:
                    p.data = p.data - e_w.clone()
                except Exception:
                    pass
        try:
            del p._sam_e_w
        except Exception:
            try:
                p._sam_e_w = None
            except Exception:
                pass

# -----------------------------
def resize_pos_embed_if_needed(model, new_img_size, patch_size=4, device='cpu'):
    sd = model.state_dict()
    changed = False
    for key in list(sd.keys()):
        if ('pos_embed' in key or 'pos_embedding' in key) and sd[key].ndim == 3:
            pe = sd[key].to(device)
            N = pe.shape[1]; C = pe.shape[2]
            model_img_size = getattr(model, "img_size", None)
            if model_img_size is None:
                model_img_size = new_img_size
            expected_grid = (model_img_size // patch_size)
            if (expected_grid*expected_grid + 1) == N:
                cls_tok = pe[:, 0:1, :].clone()
                grid = pe[:, 1:, :].reshape(1, expected_grid, expected_grid, C).permute(0, 3, 1, 2)
            elif (expected_grid*expected_grid) == N:
                cls_tok = None
                grid = pe.reshape(1, expected_grid, expected_grid, C).permute(0, 3, 1, 2)
            else:
                possible = int(math.sqrt(N))
                if possible * possible == N:
                    cls_tok = None
                    grid = pe.reshape(1, possible, possible, C).permute(0, 3, 1, 2)
                else:
                    continue

            new_grid_size = new_img_size // patch_size
            if new_grid_size <= 0:
                continue
            with torch.no_grad():
                new_grid = F.interpolate(grid, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
                new_grid = new_grid.permute(0, 2, 3, 1).reshape(1, new_grid_size*new_grid_size, C).to(pe.dtype)
                if cls_tok is not None:
                    new_pe = torch.cat([cls_tok.to(pe.dtype), new_grid], dim=1)
                else:
                    new_pe = new_grid
                sd[key] = new_pe.to(sd[key].device)
                changed = True
                print(f"[pos-embed] resized key '{key}' -> grid {new_grid_size}x{new_grid_size} (cls_token={'yes' if cls_tok is not None else 'no'})")
                break
    if changed:
        model.load_state_dict(sd, strict=False)
    return changed

# -----------------------------
def _count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_model_flops_and_throughput(model, img_size, batch_size=8, device=None, iters=30, warmup=5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    flops = 0
    counts = {"conv":0, "linear":0}
    hooks = []

    def conv_hook(self, inp, out):
        nonlocal flops, counts
        x = inp[0]
        B = x.shape[0]
        _, C_in, H_in, W_in = x.shape
        out_h, out_w = out.shape[2], out.shape[3]
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // max(1, self.groups))
        macs = kernel_ops * self.out_channels * out_h * out_w * B
        flops += 2 * macs
        counts["conv"] += 1

    def linear_hook(self, inp, out):
        nonlocal flops, counts
        x = inp[0]
        B = x.shape[0]
        in_f = self.in_features
        out_f = self.out_features
        macs = in_f * out_f * B
        flops += 2 * macs
        counts["linear"] += 1

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    try:
        fake = torch.randn(1, 3, img_size, img_size, device=device)
        with torch.no_grad():
            _ = model(fake)
    except Exception:
        flops = float("nan")
    finally:
        for h in hooks:
            try: h.remove()
            except Exception: pass

    params = _count_params(model)

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    synthetic = torch.randn(batch_size, 3, img_size, img_size, device=device)
    label = torch.zeros(batch_size, dtype=torch.long, device=device)
    for _ in range(warmup):
        opt.zero_grad()
        out = model(synthetic)
        loss = F.cross_entropy(out, label)
        loss.backward()
        opt.step()
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad()
        out = model(synthetic)
        loss = F.cross_entropy(out, label)
        loss.backward()
        opt.step()
    t1 = time.time()
    secs = (t1 - t0) / max(1, iters)
    images_per_sec = batch_size / secs if secs > 0 else float("nan")

    model.eval()
    return {"params": params, "flops": flops, "throughput_ips": images_per_sec, "flops_conv_linear_count": counts}

def save_model_stats_to_csv(outdir, stats_row):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(outdir, "model_stats.csv")
    header = list(stats_row.keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(stats_row)
    return csv_path

# -----------------------------
def epoch_validate_and_collect(model, loader, device, num_classes, out_dir, epoch, save_attention_samples=0):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total = correct = 0; total_loss = 0.0
    y_true = []; y_pred = []; y_probs = []; attention_samples = []

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.no_grad():
            out = model(x)
            probs = torch.nn.functional.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            total += y.numel(); correct += (preds == y).sum().item()
            total_loss += float(loss_fn(out, y).item())
            y_true.extend(y.cpu().numpy().tolist()); y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.extend(probs.cpu().numpy().tolist())
            if save_attention_samples and len(attention_samples) < save_attention_samples:
                batch_take = min(x.size(0), save_attention_samples - len(attention_samples))
                for b in range(batch_take):
                    attention_samples.append((x[b].cpu(), int(y[b].cpu().item())))

    avg_loss = total_loss / max(1, total)
    top1 = correct / max(1, total)
    try:
        top5 = top_k_accuracy_score(y_true, np.array(y_probs), k=5, labels=list(range(num_classes)))
    except Exception:
        top5 = float("nan")
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_csv, cm_png = compute_confusion_and_per_class_stats(y_true, y_pred, loader.dataset.classes, out_dir, epoch)
    attention_files = save_attention_maps(model, attention_samples, out_dir, epoch, device) if save_attention_samples and len(attention_samples) else []
    np.savez_compressed(os.path.join(out_dir, f"val_logits_epoch{epoch:03d}.npz"),
                        y_true=np.array(y_true, dtype=np.int32),
                        y_probs=np.array(y_probs, dtype=np.float32))
    return {"val_loss": avg_loss, "val_top1": top1, "val_top5": top5, "val_f1": f1,
            "val_precision": precision, "val_recall": recall,
            "confusion_png": cm_png, "per_class_csv": per_class_csv, "attention_files": attention_files, "val_samples": total}

# -----------------------------
def apply_grad_clip(model, optimizer, args, scaler=None):
    """
    Applies gradient clipping to model.parameters() with args.grad_clip > 0.
    If scaler is provided, unscale the optimizer before clipping (required for AMP).
    Returns the clipped norm (float) or 0.0 if not applied.
    """
    if not hasattr(args, "grad_clip") or args.grad_clip is None or float(args.grad_clip) <= 0.0:
        return 0.0
    try:
        if scaler is not None:
            try:
                scaler.unscale_(optimizer)
            except Exception:
                pass
        max_norm = float(args.grad_clip)
        # clip_grad_norm_ returns the total norm before clipping (torch >=1.7)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        try:
            return float(total_norm)
        except Exception:
            return 0.0
    except Exception:
        return 0.0

def train_one_epoch(model, loader, opt, device, epoch, args, ema_model=None, logger=None, scaler=None, is_distributed=False, train_sampler=None):
    model.train()
    total_loss = 0.0; n = 0; correct = 0; total = 0
    y_true_all = []; y_pred_all = []

    start_time = time.time()
    last_log_time = start_time

    def sched_val(base, start, end, ep):
        if base <= 0: return 0.0
        if ep <= start: return base
        if ep >= end: return 0.0
        frac = (ep - start) / float(max(1, end - start))
        return base * (1.0 - frac)

    mix_alpha = sched_val(args.mixup_alpha, args.mixup_start, args.mixup_end, epoch)
    cut_alpha = sched_val(args.cutmix_alpha, args.cutmix_start, args.cutmix_end, epoch)
    ls_val = sched_val(args.label_smoothing, args.ls_start, args.ls_end, epoch)

    if is_distributed and train_sampler is not None:
        try:
            train_sampler.set_epoch(epoch)
        except Exception:
            pass

    total_batches = len(loader)
    for i, (x, y) in enumerate(loader):
        batch_t0 = time.time()
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        opt.zero_grad()

        amp_enabled = (not args.no_amp) and (scaler is not None)

        if cut_alpha and cut_alpha > 0:
            x_mixed, y_pair, lam = cutmix_data(x.clone(), y.clone(), alpha=cut_alpha, device=x.device)
            with autocast(enabled=amp_enabled):
                out = model(x_mixed)
                n_cls = out.size(1)
                if ls_val > 0:
                    t1 = make_smoothed_targets(y_pair[0], n_cls, smoothing=ls_val, device=x.device)
                    t2 = make_smoothed_targets(y_pair[1], n_cls, smoothing=ls_val, device=x.device)
                else:
                    t1 = F.one_hot(y_pair[0], num_classes=n_cls).float().to(x.device)
                    t2 = F.one_hot(y_pair[1], num_classes=n_cls).float().to(x.device)
                soft_targets = lam * t1 + (1.0 - lam) * t2
                loss = soft_cross_entropy(out, soft_targets)

            # Non-AMP path
            if scaler is None:
                loss.backward()
                # apply grad clip BEFORE SAM first step / BEFORE optimizer.step
                clip_norm = apply_grad_clip(model, opt, args, scaler=None)
                if args.use_sam:
                    sam_first_step(model, rho=0.05)
                    with autocast(enabled=amp_enabled):
                        out2 = model(x_mixed)
                        loss2 = soft_cross_entropy(out2, soft_targets)
                    opt.zero_grad(); loss2.backward()
                    # clip for second backward as well
                    _ = apply_grad_clip(model, opt, args, scaler=None)
                    sam_second_step(model)
                    opt.step()
                else:
                    opt.step()
            else:
                # AMP / GradScaler path
                scaler.scale(loss).backward()
                # unscale and clip if requested (before SAM first step)
                if args.grad_clip and args.grad_clip > 0.0:
                    try:
                        scaler.unscale_(opt)
                    except Exception:
                        pass
                    _ = apply_grad_clip(model, opt, args, scaler=None)
                if args.use_sam:
                    try:
                        scaler.unscale_(opt)
                    except Exception:
                        pass
                    sam_first_step(model, rho=0.05)
                    with autocast(enabled=amp_enabled):
                        out2 = model(x_mixed)
                        loss2 = soft_cross_entropy(out2, soft_targets)
                    opt.zero_grad()
                    scaler.scale(loss2).backward()
                    # unscale and clip second backward
                    if args.grad_clip and args.grad_clip > 0.0:
                        try:
                            scaler.unscale_(opt)
                        except Exception:
                            pass
                        _ = apply_grad_clip(model, opt, args, scaler=None)
                    sam_second_step(model)
                    scaler.step(opt)
                    scaler.update()
                else:
                    scaler.step(opt)
                    scaler.update()

            preds = out.argmax(dim=1)
            y_ref = y_pair[0] if lam >= 0.5 else y_pair[1]

        elif mix_alpha and mix_alpha > 0:
            x_mixed, y_pair, lam = mixup_data(x, y, alpha=mix_alpha, device=x.device)
            with autocast(enabled=amp_enabled):
                out = model(x_mixed)
                n_cls = out.size(1)
                if ls_val > 0:
                    t1 = make_smoothed_targets(y_pair[0], n_cls, smoothing=ls_val, device=x.device)
                    t2 = make_smoothed_targets(y_pair[1], n_cls, smoothing=ls_val, device=x.device)
                else:
                    t1 = F.one_hot(y_pair[0], num_classes=n_cls).float().to(x.device)
                    t2 = F.one_hot(y_pair[1], num_classes=n_cls).float().to(x.device)
                soft_targets = lam * t1 + (1.0 - lam) * t2
                loss = soft_cross_entropy(out, soft_targets)

            if scaler is None:
                loss.backward()
                clip_norm = apply_grad_clip(model, opt, args, scaler=None)
                if args.use_sam:
                    sam_first_step(model, rho=0.05)
                    with autocast(enabled=amp_enabled):
                        out2 = model(x_mixed)
                        loss2 = soft_cross_entropy(out2, soft_targets)
                    opt.zero_grad(); loss2.backward()
                    _ = apply_grad_clip(model, opt, args, scaler=None)
                    sam_second_step(model)
                    opt.step()
                else:
                    opt.step()
            else:
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0.0:
                    try:
                        scaler.unscale_(opt)
                    except Exception:
                        pass
                    _ = apply_grad_clip(model, opt, args, scaler=None)
                if args.use_sam:
                    try:
                        scaler.unscale_(opt)
                    except Exception:
                        pass
                    sam_first_step(model, rho=0.05)
                    with autocast(enabled=amp_enabled):
                        out2 = model(x_mixed)
                        loss2 = soft_cross_entropy(out2, soft_targets)
                    opt.zero_grad()
                    scaler.scale(loss2).backward()
                    if args.grad_clip and args.grad_clip > 0.0:
                        try:
                            scaler.unscale_(opt)
                        except Exception:
                            pass
                        _ = apply_grad_clip(model, opt, args, scaler=None)
                    sam_second_step(model)
                    scaler.step(opt)
                    scaler.update()
                else:
                    scaler.step(opt)
                    scaler.update()

            preds = out.argmax(dim=1)
            y_ref = y_pair[0] if lam >= 0.5 else y_pair[1]

        else:
            with autocast(enabled=amp_enabled):
                out = model(x)
                if ls_val and ls_val > 0:
                    n_cls = out.size(1)
                    soft_targets = make_smoothed_targets(y, n_cls, smoothing=ls_val, device=x.device)
                    loss = soft_cross_entropy(out, soft_targets)
                else:
                    loss = F.cross_entropy(out, y)

            if scaler is None:
                loss.backward()
                clip_norm = apply_grad_clip(model, opt, args, scaler=None)
                if args.use_sam:
                    sam_first_step(model, rho=0.05)
                    with autocast(enabled=amp_enabled):
                        out2 = model(x)
                        if ls_val and ls_val > 0:
                            loss2 = soft_cross_entropy(out2, soft_targets)
                        else:
                            loss2 = F.cross_entropy(out2, y)
                    opt.zero_grad(); loss2.backward()
                    _ = apply_grad_clip(model, opt, args, scaler=None)
                    sam_second_step(model)
                    opt.step()
                else:
                    opt.step()
            else:
                scaler.scale(loss).backward()
                # unscale + clip before SAM/step
                if args.grad_clip and args.grad_clip > 0.0:
                    try:
                        scaler.unscale_(opt)
                    except Exception:
                        pass
                    _ = apply_grad_clip(model, opt, args, scaler=None)
                if args.use_sam:
                    try:
                        scaler.unscale_(opt)
                    except Exception:
                        pass
                    sam_first_step(model, rho=0.05)
                    with autocast(enabled=amp_enabled):
                        out2 = model(x)
                        if ls_val and ls_val > 0:
                            loss2 = soft_cross_entropy(out2, soft_targets)
                        else:
                            loss2 = F.cross_entropy(out2, y)
                    opt.zero_grad()
                    scaler.scale(loss2).backward()
                    if args.grad_clip and args.grad_clip > 0.0:
                        try:
                            scaler.unscale_(opt)
                        except Exception:
                            pass
                        _ = apply_grad_clip(model, opt, args, scaler=None)
                    sam_second_step(model)
                    scaler.step(opt)
                    scaler.update()
                else:
                    scaler.step(opt)
                    scaler.update()

            preds = out.argmax(dim=1)
            y_ref = y

        if ema_model is not None:
            update_ema(ema_model, model, args.ema_decay)

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
        correct += (preds == y_ref).sum().item()
        total += y.numel()
        y_true_all.extend(y_ref.detach().cpu().numpy().tolist())
        y_pred_all.extend(preds.detach().cpu().numpy().tolist())

        if (i % args.log_interval == 0) or (i == total_batches - 1):
            now = time.time()
            elapsed = now - start_time
            batches_done = i + 1
            avg_time = elapsed / max(1, batches_done)
            remaining = max(0, total_batches - batches_done)
            eta_secs = remaining * avg_time
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_secs))
            batch_time = now - batch_t0
            grad_norm = 0.0
            try:
                if scaler is not None:
                    try:
                        scaler.unscale_(opt)
                    except Exception:
                        pass

                for p in model.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    try:
                        gn = float(g.norm().item())
                        grad_norm += gn * gn
                    except Exception:
                        try:
                            grad_norm += float((g ** 2).sum().item())
                        except Exception:
                            pass
                grad_norm = math.sqrt(max(0.0, grad_norm))
            except Exception:
                grad_norm = 0.0
            mem = 0.0
            if torch.cuda.is_available():
                try:
                    mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                except Exception:
                    mem = 0.0

            # FIX: ensure lr variable exists in this scope for logging
            lr = opt.param_groups[0].get("lr", getattr(args, "lr", 0.0))

            msg = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Train: [{epoch}/{args.epochs}][{i}/{total_batches}] "
                   f"eta {eta_str} lr {lr:.6e}  wd {args.weight_decay:.4f} time {batch_time:.4f} ({avg_time:.4f}) "
                   f"loss {loss.item():.4f} grad_norm {grad_norm:.4f} mem {mem:.0f}MB")
            print(msg)
            if logger:
                logger.write(msg + "\n"); logger.flush()

    avg_loss = total_loss / max(1, n)
    acc = correct / max(1, total)
    try:
        train_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
        train_precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
        train_recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    except Exception:
        train_f1 = train_precision = train_recall = float("nan")

    return avg_loss, acc, train_precision, train_recall, train_f1

# -----------------------------
def ensure_metrics_csv(metrics_csv):
    if os.path.exists(metrics_csv):
        try: return pd.read_csv(metrics_csv)
        except Exception: os.remove(metrics_csv)
    mh = ["epoch","train_loss","train_acc","train_precision","train_recall","train_f1",
          "val_loss","val_top1","val_top5","val_precision","val_recall","val_f1","lr","val_samples","time_s"]
    df = pd.DataFrame(columns=mh); df.to_csv(metrics_csv, index=False); return df

# -----------------------------
def safe_load_checkpoint(model, ck_path, optimizer=None, scheduler=None):
    ck = torch.load(ck_path, map_location="cpu")
    sd = ck.get("model", ck)
    msd = model.state_dict()
    filtered = {}
    mismatches = []
    for k, v in sd.items():
        if k in msd:
            if v.shape == msd[k].shape:
                filtered[k] = v
            else:
                mismatches.append(k)
        else:
            pass
    model.load_state_dict(filtered, strict=False)
    if mismatches:
        print("Warning: skipped loading keys with shape mismatch:", mismatches)
    if optimizer is not None and "optimizer" in ck:
        try: optimizer.load_state_dict(ck.get("optimizer"))
        except Exception: pass
    if scheduler is not None and "scheduler" in ck:
        try: scheduler.load_state_dict(ck.get("scheduler"))
        except Exception: pass
    return ck

# -----------------------------
def _cuda_compute_capability_ok(device_index=0, min_major=3, min_minor=7):
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability(device_index)
        if cap[0] > min_major:
            return True
        if cap[0] == min_major and cap[1] >= min_minor:
            return True
        return False
    except Exception:
        return False

# -----------------------------
def enforce_hybrid_rule(model, is_main=True):
    """
    Enforces hybrid rule:
      - W-MSA  (shift_size == 0)  -> Std_MLP_Conv
      - SW-MSA (shift_size > 0)   -> GR_KAN_Conv
    """
    problems = []

    for name, m in model.named_modules():
        if isinstance(m, SwkatBlockOpt):
            mlp = getattr(m, "mlp_conv", None)

            if mlp is None:
                problems.append(f"{name}: missing mlp_conv")
                continue

            if m.shift_size == 0:
                # Expect Std_MLP_Conv (NOT KAN)
                if hasattr(mlp, "is_kan") and getattr(mlp, "is_kan") is True:
                    problems.append(
                        f"{name}: shift_size=0 (W-MSA) but using GR_KAN_Conv (expected Std_MLP_Conv)"
                    )
            else:
                # Expect GR_KAN_Conv
                if not (hasattr(mlp, "is_kan") and getattr(mlp, "is_kan") is True):
                    problems.append(
                        f"{name}: shift_size>0 (SW-MSA) but using {type(mlp).__name__} (expected GR_KAN_Conv)"
                    )

    if problems:
        msg = "HYBRID enforcement failed:\n" + "\n".join(problems)
        if is_main:
            print(msg)
        raise RuntimeError(msg)
    else:
        if is_main:
            print("[HYBRID-ENFORCE] Verified: W-MSA?MLP, SW-MSA?GR_KAN.")
# -----------------------------
def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        try:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
        except Exception as e:
            print("Warning: failed to init process group:", e)
            is_distributed = False

    device = None
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        except Exception:
            try:
                device = torch.device("cuda")
            except Exception:
                device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    rank = 0
    if is_distributed:
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = local_rank
    is_main = (rank == 0)

    seed = args.seed + rank
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    if is_main:
        print("Running on device:", device, "WORLD_SIZE=", world_size, "LOCAL_RANK=", local_rank)

    amp_safe = False
    if (not args.no_amp) and torch.cuda.is_available():
        try:
            dev_idx = device.index if hasattr(device, "index") and device.index is not None else 0
            if _cuda_compute_capability_ok(dev_idx, min_major=3, min_minor=7):
                amp_safe = True
            else:
                if is_main:
                    print("[AMP] Detected CUDA device with compute capability below 3.7; disabling AMP/GradScaler for safety.")
        except Exception:
            amp_safe = False
    else:
        amp_safe = False

    scaler = None
    if amp_safe:
        try:
            scaler = GradScaler(enabled=True)
        except Exception:
            scaler = None

    torch.backends.cudnn.benchmark = True

    # Track KAT request vs actual compiled backend availability
    kat_requested = bool(args.use_kat)
    kat_compiled_available = False

    if kat_requested:
        try:
            import importlib
            importlib.import_module("kat_rational")
            try:
                set_prefer_kat(True)
            except Exception:
                pass
            kat_compiled_available = True
            if is_main:
                print("[KAT] kat_rational import OK - requested KAT enabled (compiled backend found).")
        except Exception as e:
            # Do not allow fall back to a 'vanilla MLP' â€” but compiled kat is optional for perf only.
            # The model itself uses GR_KAN_Conv (pure-PyTorch) so we only report compiled availability here.
            try:
                set_prefer_kat(False)
            except Exception:
                pass
            if is_main:
                print("[KAT] WARNING: requested --use-kat but 'kat_rational' is not importable or failed to initialize.")
                print("       Continuing with pure-PyTorch GR_KAN_Conv implementation (no compiled kernels). Error:", repr(e))

    train_img_size = max(args.min_res, int(args.img_size * 0.75))
    val_img_size = args.img_size

    train_ds, val_ds = make_datasets(args.data, train_img_size, val_img_size, rand_augment_spec=args.rand_augment if args.ra_start==0 else None)
    train_loader, val_loader = make_loaders(train_ds, val_ds, args.batch_size, args.workers, is_distributed=is_distributed)
    num_classes = args.num_classes if args.num_classes is not None else len(train_loader.dataset.classes)

    # instantiate model
    if args.model_variant != "custom":
        try:
            model = SwkatGRKAN_Opt.from_variant(variant=args.model_variant, img_size=args.img_size, num_classes=num_classes)
        except Exception:
            if args.model_variant == "swkat-tiny":
                model = SwkatGRKAN_Opt(img_size=args.img_size, num_classes=num_classes, embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24))
            elif args.model_variant == "swkat-small":
                model = SwkatGRKAN_Opt(img_size=args.img_size, num_classes=num_classes, embed_dim=96, depths=(2,2,18,2), num_heads=(3,6,12,24))
            elif args.model_variant == "swkat-base":
                model = SwkatGRKAN_Opt(img_size=args.img_size, num_classes=num_classes, embed_dim=128, depths=(2,2,18,2), num_heads=(4,8,16,32))
            elif args.model_variant == "swkat-large":
                model = SwkatGRKAN_Opt(img_size=args.img_size, num_classes=num_classes, embed_dim=192, depths=(2,2,18,2), num_heads=(6,12,24,48))
            else:
                model = SwkatGRKAN_Opt(img_size=args.img_size, num_classes=num_classes, embed_dim=args.embed_dim)
    else:
        depths = tuple(int(x) for x in args.depths.split(","))
        num_heads = tuple(int(x) for x in args.num_heads.split(","))
        model = SwkatGRKAN_Opt(img_size=args.img_size, num_classes=num_classes,
                              embed_dim=args.embed_dim, depths=depths, num_heads=num_heads)

    # ENFORCE KAN-only if user requested it (fail early if any SwkatBlockOpt does not use GR_KAN_Conv)
    # <-- ADDED: only run enforcement if checkbox set
    if args.use_custom_kan:
        enforce_hybrid_rule(model, is_main=is_main)
    else:
        if is_main:
            print("[HYBRID-ENFORCE] Skipped (use --use-custom-kan to verify hybrid structure).")

    try:
        resize_pos_embed_if_needed(model, new_img_size=train_img_size, patch_size=args.patch_size, device='cpu')
    except Exception as e:
        if is_main: print("pos-embed resize initial attempt failed (continuing):", e)

    model.to(device)

    try:
        grid_h = max(1, train_img_size // args.patch_size)
        grid_w = max(1, train_img_size // args.patch_size)
        if hasattr(model, "update_input_resolution"):
            model.update_input_resolution(grid_h, grid_w, device=device)
            if is_main: print(f"[init] model.update_input_resolution({grid_h},{grid_w}) called")
    except Exception as e:
        if is_main: print("Warning: update_input_resolution initial call failed:", e)

    if is_distributed:
        try:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            if is_main:
                print(f"[DDP] DistributedDataParallel initialized (world_size={world_size})")
        except Exception as e:
            if is_main:
                print("Warning: failed to wrap model in DDP:", e)

    logger = None
    if args.log_to_file and is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logfile = os.path.join(args.output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logger = open(logfile, "a", buffering=1)
        logger.write(f"Logging to {logfile}\n")

    stats_row = None
    if args.measure_stats and is_main:
        print("[stats] measuring params/flops/throughput (this may take a while)...")
        stats = compute_model_flops_and_throughput(model, img_size=args.img_size,
                                                   batch_size=args.throughput_batch, device=device,
                                                   iters=args.throughput_iters)
        stats_row = {"variant": args.model_variant, "img_size": args.img_size,
                     "params": int(stats["params"]),
                     "flops_estimate": float(stats["flops"]),
                     "throughput_ips": float(stats["throughput_ips"])}
        print("[stats] params:", stats_row["params"], "flops(estimate):", stats_row["flops_estimate"],
              "throughput img/s:", stats_row["throughput_ips"])
        save_model_stats_to_csv(args.output_dir, stats_row)

    if args.opt.lower() == 'adamw':
        base_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        base_opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    opt = base_opt

    epochs = args.epochs; warmup_epochs = args.warmup_epochs
    def combined_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(1, warmup_epochs))
        progress = float(ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(opt, lr_lambda=combined_lambda)
    # ensure lr defined (fix resume UnboundLocalError)
    lr = opt.param_groups[0].get("lr", args.lr)

#  (fix resume UnBoundLocalError)


    swa_model = None; swa_update_bn = None; swa_scheduler = None
    if args.use_swa and is_main:
        try:
            from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
            swa_model = AveragedModel(model.module if hasattr(model, "module") else model); swa_update_bn = update_bn
            swa_scheduler = SWALR(opt, swa_lr=args.swa_lr)
            if is_main: print("SWA enabled: start", args.swa_start, "swa_lr", args.swa_lr)
        except Exception as e:
            if is_main: print("SWA unavailable - continuing without. reason:", e)
            args.use_swa=False; swa_model=None

    ema_model = None
    if args.use_ema and is_main:
        ema_model = create_ema_model(model.module if hasattr(model, "module") else model)
        print("EMA enabled with decay", args.ema_decay)

    start_epoch = 0; best_val = -1.0; best_epoch = -1
    outdir = args.output_dir
    if is_main: os.makedirs(outdir, exist_ok=True)

    if args.resume and is_main:
        print("Resuming from", args.resume)
        try:
            ck = safe_load_checkpoint(model.module if hasattr(model, "module") else model, args.resume, optimizer=opt, scheduler=scheduler)
            start_epoch = ck.get("epoch", 0) + 1
            best_val = ck.get("best_val", best_val)
            print("Resumed state: epoch", start_epoch, "best_val", best_val)
        except Exception as e:
            print("Resume failed:", e)

    if is_distributed:
        try:
            for param in (model.parameters() if not hasattr(model, "module") else model.module.parameters()):
                torch.distributed.broadcast(param.data, src=0)
        except Exception:
            pass

    if args.wandb and WANDB_AVAILABLE and is_main:
        wandb.init(project="swkan-imagenet", config=vars(args))
        wandb.config.update({"num_classes": num_classes})

    metrics_csv = os.path.join(outdir, "metrics.csv") if is_main else None
    if is_main: df = ensure_metrics_csv(metrics_csv)

    prior_crop = train_img_size

    training_start_time = time.time()

    train_sampler = None
    if is_distributed:
        try:
            train_sampler = train_loader.sampler
        except Exception:
            train_sampler = None

    for epoch in range(start_epoch, epochs):
    #  even if no epochs run (fix resume UnBoundLocalError)
        t0 = time.time()

        if args.ramp_epochs and epoch < args.ramp_epochs:
            frac = (epoch + 1) / float(max(1, args.ramp_epochs))
            cur_size = int(round(args.min_res + frac * (args.img_size - args.min_res)))
        else:
            cur_size = args.img_size

        ra_spec = None
        if args.rand_augment:
            if epoch <= args.ra_start:
                ra_spec = args.rand_augment
            elif epoch >= args.ra_end:
                ra_spec = args.rand_augment
            else:
                ra_spec = args.rand_augment

        reinit_loader = False
        if cur_size != prior_crop:
            reinit_loader = True

        if reinit_loader and is_main:
            print(f"[ramp] epoch {epoch}: changing train crop from {prior_crop} -> {cur_size}")
        if reinit_loader:
            train_ds, val_ds = make_datasets(args.data, cur_size, val_img_size, rand_augment_spec=ra_spec)
            train_loader, val_loader = make_loaders(train_ds, val_ds, args.batch_size, args.workers, is_distributed=is_distributed)
            try:
                resized = resize_pos_embed_if_needed(model.module if hasattr(model, "module") else model, new_img_size=cur_size, patch_size=args.patch_size, device=device)
                if resized and is_main:
                    print(f"[ramp] resized pos-embed for crop {cur_size}")
                try:
                    grid_h = max(1, cur_size // args.patch_size)
                    grid_w = max(1, cur_size // args.patch_size)
                    base_model = model.module if hasattr(model, "module") else model
                    if hasattr(base_model, "update_input_resolution"):
                        base_model.update_input_resolution(grid_h, grid_w, device=device)
                        if is_main: print(f"[ramp] model.update_input_resolution({grid_h},{grid_w}) called")
                except Exception as e:
                    if is_main: print("[ramp] update_input_resolution failed:", e)
            except Exception as e:
                if is_main: print("[ramp] pos-embed resize failed:", e)
            prior_crop = cur_size

        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
            model, train_loader, opt, device, epoch, args, ema_model=ema_model, logger=logger, scaler=scaler,
            is_distributed=is_distributed, train_sampler=train_sampler
        )

        # Validation selection according to --validate-with
        val_stats = None
        if args.validate_with == 'ema' and args.use_ema and ema_model is not None and is_main:
            val_stats = epoch_validate_and_collect(ema_model, val_loader, device, num_classes, outdir, epoch, save_attention_samples=args.save_attention_samples)
        elif args.validate_with == 'model':
            val_stats = epoch_validate_and_collect(model.module if hasattr(model, "module") else model, val_loader, device, num_classes, outdir, epoch, save_attention_samples=args.save_attention_samples)
        elif args.validate_with == 'both':
            # run model then ema (ema if present) and merge stats - prefer ema for best_val decisions
            m_stats = epoch_validate_and_collect(model.module if hasattr(model, "module") else model, val_loader, device, num_classes, outdir, epoch, save_attention_samples=0)
            if args.use_ema and ema_model is not None and is_main:
                e_stats = epoch_validate_and_collect(ema_model, val_loader, device, num_classes, outdir, epoch, save_attention_samples=args.save_attention_samples)
            else:
                e_stats = None
            # choose which to treat as primary for checkpointing: prefer ema if present
            val_stats = e_stats if e_stats is not None else m_stats
            # log secondary as well
            if is_main and e_stats is not None:
                print("  (validate-with both) model val_top1: {:.4f}  ema val_top1: {:.4f}".format(m_stats["val_top1"], e_stats["val_top1"]))
        else:
            # fallback to live model
            val_stats = epoch_validate_and_collect(model.module if hasattr(model, "module") else model, val_loader, device, num_classes, outdir, epoch, save_attention_samples=args.save_attention_samples)

        t1 = time.time()

        if is_main:
            # FIX: refresh lr from optimizer before writing metrics (accurate current LR)
            lr = opt.param_groups[0].get("lr", getattr(args, "lr", 0.0))

            row = {"epoch": int(epoch),
                   "train_loss": float(train_loss),
                   "train_acc": float(train_acc),
                   "train_precision": float(train_precision) if not np.isnan(train_precision) else "",
                   "train_recall": float(train_recall) if not np.isnan(train_recall) else "",
                   "train_f1": float(train_f1) if not np.isnan(train_f1) else "",
                   "val_loss": float(val_stats["val_loss"]),
                   "val_top1": float(val_stats["val_top1"]),
                   "val_top5": float(val_stats["val_top5"]) if not np.isnan(val_stats["val_top5"]) else "",
                   "val_precision": float(val_stats.get("val_precision", float("nan"))) if not np.isnan(val_stats.get("val_precision", float("nan"))) else "",
                   "val_recall": float(val_stats.get("val_recall", float("nan"))) if not np.isnan(val_stats.get("val_recall", float("nan"))) else "",
                   "val_f1": float(val_stats["val_f1"]),
                   "lr": float(lr),
                   "val_samples": int(val_stats["val_samples"]),
                   "time_s": float(t1 - t0)}
            try: df = pd.read_csv(metrics_csv)
            except Exception: df = pd.DataFrame(columns=list(row.keys()))
            df.loc[len(df)] = row; df.to_csv(metrics_csv, index=False)

            epoch_msg = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] EPOCH {epoch} SUMMARY\n"
                         f"  Train Loss: {train_loss:.4f}   Train Acc: {train_acc*100:.2f}%\n"
                         f"  Train Precision: {train_precision:.4f}   Train Recall: {train_recall:.4f}   Train F1: {train_f1:.4f}\n"
                         f"  Val Loss:   {val_stats['val_loss']:.4f}   Val Acc@1: {val_stats['val_top1']*100:.2f}%   Val Acc@5: {val_stats['val_top5']*100 if not np.isnan(val_stats['val_top5']) else float('nan'):.2f}%\n"
                         f"  Val Precision: {val_stats.get('val_precision', float('nan')):.4f}   Val Recall: {val_stats.get('val_recall', float('nan')):.4f}   Val F1: {val_stats['val_f1']:.4f}\n"
                         f"  LR: {lr:.6e}  Epoch time: {t1-t0:.1f}s\n"
                         f"  confusion saved: {val_stats['confusion_png']}\n")
            print(epoch_msg)
            if logger:
                logger.write(epoch_msg + "\n"); logger.flush()

            if val_stats["attention_files"]:
                print("  attention maps:", ", ".join(val_stats["attention_files"][:8]))
                if logger:
                    logger.write("  attention maps: " + ", ".join(val_stats["attention_files"][:8]) + "\n")

            if args.wandb and WANDB_AVAILABLE:
                wandb.log(row)

            if val_stats["val_top1"] > best_val:
                best_val = val_stats["val_top1"]
                best_epoch = epoch
                save_checkpoint({"epoch": epoch, "model": (model.module if hasattr(model, "module") else model).state_dict(), "optimizer": opt.state_dict(),
                                 "scheduler": getattr(scheduler, 'state_dict', lambda: None)(), "best_val": best_val},
                                outdir, name="checkpoint_best.pth", is_main=is_main)
                if args.use_ema and ema_model is not None:
                    save_checkpoint({"epoch": epoch, "model": ema_model.state_dict(), "optimizer": opt.state_dict(), "best_val": best_val},
                                    outdir, name="checkpoint_best_ema.pth", is_main=is_main)
                print(f"  New best val {best_val:.4f} -> checkpoint_best.pth")
                if logger:
                    logger.write(f"  New best val {best_val:.4f} -> checkpoint_best.pth\n")

            save_checkpoint({"epoch": epoch, "model": (model.module if hasattr(model, "module") else model).state_dict(), "optimizer": opt.state_dict(),
                             "scheduler": getattr(scheduler, 'state_dict', lambda: None)(), "best_val": best_val},
                            outdir, name="checkpoint_last.pth", is_main=is_main)
            if args.use_ema and ema_model is not None:
                save_checkpoint({"epoch": epoch, "model": ema_model.state_dict(), "optimizer": opt.state_dict(), "best_val": best_val},
                                outdir, name="checkpoint_last_ema.pth", is_main=is_main)

        try: scheduler.step()
        except Exception: pass

        if args.use_swa and swa_model is not None and epoch >= args.swa_start and is_main:
            try:
                swa_model.update_parameters(model.module if hasattr(model, "module") else model)
                if swa_scheduler is not None: swa_scheduler.step()
            except Exception:
                pass

    if args.use_swa and swa_model is not None and is_main:
        print("Finalizing SWA: updating BN and saving SWA checkpoint")
        try:
            swa_update_bn(train_loader, swa_model, device=device)
        except Exception as e:
            print("SWA BN update failed:", e)
        swa_state = swa_model.module.state_dict() if hasattr(swa_model, "module") else swa_model.state_dict()
        save_checkpoint({"epoch": args.epochs-1, "model": swa_state, "optimizer": opt.state_dict(), "best_val": best_val},
                        outdir, name="checkpoint_swa.pth", is_main=is_main)
        print("Saved SWA checkpoint: checkpoint_swa.pth")

    total_training_time = time.time() - training_start_time

    if is_main:
        final_msg = "\n================ FINAL TRAINING SUMMARY ================\n"
        final_msg += f"Model Variant:       {args.model_variant}\n"
        final_msg += f"Image Size:          {args.img_size}\n"
        final_msg += f"Num Classes:         {num_classes}\n"
        if stats_row:
            final_msg += f"Parameters:          {stats_row.get('params','NA')}\n"
            final_msg += f"FLOPs (est):         {stats_row.get('flops_estimate','NA')}\n"
            final_msg += f"Throughput (img/s):  {stats_row.get('throughput_ips','NA')}\n"
        final_msg += f"\nBest Val Acc@1:      {best_val*100 if best_val >= 0 else float('nan'):.2f}%\n"
        final_msg += f"Best Epoch:          {best_epoch}\n"
        final_msg += f"Final LR:            {lr:.3e}\n"
        final_msg += f"Total Time (s):      {total_training_time:.1f}\n"
        final_msg += f"SAM Enabled:         {args.use_sam}\n"
        final_msg += f"EMA Enabled:         {args.use_ema}\n"
        final_msg += f"SWA Enabled:         {args.use_swa}\n"
        final_msg += f"Dynamic Resolution:  Ramp {args.min_res}->{args.img_size} over {args.ramp_epochs} epochs\n"
        final_msg += f"Mixup a:             {args.mixup_alpha}\n"
        final_msg += f"CutMix a:            {args.cutmix_alpha}\n"
        final_msg += f"Label Smoothing:     {args.label_smoothing}\n"
        final_msg += f"RandAug:             {args.rand_augment}\n"
        final_msg += f"KAT Requested:       {kat_requested}\n"
        final_msg += f"KAT Compiled Active: {kat_compiled_available}\n"
        final_msg += f"Custom KAN Enforced: {bool(args.use_custom_kan)}\n"
        final_msg += "=========================================================\n"
        print(final_msg)
        if logger:
            logger.write(final_msg + "\n"); logger.close()

        print("Training finished. Best val (top1):", best_val)
        print("Metrics CSV:", metrics_csv)

if __name__ == "__main__":
    main()
