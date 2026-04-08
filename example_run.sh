#!/bin/bash

# ==========================================
# DeKAHT Example Training Script
# ==========================================

# NOTE:
# This script demonstrates how to train the
# DeKAHT (SwKAT in code) model on ImageNet-10.

# IMPORTANT:
# Update dataset and output paths before running.

# ------------------------------------------
# GPU configuration
# ------------------------------------------

export CUDA_VISIBLE_DEVICES=0,1

# ------------------------------------------
# Training command
# ------------------------------------------

torchrun --nproc_per_node=2 \
train.py \
--data /path/to/imagenet10 \
--model-variant swkat-tiny \
--img-size 224 \
--epochs 300 \
--batch-size 64 \
--lr 1e-4 \
--weight-decay 0.05 \
--warmup-epochs 5 \
--opt adamw \
--num-classes 10 \
--workers 8 \
--output-dir ./outputs/imagenet10/tiny \
--use-custom-kan \
--grad-clip 1.0 \
--no-amp \
--mixup-alpha 0.0 \
--cutmix-alpha 0.0 \
--label-smoothing 0.0 \
--ramp-epochs 0
