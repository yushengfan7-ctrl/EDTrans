# Reproduction Guide

This document explains how to fully reproduce the results reported in the paper.

## Hardware

All experiments were conducted on a single NVIDIA GPU with ≥ 16 GB VRAM.
Training time is approximately **4–6 hours** per dataset on an RTX 3090.

## Step 1 — Prepare datasets

Follow the dataset preparation instructions in [README.md](../README.md#dataset-preparation).

Verify the directory layout:

```
dataset/
├── Kvasir-SEG/
│   ├── images/   (1000 images)
│   └── masks/    (1000 masks)
└── CVC-ClinicDB/
    ├── Original/       (612 images)
    └── Ground Truth/   (612 masks)
```

## Step 2 — Train on Kvasir-SEG

```bash
python train.py --config configs/config.yaml --dataset Kvasir-SEG
```

Expected best validation Dice: **~0.88–0.89** (may vary ±0.005 across runs).

## Step 3 — Train on CVC-ClinicDB

Edit `configs/config.yaml` to set `dataset: CVC-ClinicDB`, then:

```bash
python train.py --config configs/config.yaml --dataset CVC-ClinicDB
```

Expected best validation Dice: **~0.93** (may vary ±0.005 across runs).

## Step 4 — Evaluate

```bash
# Kvasir-SEG
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --dataset Kvasir-SEG

# CVC-ClinicDB
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --dataset CVC-ClinicDB
```

## Step 5 — Expected results

### Kvasir-SEG

| Metric | Expected |
|--------|----------|
| Dice | 0.8898 |
| IoU | 0.8060 |
| Precision | 0.9320 |
| Recall | 0.8530 |
| HD95 | 0.952 |

### CVC-ClinicDB

| Metric | Expected |
|--------|----------|
| Dice | 0.9315 |
| IoU | 0.8724 |
| Precision | 0.9379 |
| Recall | 0.9263 |
| HD95 | 0.599 |

## Reproducibility notes

- The random seed is fixed at `42` (see `configs/config.yaml`).
- Results may differ slightly (< 0.005 Dice) across different GPU models or PyTorch versions due to floating-point non-determinism in CUDA operations.
- Set `torch.backends.cudnn.deterministic = True` (already done in `train.py`) to maximise reproducibility within the same hardware.

## Configuration reference

All hyperparameters are in `configs/config.yaml`:

| Parameter | Value |
|-----------|-------|
| Image size | 256 × 256 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR scheduler | CosineAnnealingWarmRestarts (T₀=10) |
| Max epochs | 100 |
| Early stopping patience | 15 |
| Edge loss weight | 0.5 |
| Gradient clip norm | 1.0 |
| Encoder | ResNet-34 (ImageNet pre-trained) |
