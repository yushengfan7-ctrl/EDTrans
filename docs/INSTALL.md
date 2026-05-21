# Installation Guide

## Requirements

- Python ≥ 3.8
- CUDA ≥ 11.3 (recommended for GPU training)
- PyTorch ≥ 1.12

## Step 1 — Create a virtual environment

```bash
conda create -n edtrans python=3.9 -y
conda activate edtrans
```

Or with venv:

```bash
python -m venv edtrans_env
# Linux/macOS
source edtrans_env/bin/activate
# Windows
edtrans_env\Scripts\activate
```

## Step 2 — Install PyTorch

Install PyTorch matching your CUDA version from https://pytorch.org/get-started/locally/.

Example (CUDA 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Step 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Step 4 — Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import segmentation_models_pytorch; print('smp OK')"
```

## Tested configurations

| OS | Python | PyTorch | CUDA | GPU |
|----|--------|---------|------|-----|
| Ubuntu 20.04 | 3.9 | 1.13.1 | 11.7 | RTX 3090 |
| Windows 11 | 3.10 | 2.0.1 | 11.8 | RTX 4070 |
