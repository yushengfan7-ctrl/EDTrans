# EDTrans: Edge-aware Dual-attention Transformer for Polyp Segmentation

> **Official code for the paper submitted to *The Visual Computer*.**
>
> If you use this code or our results in your research, please cite our paper .

---

## Highlights

- **Edge-Aware Module (EAM)**: fixed Sobel operators + learnable gating for boundary-sensitive feature enhancement with deep supervision.
- **Dual Attention Module (DAM)**: joint channel attention, spatial attention, and dilated context modeling in a single residual block.
- **Hybrid CNN-Transformer Block**: parallel CNN and patch-ViT branches fused via cross-branch attention to capture both local texture and global context.
- **Spatial Pyramid Pooling (SPP)**: multi-scale adaptive pooling with residual connection for robust multi-resolution context aggregation.
- Achieves **Dice 0.9315** on CVC-ClinicDB and **Dice 0.8898** on Kvasir-SEG with a ResNet-34 backbone.

---

## Architecture

```
[ARCHITECTURE_DIAGRAM]
```

*Fig. 1: Overall architecture of EDTrans. The encoder features pass through EAM (shallow), DAM (mid-level), and the Feature Transformer (deep) before decoding.*

---

## Installation

See [docs/INSTALL.md](docs/INSTALL.md) for detailed environment setup.

**Quick start:**

```bash
git clone https://github.com/[USERNAME]/EDTrans.git
cd EDTrans
pip install -r requirements.txt
```

---

## Dataset Preparation

Download the datasets from their official sources:

| Dataset | Source |
|---------|--------|
| Kvasir-SEG | https://datasets.simula.no/kvasir-seg/ |
| CVC-ClinicDB | https://polyp.grand-challenge.org/CVCClinicDB/ |
| ETIS-LaribPolypDB | https://polyp.grand-challenge.org/ |

Organize data as follows:

```
dataset/
├── Kvasir-SEG/
│   ├── images/    # *.jpg
│   └── masks/     # *.jpg
└── CVC-ClinicDB/
    ├── Original/       # *.tif or *.png
    └── Ground Truth/   # *.tif or *.png
```

---

## Training

```bash
# Train on Kvasir-SEG (default)
python train.py --config configs/config.yaml

# Train on CVC-ClinicDB
python train.py --config configs/config.yaml --dataset CVC-ClinicDB
```

The best checkpoint is saved to `checkpoints/best_model.pth`.

---

## Evaluation

```bash
python test.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --dataset Kvasir-SEG
```

Results are printed to stdout and saved as a CSV in `results/`.

---

## Inference Demo

```bash
python inference.py \
  --image path/to/image.jpg \
  --checkpoint checkpoints/best_model.pth \
  --output prediction.png
```
## ⚠️ Important Notice

This repository contains the official PyTorch implementation directly 
associated with our manuscript currently **under review at The Visual 
Computer (Springer Nature)**:

> **"Boundary-Guided Hybrid CNN-Transformer Architecture for Robust 
Medical Image Segmentation"**

The code, models, and benchmarks in this repository are provided 
specifically to support reproducibility of the experiments reported 
in the above manuscript.

---

## 📜 Citation Request

If you use this code, the trained models, or any derived artifacts 
in your research, please **cite our manuscript** to acknowledge this 
work:

```bibtex
@article{edtrans2026,
  title={Boundary-Guided Hybrid CNN-Transformer Architecture for 
         Robust Medical Image Segmentation},
  author={Fan, Yusheng and Yan, Shaoliang and Shi, Jianfeng and 
          Ying, Xuqing and Lu, Xufeng and Fu, Kaiye and Shen, Zhifa 
          and Zhang, Xin and Guo, Ganhua and Ru, Jiaxi},
  journal={The Visual Computer},
  year={2026},
  note={Manuscript under review}
}
```

Citation details will be updated upon publication.

---
