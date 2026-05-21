# EDTrans: Edge-aware Dual-attention Transformer for Polyp Segmentation

> **Official code for the paper submitted to *The Visual Computer*.**
>
> If you use this code or our results in your research, please cite our paper (BibTeX below).

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

---

## Results

### Kvasir-SEG

| Method | Dice | IoU | Precision | Recall | HD95 |
|--------|------|-----|-----------|--------|------|
| U-Net | 0.8326 | 0.7161 | 0.8606 | 0.8097 | 1.460 |
| U-Net++ | 0.8415 | 0.7289 | 0.8660 | 0.8215 | 1.586 |
| PraNet | 0.8917 | 0.8069 | 0.9191 | 0.8690 | 0.948 |
| **EDTrans (Ours)** | **0.8898** | **0.8060** | **0.9320** | **0.8530** | **0.952** |

### CVC-ClinicDB

| Method | Dice | IoU | Precision | Recall | HD95 |
|--------|------|-----|-----------|--------|------|
| U-Net++ (Basic) | 0.9164 | 0.8462 | 0.9143 | 0.9201 | 0.626 |
| **EDTrans (Ours)** | **0.9315** | **0.8724** | **0.9379** | **0.9263** | **0.599** |

### Model Complexity

| Model | Params (M) | FLOPs (G) | FPS |
|-------|-----------|-----------|-----|
| EDTrans | 62.23 | 61.06 | 77.9 |

---

## Ablation Study

| Variant | Dice (CVC) | IoU (CVC) |
|---------|-----------|----------|
| Full Model (EDTrans) | 0.9315 | 0.8724 |
| w/o EAM | 0.9008 | 0.8217 |
| w/o DAM | 0.9311 | 0.8716 |
| w/o Hybrid CNN-Trans + SPP | 0.9321 | 0.8739 |
| Basic UNet++ (w/o All) | 0.9164 | 0.8462 |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{[CITATION_KEY],
  title   = {[PAPER_TITLE]},
  author  = {[AUTHOR_NAME] and others},
  journal = {The Visual Computer},
  year    = {2025},
  doi     = {[DOI]}
}
```

---

## Acknowledgements

This work was supported by [GRANT_NAME] (Grant No. [GRANT_NUMBER]).

We thank the authors of [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) for their excellent library, and the creators of Kvasir-SEG and CVC-ClinicDB for providing publicly available benchmarks.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions or issues, please open a GitHub Issue or contact [AUTHOR_EMAIL].
