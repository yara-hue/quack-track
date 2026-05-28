# quack-track

Lightweight Siamese Region Proposal Network for aerial object tracking (AIC-4 competition).

## Architecture

- **Backbone:** MobileNetV2 (ImageNet-pretrained, frozen) — extracts stride 8/16/32 features
- **Neck:** Feature Pyramid Network with lateral connections + top-down pathway
- **Correlation:** Depthwise separable cross-correlation
- **Head:** RPN with 5 anchors (ratios: 0.33, 0.5, 1.0, 2.0, 3.0)
- **Loss:** BCE (cls) + Smooth L1 (reg), weight ratio 1.0:1.2
- **Inference:** Single-scale (stride 16), cosine window penalty (weight 0.4)

## Setup

```bash
git clone https://github.com/<YOUR_USERNAME>/quack-track.git
cd quack-track
pip install -r requirements.txt
```

### Datasets

Place UAV123_10fps and UAV20L in the `data/` directory:

```
data/
├── UAV123_10fps/
│   ├── data_seq/
│   └── anno/
├── UAV20L/
│   ├── data_seq/
│   └── anno/
└── train2017/   (optional, for COCO negative pairs)
```

Or use symlinks:
```bash
python data/prepare_datasets.py --uav123 /path/to/UAV123_10fps --uav20l /path/to/UAV20L
```

## Training

```bash
# Debug on UAV123_10fps
python scripts/train.py --config configs/default.yaml

# With COCO negative pairs
python scripts/train.py --config configs/default.yaml  # set coco_root in config
```

## Evaluation

```bash
python scripts/test.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth
```

## Colab

Open `colab/setup.ipynb` in Google Colab for one-click setup.

## Results

| Metric | Value |
|--------|-------|
| Precision (20px) | — |
| Success (IoU=0.5) | — |
| AUC | — |
| Params | — |
| FLOPs | — |

## References

- [AIC-4 Competition](https://aiaerialtracking.com/)
- [SiamRPN++: Evolution of Siamese Visual Tracking](https://arxiv.org/abs/1812.11703)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
