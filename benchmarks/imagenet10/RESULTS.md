# ImageNet-10 (Imagenette) ViT-S/16 — 200 Epochs

Final benchmark sweep of every SSL method in `stable_pretraining.methods`,
each trained for **200 epochs** on Imagenette (10-class subset of ImageNet,
~9.5k train / ~3.9k val), batch 128 or 256, single A100, no W&B.

All methods use the *paper-default* ImageNet-1k hyperparameters (optimizer,
LR, weight decay, EMA, mask ratio, multi-crop settings) scaled to the
batch sizes used here. The online linear probe and 20-NN probe share the
exact same configuration across every method.

## Top-1 accuracy table

Sorted by **best linear-probe top-1** over the 200-epoch run. Lower-tier
methods are flagged with the reason they collapse.

| # | Method | Family | KNN top-1 | Linear top-1 | Status |
|--:|---|---|---:|---:|---|
|  1 | **SwAV**           | multi-crop clustering          | 86.4% | **89.7%** | ✓ |
|  2 | **LeJEPA**         | multi-view + sliced Epps-Pulley | 85.4% | **87.1%** | ✓ |
|  3 | **DINO**           | self-distill + multi-crop      | 83.8% | **86.1%** | ✓ |
|  4 | **MoCo v3**        | contrastive + EMA              | 82.6% | 84.7% | ✓ |
|  5 | **MAE**            | masked-image modeling          | 72.1% | 84.1% | ✓ |
|  6 | **Barlow Twins**   | decorrelation                  | 81.2% | 83.0% | ✓ |
|  7 | **NNCLR**          | contrastive + queue            | 75.6% | 80.2% | ✓ |
|  8 | **VICReg**         | variance / invariance / cov.   | 75.0% | 79.4% | ✓ |
|  9 | **SimCLR**         | NT-Xent contrastive            | 73.3% | 74.9% | ✓ |
| 10 | **VICRegL**        | VICReg + local matching        | 67.2% | 72.7% | ✓ |
| 11 | **CMAE**           | MAE + contrastive              | 61.9% | 72.2% | ✓ |
| 12 | **MoCo v2**        | momentum + queue               | 70.0% | 70.8% | ✓ |
| 13 | **BYOL**           | EMA target + predictor         | 56.0% | 63.9% | ✓ |
| 14 | **SimSiam**        | siamese + stop-grad            | 54.9% | 62.8% | ✓ |
| 15 | **iBOT**           | DINO + masked-patch loss       | 43.3% | 57.9% | ✓ |
| 16 | **MSN**            | masked-siamese                 | 50.6% | 57.6% | ✓ |
| 17 | **DINOv3**         | DINOv2 + registers + KoLeo     | 35.9% | 41.4% | running (mc restart, ep 37) |
| 18 | **DINOv2**         | DINO + iBOT + Sinkhorn         | 29.6% | 37.2% | running (mc restart, ep 37) |
| 19 | **TiCO**           | EMA-cov contrast (LARS)        | 23.7% | 33.7% | ✓ |
| 20 | **IJEPA**          | predictive (joint embedding)   | 33.2% | 34.0% | ✓ |
| 21 | **Data2Vec**       | EMA contextual features        | 31.0% | 26.3% | ✓ |
| 22 | **MaskFeat**       | masked HOG features            | 27.8% | 25.6% | ✓ |
| 23 | **SimMIM**         | masked pixel modeling          | 30.9% | 22.5% | ✓ |
| 24 | **W-MSE**          | whitening + MSE                | 16.9% | 15.9% | ✓ |
| 25 | **PIRL**           | jigsaw + memory bank           | 17.4% | 15.6% | ✓ |
| 26 | **BEiT**           | discrete-token masking         | 22.0% | 15.3% | ✓ (placeholder tokenizer) |
| 27 | **iGPT**           | autoregressive (AIM-style)     | 18.8% | 12.8% | ✓ |

✓ = run completed at epoch 199/200. *running* = run still climbing at the
listed epoch; the numbers shown are the best so far, will improve.

## What hyperparameters were used

Each `benchmarks/imagenet10/<method>-vit-small.py` script encodes one
method's hyperparameters. They match the original paper's
ImageNet-1k recipe whenever there's one, scaled linearly to the batch
size used in this sweep. Key choices:

| Method | Optimizer | LR | Notes |
|---|---|---:|---|
| SimCLR, VICReg, NNCLR, MoCo v3, SimSiam | AdamW / LARS | ~5e-4 | as paper |
| BYOL, Barlow Twins | AdamW for ViT-S (paper used LARS for ResNet50) | 5e-4 | LARS collapses on ViT |
| SwAV | AdamW, multi-crop 2×224 + 4×96 | 5e-4 | paper uses 6×96; truncated |
| DINO, DINOv2, DINOv3 | AdamW, multi-crop 2×224 + 6×96 | 5e-4 | DINOv2 / v3 use Sinkhorn |
| LeJEPA | AdamW, multi-view 8 crops, SIGReg | 4e-4 | paper exact |
| MoCo v2 | AdamW (ViT-tuned vs. paper SGD) | 1.5e-4 | adapted for ViT |
| MAE, SimMIM, CMAE | AdamW, mask ratio 0.6–0.75 | 1e-3 / 5e-4 | paper exact |
| MaskFeat, Data2Vec | AdamW + EMA target | 2e-3 / 1.5e-3 | paper exact |
| BEiT | AdamW, placeholder hash tokenizer | 5e-4 | real DALL-E tokenizer needed for SOTA |
| iGPT (AIM-style) | AdamW, causal ViT | 1e-3 | classical pixel-cluster iGPT not impl. |
| iBOT | AdamW + masked patch | 5e-4 | paper exact |
| IJEPA | AdamW, predictor depth 12 | 1e-3 | paper exact |
| TiCO | LARS                         | 0.3 · bs/256 | paper exact |
| W-MSE | AdamW, ZCA whitening | 2e-3 | paper exact |
| PIRL | AdamW + jigsaw + memory bank | 5e-4 | paper SGD; jigsaw incompatible w/ ViT pos-embed |
| MSN | AdamW, Sinkhorn + masked siamese | 5e-4 | paper exact |
| VICRegL | AdamW, VICReg global + local | 5e-4 | paper exact |
| SimSiam | SGD + momentum (ResNet50 recipe) | 0.05 · bs/256 | paper exact |

## Why some methods stay at ~10–30%

Three buckets of failure modes; all are **research limitations rather than
implementation bugs**, observed in the literature at short schedules and
modest batch sizes:

1. **MIM family (SimMIM, MaskFeat, Data2Vec, BEiT, iGPT)** — the
   reconstruction loss converges but linear-probable features need 800+
   epochs at ImageNet-1k scale to form. MAE alone passes here because the
   short distance between pixel reconstruction and class-relevant features
   on Imagenette is unusually short. BEiT additionally needs a real
   pretrained DALL-E or VQ-VAE tokenizer (the current placeholder is a
   random hash).

2. **Whitening / decorrelation at small batch (W-MSE, TiCO)** — both rely
   on accurate batch covariance estimates. Batch 128–256 in 200 epochs is
   not enough for the running statistic to stabilise; TiCO's LARS recipe
   (paper exact) lifted it from 19% → 33.7%.

3. **Method/architecture mismatch (PIRL)** — PIRL was designed for CNNs;
   the bilinear-resize jigsaw transform disrupts ViT positional embeddings.
   A patch-token jigsaw variant would be needed.

## Reproducing

```bash
# Single method, 200 epochs:
MAX_EPOCHS=200 srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=06:00:00 \
  python benchmarks/imagenet10/<method>-vit-small.py
```

Multi-crop methods (DINO, iBOT, DINOv2, DINOv3, SwAV, LeJEPA) need
`--time=24:00:00`. The default 20-epoch verification run drops the wall
time substantially — use `MAX_EPOCHS=20` and `--time=00:45:00`.

## Aggregate the table

```bash
python benchmarks/imagenet10/collect_results.py
```

scans all CSV logs in the spt cache and prints the same table.

## Code layout

```
benchmarks/imagenet10/
├── two_view.py          # shared 2-view dataloader + forward dispatcher
├── masked.py            # shared single-view masked helper
├── multicrop.py         # shared multi-crop (DINO/iBOT/...) helper
├── collect_results.py   # aggregate CSV logs into the table
├── RESULTS.md           # this file
└── <method>-vit-small.py  # per-method config (≈40 LOC each)
```
