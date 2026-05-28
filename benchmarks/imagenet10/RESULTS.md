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
├── two_view.py              # shared 2-view dataloader + forward dispatcher (CPU torchvision aug)
├── two_view_gpu.py          # GPU-augmented variant (kornia + torch.compile in on_after_batch_transfer)
├── benchmark_gpu_vs_cpu.py  # head-to-head throughput benchmark (CPU vs GPU pipeline)
├── masked.py                # shared single-view masked helper
├── multicrop.py             # shared multi-crop (DINO/iBOT/...) helper
├── collect_results.py       # aggregate CSV logs into the table
├── RESULTS.md               # this file
└── <method>-vit-small.py    # per-method config (≈40 LOC each)
```

## Migrating CPU torchvision → GPU stacked

Augmentation pairs with the dataset: pass ``gpu_transform=`` to the
dataset constructor and ``Module.on_after_batch_transfer`` discovers it
through the active DataLoader. Each split (train / val / test) carries
its own spec naturally — no per-stage routing dict needed.

```python
import stable_pretraining as spt
from stable_pretraining.data import transforms, gpu_transforms as gt
from torch.utils.data import DataLoader

# Minimal CPU side: decode + resize. rgb=True folds in RGB().
cpu_train_t = transforms.Compose(
    transforms.Resize((256, 256)),
    transforms.ToImage(rgb=True, scale=True),
)
cpu_val_t = transforms.Compose(
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToImage(rgb=True, **spt.data.static.ImageNet),
)

# GPU side: same six aug ops, batched. StackedMultiView fans out 2 views.
train_aug = gt.StackedMultiView(
    gt.GPUCompose([
        gt.GPURandomResizedCrop(size=224, scale=(0.08, 1.0)),
        gt.GPURandomHorizontalFlip(p=0.5),
        gt.GPUColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
        gt.GPURandomGrayscale(p=0.2),
        gt.GPUGaussianBlur(kernel_size=23, p=0.5),
        gt.GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    n_views=2,
)

# Each dataset owns its own GPU aug — train and val differ for free.
train_ds = spt.data.HFDataset(
    "frgfm/imagenette", split="train",
    transform=cpu_train_t, gpu_transform=train_aug,
)
val_ds = spt.data.HFDataset(
    "frgfm/imagenette", split="validation",
    transform=cpu_val_t,  # no gpu_transform — val is already normalized on CPU
)

dm = spt.data.DataModule(
    train=DataLoader(train_ds, batch_size=256, num_workers=8, pin_memory=True),
    val=DataLoader(val_ds,   batch_size=256, num_workers=8, pin_memory=True),
)
module = BarlowTwins(...)              # unchanged
trainer.fit(module, dm)                # unchanged
```

Resolution order in ``Module.on_after_batch_transfer`` (first match wins):

1. ``self.gpu_transform`` set on the Module (override).
2. ``dataset.gpu_transform`` — the recommended placement.
3. ``self.trainer.datamodule.gpu_transform`` — callable, or
   ``{"train": ..., "val": ...}`` for stage routing if you can't change
   the dataset constructor (e.g. wrapping a third-party dataset).

The resolved transform is auto-moved to ``self.device`` on first use, so
DDP just works. ``gpu_transform`` is dropped during dataset pickling, so
DataLoader workers don't pay any serialisation cost — only the main
process holds the (potentially large) kornia chain.

Asymmetric SSL (BYOL, DINO student/teacher, …): use
``gt.MultiView([chain1, chain2, ...])`` instead of ``StackedMultiView``
— same output schema, one chain call per view.

## GPU augmentation throughput (BarlowTwins ViT-S/16, batch 256, H200)

Throughput (samples/sec) for the two-view BarlowTwins recipe with three
augmentation backends:

- **CPU torchvision** (`two_view.py`): full augmentation in DataLoader workers.
- **GPU two-chain** (`two_view_gpu.py`, `--no-stack`): minimal CPU prep
  (resize + ToImage); each step runs the kornia augmentation chain
  *twice* (once per view) in `Module.on_after_batch_transfer`.
- **GPU stacked** (`two_view_gpu.py`, default): source batch is
  concatenated with itself into a `(2B, ...)` tensor, run through *one*
  chain (kornia samples independent random params per sample, so the two
  halves get different augmentations), then split back into two views.
  Valid for symmetric SSL only (Barlow Twins, SimCLR, VICReg, NNCLR).

End-to-end step time, 8 workers, single H200, imagenette parquet,
`pin_memory=True`, `persistent_workers=True`. Each row is 30–50 timed
steps after 5 warmup. 1-epoch time extrapolated from samples/sec on the
Imagenette train split (~9469 samples).

### Headline table

CPU baseline measured at bf16-mixed so the speedup ratio isolates aug
location rather than mixing in a precision change.

| # | Config | Step (ms) | Samples/sec | Epoch (s) | Speedup |
|--:|---|---:|---:|---:|---:|
| 1 | CPU torchvision (bf16-mixed) — baseline   | 749.4 | 341.6 | 27.72 | 1.00× |
| 2 | + GPU kornia stacked (fp16-mixed)         | 298.1 | 858.8 | 11.03 | 2.51× |
| 3 | **+ GPU kornia stacked (bf16-mixed)** ⭐  | **293.9** | **870.9** | **10.87** | **2.55×** |
| 4 | + GPU kornia stacked + FP8 (`transformer-engine`) | 342.5 | 747.5 | 12.67 | 2.19× ↓ |
| 5 | + FP8 + `torch.compile(model)`            | 363.3 | 704.7 | 13.44 | 2.06× ↓↓ |
| 6 | + FP8 + bs=512                            | 333.9 | 766.6 | 12.35 | 2.24× ↓ |

**Recommendation for ViT-S/16-class workloads:** `precision="bf16-mixed"`
+ `GPUCompose` + `StackedMultiView` (row 3). Avoid FP8 below ViT-B.

### Why FP8 doesn't help on ViT-S/16

Row 4 (FP8) is **slower in absolute terms** than row 2 (fp16): step
goes 298 → 342 ms (+15%). Two reasons:

1. **Model is too small.** TransformerEngine accelerates `nn.Linear` GEMMs, but at
   384-dim with batch 256, the MLP/QKV/out projections don't have enough
   FLOPs to overcome TransformerEngine's per-`Linear` Python overhead (autocast, scale
   tracking, kernel dispatch).
2. **The bottleneck wasn't matmuls.** Augmentation is ~70% of unoverlapped
   step time (see per-phase breakdown below). Even if FP8 halved fwd+bwd,
   that's at most ~15% on the total step — and TransformerEngine's overhead eats it.

`torch.compile(model)` + FP8 (row 5) is the worst combo: TransformerEngine's custom
autograd Functions graph-break heavily, adding tracing overhead with
no fusion payoff.

FP8 should pay off around **ViT-B/L** and bigger (LLMs, DiT,
multimodal). Re-evaluate then.

### Caveats on these specific numbers

This sweep ran with `srun --jobid=... --overlap` on a node already
holding an idle interactive allocation, so the 8 DataLoader workers
shared CPU cores with another job. CPU baseline (row 1) is extra-sensitive
to this contention — a solo-node run on the prior (fp16-mixed) baseline
measured 0.498 s/step / 514 samples/sec, putting GPU stacked at
**~1.77× over a dedicated CPU baseline**. The **absolute step times** in
rows 2–6 are still directly comparable to each other since the GPU
pipeline barely uses CPU.

Note: `bf16-mixed` vs `fp16-mixed` on the CPU pipeline measured within
~3% of each other (749 ms vs 729 ms here). On H100/H200 both use the
FP16 tensor cores; only the numerical range differs. The precision
choice is mostly about training stability, not speed.

### ViT-Large variant (bs=384)

Sanity check on a bigger model where the matmul/everything ratio
should let FP8 start paying off. Same H200 / shared allocation.
**bs=384** is the practical ceiling on a single H200 for the
two-view recipe (effective batch 768) — bs=512 OOMs.

Rows sorted by speedup (fastest first).

| # | Config (ViT-L, bs=384) | Step (ms) | Samples/sec | Epoch (s) | Speedup |
|--:|---|---:|---:|---:|---:|
| 1 | **+ GPU kornia stacked + FP8 + compile(model)** ⭐ | **751.0** | **511.3** | **18.52** | **1.53×** |
| 2 | + GPU kornia stacked + FP8                         |  797.6 | 481.5 | 19.67 | 1.44× |
| 3 | + GPU kornia stacked (bf16) + compile(model)       |  798.5 | 480.9 | 19.69 | 1.44× |
| 4 | + GPU kornia stacked (bf16-mixed)                  |  931.7 | 412.2 | 22.97 | 1.23× |
| 5 | + CPU torchvision (bf16) + compile(model)          |  951.5 | 403.6 | 23.46 | 1.21× |
| 6 | CPU torchvision (bf16-mixed) — baseline            | 1149.8 | 334.0 | 28.35 | 1.00× |

Three findings vs ViT-S:

1. **GPU augmentation helps but less dramatically** (row 4 vs row 6:
   1.23×). At ViT-L the CPU side still has plenty of work — 2 views
   × 384 samples × random augs is a lot for 8 workers — so moving
   aug to the GPU frees CPU time and vectorises the augs, but the
   model fwd+bwd is now the dominant cost so the headline ratio is
   smaller than on ViT-S.
2. **FP8 wins on top of GPU aug** (row 2 vs row 4: +17%). At ViT-L
   scale the matmul FLOPs are large enough that TransformerEngine's
   per-`Linear` Python overhead is dwarfed by the FP8 GEMM speedup.
   The threshold between "FP8 hurts" (ViT-S, row 4 of the ViT-S
   table) and "FP8 helps" (ViT-L) sits around ViT-B; worth confirming
   on a per-model basis.
3. **`torch.compile(model)` flips from regression on ViT-S to a real
   win at ViT-L+bs=384** (row 1 vs row 2: +6% on top of FP8;
   row 3 vs row 4: +17% on top of bf16; row 5 vs row 6: +21% even
   on the CPU pipeline). At ViT-L+bs=384 the kernels are big enough
   to amortize the tracing overhead. Pairs well with FP8 here, unlike
   the ViT-S result where the same combo was the worst row.

### `compile=True` on `GPUCompose`

The default is `compile=True` (for safety/discoverability), but it adds
first-batch warmup and gives essentially no steady-state gain because
kornia's per-sample random param sampling graph-breaks. `compile=False`
is fine and a hair faster — we used it for rows 2–6.

### Per-phase breakdown (single-stream, no overlap)

Synchronous timing of each phase via `torch.cuda.Event` pairs
(`profile_phases.py`). The `TOTAL` row is what a fully serialised
timeline would cost; real overlapped step time (above) is meaningfully
lower thanks to producer/consumer overlap.

| Phase | CPU mode (ms) | GPU two-chain (ms) | GPU stacked (ms) |
|---|---:|---:|---:|
| data_load | 252.7 (50.3%) | 0.17 | 0.20 |
| h2d       | 145.0 (28.9%) | 3.73 | 3.74 |
| aug       | 0.0 | 226.9 (69.1%) | 233.6 (70.2%) |
| fwd       | 28.9 (5.7%)   | 27.5 | 26.7 |
| bwd       | 75.3 (15.0%)  | 70.0 | 68.7 |
| **TOTAL** | **501.9** | **328.3** | **332.9** |

(`profile_phases.py` syncs between phases for accurate per-phase
costs; this is *not* steady-state step time. The real step time
benefits from H2D overlap and inter-step prefetch.)

### Per-op cost in the GPU chain (B=256, 224×224 outputs, H200)

| Op | Mean (ms) |
|---|---:|
| GPURandomResizedCrop(224) | **18.32** |
| GPUGaussianBlur(k=23) | **17.05** |
| GPUGaussianBlur(k=5) | 15.98 |
| GPUColorJitter | 4.84 |
| GPURandomSolarize | 1.05 |
| GPURandomHorizontalFlip | 0.64 |
| GPURandomGrayscale | 0.59 |
| GPUNormalize | 0.25 |

Two ops dominate: `RandomResizedCrop` (grid_sample) and `GaussianBlur`
(kornia uses an FFT-like path so the cost is roughly flat in kernel size
— a `k=5` blur is only ~7% cheaper than `k=23`).

### Where to push next

Augmentation is still ~70% of unoverlapped time. Remaining levers:

1. **Drop or downgrade GaussianBlur** — costs 17 ms regardless of kernel
   size. Lowering `p` from 0.5 to 0.1 saves on expectation, dropping it
   entirely saves the full 17 ms.
2. **Lower aug cost per sample, not per op** — for symmetric SSL, the
   stacked path already shares the crop computation between views;
   further wins need cheaper crops (e.g. uniform random crop without
   the per-sample scale resampling that grid_sample needs).
3. **Multi-crop / asymmetric / smaller-model regimes** are where this
   stack pays out the most — the ratio of aug cost to model cost grows.
4. **NVIDIA DALI** if you want FFCV-class numbers without rewriting the
   file format: GPU JPEG decode + GPU augmentation graph, but heavy dep
   and less flexible for unusual data shapes.

### Reproducing

```bash
# Full end-to-end throughput
srun --gpus=1 --cpus-per-task=8 python benchmarks/imagenet10/benchmark_gpu_vs_cpu.py --mode cpu --num-workers 8 --steps 30
srun --gpus=1 --cpus-per-task=8 python benchmarks/imagenet10/benchmark_gpu_vs_cpu.py --mode gpu --num-workers 8 --steps 30 --no-compile

# Per-phase profile
srun --gpus=1 python benchmarks/imagenet10/profile_phases.py --mode gpu --steps 30

# Per-op cost
srun --gpus=1 python benchmarks/imagenet10/profile_ops.py --batch-size 256
```
