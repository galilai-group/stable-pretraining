# SSL-EY on CIFAR-10

This contribution integrates SSL-EY from [Unconstrained Stochastic CCA: Unifying Multiview and Self-Supervised Learning (ICLR 2024)](https://arxiv.org/abs/2310.01012) into stable-pretraining with a loss, composable forward function and method class, plus a single-GPU CIFAR-10 / ResNet-18 benchmark.

## Library interface

```python
import torch
import stable_pretraining as spt

model = spt.SSLEY(
    backbone=spt.backbone.from_torchvision("resnet18", low_resolution=True),
    projector=torch.nn.Sequential(
        torch.nn.Linear(512, 2048),
        torch.nn.BatchNorm1d(2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 2048),
    ),
    optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
)
```

The example above demonstrates the generic API, not the complete experiment recipe. For custom composition use `spt.Module(forward=spt.forward.ssley, ssley_loss=spt.losses.SSLEYLoss(), backbone=..., projector=..., optim=...)`.
Training accepts exactly two views as a list or named mapping, with `image` tensors in each view. Labels are optional and never enter the SSL objective. Evaluation returns backbone embeddings without running the projector.

For centered projections X and Y, the loss is `-2 trace(C) + trace(V @ V)`, where `C = 2 X.T @ Y / (N - 1)` and `V = (X.T @ X + Y.T @ Y) / (N - 1)`. Negative losses are expected. The generic loss promotes half/bfloat16 covariance arithmetic to FP32 by default and retains float64 inputs. Autograd-aware gathering requires equal local batch shapes; use `drop_last=True`. Multi-GPU runtime validation is not claimed.

## Benchmark files

| File | Purpose |
|---|---|
| `ssley-paper.py` | Training entry point: data, Manager, logging, numerical guards, resume |
| `ssley_recipe.py` | Model, augmentations, optimizer and stability options |
| `ssley_reference_lars.py`, `ssley_reference_scheduler.py` | Author-fork LARS and warmup/cosine schedule, with attribution |
| `ssley-paper.json` | Released author arithmetic (ambient autocast, no clipping); default config |
| `ssley-stable.json` | FP32 loss + vector-gradient clipping; the successful 1000-epoch run |
| `ssley-fp32-only.json` | FP32 loss without clipping; controlled ablation |
| `results/ssley-cifar10.json`, `results/ssley-cifar10.png` | Machine-readable results and raw epoch curves |

The three configurations share every setting except the two stability options and their name/revision metadata; `test_ssley_recipe.py` checks this.

## Recipe

The reference is the author's `ssley` branch of [solo-learn at commit 11eac929a3cfb8fe28cefa573ef87e08fcf1eb27](https://github.com/jameschapman19/solo-learn/tree/11eac929a3cfb8fe28cefa573ef87e08fcf1eb27), including its CIFAR configuration, loss, LARS and recurrent warmup/cosine schedule. Source notices are preserved in the adapted files and the pinned test fixtures.

| Setting | Value |
|---|---|
| Encoder | ResNet-18, CIFAR 3x3/stride-1 stem with padding 2, no initial max pool |
| Projector | 512 -> 2048 -> 2048 -> 2048; hidden BN/ReLU; Linear biases enabled |
| Training | 1000 epochs; batch 256; `drop_last=True` (195 steps per epoch); seed 7 |
| Optimizer | Author-fork LARS; LR 0.3 encoder/projector, 0.1 classifier; momentum 0 |
| LARS options | eta 0.02, clip_lr, exclude_bias_n_norm; weight decay 1e-4 except classifier 0 |
| Scheduler | 10-epoch warmup from 3e-5, recurrent step-wise cosine to zero |
| Precision | Network FP16 mixed; loss precision depends on the configuration |
| Augmentation | Bicubic crop scale 0.2–1; jitter 0.4/0.4/0.2/0.1 with p=0.8; grayscale p=0.2; flip p=0.5; solarization p=0.1 on each view; no blur |
| Evaluation | Detached online linear classifier; top-1/top-5 on all 10000 test images |

The stable configuration computes the loss in FP32 with autocast disabled and jointly clips, to norm 1, the gradients of the one-dimensional parameters excluded from LARS adaptation (BN affine parameters and biases), after AMP unscaling. Matrix gradients keep the reference LARS update, and invalid vector gradients are left untouched so GradScaler can skip the step. This deliberately changes optimization.

The online classifier receives detached features, so supervised gradients never reach the encoder/projector; this matches the author's pretraining evaluation pathway and is not a separate frozen-encoder probe. The author fork adapts 2D zero-decay classifier weights, unlike the library LARS; `ssley_reference_lars.py` preserves that behavior. Training uses CIFAR standard deviations (0.2470, 0.2435, 0.2616), while evaluation follows the released rounded values (0.247, 0.243, 0.261). Seed 7 is our explicit choice, not a recovered paper seed.

## Running

Install a CUDA-compatible torch/torchvision build, then from the repository root:

```bash
python -m pip install -e '.[dev]'
```

Download CIFAR-10 once (exits before training; training runs use `download=False`):

```bash
python benchmarks/cifar10/ssley-paper.py --config benchmarks/cifar10/ssley-stable.json \
    --data-dir /path/to/cifar10 --output /path/to/runs/prepare --download-only
```

Train on one GPU:

```bash
python benchmarks/cifar10/ssley-paper.py --config benchmarks/cifar10/ssley-stable.json \
    --data-dir /path/to/cifar10 --output /path/to/runs/ssley-stable-seed7 \
    --accelerator gpu --workers 4 --threads 4
```

Useful options:

- `--resume /path/to/last.ckpt` continues a run with the same configuration and output directory; recipe mismatches are rejected. Prefer epoch-boundary checkpoints, since mid-epoch recovery can replay data.
- `--max-time DD:HH:MM:SS` stops cleanly and writes a resumable checkpoint before a job time limit; each resumed job gets a fresh time budget while the training schedule is kept.
- `--stop-after-epochs N` stops early without changing the 1000-epoch schedule. The ablation was run fresh with `--config benchmarks/cifar10/ssley-fp32-only.json --stop-after-epochs 50` in its own output directory.
- `--smoke` swaps in synthetic data for a quick wiring check; it is not a representation-quality experiment and was not used for the reported results.

Each run directory contains `resolved.json` (actual recipe, arguments and versions), `history.jsonl` (per-epoch metrics), `progress.json`, `run-status.json` (running, paused, complete or failed), `numerics.jsonl` (AMP skips and invalid-loss events), `checkpoints/last.ckpt` and `backbone.pt` on clean exit. The guard stops training on damaged model state, invalid validation metrics or 32 consecutive non-finite loss batches.

## Results

| Run | Completed epochs | Framework steps | Final top-1 | Final top-5 | Status |
|---|---:|---:|---:|---:|---|
| `ssley-stable.json` | 1000 | 195000 | **89.46%** | **99.63%** | Complete |
| `ssley-fp32-only.json` | 28 of planned 50 | 5460 at last epoch | 10.62% | 56.74% | Failed at step 5581 during epoch 29 |

Scores are the final recorded epoch on the CIFAR-10 test split, not best-epoch selections. The stable run reached 79.03%, 84.12%, 86.55%, 88.60% and 89.46% at epochs 50, 100, 200, 500 and 1000; the last 100 epochs ranged from 89.38% to 89.70%.

![Raw epoch metrics](results/ssley-cifar10.png)

The ablation scored 71.36% at epoch 27, then collapsed to 10.62% at epoch 28 with mean SSL loss about 4.2e15, while the stable run scored 73.32% at the same epoch. The guard then raised `FloatingPointError: Non-finite model state` on backbone/projector BatchNorm running statistics. This is stability evidence for this seed and environment, not a matched-duration accuracy comparison. No run with `ssley-paper.json` completed successfully in our environment.

All 1000 stable epoch records are contiguous and finite, with zero non-finite loss batches. AMP recorded 95 gradient-overflow skips (13 in the first 50 epochs), so 195000 framework steps are not 195000 parameter updates; the final AMP scale was 32768 and the final mean SSL loss -210.34. The final checkpoint matched the recipe and step count, all tensors were finite, strict state loading passed, and the exported backbone exactly matched the checkpoint.

The stable run trained 50 epochs fresh (19m28s) and then resumed from the epoch-50 checkpoint with optimizer, scheduler and scaler state through epoch 1000 (5h15m26s). Recorded environment: Python 3.12.3, torch 2.8.0+cu126, torchvision 0.23.0+cu126, Lightning 2.6.6, one RTX A6000, 8 CPUs and 32 GB host RAM. The [machine-readable summary](results/ssley-cifar10.json) is derived from the epoch histories, run status and final verification output; no dataset or model weights are included.

**Limits.** FP32 loss and vector-gradient clipping change the released arithmetic and optimization, so this is a stability variant, not a strict reproduction. The paper's Table 1 reports 91.43% top-1, 1.97 points above this result; historical software, seeds and checkpoint selection are not fully recovered, and the gap should not be attributed to unverified author practices. This is one seed on one GPU, without confidence intervals or multi-GPU validation.

## Tests

```bash
python -m pytest stable_pretraining/tests/unit/test_ssley*.py stable_pretraining/tests/unit/test_methods_smoke.py -k ssley -q
```

Coverage includes mathematical values/gradients and invariances, invalid inputs, low precision, label-free training, forward layouts, Manager/checkpoint integration, recipe and configuration semantics, detached classification, vector clipping, AMP recovery and failure guards. Pinned fixtures check author loss/gradient parity, initialization, two optimizer updates and the complete 195000-step learning-rate schedule. CUDA-specific cases skip without CUDA. Source parity does not establish identical historical training trajectories.
