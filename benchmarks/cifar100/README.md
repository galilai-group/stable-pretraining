# Rectified LpJEPA on CIFAR-100

[lpjepa-resnet18.py](lpjepa-resnet18.py) reproduces the ResNet-18 experiment
with `p=1`, `mu=0`, and the generalized-normal scale from
[Table 5 of the paper](https://arxiv.org/html/2602.01456v2#A10.T5).
It uses `spt.forward.lpjepa`, two online linear probes, and the `Sparsity` callback.

## Results

All runs use the same hyperparameters; only the seed changes. Accuracy is
**final-epoch online linear-probe top-1** on the 10,000-image CIFAR-100 test split,
used as validation during training. The probes learn from detached features;
these are not separately trained offline probes or best-checkpoint scores.
Mean ± standard deviation uses the sample standard deviation across three seeds.

| Run | Encoder top-1 (%) | Projector top-1 (%) | L0 sparsity | L1 sparsity |
| --- | ---: | ---: | ---: | ---: |
| Paper, Table 5 | 65.97 | 62.22 | 0.6474 | 0.3019 |
| [Seed 5](https://wandb.ai/rwgao_b-brown-university/spt-lpjepa-cifar100/runs/9387c9d3) | 65.60 | 62.43 | 0.6466 | 0.3012 |
| [Seed 6](https://wandb.ai/rwgao_b-brown-university/spt-lpjepa-cifar100/runs/662281a7) | 65.42 | 62.30 | 0.6473 | 0.3004 |
| [Seed 7](https://wandb.ai/rwgao_b-brown-university/spt-lpjepa-cifar100/runs/20b713b1) | 65.49 | 62.41 | 0.6435 | 0.2991 |
| **Mean ± SD** | **65.50 ± 0.09** | **62.38 ± 0.07** | **0.6458 ± 0.0020** | **0.3003 ± 0.0010** |

The encoder mean is 0.47 percentage points below the paper's reported value;
the projector mean is 0.16 points above it. The paper row is reported as published,
without assuming the same checkpoint-selection rule.

For rectified projections `z` of dimension `D`, L0 is the mean fraction of nonzero
coordinates, and L1 is the mean of `(||z||_1 / (||z||_2 + 1e-12))^2 / D`.
Lower values indicate greater sparsity; zero vectors contribute zero.
Validation statistics use all samples with distributed sum/count reduction.

## Reproduce

From the repository root, in an environment with CUDA-enabled PyTorch:

```bash
pip install -e .
python benchmarks/cifar100/lpjepa-resnet18.py --download-only --data-dir ./data

for seed in 5 6 7; do
    torchrun --standalone --nproc-per-node=2 benchmarks/cifar100/lpjepa-resnet18.py \
        --data-dir ./data --output-dir "./outputs/lpjepa-seed${seed}" --seed "$seed"
done
```

This runs the three seeds sequentially on two GPUs on one node. Add
`--wandb-project YOUR_PROJECT --wandb-entity YOUR_ENTITY` to each training command
for W&B logging; the local registry is enabled without W&B.

| Setting | Value |
| --- | --- |
| Data | CIFAR-100: 50,000 training images, 10,000 evaluation images |
| Duration | 1,000 epochs; 195,000 training steps |
| Batch size | 256 globally, 128 per GPU; two augmented views; drop incomplete training batches |
| Encoder | Randomly initialized ResNet-18; 3×3 stride-1 CIFAR stem, padding 2, no max pool |
| Projector | 512 → 512 → 512 → 512, BatchNorm/ReLU between layers, final ReLU |
| RDMReg | `p=1`, `mu=0`, automatic `sigma=1/sqrt(2)`, 8,192 random directions |
| Loss weights | MSE: 25; RDMReg: 250 |
| SSL optimizer | LARS, LR 0.03, momentum 0.9, weight decay 1e-4, trust coefficient 0.02, clipped adaptation |
| Probe optimizers | LARS, LR 0.01, momentum 0.9, zero weight decay, trust coefficient 0.02, weight adaptation enabled |
| Schedule | 10-epoch linear warmup from 3e-5, then per-step cosine decay to zero |
| Distributed training | Two GPUs, NCCL DDP, SyncBatchNorm, FP16 mixed precision, no gradient accumulation |

The augmentations match the reference recipe: 32-pixel random resized crops
(scale 0.2–1.0, bicubic), color jitter `(0.4, 0.4, 0.2, 0.1)` with probability 0.8,
grayscale with probability 0.2, solarization at 128 with probability 0.1,
and horizontal flips with probability 0.5.
Training normalization uses mean `(0.5071, 0.4865, 0.4409)` and standard deviation
`(0.2673, 0.2564, 0.2762)`; evaluation uses mean `(0.4914, 0.4822, 0.4465)` and
standard deviation `(0.247, 0.243, 0.261)`, preserving the reference code's
different training/evaluation normalizations.

[lpjepa_precision.py](lpjepa_precision.py) updates the FP16 gradient scaler once
after all three optimizers have stepped, matching the training runs. Using
Lightning's default scaler update after each optimizer changes this behavior.

The reported runs used **2× NVIDIA L40S** on Brown's Oscar cluster, Python 3.10.19,
PyTorch 2.10.0, torchvision 0.25.0, Lightning 2.6.1, timm 1.0.25,
torchmetrics 1.9.0, CUDA 12.8, and NCCL 2.27.5. W&B reports runtimes of
7h 17m, 6h 05m, and 5h 48m for seeds 5, 6, and 7 respectively.
There was no hyperparameter search and no checkpoint resume in these runs.

## Outputs and checkpoints

Metrics and full-state checkpoints are saved beneath
`<output-dir>/spt-cache/runs/`. The script retains the checkpoint with the best
encoder validation accuracy (`best-encoder.ckpt`) and a resumable `last.ckpt`.
The table above uses the final validation metrics, not the best checkpoint.
Resume with the same training arguments and `--resume /path/to/last.ckpt`.
Manual resume creates a new registry run and restarts best-checkpoint selection;
exact replay of random augmentations after resume is not guaranteed.

The completed runs used this recipe through the original cluster runner. This
portable entry point was packaged afterward: initialized models, augmentations,
optimizer settings, scheduler trajectory, and AMP helper behavior were checked
against that runner. The packaged script also passed a synthetic CPU training
and full-state resume check; it has not yet been rerun for 1,000 epochs or
launched on GPUs in this entry-point form. The linked W&B runs retain the
original launch configuration and source hashes.
