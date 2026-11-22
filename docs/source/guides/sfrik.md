---
orphan: true
---

# SFRIK

SFRIK (Zheng et al., ICLR 2023) matches augmented views while encouraging their
representations to spread over the unit sphere. This implementation covers the
**two-view, degree-three** objective, not the paper's multi-crop extensions.

- [Paper](https://arxiv.org/abs/2208.00789)
- [Pinned reference implementation](https://github.com/valeoai/sfrik/blob/c6ce16ed36ecb9c68a3e3c42dce3ec1a27e6c9d9/src/sfrik/ssl.py)

## Objective

For normalized projections {math}`u_i,v_i\in\mathbb{R}^d`,

```{math}
L = \frac{\alpha}{Bd}\sum_i\|u_i-v_i\|_2^2
    + \frac{\beta}{2B^2}\sum_{i,j}
      [k(u_i^\top u_j)+k(v_i^\top v_j)],

```

where {math}`k(t)=w_1P_1(t)+w_2P_2(t)+w_3P_3(t)` and

```{math}
P_1(t)=t,\quad P_2(t)=\frac{dt^2-1}{d-1},\quad
P_3(t)=\frac{(d+2)t^3-3t}{d-1}.

```

The kernel means **include diagonal entries**, matching the reference's biased
MMD estimator. These polynomials integrate to zero under the uniform target,
so no target samples are needed. The three-degree kernel constrains selected
low-order moments; zero discrepancy does not uniquely identify uniformity.

`SFRIKLoss` normalizes inputs internally. Its defaults (`sim_coeff=4000`,
`mmd_coeff=1`, `kernel_weights=(1, 40, 40)`) match the reference's
8192-dimensional small-dataset recipe. Since MSE averages over dimensions,
retune its coefficient when changing projector width. Our 512-dimensional
benchmark uses 256, approximately preserving the coefficient-to-width ratio.

Float16/bfloat16 inputs are promoted to float32 and the loss disables autocast;
float64 is preserved. Normalization uses an epsilon denominator for zero vectors,
which are not points on the sphere and do not represent a valid uniform solution.

## Usage

Supply an encoder and projector. The reference projector uses Linear-BN-ReLU
hidden layers and a final bias-free linear layer:

```python
import torch
import stable_pretraining as spt

backbone = spt.backbone.from_torchvision(
    "resnet18", low_resolution=True, weights=None
)
backbone.fc = torch.nn.Identity()
projector = torch.nn.Sequential(
    torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU(),
    torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU(),
    torch.nn.Linear(512, 512, bias=False),
)
model = spt.SFRIK(
    backbone=backbone, projector=projector, sim_coeff=256.0,
    optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
)

```

Train through `spt.Manager` as in [the quickstart](quickstart.md). Batches contain two
view dictionaries under `views` (a list or named mapping); each view has
`image` and optional `label`. Training returns concatenated `embedding`,
raw `projection`, optional labels and scalar `loss`. Labels are for probes,
never for the SSL objective. Single-view evaluation bypasses the projector and
loss. `OnlineProbe` and `OnlineKNN` can consume `embedding` and `label`.

For custom composition, use `spt.Module(forward=spt.forward.sfrik, ...,
sfrik_loss=spt.losses.SFRIKLoss(sim_coeff=256.0))` with the same components.
Loss settings are saved in checkpoint hyperparameters; supply backbone/projector
instances and optimizer configuration when using `SFRIK.load_from_checkpoint`.
For full training continuation, pass the checkpoint to `Manager` with
`weights_only=False`.

## Distributed behavior

The loss gathers both views with autograd and evaluates the global objective
on every rank. Gather-backward summation and DDP gradient averaging reproduce
a single-process global-batch gradient; no world-size multiplier is needed.
Workers must have **equal nonempty local batch sizes** and matching feature
shapes. Use a distributed sampler and `drop_last=True`. Set
`gather_distributed=False` only for an intentional local-batch objective.
Memory is quadratic in global batch size; gradient accumulation does not enlarge
the batch seen by the kernel penalty.

## Learning evidence and reproduction

The primary experiment trains ResNet18 with a CIFAR stem for **200 epochs**
on 45,000 CIFAR-10 training images, holding out the remaining 5,000 for
validation. The official test set is unused. SFRIK and SimCLR share seed 42,
the exact split, initialization, augmentations and optimizer recipe.

Both use a 512-512-512 projector, batch 128, AdamW (lr 0.001, weight decay
0.0001), and cosine decay over 200 epochs: 70,200 optimizer steps each.
SFRIK uses `sim_coeff=256`, `mmd_coeff=1` and weights `(1, 40, 40)`;
SimCLR uses temperature 0.5. Runs used float32 MPS on an Apple M2 Max,
Python 3.12.15, PyTorch 2.14.1 and Lightning 2.6.6.

| Encoder | Linear | 20-NN |
| --- | ---: | ---: |
| Shared initialization | 31.66% | 36.86% |
| SFRIK, epoch 200 | 79.50% | 78.28% |
| SimCLR, epoch 200 | 73.94% | 73.78% |

The linear probe is a fresh logistic regression (C=1, L-BFGS, maximum 1,000
iterations), fitted on L2-normalized frozen backbone features from the training
subset. The 20-NN probe uses cosine similarity and unweighted votes. Both use
deterministic image preprocessing and evaluate on the held-out subset; these
are **frozen-probe**, not online-probe, accuracies. Labels never enter SSL training.

```{figure} images/sfrik-learning-curve.svg
:alt: Frozen linear accuracy increases from 31.66 percent to 79.50 percent for SFRIK and 73.94 percent for SimCLR over 200 epochs.
:width: 700px

Evaluations of saved encoders at epochs 0, 1, 20, 50, 100 and 200.
Lines connect measurements; they are not per-epoch evaluations.

```

Both runs completed with finite losses and gradients. SFRIK's final feature
effective rank was 269.3 (initially 341.9). Rank is a collapse diagnostic,
not an accuracy score: random features can have higher rank than learned ones.
This single-seed comparison demonstrates learning under a shared recipe;
it does not establish general superiority over SimCLR or reproduce the paper.

Run these commands from the repository root after installing its dependencies.
Each output directory must be new:

```bash
python benchmarks/cifar10/sfrik-resnet18.py \
    --accelerator mps --data-dir /path/to/cifar-cache \
    --output-dir /path/to/results/sfrik-200ep \
    --method sfrik --epochs 200 --train-size 45000 --val-size 5000 --seed 42

python benchmarks/cifar10/sfrik-resnet18.py \
    --accelerator mps --data-dir /path/to/cifar-cache \
    --output-dir /path/to/results/simclr-200ep \
    --method simclr --epochs 200 --train-size 45000 --val-size 5000 --seed 42

```

These commands reproduce the training recipe and initial/final evaluations;
exact results can vary with hardware and package versions. Use `--accelerator
cpu` or `cuda` for other devices. The script writes `results.json` with
configuration, exact split indices, epoch losses and metrics, plus `last.ckpt`.
The displayed intermediate measurements came from additional encoder exports
in the experimental run; the benchmark evaluates only before and after training.

For a shorter check, omit `--epochs`, `--train-size` and `--val-size`:
defaults are 20 epochs, 10,000 training images and 2,000 held-out images.
SFRIK's frozen linear accuracy improved from 27.20% to 44.55% for seed 42 and
27.40% to 45.20% for seed 7. These use different data sizes and schedules from
the primary experiment and are not points on its learning curve.

{download}`Download the compact results <../../../benchmarks/cifar10/sfrik-results.json>`
for both protocols, including the plotted measurements and evaluation settings.

## Tests

```bash
python -m pytest stable_pretraining/tests/unit/test_sfrik.py -q
python -m pytest stable_pretraining/tests/regression/test_sfrik.py -q

```

Tests cover independent moment-form values/gradients, gradcheck, geometric
invariances, precision handling, available accelerator parity, callback
integration, checkpoint continuation and real two-process Gloo/DDP gradient
and optimizer-step equivalence, plus Manager training with spawned DDP workers.
These checks passed on macOS/Python 3.12 and Linux/Python 3.11 (CPU/Gloo).
CUDA/NCCL has not been validated.
