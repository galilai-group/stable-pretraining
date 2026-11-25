# SimCo on CIFAR-10 and CIFAR-100

SimCo is the symmetric, shared-encoder method from Zhang et al.,
[Dual Temperature Helps Contrastive Learning Without Many Negative Samples:
Towards Understanding and Simplifying MoCo (CVPR 2022)](https://arxiv.org/abs/2203.17248).
The numerical reference is the authors' [loss](https://github.com/ChaoningZhang/Dual-temperature/blob/e8d5e768255721b23643d948631f80c0b8a0851b/solo/losses/dual_temperature_loss.py)
and [SimCo method](https://github.com/ChaoningZhang/Dual-temperature/blob/e8d5e768255721b23643d948631f80c0b8a0851b/solo/methods/simco_dual_temperature.py).

```{figure} ../figures/simco_overview.svg
:alt: Two independently augmented views use a shared encoder and projector; normalized projections are gathered across ranks for the symmetric dual-temperature SimCo loss.
:width: 100%

SimCo overview. Both views receive gradients; only the dual-temperature hardness
ratio is detached. The online probe uses detached encoder features.
```

## Objective and framework wiring

For row-normalized projections `Q, K`, form `S = Q @ K.T`. Matching rows
are positives; only other images from the opposite view are negatives.
For each row, with `a = intra_temperature` and `b = inter_temperature`:

```text
p_a = softmax(S / a)[positive]
p_b = softmax(S / b)[positive]
weight = stop_gradient((1 - p_b) / (1 - p_a))
directional_loss = mean(weight * -log(p_a))
loss = (directional_loss(S) + directional_loss(S.T)) / 2
```

Defaults are `a=0.1, b=1.0`, equivalent to the authors' `temperature=0.1,
dt_m=10`. Both views receive gradients. The hardness ratio is detached;
removing that detach changes the method. When temperatures match, the loss
reduces to symmetric cross-view InfoNCE with `N-1` negatives per anchor,
not SimCLR's `2N-2` negatives. There is no teacher, queue or predictor.

The implementation evaluates negative probability mass and the weighted
cross-entropy in log space to avoid saturated-softmax `0/0`. Half and BF16
inputs use FP32 loss arithmetic; float64 is preserved. Existing differentiable
`all_gather` builds a global batch in rank order for both views. Local batch
shapes must match, and the global batch must contain at least two examples.
Every rank evaluates the global loss. Gather's backward sum and DDP's
parameter-gradient average together yield the global-reference parameter
gradient; no extra world-size factor is applied to the loss.

The benchmark uses five framework components.

1. `DataModule` supplies two independent augmented view dictionaries.
2. `spt.SimCo` combines a backbone, projector, `SimCoLoss` and `forward.simco`.
3. `OnlineProbe` trains a separate classifier on detached backbone features.
4. `OnlineWriter` records validation IDs, labels and classifier logits.
5. `Manager(...)()` runs Lightning and manages the run registry/checkpoints.

Each training view contains `image` with shape `(B, 3, 32, 32)`. Optional
`label` and `sample_idx` tensors are concatenated in the same view order as
the embeddings. Missing or all-`None` optional fields are omitted. If a field
is supplied, both views must provide tensors with matching batch/trailing
shapes. Labels are consumed only by the detached probe, never by `SimCoLoss`.
Validation accepts a single view dictionary.

## Recipe and step budget

`benchmarks/cifar/simco/simco_cifar.py` uses a CIFAR-stem ResNet18, a biased
`512 -> 128 -> 128` projector with ReLU and no projector BatchNorm, SGD
learning rate 0.03, momentum 0.9 and weight decay 5e-4. The default **500-epoch**,
**global batch 256** recipe warms up for 10 epochs and then applies cosine
decay. The scheduler receives optimizer-step counts: 175 steps per epoch,
**87,500 steps total**, and 1,750 warmup steps. Short checks explicitly set
`--warmup-epochs 0`; warmup must be shorter than the full step budget.

This is an adaptation of the authors' CIFAR configuration, not a reproduction
of its accuracy. Differences include a fixed 45,000/5,000 split from the
official training set, FP32 by default, and probe SGD learning rate 0.1
(configurable using `--probe-learning-rate`; the released launcher uses 0.3).
The resized-crop/color-jitter settings are saved completely in `config.json`.
SSL and the probe see only the training partition; the official 10,000-image
test split is unused. Keep `--split-seed` fixed across runs and methods.

`--batch-size` always denotes the global batch. With `--devices 2`, each
process receives 128 examples and a complete 500-epoch run still takes
87,500 optimizer steps. Lightning installs distributed samplers, shuffles the
training partition each epoch, and synchronizes validation metrics. GPU DDP
uses SyncBatchNorm in the backbone; CPU DDP leaves ordinary BatchNorm in place.
Both partition sizes must be divisible by the device count so sampler padding
cannot duplicate validation examples. Training drops the incomplete last
batch: 44,800 examples per epoch at batch 256.

## Run with one or two devices

Install this checkout following the project installation instructions. Prepare
both datasets once in an existing data directory:

```bash
export CIFAR_DATA_DIR=/absolute/path/to/cifar
python benchmarks/cifar/simco/simco_cifar.py --dataset cifar10 --data-dir "$CIFAR_DATA_DIR" --download-only
python benchmarks/cifar/simco/simco_cifar.py --dataset cifar100 --data-dir "$CIFAR_DATA_DIR" --download-only
```

Within an allocation containing two GPUs, run a short real-data check first:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  benchmarks/cifar/simco/simco_cifar.py --devices 2 --dataset cifar10 \
  --data-dir "$CIFAR_DATA_DIR" --output-dir outputs/simco-cifar10-check \
  --batch-size 256 --epochs 1 --max-steps 10 --warmup-epochs 0
```

Then start independent full experiments, using distinct output directories:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  benchmarks/cifar/simco/simco_cifar.py --devices 2 --dataset cifar10 \
  --data-dir "$CIFAR_DATA_DIR" --output-dir outputs/simco-cifar10-500 \
  --batch-size 256 --epochs 500 --warmup-epochs 10

python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  benchmarks/cifar/simco/simco_cifar.py --devices 2 --dataset cifar100 \
  --data-dir "$CIFAR_DATA_DIR" --output-dir outputs/simco-cifar100-500 \
  --batch-size 256 --epochs 500 --warmup-epochs 10
```

The benchmark supports a single node. Use exactly one launcher process; torchrun
creates the two DDP workers. On Slurm, one task requesting two GPUs can invoke
the command through `srun --ntasks=1`. Do not create two Slurm launcher tasks.
For one GPU, invoke `python benchmarks/cifar/simco/simco_cifar.py --devices 1 ...` directly.
See the official [torchrun documentation](https://docs.pytorch.org/docs/stable/elastic/run.html)
for launcher configuration.

To resume an interrupted run, pass `--resume /absolute/path/to/latest.ckpt` and
keep the original dataset, model, split, device count and scheduling horizon.
The learning-rate schedule depends on the total training budget.

## Artifacts and validation

Each output directory contains:

- `config.json`: arguments, complete protocol, SHA256 and metric definitions.
- `split_indices.json`: exact train/validation partition, hashed canonically.
- `summary.json`: method/dataset, final metrics, best validation online accuracy,
  step count, duration, actual world size and checkpoint paths.
- `validation_records/`: saved `sample_idx`, `label` and `linear_probe_preds`.
- `runs/`: Manager registry, environment information, metric CSV and checkpoints.

Only rank zero writes config, split and summary JSON. `summary.distributed`
records the actual world size, strategy, global/per-device batch, and
SyncBatchNorm setting. A completed full run should report world size 2 and
87,500 optimizer steps.

Each rank writes separate validation files named with `device=LOCAL_RANK`.
Concatenate **all ranks for the same epoch**, then use `sample_idx` to map
each prediction to `split_indices.json`'s validation list. At two devices,
the 5,000 validation examples are covered exactly once in total. Do not
interpret one rank's records as the complete validation set.

Two standard `ModelCheckpoint` callbacks save the best validation model and
overwrite `latest.ckpt` every training epoch. `summary.last_checkpoint` points
to the latter. Manager redirects checkpoint directories into its nested run
directory; read the actual summary paths.

`fit/loss` is the SSL objective before adding detached probe cross-entropy
(lower is better for this fixed objective). Validation accuracy is a fraction
in `[0,1]`, higher is better. Summary metrics are final values; they need not
match the best checkpoint. CPU/RAM hardware readings describe the whole node,
and GPU readings describe the device; neither is process peak memory.

Online probe accuracy measures the classifier trained alongside the encoder.
Frozen-encoder evaluation, matched baselines and multiple seeds can support
broader representation comparisons. Reserve official test evaluation until
the protocol is fixed. Equal-temperature cross-view InfoNCE is a useful
ablation supported by `--inter-temperature 0.1`, but is not full SimCLR.

## Tests

Run only unit tests locally, following `TESTING.md`:

```bash
python -m pytest stable_pretraining/tests -m unit
```

The distributed integration suite on Linux runs two-process
Gloo collectives. It compares the loss, both-view input gradients, DDP
parameter gradients, two SGD updates and momentum state against an independent
global-batch reference. A separate two-process Manager check verifies shared
artifacts, checkpoints, distributed validation records and aggregate accuracy:

```bash
python -m pytest stable_pretraining/tests/integration/test_simco_ddp.py \
  -m 'integration and distributed' -q
```

The Manager integration suite also checks a short checkpoint continuation:

```bash
python -m pytest stable_pretraining/tests/integration/test_simco_cifar.py \
  stable_pretraining/tests/integration/test_simco_ddp.py -m integration -q
```

## Reported 500-epoch results

Each dataset used a separate training run with seed 0, two RTX A5000 GPUs,
DDP with SyncBatchNorm, and global batch size 256. The table reports the
online linear probe's highest and final top-1 accuracy on the 5,000 held-out
validation images. Each run completed 87,500 optimizer steps.

| Dataset | Best online validation top-1 | Final online validation top-1 | Final SSL loss | Wall time |
|---|---:|---:|---:|---:|
| CIFAR-10 | 88.58% | 88.18% | 1.0838 | 2.69 h |
| CIFAR-100 | 60.76% | 60.34% | 1.0864 | 2.88 h |

The benchmark directory contains the full-precision metrics, environment
versions, and protocol and split hashes in `benchmarks/cifar/simco/results.json`.
