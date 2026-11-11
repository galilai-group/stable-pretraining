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

The example uses existing framework components:

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

`examples/simco_cifar.py` uses a CIFAR-stem ResNet18, a biased
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
python examples/simco_cifar.py --dataset cifar10 --data-dir "$CIFAR_DATA_DIR" --download-only
python examples/simco_cifar.py --dataset cifar100 --data-dir "$CIFAR_DATA_DIR" --download-only
```

Within an allocation containing two GPUs, run a short real-data check first:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  examples/simco_cifar.py --devices 2 --dataset cifar10 \
  --data-dir "$CIFAR_DATA_DIR" --output-dir outputs/simco-cifar10-check \
  --batch-size 256 --epochs 1 --max-steps 10 --warmup-epochs 0
```

Then start independent full experiments, using distinct output directories:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  examples/simco_cifar.py --devices 2 --dataset cifar10 \
  --data-dir "$CIFAR_DATA_DIR" --output-dir outputs/simco-cifar10-500 \
  --batch-size 256 --epochs 500 --warmup-epochs 10

python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  examples/simco_cifar.py --devices 2 --dataset cifar100 \
  --data-dir "$CIFAR_DATA_DIR" --output-dir outputs/simco-cifar100-500 \
  --batch-size 256 --epochs 500 --warmup-epochs 10
```

The example supports a single node. Use exactly one launcher process; torchrun
creates the two DDP workers. On Slurm, one task requesting two GPUs can invoke
the command through `srun --ntasks=1`. Do not create two Slurm launcher tasks.
For one GPU, invoke `python examples/simco_cifar.py --devices 1 ...` directly.
See the official [torchrun documentation](https://docs.pytorch.org/docs/stable/elastic/run.html)
for launcher configuration.

Start each training horizon independently. A two-epoch or ten-step checkpoint
was built with a different cosine schedule and cannot establish a 500-epoch
continuation result. Resume an interrupted SimCo run using
`--resume /absolute/path/to/latest.ckpt` with its original dataset, model,
split, device count and full scheduling horizon. The synthetic Manager
integration test verifies complete-state checkpoint continuation. Continuation
from either completed 500-epoch checkpoint and automatic Slurm requeue remain
unverified.

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

## Tests and experiment status

Run only unit tests locally, following `TESTING.md`:

```bash
python -m pytest stable_pretraining/tests -m unit
```

On Linux CI or Oscar, the distributed regression suite runs actual two-process
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

The historical Linux validation output from job 7816077 reports all five SimCo
integration tests passed with no skips, including the full-state synthetic continuation test and
real two-process gradient/update checks. The training logs also report three
distributed tests passed before each 500-epoch run.

Historical validation job 7822625, before the subsequent utility/test fixes,
passed all 81 SimCo unit cases on Linux CPU. The whole unit suite
reported 3,820 passed, 11 failed and 20 skipped. Pristine upstream at
`ab836bf699a2be712dfcf980a5eb70d36391e876`, run in the same environment, reported
3,739 passed, the same 11 failed, and 20 skipped. Failure-node sets and error
causes match: one home-directory symlink assertion and ten existing statistics
tests using unsupported Gloo `ReduceOp.AVG` under PyTorch 2.6. There are no
additional failing cases in that SimCo comparison, but the whole suite returned
a nonzero exit code.

The statistics helper now uses differentiable `SUM / world_size` instead of
unsupported Gloo `AVG`. The tilde-expansion regression compares resolved cache
paths and exercises isolated direct, parent-component and symlink home layouts.
A subsequent user-provided `unit-comparison.json` reports that both the SimCo
unit command and complete unit-suite command exited with code 0 after these
fixes, with no failing node IDs and no baseline run needed. This run used CPU
with inherited Slurm/elastic/MPI/rank variables cleared. Its job ID, test/skip
counts, duration, source hashes and integration results were not supplied;
post-fix integration execution and CI status remain unverified. The historical
counts above remain the record of the pre-fix run. These utility/test fixes do
not change the SimCo objective or recipe, so the completed 500-epoch training
runs do not need to be repeated.

### Reported 500-epoch results

On 2026-10-22, user-provided Oscar configurations and summaries recorded the
following independently initialized runs, each with 500 epochs, 87,500 updates,
two RTX A5000 GPUs, DDP, SyncBatchNorm, global batch 256, and seed 0:

| Dataset | Best online validation top-1 | Final online validation top-1 | Final SSL loss | Wall time |
|---|---:|---:|---:|---:|
| CIFAR-10 | 88.58% | 88.18% | 1.0838 | 2.69 h |
| CIFAR-100 | 60.76% | 60.34% | 1.0864 | 2.88 h |

The environment records PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, and
Lightning 2.6.6. Both complete protocol hashes agree between config and summary.
These are **online probe results on held-out training-set validation images**,
not frozen linear evaluation, official test accuracy, or a reproduction of
paper accuracy. The best checkpoints were selected using validation accuracy;
the table reports final accuracy separately.

The training logs confirm both runs reached `max_epochs=500`. A subsequent
Oscar validation checked the original protocol/split hashes, complete disjoint
45,000/5,000 partitions, and final checkpoint epoch 499 / step 87,500 with two
populated optimizer states and saved probe parameters. It also confirmed metric
CSV files exist. The received local split copies were truncated during transfer;
the original scratch files passed these checks.

Full metric CSV, checkpoint files and prediction shards have not been transferred
for local inspection. CSV contents and prediction-derived accuracy remain
unchecked. The passing continuation test uses a new synthetic checkpoint rather
than resuming either completed 500-epoch checkpoint. Compact results and provenance
are recorded under `benchmarks/cifar/simco/`.

Earlier ten-step and two-epoch CIFAR-10 checks were separate runs with different
scheduler horizons and must not be joined into a continuous training curve.
Copy final artifacts to durable storage when using temporary scratch.
