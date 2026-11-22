# ADNCE: adaptive contrastive comparisons

ADNCE learns from two independently augmented views of each image. It uses the
same encoder/projector pipeline as SimCLR, but adjusts the contribution of each
negative comparison using a Gaussian function of its cosine similarity. The
center (`mu`) and width (`sigma`) determine which similarities receive more
weight. Image labels are unnecessary for this objective.

The method is from [Wu et al., *Understanding Contrastive Learning via
Distributionally Robust Optimization*, NeurIPS 2023](https://arxiv.org/abs/2310.11048).
The implementation follows the authors' image-loss convention at
[commit `cdd110f`](https://github.com/junkangwu/ADNCE/blob/cdd110f85402641d52762ba63aa2a17ba907a6f9/image/model.py),
with an independent implementation using stable logarithmic operations.

## Reference conventions

For a global batch of `B` image pairs, every one of the `2B` views is an anchor. Its
positive is the other view of the same image. Its `2B - 2` negatives exclude
itself and that positive. Features are normalized before computing similarities.

For negative cosine similarities `s`, the unnormalized weights are proportional
to `exp(-0.5 * ((s - mu) / sigma)**2)`. Each anchor's weights are normalized to
have mean one. Those weights are detached before use: gradients flow through
similarities, but not through the weight calculation. The positive exponential
score remains in the denominator, matching the authors' image implementation.

These details matter when comparing implementations: the upstream README's
short pseudocode applies the Gaussian to a variable described as an exponential
score, while the actual image code converts it back to cosine similarity first.
The paper's theoretical objective also has a different denominator presentation.
Our reference is the executable image implementation, not that pseudocode.

`ADNCELoss(temperature=0.4, mu=0.7, sigma=0.5)` uses the CIFAR-10 settings from
the authors' recipe. It returns one scalar and accepts unnormalized `[B, D]`
feature tensors. Distributed training gathers features with gradients across
processes; each process must provide the same positive local batch size, and
the global batch must contain at least two images. Use `drop_last=True` and the
normal distributed sampler. Accumulating gradients does not increase the number
of negatives participating in any one forward pass.

## Library usage

```python
import torch
import stable_pretraining as spt

model = spt.ADNCE(
    backbone=spt.backbone.from_torchvision("resnet18", low_resolution=True),
    projector=torch.nn.Sequential(
        torch.nn.Linear(512, 512, bias=False),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
    ),
    temperature=0.4,
    mu=0.7,
    sigma=0.5,
    optim={
        "optimizer": {"type": "Adam", "lr": 1e-3, "weight_decay": 1e-6},
        "scheduler": {"type": "ConstantLR", "factor": 1.0},
    },
)
```

Alternatively, compose `spt.Module(forward=spt.forward.adnce, backbone=...,
projector=..., adnce_loss=spt.losses.ADNCELoss(...), optim=...)`.
`MultiViewTransform([augment, augment])` supplies the two views. Training returns
`loss` and concatenated encoder `embedding` values; labels are aligned with the
embeddings when present. Validation/prediction accepts a single image batch.
Use `spt.Manager` to run training and existing `OnlineProbe` or `OnlineKNN`
callbacks to evaluate features without changing the encoder's gradients.

## Paired CIFAR development experiment

From a checkout with the package installed, run the default CIFAR-10 experiment:

```bash
python examples/adnce_cifar10.py --method adnce --download \
    --data-dir ~/data --output-dir outputs/adnce --accelerator gpu \
    --epochs 50 --seed 0
python examples/adnce_cifar10.py --method simclr \
    --data-dir ~/data --output-dir outputs/simclr --accelerator gpu \
    --epochs 50 --seed 0
```

Both modes share the seed, initialization, architecture, augmentations, optimizer,
temperature, data split, and evaluation procedure. Only the loss changes. Use
identical options on both commands, then repeat with additional seeds before
claiming a reliable improvement. Hyperparameter comparisons should give the
baseline a comparable tuning budget, including its temperature.

For CIFAR-100, add `--dataset cifar100`. The online probe then predicts 100
classes, and the example keeps the same fixed 45,000/5,000 training-only split.
The authors' [CIFAR-100 recipe](https://github.com/junkangwu/ADNCE/blob/cdd110f85402641d52762ba63aa2a17ba907a6f9/image/README.md)
uses temperature `0.3`, `mu=0.5`, and `sigma=1.0`; select these explicitly because
the CLI retains its CIFAR-10 loss defaults:

```bash
python examples/adnce_cifar10.py --dataset cifar100 --method adnce --download \
    --temperature 0.3 --mu 0.5 --sigma 1.0 --epochs 50 --seed 0 \
    --data-dir ~/data --output-dir outputs/adnce-cifar100 --accelerator gpu
python examples/adnce_cifar10.py --dataset cifar100 --method simclr \
    --temperature 0.3 --epochs 50 --seed 0 \
    --data-dir ~/data --output-dir outputs/simclr-cifar100 --accelerator gpu
```

Both datasets use mean `(0.4914, 0.4822, 0.4465)` and standard deviation
`(0.2023, 0.1994, 0.2010)`, matching the shared transform in the authors'
[`image/utils.py`](https://github.com/junkangwu/ADNCE/blob/cdd110f85402641d52762ba63aa2a17ba907a6f9/image/utils.py).
The held-out split and ridge diagnostic remain development adaptations on
CIFAR-100 as well; this option does not reproduce the paper's test protocol.

The defaults use the reference architecture and optimizer: low-resolution
ResNet-50 (3-by-3 first convolution, no max-pool), a `2048 → 512 → 128` projector
with batch normalization and ReLU in the hidden layer, Adam at constant learning
rate `1e-3`, weight decay `1e-6`, batch size 256, and 400 epochs. The example's
Gaussian blur uses the existing torchvision-based transform with the same
kernel size, probability, and sigma range as the authors' OpenCV blur; those
backends need not produce pixel-identical results. `--backbone resnet18` is a
smaller architecture adaptation, not the reference model.

This is a **development benchmark**, not a paper-results reproduction:

- A fixed, stratified 45,000/5,000 split of the selected dataset's training set keeps held-out
  images out of SSL training and probe fitting. `--split-seed` controls that split
  independently of the model seed. The official test set is never accessed.
- An existing online linear probe monitors progress. Its predictions do not
  backpropagate into the encoder. At the start and end, a fresh ridge classifier
  is fit to frozen, L2-normalized encoder features. Standardization is fit only
  on training features; ridge regularization is fixed by `--probe-alpha`.
- The resulting validation accuracy is a frozen-feature diagnostic, **not** the
  authors' weighted kNN score and not a test-set score. Initial accuracy and
  mean per-feature standard deviation help detect failure to learn or collapse.
- Shorter schedules, training subsets, and ResNet-18 must be reported as
  adaptations. A successful smoke test alone does not demonstrate useful learning.

`--batch-size` is per process; changing `--devices` changes the global batch and
negative count. The single-device command is closest to the reference batch
normalization behavior. Do not compare runs with different device counts as a
controlled loss-only experiment.

For a bounded debugging run after downloading the data:

```bash
python examples/adnce_cifar10.py --backbone resnet18 --accelerator cpu \
    --epochs 1 --max-steps 2 --batch-size 16 --num-workers 0 \
    --train-samples 128 --val-size 100 --probe-samples 100 \
    --output-dir outputs/adnce-smoke
```

Use `--no-probe` to skip the initial/final frozen evaluation; the online probe
remains available. GPU mixed precision is selectable with `--precision bf16-mixed`
on supported hardware. The default `32-true` avoids relying on such support.

## Outputs and resume

`Manager` creates a unique run folder beneath `--output-dir`, with logs and
checkpoints. `evaluation.json` records the arguments, versions, reference commit,
completed steps, validation diagnostics, elapsed time for the current invocation,
and rank-zero peak CUDA allocation. Elapsed time includes training and online
validation, but excludes initial/final frozen-probe fitting. With multiple
processes, the frozen probe runs on rank zero while other processes wait.
After training, evaluation runs on the training device before teardown. The
report records that device as `final_frozen_probe_device`. If an invocation does
no training (for example, a resumed checkpoint already meets the requested step
limit), a fallback still writes the report, but evaluates on CPU after teardown.

The final probe does not overwrite the original score after resume; the initial
score is stored in callback checkpoint state. An unrelated checkpoint lacking
that state reports no initial score rather than incorrectly evaluating trained
features as an untrained baseline.

Resume a **trusted** local checkpoint with the same dataset, architecture, loss
settings, split, and batch configuration, setting the desired total epoch count:

```bash
python examples/adnce_cifar10.py --method adnce --accelerator gpu \
    --epochs 50 --resume /absolute/path/to/checkpoints/last.ckpt \
    --data-dir ~/data --output-dir outputs/adnce-resumed --seed 0
```

Resume restores optimizer and callback state through `Manager`. The example's
seeded data loader restarts its random stream after a fresh invocation; resume
is not promised to reproduce the exact uninterrupted minibatch/augmentation
sequence, especially for a checkpoint saved partway through an epoch. The JSON
report includes a dirty-working-tree flag: a commit ID alone does not identify
uncommitted changes.

Keep generated
datasets, checkpoints, and personal cluster launchers outside a contribution.
Use `spt web outputs/` to inspect locally recorded training curves.

## Evidence to report

Record each run's actual schedule, split, model, global batch size, seed, software
version, code revision, validation accuracy, elapsed time, and memory use. Report
unfinished, skipped, or failed checks explicitly. This example contains no
pre-filled accuracy claims; fill an experiment table only after runs finish.
