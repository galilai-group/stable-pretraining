# FlatNCE

Implementation of [Simpler, Faster, Stronger: Breaking The log-K Curse On
Contrastive Learners With FlatNCE](https://arxiv.org/abs/2107.01152), using the
repository's SimCLR encoder, projector, and two-view training pipeline.

## Objective

For normalized projections, let `s[i,j] = cosine(z[i], z[j]) / temperature`
and `p(i)` be anchor `i`'s positive (the other view of the same image). The
implementation follows Algorithm 1 and Equation 6:

```text
c[i] = logsumexp(s[i,j] - s[i,p(i)]) over j != i and j != p(i)
loss = mean(exp(c - c.detach()))
```

Both views are anchors. In one process, a batch of `B` images contributes `2B`
anchors and `2B - 2` negatives per anchor. With `W` distributed ranks, candidates
are gathered with autograd and there are `2BW - 2` negatives per local anchor.
Ranks must have equal batch sizes. At least two images are required globally.

**The loss value is always one, while its gradient trains the model.** Do not
use decreasing loss, finite differences of the detached expression, or the loss
value alone to judge correctness. Unit tests compare autograd gradients with an
independent negative-only log-partition reference. Keep both the self-pair and
the positive out of the negative sum; including the positive changes the gradient.

The loss promotes float16/bfloat16 projections to float32 and disables autocast
for its similarity and reduction operations. Float64 inputs retain their precision.

## Python API

Use the composable interface to supply your own backbone and projector:

```python
import stable_pretraining as spt
from torch import nn

backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
backbone.fc = nn.Identity()

module = spt.Module(
    backbone=backbone,
    projector=nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)),
    forward=spt.forward.flatnce,
    flatnce_loss=spt.losses.FlatNCELoss(temperature=0.5),
    optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
)
# spt.Manager(trainer=trainer, module=module, data=data_module)()
```

Training batches contain exactly two view dicts:

```python
batch = {"views": [
    {"image": first_view, "label": labels},
    {"image": second_view, "label": labels},
]}
```

Named view dicts are also accepted. Training returns concatenated `embedding`
and optional `label` tensors plus `loss`; single-view evaluation takes
`{"image": images, "label": labels}` and skips the projector and loss. Labels
are used for evaluation, not the contrastive objective.

The method class constructs SimCLR's standard MLP projector and also supports
training through `Manager`. Its batch-normalized projector requires at least
two local training images per rank:

```python
model = spt.FlatNCE(
    encoder_name="resnet18",
    low_resolution=True,
    projector_dims=(2048, 2048, 256),
    temperature=0.5,
    optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
)
# Direct tensor interface, before attaching evaluation callbacks:
# output = model(first_view, second_view)
# output.loss.backward()
```

For a minimal objective replacement, `spt.forward.simclr` also accepts
`simclr_loss=spt.losses.FlatNCELoss(...)`.

## CIFAR-10 experiment

From the repository root, with the package installed in your environment:

```bash
python benchmarks/flatnce/flatnce-resnet18.py --download --method flatnce \
  --batch-size 256 --temperature 0.5 --epochs 100 --seed 0

python benchmarks/flatnce/flatnce-resnet18.py --method simclr \
  --batch-size 256 --temperature 0.5 --epochs 100 --seed 0
```

`--method` selects only the objective and its forward function. Both runs use
the CIFAR-adapted ResNet-18, the same augmentations, and the original CIFAR-10
benchmark's `512 -> 2048 -> 2048 -> 256` projector (no final batch norm).
Optimization uses LARS (`lr=5`, `weight_decay=1e-6`) and step-based linear warmup
plus cosine decay. The learning rate is not automatically rescaled by batch size;
use `--lr` to select and document another policy.

The default evaluation set is a fixed 5,000-image holdout from CIFAR-10's
training split. The remaining 45,000 images train the encoder and probes.
`--eval-split test` selects the official test split for final reporting, while
keeping the same 45,000-image training subset. Choose hyperparameters using
the validation split. The split seed is fixed independently of training seed.

Useful options:

| Option | Default | Purpose |
|---|---|---|
| `--data-dir` | `${STABLE_PRETRAINING_DATA_DIR:-data}/cifar10` | CIFAR-10 download/cache directory. |
| `--output-dir` | `outputs/flatnce` | Local registry, metrics, and checkpoints. |
| `--accelerator`, `--devices` | `auto`, `1` | CPU, GPU, MPS, or multiple devices. |
| `--precision` | `32-true` | Also accepts `16-mixed` and `bf16-mixed` on supported hardware. |
| `--num-workers` | `4` | Use `0` for simple CPU smoke runs. |
| `--train-samples`, `--max-steps` | all, unlimited | Bound a short implementation check. |
| `--resume` | none | Absolute or relative checkpoint path; restores full training state. |

Resume interrupted runs with the original epoch/step horizon to preserve the
saved scheduler state.

Runs are saved under `OUTPUT_DIR/registry/runs/.../`. Each run contains:

- `metrics.csv`, `hparams.yaml`, and registry metadata;
- `experiment_summary.json` with settings, dataset identity, accuracy metrics,
  elapsed time, update count, and peak CUDA memory when applicable;
- `checkpoints/last.ckpt` for resuming, refreshed after every completed epoch.

Evaluation uses an online linear probe and a k-NN probe on backbone features.
The runner also logs model gradient norm, local positive cosine similarity,
and local embedding standard deviation. Under mixed precision the gradient
diagnostic is measured in `on_after_backward` and may include AMP loss scaling.

Inspect runs locally with:

```bash
spt web outputs/flatnce/registry
```

See [RESULTS.md](RESULTS.md) for the checked 10-epoch, seed-0 comparison and
its per-epoch data. This short comparison validates learning behavior; it is
not a reproduction of the paper's full experimental protocol.

## Reproducible experiment plan

Compare both objectives at batch sizes `32, 64, 128, 256` and temperatures
`0.1, 0.2, 0.5`. Start with a short run, then use the same epoch budget for each
main comparison. Repeat selected settings with seeds `0, 1, 2` and report mean
and standard deviation of held-out accuracy, runtime, and memory. This is a
suggested experiment matrix, not a claim of reproduced paper results.

Keep architecture, initialization seed, augmentations, split, optimizer,
learning-rate policy, and evaluation settings matched between objectives.
Gradient accumulation does not enlarge the negatives available in one forward.
The online linear probe is a training diagnostic; it is not a separately trained
post-pretraining frozen linear evaluation.

## Verification

A synthetic end-to-end run requires no dataset download:

```bash
python benchmarks/flatnce/flatnce-resnet18.py --smoke-test \
  --batch-size 4 --train-samples 8 --epochs 1 --max-steps 2 \
  --num-workers 0 --accelerator cpu
```

Synthetic accuracy is not CIFAR-10 accuracy. For a small real-data check:

```bash
python benchmarks/flatnce/flatnce-resnet18.py --download \
  --batch-size 8 --train-samples 16 --validation-size 16 \
  --epochs 1 --max-steps 2 --num-workers 0 --accelerator cpu
```

Run loss, forward, and training tests:

```bash
python -m pytest stable_pretraining/tests/unit/test_flatnce.py \
  stable_pretraining/tests/integration/test_flatnce.py -q

python -m pytest stable_pretraining/tests/distributed/test_collectives_dist.py \
  -k flatnce -q
```

The distributed test starts two CPU/Gloo ranks and compares DDP parameter
gradients with an independent global reference, including one image per rank.
Existing SimCLR forward, loss, method smoke, and boundary tests cover shared
code compatibility.

See [VALIDATION.md](VALIDATION.md) for the implementation summary and the checks
actually performed, including their limits.
