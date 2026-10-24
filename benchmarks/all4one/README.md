# All4One: reproducible training and recorded results

This guide provides training commands and recorded results for the All4One example. It distinguishes the released-code recipe from exact paper
reproduction. See the [reference audit](../../docs/source/guides/all4one.rst)
for algorithmic differences and evaluation limitations.

## Install and prepare data

From a checkout of this repository, create a Python environment with a
CUDA-enabled PyTorch/torchvision pair appropriate for your GPU, then install:

```bash
python -m pip install -e .
```

The recorded runs used Python 3.10.20, PyTorch 2.11.0+cu126 and torchvision
0.26.0+cu126.

Set the Python environment and dataset directory from the repository root:

```bash
export ALL4ONE_PYTHON="$PWD/.venv/bin/python"
export ALL4ONE_DATA_ROOT="/path/to/datasets/cifar"
```

The dataset directory must contain torchvision's `cifar-10-batches-py/` or
`cifar-100-python/`. Use `--download` on the pretraining command if an explicit
CIFAR download is needed. ImageNet-100 requires existing `train/<class>/` and
`val/<class>/` folders with identical 100-class mappings.

## Training commands

Run these commands on an allocated GPU, using the existing example:

```bash
"$ALL4ONE_PYTHON" examples/all4one.py --stage pretrain --dataset cifar10 \
  --data-root "$ALL4ONE_DATA_ROOT" --cache-dir outputs/all4one/local/pretrain \
  --accelerator gpu --precision 16-mixed --num-workers 4 --seed 42 \
  --epochs 1000 --batch-size 256 --warmup-epochs 10 \
  --lr 1.0 --attention-lr 0.1 --cifar-stem-padding 2 \
  --export-encoder outputs/all4one/local/encoder.pt

"$ALL4ONE_PYTHON" examples/all4one.py --stage linear --dataset cifar10 \
  --data-root "$ALL4ONE_DATA_ROOT" --cache-dir outputs/all4one/local/linear \
  --accelerator gpu --precision 16-mixed --num-workers 4 --seed 42 \
  --epochs 100 --batch-size 256 --warmup-epochs 0 --lr 0.1 \
  --encoder-path outputs/all4one/local/encoder.pt
```

For CIFAR-100, change `--dataset` to `cifar100` and use a separate output
folder. For ImageNet-100, use `--dataset imagenet100`, point `--data-root` to
its folder tree, and set pretraining to `--epochs 400`. Change `--seed` and the
output folder for another independent run. Dataset-specific defaults are handled
by the existing example; no new launcher is required.

## Recipe and evaluation

The recipe identifier is `all4one-release-00eeb30-v1`, referencing the authors'
commit `00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08`.

| Setting | Value |
| --- | --- |
| Backbone | torchvision ResNet-18, trained from scratch |
| CIFAR stem | 3x3, stride 1, padding 2, no max-pool |
| Pretraining | CIFAR: 1,000 epochs; ImageNet-100: 400; batch 256 |
| Optimizers | LARS with momentum 0.9, weight decay 1e-5, eta 0.02 |
| Learning rates | Representation 1.0 at batch 256; attention 0.1 |
| Schedule | 10-epoch warmup from 3e-5, cosine decay to zero |
| Projector / predictors | 2048-2048-256 / 4096-256 |
| Queue / neighbors | 98,304 / 5 |
| Attention | 3 layers, 8 heads, width 512, dropout 0.1 |
| EMA | Parameter-only, cosine 0.99 to 1.0 |
| Frozen evaluation | 100 epochs, SGD LR 0.1, momentum 0.9, cosine decay |

The example prints and records effective learning rates, preprocessing statistics,
class mappings, and stage settings. CIFAR training uses dataset-specific
normalization; evaluation preserves the release's classification-loader settings.
See the reference audit for full augmentation details and known differences.

Report `eval/linear_accuracy` and `eval/linear_top5` at the **final epoch**, not
the best observed test accuracy. Pretraining values are online classifier results;
the `linear` stage trains a new classifier with a frozen encoder. These are distinct
protocols. CIFAR uses the test split for evaluation; ImageNet-100 uses validation.
The frozen protocol is our baseline, not a verified authors' evaluation recipe.

## Recorded single-seed results

All rows use seed 42, batch 256, one GPU, and FP16. No mean or standard deviation
is claimed. Values come from the original runner's four-decimal metric output.

| Dataset | Pretraining GPU | Epochs | Online top-1 / top-5 (%) | Frozen top-1 / top-5 (%) |
| --- | --- | ---: | ---: | ---: |
| CIFAR-10 | RTX A5000 | 1,000 | 93.08 / 99.84 | 92.55 / 99.83 |
| CIFAR-100 | A40 | 1,000 | 71.63 / 93.03 | 70.31 / 92.59 |
| ImageNet-100 | A40 | 400 | 81.52 / 95.78 | 81.52 / 95.78 |

CIFAR-100 records commit `a96a92326fc71e5a9620a86375c30cebaa675946`, with only
launcher files untracked. CIFAR-10 and ImageNet-100 record dirty implementation
files. Historical source archives are unavailable, so the exact uncommitted code
cannot be reconstructed from a commit ID alone. Matching recipe
identifiers do not prove byte-identical source. These results are not presented as
verified runs of an identical source snapshot or as exact reproduction of the paper.

## Resume and validation

Resume pretraining with `--ckpt-path /path/to/checkpoints/last.ckpt`, preserving
the original total epochs, model, optimizer, preprocessing, and scheduler settings.
An exported `encoder.pt` is for frozen evaluation, not full training resume.
A two-epoch smoke-test checkpoint must not be treated as the start of a
1,000-epoch schedule.

The focused suite passed **61 CPU tests**, and the two-process CPU/Gloo test
passed with local socket access enabled. Three additional single-GPU FP16 checks
passed using a tiny synthetic model and an initial gradient scale of 128:
checkpoint resume matched uninterrupted training (including optimizer, scheduler,
gradient-scaler, EMA, queue, probe, and step state), and both optimizers updated
with one EMA update per step, with automatic and explicit EMA callbacks.
Losses, model state, and unscaled gradients remained finite. These checks do not
cover multi-GPU/NCCL, overflow recovery, or a multiple-seed benchmark study.

The earlier full unit suite reported **3,801 passed, 5 failed, 13 skipped, and
170 deselected**. All five failing cases were rerun on All4One commit `c4ab138`
and unchanged upstream `ab836bf` in the same Quadro RTX 6000 environment.
Both revisions produced **4 failures and 1 pass**, with matching failure causes:
home-path alias comparison, two CPU/CUDA tensor mismatches in visualizer tests,
and CUDA reinitialization in a forked worker. The scanner timeout did not
reproduce. This is not a clean full-suite pass.
