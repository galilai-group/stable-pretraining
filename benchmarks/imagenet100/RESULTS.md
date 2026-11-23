# DirectCLR on ImageNet-100

DirectCLR uses a fixed subvector for contrastive learning and the full encoder output for frozen evaluation ([paper](https://arxiv.org/abs/2110.09348)). This is a small-scale ImageNet-100 validation with ResNet-50.

| Experiment | Encoder epochs | Evaluation | Top-1 | Top-5 |
|---|---:|---|---:|---:|
| DirectCLR, 256 dimensions, temperature 0.2 | 400 | Center-cache probe, dev-selected LR, final validation (5,000 images) | **78.36%** | **94.24%** |
| Earlier matched DirectCLR control | 100 | Frozen classifier, final validation | 62.26% | — |
| Earlier matched linear-projector control | 100 | Frozen classifier, final validation | 64.42% | — |
| Earlier matched no-projector, full-dimensional control | 100 | Frozen classifier, final validation | 59.02% | 85.90% |

The 100-epoch comparison favors the linear projector by 2.16 points over DirectCLR. The 400-epoch center-cache refit scores 3,918/5,000 Top-1 and 4,712/5,000 Top-5 on final validation. There is no matched 400-epoch projector control.

## Matched 100-epoch projector ablation

The no-projector control is already implemented as `--method full`: the entire
2,048-dimensional ResNet-50 output enters the same symmetric cross-view InfoNCE
loss. There is no trainable projector or feature truncation. `--method directclr`
uses the first 360 coordinates in this historical comparison; `--method linear`
adds a bias-free 2,048-to-360 trainable linear layer. The loss and candidate
definition stay fixed across all three conditions.

These completed encoders share the original 100-epoch / 49,400-update recipe:
global batch 256, temperature 0.1, LARS peak LR 0.3, seed 0, FP16 mixed precision,
ten warmup epochs, cosine decay and the symmetric `legacy` augmentation.
All readouts use frozen full-width, L2-normalized center-crop features and
100-epoch SGD classifiers with batch 1,024. The fixed protocol uses LR 1;
the second protocol selects LR from 1/3/10 on the same stratified training/dev
split, then fits a fresh classifier on all training features. All three select
LR 10. The table restores previously recorded results; it adds no training or
held-out evaluation.

| Encoder condition | Contrastive dimensions | Fixed LR 1: val Top-1 | Dev-selected LR: val Top-1 | Dev-selected LR: val Top-5 |
|---|---:|---:|---:|---:|
| DirectCLR, fixed leading subspace | 360 | 53.98% | 62.26% | 86.84% |
| Trainable linear projector | 360 | **57.80%** | **64.42%** | **88.54%** |
| No projector, full-dimensional objective | 2,048 | 57.42% | 59.02% | 85.90% |

With dev-selected classifiers, DirectCLR exceeds the full-dimensional
no-projector control by 3.24 percentage points, while remaining 2.16 points
below the trainable linear projector. With fixed LR 1, DirectCLR is 3.44 points
below the no-projector control. The ordering depends on the readout protocol;
these single-seed measurements do not establish general method superiority.
The full-dimensional control is a separately trained SSL encoder, distinct
from the full-2,048 readout of the fixed DirectCLR encoder below.

To run this historical no-projector configuration on fresh data, use:

```bash
python benchmarks/imagenet100/directclr.py train --data-dir /data/imagenet100 --data-revision 0519dc2f402a3a18c6e57f7913db059215eee25b --method full --backbone resnet50 --temperature 0.1 --epochs 100 --batch-size 256 --lr 0.3 --augmentation legacy --precision 16-mixed --devices 1 --seed 0 --output /data/runs/imagenet100-r50-100ep-full-seed0
```

For a matched fresh comparison, change only `--method` to `directclr` or `linear`
and name each output directory with that method; the default `--dim 360` applies
to those two methods, while `full` always uses the encoder width. Cache each
encoder using the evaluation command below. For the dev-selected readout,
follow the split/sweep commands with `--encoder-epochs 100`,
`--methods directclr linear full` and `--learning-rates 1 3 10`.
The sweep selects on dev and refits a fresh classifier on all training rows.

## Training and selection

The subspace width **256 was fixed before these experiments**. API defaults
360/0.1 follow the reference implementation. Temperature 0.2 and the 400-epoch
budget were selected during earlier exploratory dev experiments and fixed
before this center-cache evaluation.

The fixed stratified split uses seed 2026 (114,020 fit / 12,669 dev rows). Encoder seed is 0, global batch 512, LARS peak LR 1.2, weight decay 1e-6, 10-epoch warmup then cosine. The 400-epoch run starts from a new encoder with 98,800 updates; it does not extend a completed 200-epoch cosine schedule. Source images are pre-resized to 160 px, with actual SSL/evaluation crops at 224 px.

The center-cache evaluation uses that frozen 400-epoch checkpoint.
Classifier LR 15/30/60 gives dev Top-1 80.71% / 81.15% / 81.17%, selecting
LR 60. A fresh 100-epoch classifier refit on all 126,689 training feature rows
then gives final-validation Top-1 78.36% (3,918/5,000) and Top-5 94.24%
(4,712/5,000). All heads use L2-normalized features, batch 1,024, seed 0, SGD
with momentum 0.9 and per-update cosine decay. Candidate training uses 11,200
updates per head and the full-train refit uses 12,400 updates. The committed
cache-only evaluator and probe sweep produced these measurements. An independent
CPU calculation reproduces the three candidate development scores and both
final validation counts. The full2048 diagnostic below remains a dev result.
The validation set had been used in earlier experiments; this is a repeated
evaluation on the same dataset.

## Fixed-encoder subspace and spectrum diagnostic

The pre-CIFAR SSL encoder and center-view cache stay fixed. First256, last256,
fixed random orthogonal Q256 and full2048 each receive the same fit/dev rows,
seed 0, zero-initialized head, LR 15/30/60 grid and 100 epochs / 11,200 updates.
The learning rate is selected using final-epoch dev only.

| Readout | Head parameters | Dev Top-1 | Effective rank / dimension |
|---|---:|---:|---:|
| First256 | 25,700 | 70.94% | 43.55% |
| Last256 | 25,700 | 75.07% | 84.99% |
| Fixed Q256 | 25,700 | 77.13% | 80.12% |
| Full2048 | 204,900 | 81.19% | 47.43% |

The first three heads have equal trainable capacity; full2048 has more
parameters. Q is a fixed posthoc readout, not a projector added to SSL training.
All spectra use the same 4,096 fit-only rows, L2-normalized features and
FP64-centered covariance; effective rank uses entropy of data singular values.
Last256 has a more uniform spectrum than Q256 but lower accuracy, so spectral
uniformity alone does not explain readout performance. One encoder/projection
seed and reused dev limit interpretation. These diagnostic scores use
zero-initialized heads and are measured on dev. The main result uses a fresh
randomly initialized head refitted on all training rows and scores final
validation.

## Separate supervised transfer

The selected SSL encoder was also fine-tuned on CIFAR-10 with a new 10-class head: 45,000/5,000 train/dev split (seed 2026), input 160, seed 0, 100 epochs / 17,600 updates, SGD encoder/head LR 0.01/0.1, momentum 0.9, weight decay 5e-4, five-epoch warmup then cosine. Dev is 93.86%; one final test is 93.57% (9,357/10,000). Recovery retains the first 45 epochs and the original remaining schedule. This supervised transfer score is separate from ImageNet frozen evaluation.

### Fixed-budget CIFAR development controls

Two new controls use the same 45,000/5,000 training/dev split, seed 0,
head initialization, row order, input 160 and 100-epoch / 17,600-update
schedule as the historical pretrained fine-tuning recipe. Fixed-epoch
evaluation gives 88.38% (4419/5000) for random-init full supervised
training and 80.40% (4020/5000) for a frozen pretrained encoder/BN with
head-only training. Historical pretrained full fine-tuning is 93.86%
(4693/5000) on this same dev split, 5.48 and 13.46 percentage points higher.
Each control retains the fixed recipe rather than receiving separate tuning.
These two CIFAR controls use training and dev data only.

Each control records five bounded same-batch AMP retries with no skipped
successful updates. These single-seed results support transfer under this
recipe; they do not determine each control's best achievable accuracy.
The 400-epoch SSL projector control remains unavailable.

## Fresh-run commands

Install the library with `python -m pip install .`. Prepare the pinned data before allocating training hardware, then run the selected SSL configuration:

```bash
python benchmarks/imagenet100/directclr.py prepare --data-dir /data/imagenet100 --data-revision 0519dc2f402a3a18c6e57f7913db059215eee25b
python benchmarks/imagenet100/directclr.py train --data-dir /data/imagenet100 --data-revision 0519dc2f402a3a18c6e57f7913db059215eee25b --method directclr --backbone resnet50 --dim 256 --temperature 0.2 --epochs 400 --batch-size 512 --lr 0.6 --augmentation reference --precision 16-mixed --devices 4 --seed 0 --output /data/runs/imagenet100-r50-400ep-directclr-seed0
```

The CLI LR is scaled by global batch / 256, giving peak 1.2. Extract center-crop
features from the frozen 400-epoch checkpoint, select LR on dev, and refit on
all training features using the committed evaluator and probe sweep:

```bash
RUN=/data/runs/imagenet100-r50-400ep-directclr-seed0
CHECKPOINT=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$RUN/training_summary.json")
python benchmarks/imagenet100/directclr.py evaluate --data-dir /data/imagenet100 --data-revision 0519dc2f402a3a18c6e57f7913db059215eee25b --backbone resnet50 --checkpoint "$CHECKPOINT" --cache-only --batch-size 256 --workers 4 --seed 0 --output "$RUN/evaluation"
python benchmarks/imagenet100/probe_sweep.py prepare-split --source /data/runs --encoder-epochs 400 --split /data/runs/center400-split.pt --output /data/runs/center400-readouts
python benchmarks/imagenet100/probe_sweep.py sweep --source /data/runs --encoder-epochs 400 --split /data/runs/center400-split.pt --output /data/runs/center400-readouts --methods directclr --learning-rates 15 30 60 --epochs 100 --batch-size 1024 --seed 0
```

Report the final `linear_top1` and `linear_top5` from
`/data/runs/center400-readouts/directclr/evaluation.json`. The classifier
selection is saved before final validation is scored. For the three matched
100-epoch encoders, repeat the cache command for each checkpoint, then use
`--encoder-epochs 100 --methods directclr linear full --learning-rates 1 3 10`
with a separate output directory.

## Supplementary analysis

Separate CPU experiments examine directional suppression, factorization and
finite-time layer alignment. Spectral effects are observed, while perfect
alignment remains unobserved under both the released-code and Eq2 protocols.
The [scripts, figures and reports](https://github.com/realwooha/stable-pretraining/tree/2d4c5291a30b6f1418bdd32c631266df21ef4f39/benchmarks/toy)
are preserved separately from this method contribution. These toy results do
not establish the full asymptotic theory or explain the ResNet accuracy order.

## Interpretation limits

One encoder seed and the missing matched 400-epoch baseline limit conclusions about method superiority. The early matched comparison is a negative result. Repeated dev exploration, earlier val/test exposure and unlabeled dev images in SSL pretraining limit independence; reduced source resolution, AMP/DDP and recovery boundaries also affect comparisons. This validation uses ImageNet-100 rather than the paper's ImageNet-1K protocol.
