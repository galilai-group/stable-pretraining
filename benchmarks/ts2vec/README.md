# Matched TS2Vec UCR benchmark

This directory contains a small reproducible comparison between
`examples/ts2vec_ucr.py` and the author implementation at
`zhihanyue/ts2vec` commit `b0088e14a99706c05451316dc6db8d3da9351163`.
It is a short training smoke benchmark, not a claim about state-of-the-art UCR
performance.

The recorded runs use Python 3.12.14 on CPU with one thread. From the
repository root, install the measured package versions in a fresh environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.14.1+cpu torchvision==0.29.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]' numpy==2.5.2 scikit-learn==1.9.1 lightning==2.6.6
```

After creating the reference checkout below, run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/ts2vec/run_ucr.py --results /tmp/ts2vec-results.jsonl
```

The runner expects the reference checkout at `../ts2vec-reference` by default.
Create it with:

```bash
git clone https://github.com/zhihanyue/ts2vec.git ../ts2vec-reference
git -C ../ts2vec-reference checkout b0088e14a99706c05451316dc6db8d3da9351163
```

The default runs seeds 0, 1, and 2, with 200 optimizer updates, batch size 16,
output dimension 320, hidden dimension 64, depth 10, temporal unit 0, and
AdamW at learning rate 0.001 and weight decay 0.01. Both encoders include the
initial weights and every completed update in their cumulative average.
The runner downloads each official archive from
`https://www.timeseriesclassification.com/aeon-toolkit/<dataset>.zip`, records
the archive and converted TSV SHA-256 checksums, then runs both implementations
on the same files. Results append as JSON Lines to `results.jsonl`. Use
`--reference PATH` to point at the pinned author checkout, `--data-dir PATH`
to choose the download/cache directory, or `--results PATH` for another output
file. The reference adapter deliberately runs on CPU.

The archive's univariate equal-length `.ts` records are converted to the
label-first TSV format consumed by the existing example. Both sides use the
same max-pooled full-series representations and SVC protocol: for datasets
with at least 50 train cases and five examples per class, five-fold training
set-only selection over the example's C grid; otherwise the example's fixed
`C=inf` fallback. Test labels are used only for final accuracy.

The runner supports ECG200 and GunPoint only. Neither dataset receives extra
normalization in the pinned author's `datautils.load_UCR`; both paths therefore
use the archive values directly. Both datasets take the five-fold SVM grid-search
branch in the author's protocol and this example. The example's minimum-class
fallback differs from the author's average-class-size gate on some other
datasets; this comparison does not cover those datasets.

The comparison matches architecture, optimizer defaults, batch size, update
count, initialization seed, preprocessing, representation pooling, and SVM
evaluation. It is not a bitwise training replay. The reference samples training
rows using PyTorch's shuffled `DataLoader` and uses NumPy RNG for a few series
operations, while the project uses PyTorch's shuffled loader and Lightning's
seed setup. Consequently the order of minibatches and RNG consumption can
diverge even with the same seed. The result JSON captures the versions and
settings needed to interpret that difference.

For the project's distributed TS2Vec test (which checks that instance
negatives are local to each rank), run:

```bash
python -m pytest stable_pretraining/tests/integration/test_ts2vec_ddp.py -v
```

Instance negatives are **local to each rank**, not global. Each anchor has
2(B - 1) instance negatives for per-rank batch size B. DDP averages gradients
without gathering representations. Increasing world size does not increase
the negative pool, and per-rank batch size one gives zero instance contrast.

The test launches two CPU/Gloo processes, performs three real DDP optimizer
steps, and checks synchronized backbone and averaged weights, finite nonzero
gradients, and the exact cumulative average. An independent cross-entropy
calculation distinguishes local from global instance negatives. It is selected
by the existing integration CI job; no GPU or dataset download is needed.

Validation status: the two-process CPU/Gloo test passed in the fork's
[Testing workflow](https://github.com/qinok01/stable-pretraining/actions/runs/39208936086)
on commit `f6f74436e495fbde349ab8bb970920f90f413f18`. That run also passed
regression tests, all four unit-test shards, JAX unit tests, and the combined
coverage gate (minimum 95%). The earlier local sandbox attempt was blocked by
Gloo socket permissions; the successful CI run completes CPU/Gloo DDP validation.
Upstream PR checks for that commit remain `action_required` pending maintainer
approval and are not reported as passing.

## Recorded results

Test accuracy (%), 200 updates per run; seed order is 0, 1, 2. These are
small-dataset checks, not evidence of equivalence across the full UCR archive.

| Dataset (train/test) | Project, seeds 0/1/2 | Reference, seeds 0/1/2 | Project mean | Reference mean |
| --- | --- | --- | --- | --- |
| ECG200 (100/100) | 91.00 / 89.00 / 91.00 | 90.00 / 90.00 / 90.00 | 90.33 | 90.00 |
| GunPoint (50/150) | 98.67 / 98.00 / 98.00 | 99.33 / 99.33 / 98.00 | 98.22 | 98.89 |

Raw measurements, exact commands, data checksums and package versions are in
[results.jsonl](results.jsonl). Runs were completed on 2026-10-13/14 using the
training implementation at `65310e04eebd6eb071e181301ba1a869fc509165` plus
the CPU-selection flag and benchmark harness in this change. `project_commit`
records that base revision because the runs preceded the harness commit.
The first two records predate the extra reference update-counter fields; their
configured training budget is still recorded as 200 updates.

For a committed reproduction target containing the CPU flag and complete
harness, use `7e4995f089249638994ac0cbc0ef427aa9510534`. Its TS2Vec training
logic is unchanged from the recorded base revision; the intervening production
changes only expand the documentation.
