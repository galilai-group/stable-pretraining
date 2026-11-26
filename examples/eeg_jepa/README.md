# EEG-JEPA synthetic reference checks

This milestone checks a thin stable-pretraining adapter against the authors'
[EEG-JEPA implementation](https://github.com/SWF-hao/EEG-JEPA-official/tree/925b1194d5abf1e95ca2a2e7416b5a5d268845df),
pinned to **925b1194d5abf1e95ca2a2e7416b5a5d268845df**. It loads the
unmodified model, objective and N-MET sampler from an external source archive;
`reference.py` verifies hashes before loading.

The local configuration uses two encoder layers, intermediate/final supervision,
19 channels and four 200-sample patches. Inputs represent microvolts and follow
the reference trainer's division by 100. Tests cover stochastic train/eval
forwards, loss terms, every parameter gradient, input gradients, masking, loss
warmups, clipping, learning-rate schedules and post-optimizer EMA updates.
This is synthetic correctness, not a dataset or training-speed benchmark.

## Environment and reference

Run from the **stable-pretraining repository root**, on
`research/eeg-reference-integration`. All commands below leave source caches and
run artifacts outside Git. Branch management and Hydra execution belong to Pascal.

On a Mac, create a separate environment and install the checkout:

```bash
python3.12 -m venv "$HOME/.venvs/eeg-jepa"
source "$HOME/.venvs/eeg-jepa/bin/activate"
python -m pip install -e . pytest ruff \
  "torch==2.15.0" "torchvision==0.29.2" "lightning==2.6.6" "numpy==2.5.5"
```

These core versions were checked locally with Python 3.12.13. On Hydra, use a
compatible CUDA-enabled PyTorch environment for the allocated GPU and driver;
install this checkout, pytest and Ruff into that environment. Preserve its CUDA
wheel build rather than replacing it with a CPU build.

Download the pinned archive once per machine (no Git clone is needed):

```bash
study_root="$HOME/eeg-jepa-milestone"
revision=925b1194d5abf1e95ca2a2e7416b5a5d268845df
mkdir -p "$study_root/reference"
curl --fail --location \
  "https://api.github.com/repos/SWF-hao/EEG-JEPA-official/tarball/$revision" \
  --output "$study_root/reference.tar.gz"
tar -xzf "$study_root/reference.tar.gz" --strip-components=1 \
  -C "$study_root/reference"
export EEG_JEPA_REFERENCE="$study_root/reference"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$study_root/matplotlib"
python -m pip freeze > "$study_root/environment.txt"
```

Set `EEG_JEPA_REFERENCE` again in each new shell. Without it, reference-dependent
tests skip; a skipped suite is not a correctness result.

## CPU checks — Mac terminal

```bash
python -m pytest -q -s -rs -p no:cacheprovider -m "not gpu" \
  stable_pretraining/tests/integration/test_eeg_jepa_reference.py \
  stable_pretraining/tests/unit/test_callback_scheduling_and_logging.py \
  stable_pretraining/tests/unit/test_manual_optimization_trainer_args.py

ruff check --no-cache examples/eeg_jepa stable_pretraining/tests/integration/test_eeg_jepa_reference.py
ruff format --no-cache --check examples/eeg_jepa stable_pretraining/tests/integration/test_eeg_jepa_reference.py
```

The parity assertions require exact agreement (`rtol=0, atol=0`), including
the four-step Manager run. Configuration restoration is checked with both empty
and pre-existing overrides after successful fits, Trainer construction failures
and training failures.

To exercise the standalone entry point:

```bash
python -m examples.eeg_jepa.train --reference-root "$EEG_JEPA_REFERENCE" \
  --output "$study_root/cpu-run"
```

## CUDA checks — Hydra GPU allocation

Inside an allocated GPU job, with the same checkout and an external reference
archive on Hydra, run:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c 'import torch; assert torch.cuda.is_available(); print(torch.ones(1, device="cuda"))'
python -m pytest -q -s -rs -p no:cacheprovider -m gpu \
  stable_pretraining/tests/integration/test_eeg_jepa_reference.py
```

All four GPU cases must execute: three forward/loss/gradient cases and one
Manager/EMA comparison. They skip if CUDA is unavailable or unusable.
These GPU checks remain unverified locally on the Mac. Mixed precision,
distributed execution and real EEG data are outside this milestone.
