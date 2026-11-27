# EEG-JEPA reference checks and neonatal pilot

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


## Neonatal EDF pilot

The real-data entry point uses `EdfWindowReader` in stable-datasets' existing
`profiling/edf_record_loading.py`. That facade delegates to the existing native
memmap/CPU calibration and GPU record assembler; this example owns no EDF or
Lance parsing or calibration. Lance paths are not enabled in this pilot.

Preprocessing follows the transform order inspected in the pinned reference's
[record-shard preprocessing](https://github.com/SWF-hao/EEG-JEPA-official/blob/925b1194d5abf1e95ca2a2e7416b5a5d268845df/preprocessing/preprocess_tuh_eeg_record_shards.py)
and [canonical channel mapper](https://github.com/SWF-hao/EEG-JEPA-official/blob/925b1194d5abf1e95ca2a2e7416b5a5d268845df/preprocessing/preprocess_tuh_eeg_lmdb.py):

- Channel order: `FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ`.
  Case-insensitive canonical names and `EEG ...-Ref` labels are accepted;
  T7/T8/P7/P8 map to T3/T4/T5/T6. ECG/respiration are excluded. All 19 electrodes
  are required; missing/duplicate electrodes and bipolar or other suffixes fail.
  Source referential voltages are preserved; no common-average re-reference is added.
- Rates: the existing reader requires equal-rate continuous channels. The pilot
  requires an integral source rate of at least 200 Hz (eeg4 is 256 Hz). Shared
  CPU/MNE FFT resampling produces 200 Hz, followed by 0.3–75 Hz FIR filtering and
  a 60 Hz FIR notch. The notch follows the reference default, not neonatal tuning.
- Units: stable-datasets supplies calibrated float32 **volts**. MNE processes
  float64 volts; the example converts the result to float32 **microvolts**.
  The existing reference-objective forward then divides by 100, exactly once.
- Patches: each request specifies the core window's original EDF sample start.
  Starts use whole seconds. An additional 12 seconds on each side is read and
  processed; that context is removed after resampling/filtering. Consecutive
  200-sample, one-second patches form `[batch, 19, seconds, 200]`.
- Scope: this uses contextual windows rather than full-record preprocessing.
  The reference's 60-second end trimming, amplitude QC and corpus selection
  are not reproduced. It is not a complete reference-data reproduction or a
  held-out neonatal seizure evaluation; annotation labels are not consumed.
  Filter boundaries can differ from full-record preprocessing.

Both paths deliberately use the **same CPU/MNE preprocessing**, so GPU assembly
includes a device-to-host round trip followed by the usual model-device transfer.
The pilot verifies every scheduled normalized input against CPU memmap with
`rtol=2e-5, atol=1e-6`, then resets model initialization and training RNG for
each path. N-MET masks are generated once and replayed with the exact same windows.
Loss traces must agree within `rtol=2e-4, atol=2e-5`; teacher update counts are
checked. GPU numerical checks must actually run on Hydra before claiming parity.

From stable-pretraining, install the sibling checkout and medical dependencies.
The `profiling` research modules are source-tree modules, so also put the
stable-datasets repository root on `PYTHONPATH`:

```bash
sds_root="$HOME/stable-datasets"
python -m pip install -e "$sds_root" "mne==1.10.2" edfio pyedflib
export PYTHONPATH="$sds_root:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q -s -rs -p no:cacheprovider -m "not gpu" \
  stable_pretraining/tests/integration/test_eeg_jepa_neonatal.py \
  stable_pretraining/tests/integration/test_eeg_jepa_reference.py
```

The neonatal tests use generated EDF files without downloads. For a real CPU
functional run, supply the unchanged neonatal EDF already used in the loading
study; `--output` must be a new external directory:

```bash
python -m examples.eeg_jepa.pilot --source /path/to/eeg4.edf \
  --reference-root "$EEG_JEPA_REFERENCE" --output "$study_root/neonatal-cpu" \
  --paths edf_memmap --accelerator cpu --steps 3 --batch-size 2 --seconds 4 --seed 0
```

Proposed command **inside Pascal's allocated Hydra GPU job**, with the two
checkouts and reference archive available on Hydra:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q -s -rs -p no:cacheprovider -m gpu \
  stable_pretraining/tests/integration/test_eeg_jepa_reference.py \
  stable_pretraining/tests/integration/test_eeg_jepa_neonatal.py
python -m examples.eeg_jepa.pilot --source /path/on/hydra/eeg4.edf \
  --reference-root "$EEG_JEPA_REFERENCE" --output "$study_root/neonatal-gpu" \
  --paths edf_memmap edf_gpu_assembly --accelerator gpu \
  --steps 4 --batch-size 2 --seconds 4 --seed 0
```

Readers execute in the training process with zero workers; no new GPU buffers
or prefetch architecture is introduced. Each path trains on the same GPU when
`--accelerator gpu` is selected. Reports and Manager artifacts stay in the
external output directory and contain source/code hashes, window starts,
masks, initialization hashes, input differences, losses and elapsed times.
Preflight checks warm the requests. Timings include shared CPU preprocessing
and are a short correctness diagnostic, not a GPU speedup result.


## Repeated benchmark

Add `--trials 3 --workers 0 2` to the pilot command to enable the benchmark.
Each trial uses CPU memmap with 0/2 spawned workers and, when requested, GPU
assembly with 0 workers. Case order alternates forward/reverse/forward. All
trials replay the same initialization, windows, masks and update budget; model
and optimizer state reset per case. Three restarts characterize timing variation,
not independent data samples or confidence intervals.

`--startup-steps N` fixes the startup interval before measurement. It includes
worker/reader startup and the first N model updates. The sustained rate counts
only the remaining updates and elapsed time after update N. No updates or losses
are discarded, and weights are not reset at that boundary. First-batch time,
iterator setup, startup, total time and sustained windows/s are reported with
per-trial values, medians and observed ranges.

The main clock starts before creating the training DataLoader iterator and ends
after the final optimizer/teacher update. Model/Manager setup, input checks,
worker shutdown, report/plot writing and automatic checkpoint I/O are excluded.
CPU uses completed host timestamps; CUDA records nonblocking completion events,
resolved after training. Loss tensors are copied to the host after training.
The benchmark adds no per-step synchronization in main mode. The unchanged
reference objective and shared CPU preprocessing still perform their own scalar
reads/device-to-host work, so this is not a fully asynchronous trainer.

`--diagnostic` runs one **separate** instrumented replay per case. It records
reader loading/calibration, CPU preprocessing, main-process loader wait,
delivery-to-compute delay, and model-step time (forward/backward, optimizer and
EMA). Diagnostic boundaries synchronize CUDA. Worker timings can overlap model
compute and must not be summed as pipeline wall time. Delivery-to-compute also
includes framework/transfer overhead. Diagnostic rates are not included in the
main throughput summary. Every replay must pass the same loss checks.

Inside an allocated Hydra GPU job, after the environment/path setup above:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m examples.eeg_jepa.pilot --source /path/on/hydra/eeg4.edf \
  --reference-root "$EEG_JEPA_REFERENCE" --output "$study_root/benchmark-gpu" \
  --paths edf_memmap edf_gpu_assembly --accelerator gpu \
  --steps 8 --startup-steps 2 --batch-size 2 --seconds 4 --seed 0 \
  --trials 3 --workers 0 2 --diagnostic
```

For a local CPU check, use `--paths edf_memmap --accelerator cpu` and a new
output directory. Workers open independent readers after spawn and cap Torch/
Arrow thread pools; keep the environment thread limits above. GPU assembly
cannot be placed in workers. Parent and spawned-worker inputs are checked before
timing; those checks warm requests and are recorded in the report.

`report.json` retains all main/diagnostic runs, phase measurements, parameter
counts, case order, seeds, input/source/code hashes, environment and allocation.
`trials.jsonl` retains each validated completed run even if a later run fails.
`learning-progress.png` contains loss-versus-step and loss-versus-completed-wall-
time panels, with individual trials, median curves, parameter counts and source/
reference fingerprints. All artifacts are local and outside Git.


## Batched MNE preprocessing

Preprocessing packs windows as independent rows shaped `[batch * 19, time]`
in one MNE `RawArray`; it never joins windows along time. Resampling, FIR
parameters, microvolt conversion and context cropping are unchanged. The result
is restored to `[batch, 19, seconds, 200]`. A frozen per-window test oracle
requires exact agreement for batches 1, 2 and 4, including 200/256 Hz inputs,
one/four-second crops and generated EDF windows.

CPU diagnostic job 237865 measured median preprocessing of 51.24 ms per batch
before batching and 43.57 ms for the candidate (batch 2, four-second core,
12-second context on each side, 256 Hz source). Tracemalloc peak allocation
increased from 5,344,826 to 7,235,690 bytes: approximately **5.10 to 6.90 MiB**.
That is a measured allocation-workspace tradeoff, not RSS or CUDA memory;
larger batches require larger simultaneous workspaces. This change does not
alter reader/calibration code or introduce a filter cache.
