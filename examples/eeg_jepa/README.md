# EEG-JEPA reference checks and neonatal pilot

A thin `Module`/`Manager` adapter loads the authors'
[EEG-JEPA implementation](https://github.com/SWF-hao/EEG-JEPA-official/tree/925b1194d5abf1e95ca2a2e7416b5a5d268845df)
at **925b1194d5abf1e95ca2a2e7416b5a5d268845df**. `reference.py` verifies
unmodified model, objective, trainer and N-MET source hashes before loading.
The small configuration uses two encoder layers, intermediate/final supervision,
19 channels and four 200-sample patches. Tests compare outputs, loss terms,
every parameter gradient, input gradients, masks, warmups, clipping, learning
rates and post-optimizer EMA updates, including a four-step Manager run.

`config.py` contains configuration and schedules; `data.py` owns example
preprocessing and batch construction; `model.py` owns model adaptation;
`train.py` owns initialization, result validation, temporary framework settings
and Manager wiring; `benchmark.py` owns observers and timing. The `pilot.py`
entry point defers training imports so spawned readers avoid the training stack.
Dataset reading and calibration stay in stable-datasets.

## Environment and reference

Run from the stable-pretraining source checkout on
`research/eeg-reference-integration`. The neonatal example requires the companion
stable-datasets contribution on `research/gpu-medical-timeseries`, compatible
with **32cb823e** for EDF controls. The `current_cache` path additionally needs
the companion `CurrentCacheWindowReader` contribution based on **cd138094**;
that base revision alone lacks the new reader. A release installation alone
lacks the `profiling` modules: both source roots must be on
`PYTHONPATH`. Keep reference caches, environments and artifacts outside Git.

A fresh Mac CPU environment (core versions checked with Python 3.12.13):

```bash
spt_root="$HOME/stable-pretraining"
sds_root="$HOME/stable-datasets"
python3.12 -m venv "$HOME/.venvs/eeg-jepa"
source "$HOME/.venvs/eeg-jepa/bin/activate"
python -m pip install -e "$spt_root" -e "$sds_root" pytest ruff \
  "torch==2.15.0" "torchvision==0.29.2" "lightning==2.6.6" \
  "numpy==2.5.5" "scipy==1.18.1" "mne==1.10.2" \
  "edfio==0.4.18" "pyedflib==0.1.42" "pylance==14.0.3" "pyarrow==26.0.1"
cd "$spt_root"
export PYTHONPATH="$spt_root:$sds_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1 TOKIO_WORKER_THREADS=1
```

This is not a complete transitive lock. On Hydra, install both projects and the
research dependencies into an existing driver-compatible CUDA-enabled PyTorch
environment; preserve its CUDA build and record `pip freeze` and driver details.
The sibling storage study's Python 3.13/MNE 1.13.2/Lance 13/CUDA pins describe a
historical environment, separate from this EEG integration environment.

Download the pinned archive once per machine:

```bash
study_root="$HOME/eeg-jepa-study"
revision=925b1194d5abf1e95ca2a2e7416b5a5d268845df
mkdir -p "$study_root/reference"
curl --fail --location \
  "https://api.github.com/repos/SWF-hao/EEG-JEPA-official/tarball/$revision" \
  --output "$study_root/reference.tar.gz"
tar -xzf "$study_root/reference.tar.gz" --strip-components=1 \
  -C "$study_root/reference"
export EEG_JEPA_REFERENCE="$study_root/reference"
export MPLCONFIGDIR="$study_root/matplotlib"
python -m pip freeze > "$study_root/environment.txt"
```

Set `EEG_JEPA_REFERENCE` in each shell. Missing reference or optional reader
packages can skip reference-dependent tests or whole modules. Inspect `-rs`:
those skips mean incomplete validation. `-m 'not gpu'` deliberately deselects
CUDA cases on CPU; they must execute separately in a GPU allocation.

## Required CPU validation

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q -rs -p no:cacheprovider -m 'not gpu' \
  stable_pretraining/tests/integration/test_eeg_jepa_reference.py \
  stable_pretraining/tests/integration/test_eeg_jepa_neonatal.py \
  stable_pretraining/tests/integration/test_eeg_jepa_gpu_preprocessing.py \
  stable_pretraining/tests/unit/test_callback_scheduling_and_logging.py \
  stable_pretraining/tests/unit/test_manual_optimization_trainer_args.py
ruff check --no-cache examples/eeg_jepa \
  stable_pretraining/tests/integration/test_eeg_jepa_{reference,neonatal,gpu_preprocessing}.py \
  stable_pretraining/tests/unit/test_callback_scheduling_and_logging.py \
  stable_pretraining/tests/unit/test_manual_optimization_trainer_args.py
ruff format --no-cache --check examples/eeg_jepa \
  stable_pretraining/tests/integration/test_eeg_jepa_{reference,neonatal,gpu_preprocessing}.py \
  stable_pretraining/tests/unit/test_callback_scheduling_and_logging.py \
  stable_pretraining/tests/unit/test_manual_optimization_trainer_args.py
```

Reference parity uses `rtol=0, atol=0`, including the Manager run. Settings tests
cover empty/pre-existing overrides after successful fits, Trainer construction
failures and training failures. Neonatal tests generate EDF fixtures locally;
worker tests exercise real spawn/reopening and reject training imports during
bootstrap. A frozen, independent per-window preprocessing oracle requires exact
agreement with batched preprocessing for batch sizes 1/2/4, 200/256 Hz and
one/four-second cores. Run the sibling's six retained research test files too,
as listed in `profiling/medical_signal_research/REPRODUCING.md`.

Synthetic entry point:

```bash
python -m examples.eeg_jepa.train --reference-root "$EEG_JEPA_REFERENCE" \
  --output "$study_root/synthetic-cpu"
```

## Neonatal EDF preprocessing and pilot

`EdfWindowReader` delegates to the existing CPU memmap/calibration or CUDA
record assembler. `CurrentCacheWindowReader` delegates to the existing
production-format `CurrentSignalCache`. The example owns no storage parsing
or calibration. Its
transform order follows the pinned reference's
[record-shard preprocessing](https://github.com/SWF-hao/EEG-JEPA-official/blob/925b1194d5abf1e95ca2a2e7416b5a5d268845df/preprocessing/preprocess_tuh_eeg_record_shards.py)
and [canonical channel mapper](https://github.com/SWF-hao/EEG-JEPA-official/blob/925b1194d5abf1e95ca2a2e7416b5a5d268845df/preprocessing/preprocess_tuh_eeg_lmdb.py):

- Channels: `FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ`.
  Canonical names and `EEG ...-Ref` labels are case insensitive;
  T7/T8/P7/P8 map to T3/T4/T5/T6. All 19 electrodes are required. Duplicate,
  missing, bipolar or other suffixes fail. ECG/respiration are excluded;
  source referential voltages are preserved without common-average re-reference.
- Rates: continuous, equal-rate channels at an integral source rate >=200 Hz
  (eeg4: 256 Hz). CPU/MNE FFT resampling produces 200 Hz, then 0.3–75 Hz FIR
  filtering and a 60 Hz FIR notch, following the reference defaults.
- Units: calibrated float32 volts enter MNE as float64 volts; the result becomes
  float32 microvolts. The objective's forward divides by 100 exactly once.
- Windows/patches: whole-second core starts address original EDF samples. Read
  12 seconds of context on each side, process, crop context, and reshape
  consecutive 200-sample one-second patches to `[batch, 19, seconds, 200]`.
  Batching packs independent rows `[batch * 19, time]` in one MNE `RawArray`;
  windows are never joined along time.

This contextual-window pilot omits the reference corpus selection, 60-second
end trimming and amplitude QC. Filter boundaries can differ from full-record
preprocessing. It consumes no seizure labels and establishes no held-out
neonatal accuracy. All paths share CPU/MNE preprocessing; GPU assembly includes
a device-to-host round trip and subsequent model-device transfer.

Every scheduled normalized input is checked against CPU memmap at
`rtol=2e-5, atol=1e-6`. All paths replay identical initialization, windows,
N-MET masks and training RNG. Losses must match at `rtol=2e-4, atol=2e-5`;
teacher updates must equal the budget. A short real CPU functional run uses the
unchanged EDF from the sibling study and a new external output directory:

```bash
python -m examples.eeg_jepa.pilot --source /path/to/eeg4.edf \
  --reference-root "$EEG_JEPA_REFERENCE" --output "$study_root/neonatal-cpu" \
  --paths edf_memmap --accelerator cpu --steps 3 --batch-size 2 --seconds 4 --seed 0
```

### Production-cache CPU baseline

Keep the default `edf_memmap` command above unchanged. To include `current_cache`,
pass `--prepared-workspace` pointing to the existing window study workspace
containing `preparation.json` and `current/`. If no workspace exists, prepare it
once outside Git (this also builds the study's Lance layouts):

```bash
python -m profiling.benchmark_signal_windows prepare /path/to/eeg4.edf "$study_root/prepared-1024" --chunk-samples 1024
python -m examples.eeg_jepa.pilot --source /path/to/eeg4.edf \
  --prepared-workspace "$study_root/prepared-1024" \
  --reference-root "$EEG_JEPA_REFERENCE" --output "$study_root/cache-cpu" \
  --paths current_cache edf_memmap --accelerator cpu \
  --steps 4 --batch-size 2 --seconds 4 --seed 0
```

Before training, stable-datasets checks the manifest's EDF SHA256 and recording
sample count, rate, channel order, gain and offset against the source. The cache
already holds float32 volts; no second calibration is applied. Each batch makes
one grouped `CurrentSignalCache.read()` call, materializing each requested
recording once even for repeated/unordered windows. Handles reopen per spawned
process. Source verification is outside timing; worker reopen and batch delivery
remain inside the existing measured intervals.

All scheduled cache inputs pass the same normalized-input checks against
memmap, and all training runs pass the same loss/initialization/budget checks.
`report.json` adds `prepared_cache` with workspace, preparation-manifest SHA256
and verified EDF SHA256. The manifest records declared source provenance;
numerical checks validate the requested cached windows. This baseline represents
the production signal-column cache, not a complete neonatal dataset build.

## Repeated measurements

Add `--trials 3 --workers 0 2 --startup-steps 2` to compare workers 0/2 for
each requested CPU path (`edf_memmap` and/or `current_cache`) and optional GPU
assembly with 0 workers. Order alternates forward/reverse/forward.
Each run resets model/optimizer state and replays the same windows/masks/budget.
Three restarts describe observed timing variation, not independent data samples.

The main clock starts before training iterator/worker creation and ends after
the final optimizer/EMA update. Setup, preflight, worker shutdown, checkpoint
I/O and report/plot writing are excluded. Startup is the first configured
updates; sustained throughput uses all remaining updates without resetting
weights. A short sustained interval can be affected by prefetched batches and
should not be interpreted as steady state. CPU uses host timestamps; CUDA uses
nonblocking completion events resolved after fit. Loss tensors move to the host
after fit. Main mode adds no per-step synchronization; reference scalar reads
and CPU preprocessing retain their own synchronization costs.

`--diagnostic` adds separate synchronized replays measuring reader/calibration,
preprocessing, loader wait, delivery-to-compute and model-step time including
optimizer/EMA. Worker work overlaps compute and cannot be summed as pipeline
wall time. Diagnostic rates are excluded from main throughput summaries; loss
checks still apply. Spawned readers open independently and cap thread pools;
GPU assembly stays in the main process. Preflight checks warm requests.

`report.json` records starts, masks, initialization/input/source/code hashes,
losses, teacher counts, environment/allocation, parameter counts, order and all
measurements. `trials.jsonl` retains validated completed runs if a later run
fails. `learning-progress.png` plots loss versus step and completed wall time,
with trials, medians, parameter counts and source/reference fingerprints.

Historical CPU diagnostic **237865** measured preprocessing medians 51.24 ms
per batch before batching and 43.57 ms for the candidate (batch 2, four-second
core, 12-second context each side, 256 Hz). Tracemalloc peak allocation rose
from 5,344,826 to 7,235,690 bytes (**5.10 to 6.90 MiB**), an allocation-workspace
tradeoff rather than RSS/CUDA memory. Larger batches need more simultaneous
workspace. These external measurements retain their original provenance and
are not performance measurements of this cleanup.

## Proposed GPU correctness check

Inside an existing GPU allocation, after the same source/reference setup:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c 'import torch; assert torch.cuda.is_available(); print(torch.ones(1, device="cuda"))'
python -m pytest -q -rs -p no:cacheprovider -m gpu \
  stable_pretraining/tests/integration/test_eeg_jepa_reference.py \
  stable_pretraining/tests/integration/test_eeg_jepa_neonatal.py \
  stable_pretraining/tests/integration/test_eeg_jepa_gpu_preprocessing.py
python -m examples.eeg_jepa.pilot --source /path/on/hydra/eeg4.edf \
  --reference-root "$EEG_JEPA_REFERENCE" --output "$study_root/neonatal-gpu" \
  --paths edf_memmap edf_gpu_assembly --accelerator gpu \
  --steps 4 --batch-size 2 --seconds 4 --seed 0
```

All eleven selected GPU cases must execute; unavailable/unusable CUDA skips are
incomplete GPU validation. GPU checks remain unverified on the Mac. To check
the production-cache CPU pipeline against GPU assembly, retaining memmap as a
control on the same GPU, propose a four-update run per case:

```bash
python -m examples.eeg_jepa.pilot --source /path/on/hydra/eeg4.edf \
  --prepared-workspace /path/on/hydra/prepared-1024 \
  --reference-root "$EEG_JEPA_REFERENCE" --output "$study_root/cache-gpu-check" \
  --paths current_cache edf_memmap edf_gpu_assembly --accelerator gpu \
  --steps 4 --batch-size 2 --seconds 4 --seed 0 \
  --trials 3 --workers 0 2 --startup-steps 2
```

This short run checks correctness and interval accounting; two sustained updates
cannot establish steady-state performance. CUDA execution and real-recording
speedup remain unverified. The opt-in GPU preprocessing candidate below is
implemented but requires actual CUDA validation. Further Lance integration,
buffer/stream tuning, larger models and external tracking remain future work.


## Experimental GPU preprocessing: `gpu_fft_f64_v1`

This candidate builds on stable-pretraining **62cbd5ac** and uses the unchanged
stable-datasets readers at **730175f0** (or their compatible companion contribution).

Add `--gpu-preprocessing` beside `--paths ... edf_gpu_assembly --accelerator gpu`.
This adds the distinct case **`edf_gpu_assembly+gpu_fft_f64_v1`**; existing loading
paths, CPU/MNE preprocessing and defaults are unchanged. Use a new external output
directory. Each candidate result, trial directory, JSONL entry and plot series has
that distinct identifier. Reports add `experiment`, `preprocessing_cases` and
`gpu_preprocessing`; old reports remain readable. Historical timings above do
not measure this implementation.

Supported conditions are deliberately fixed: the 19 canonical referential EEG
electrodes (auxiliary source channels excluded), 256 Hz input, 200 Hz output,
four-second cores, and 12 seconds of context on each side. Source windows are
`[batch, 7168, source_channels]` float32 volts. Unsupported rates, core sizes,
montages, shapes, dtypes, devices and worker execution fail explicitly. Only the
GPU assembler feeds the candidate; CPU readers/workers retain CPU preprocessing.
The model/objective, division by 100, masks, initialization, optimizer schedules
and EMA callbacks are unchanged.

The implementation follows the inspected **MNE 1.10.2** `Raw.resample`,
`filter._resample_fft`, `cuda._fft_resample`, `cuda._smart_pad`, `notch_filter`
and zero-phase FIR convolution semantics. Other MNE versions are rejected for
this candidate, while baseline behavior is unchanged:

- FFT resampling uses float64, the default flat/boxcar window and odd reflection
  about each endpoint (`reflect_limited`). Auto padding adds 512 samples per side
  to reach 8192. Resample to 6400 with the even-length downsampling Nyquist-bin
  correction and factor `6400/8192`; remove 400 samples per side to retain 5600.
- Design the same MNE firwin/Hamming 0.3–75 Hz bandpass (2201 taps) and 60 Hz
  notch (1321 taps; 0.3 Hz notch width, 0.5 Hz transitions per side). Apply them
  sequentially using separate float64 GPU FFT linear convolutions. Each stage
  reflects `taps - 1` samples per side and compensates the single-pass zero-phase
  group delay. This is not forward/backward filtering. Full convolution replaces
  MNE's overlap-add scheduling; double-precision FFT roundoff can differ.
- Crop target samples `[2400:3200]`, multiply volts by `1e6` in float64, convert
  once to float32 and construct `[batch, 19, 4, 200]` microvolt patches on GPU.
  Per-batch processing performs no host signal copies or scalar reads. Existing
  preflight and completed-loss checks reject nonfinite inputs/results.

CPU FIR design, channel mapping, coefficient/index upload and FIR spectra are
prepared once before preflight/timing. The report records preparation seconds,
coefficient hashes, FIR/FFT lengths, MNE version and excluded work. The unchanged
CPU baseline still designs its filters within preprocessing. Do not hide this
setup/amortization difference when interpreting an end-to-end comparison.
No custom streams, buffer pools or GPU decoder are introduced. Float64 FFTs may
be slower on the available GPU; improvement remains an experimental hypothesis.

Primary timing retains the existing clock before DataLoader iterator creation
through final optimizer/teacher completion, including reading/assembly, required
transfers, preprocessing and training. Candidate data stay on GPU; its loader
uses zero workers and disables host pinning for GPU tensors. CPU controls retain
workers 0/2 and their existing pinning behavior. Main mode adds no per-step
synchronization. `--diagnostic` runs separate replays, synchronizing after GPU
assembly and preprocessing so phase durations represent completed work; its
rates stay out of the primary summary. Startup and sustained intervals are
unchanged. Two sustained updates in the smoke below are not steady-state evidence.

Local tests run the same tensor kernels **on CPU**, comparing each signal stage
and final patches with independent public MNE operations. Endpoint impulses,
context/crop boundaries, 60/90 Hz signals, Nyquist frequencies, channel order,
units and repeated/unordered windows are exercised at batches 1/2/4. Local final
normalized differences were zero; the external prototype's maximum resampling
difference was about `1.4e-18` volts. These are CPU algorithm checks, not CUDA
validation. The frozen exact CPU preprocessing oracle and reference-model tests
remain independent and unchanged. Model-input tolerances stay
`rtol=2e-5, atol=1e-6`, and loss tolerances stay `rtol=2e-4, atol=2e-5`.

Before any longer benchmark, execute the eleven GPU-marked tests above on an
actual allocated CUDA device. Three new CUDA kernel cases compare against MNE
and reject per-batch host transfers/scalar reads; the new training case compares
all six path/worker cases over matched trials and separate diagnostics. It checks
inputs, masks, initialization, finite losses, loss parity, four updates and EMA
counts. Skips are incomplete validation. Then run this four-update smoke from
stable-pretraining, after the environment/reference setup above:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$HOME/stable-datasets${PYTHONPATH:+:$PYTHONPATH}"
python -m examples.eeg_jepa.pilot --source /path/on/hydra/eeg4.edf \
  --prepared-workspace /path/on/hydra/prepared-1024 \
  --reference-root "$EEG_JEPA_REFERENCE" --output "$study_root/gpu-fft-f64-v1-smoke" \
  --paths current_cache edf_memmap edf_gpu_assembly --accelerator gpu \
  --gpu-preprocessing --steps 4 --batch-size 2 --seconds 4 --seed 0 \
  --trials 3 --workers 0 2 --startup-steps 2 --diagnostic
```

This produces six cases: cache CPU workers 0/2, memmap CPU workers 0/2, existing
GPU assembly with CPU/MNE preprocessing, and GPU assembly with GPU preprocessing.
All replay identical windows, masks and update budgets on the same model GPU;
trial order alternates. Inspect every input/loss difference, teacher count,
preparation record and phase before interpreting timing. Actual CUDA kernel
parity and the real-recording smoke remain **unvalidated locally**. Require both
to pass before calling the GPU route validated. A longer run must additionally
show whether prefetch/startup distort sustained rates and report unchanged or
slower performance honestly, including preparation and device/memory limitations.
