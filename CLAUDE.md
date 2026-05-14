# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`stable-pretraining` is a PyTorch Lightning framework for self-supervised learning (SSL) research. The key design principle is that users only define `forward()` — not `training_step()`. All data flows as dictionaries, giving callbacks and middleware access to any intermediate value.

## Commands

**Installation:**
```bash
pip install -e .                     # Core only
pip install -e ".[vision,tracking]"  # With timm/HF and wandb
pip install -e ".[all]"              # Everything
```

**Running experiments:**
```bash
spt examples/simclr_cifar10_config.yaml                   # From YAML config
spt examples/simclr_cifar10_config.yaml trainer.max_epochs=50  # With overrides
spt examples/simclr_cifar10_slurm.yaml -m                 # SLURM multirun
```

**Testing:**
```bash
python -m pytest stable_pretraining/tests -m unit --verbose   # Fast unit tests (CI default)
python -m pytest stable_pretraining/tests -m integration      # Integration tests
python -m pytest -m "not slow"                                # Skip slow tests
# Markers: unit, integration, gpu, slow, download, ddp
```

**Linting:**
```bash
ruff check stable_pretraining --fix
ruff format stable_pretraining
pre-commit run --all-files
```

**Registry CLI:**
```bash
spt registry ls                    # List runs
spt registry best val_acc -n 5     # Top 5 by metric
spt registry export sweep.csv      # Export to CSV
spt registry scan --full           # Rebuild SQLite cache
```

## Architecture

### Core 4-Component Design

1. **Data** (`stable_pretraining/data/`) — dictionary-structured datasets, multi-view transforms (`transforms.py`), `RepeatedRandomSampler`, and HuggingFace-compatible `HFDataset`.

2. **Module** (`stable_pretraining/module.py`) — `Module` extends PyTorch Lightning. Users provide a custom `forward(batch, stage)` function and the framework builds `training_step` / `validation_step` around it. Supports manual optimization for multi-optimizer methods (BYOL, DINO).

3. **Manager** (`stable_pretraining/manager.py`) — orchestrates Trainer, Module, DataModule, callbacks, and loggers. Handles SLURM preemption, atomic checkpointing, and deterministic run IDs.

4. **Callbacks** (`stable_pretraining/callbacks/`) — rich monitoring ecosystem. Notable ones:
   - `OnlineProbe` — linear evaluation during training
   - `OnlineKNN` — k-nearest neighbor evaluation
   - `RankMe` / `LiDAR` — representation quality metrics
   - `LatentViz` — 2D dimensionality reduction visualization
   - `TeacherStudent` — EMA teacher weight updates
   - `WDSchedule` — per-batch weight decay scheduling

### Key Design Patterns

- **Lazy loading** (PEP 562) in `__init__.py` for fast CLI startup — avoid importing anything heavy at module level.
- **Dictionary data flow** — batches are always `dict`, never raw tensors. This is what allows callbacks to intercept intermediate values.
- **Filesystem-first registry** — sidecars + SQLite cache under `stable_pretraining/registry/`. Zero write contention on HPC shared filesystems.
- **Atomic checkpointing** (`utils/atomic_checkpoint.py`) — write to temp file, then rename, to survive preemption.
- **Manual optimization** (`utils/lightning_patch.py`) — patches Lightning's training loop for multi-optimizer SSL methods.

### Pre-built SSL Forward Functions

`stable_pretraining/forward.py` provides ready-made `forward()` implementations for 27+ methods: SimCLR, BYOL, VICReg, Barlow Twins, SwAV, DINO, DINOv2, MAE, iBOT, MoCo v2/v3, and more. Import these directly rather than reimplementing.

### Configuration System

Two layers:
1. **Hydra / OmegaConf** — YAML configs passed to `spt` CLI, supporting `${interpolation}` and multirun sweeps.
2. **Global config** (`stable_pretraining/_config.py`) — `spt.set(key, value)` / `spt.get_config()` for runtime flags accessible anywhere without threading through call stacks.

### Losses (`stable_pretraining/losses/`)

Organized by SSL family:
- Joint-embedding: `NTXEntLoss`, `BYOLLoss`, `VICRegLoss`, `BarlowTwinsLoss`, `SwAVLoss`
- Self-distillation: `DINOv1Loss`, `DINOv2Loss`, `iBOTPatchLoss`
- Reconstruction: `MAELoss`
- Multimodal: `CLIPLoss`

### Backbones (`stable_pretraining/backbone/`)

Thin wrappers around torchvision, timm, and HuggingFace models. Built-ins include `MLP`, `ConvMixer`, `Resnet9`, `MAEDecoder`, `FlexibleTransformer`, and `TeacherStudentWrapper`.

### Loggers (`stable_pretraining/loggers/`)

Integrations for WandB, Trackio, and SwanLab. All loggers receive the same dictionary-structured data as callbacks.

## Testing Conventions

Tests live in `stable_pretraining/tests/`. Mark each test with the appropriate pytest marker (`unit`, `integration`, `gpu`, `slow`, `download`, `ddp`). CI runs only `unit` tests. Coverage is measured against `stable_pretraining/` excluding the tests themselves (see `pytest.ini`).

## Linting

Ruff is the single linter/formatter (Google docstring convention, max line length 88). Pre-commit hooks enforce this plus trailing whitespace, YAML validity, ShellCheck, and Codespell. Examples directory is excluded from Ruff.
