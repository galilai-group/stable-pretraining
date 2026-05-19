# stable-pretraining — Agent Instructions

## What this library does

`stable-pretraining` is a PyTorch Lightning framework for self-supervised learning (SSL) research. It provides composable forward functions and full `LightningModule` method classes covering SimCLR, BYOL, VICReg, Barlow Twins, SwAV, NNCLR, DINO, DINOv2, MAE, BEiT, MoCo, and more — all built on top of `lightning`. See [`METHODS.md`](./METHODS.md) for the complete catalog. The key design principle is that **users only define `forward(self, batch, stage)`** — the framework builds `training_step` and `validation_step` around it, with all data flowing as dicts so callbacks can intercept any intermediate value without modifying the forward function.

## Repository layout

```
stable_pretraining/
  __init__.py       # lazy-loaded public API (PEP 562); add new exports here
  module.py         # Module — LightningModule all methods share; wraps user forward fn
  manager.py        # Manager — programmatic entry point; prefer over Trainer directly
  forward.py        # forward functions (simclr_forward, byol_forward, …)
  methods/          # LightningModule subclasses, one per SSL method
  callbacks/        # evaluation and training callbacks (OnlineProbe, OnlineKNN, RankMe, …)
  losses/           # loss classes ({Method}Loss naming convention)
  backbone/         # encoder wrappers (torchvision, timm, HuggingFace)
  data/             # HFDataset, MultiViewTransform, RepeatedRandomSampler
  loggers/          # WandB, Trackio, SwanLab integrations
  registry/         # filesystem-first run registry (sidecars + SQLite)
  optim/            # optimizer and scheduler factories
  utils/            # atomic checkpointing, lightning patch, error handling
  _config.py        # global config: spt.set(key, value) / spt.get_config()
examples/           # runnable .py scripts and YAML configs
docs/               # Sphinx source
METHODS.md          # ground-truth index of all methods + forward functions
```

## How to import

**Top-level lazy imports (preferred):**
```python
import stable_pretraining as spt

model = spt.Module(forward=..., backbone=..., projector=...)
manager = spt.Manager(trainer=trainer, module=model, data=data_module)

# Method classes (most-used ones hoisted to top level)
model = spt.SimCLR(...)
model = spt.BYOL(...)
model = spt.DINO(...)
model = spt.DINOv2(...)
model = spt.MAE(...)
model = spt.BarlowTwins(...)
model = spt.VICReg(...)
model = spt.SwAV(...)
model = spt.NNCLR(...)

# Callbacks
probe = spt.OnlineProbe(...)
knn = spt.OnlineKNN(...)
rankme = spt.RankMe(...)
```

**Direct module imports:**
```python
from stable_pretraining.forward import simclr_forward, byol_forward
from stable_pretraining.methods import SimCLR, BYOL, DINO, VICReg, MAE  # full list in METHODS.md
from stable_pretraining.callbacks import OnlineProbe, OnlineKNN, RankMe, LiDAR
from stable_pretraining.losses import NTXEntLoss, BYOLLoss, VICRegLoss
```

**YAML config (Hydra):**
```yaml
module:
  _target_: stable_pretraining.Module
  forward: stable_pretraining.forward.simclr_forward
  backbone: ...
  projector: ...
```

## Core concepts

### 1. Forward function — stateless SSL logic

A forward function is a plain Python function with signature:
```python
def my_forward(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    ...
```

`self` is the `Module` instance (bound at runtime). `batch` is always a `dict` — never a raw tensor. `stage` is `"train"`, `"val"`, or `"test"`. The function must return a dict; during training that dict must contain `"loss"`. Forward functions are stateless and reusable — they are specified by dotted import path in YAML configs.

### 2. Module — the LightningModule wrapper

`Module` (in `stable_pretraining/module.py`) extends `pl.LightningModule`. It:
- Receives a `forward` callable at init and binds it to `self`
- Builds `training_step` and `validation_step` from that callable
- Manages optimizer and scheduler configuration (single or multi-optimizer)
- Supports manual optimization for methods with EMA teachers (BYOL, DINO)
- Stores arbitrary sub-modules (backbone, projector, predictor, etc.) as attributes

### 3. Manager — the programmatic entry point

`Manager` (in `stable_pretraining/manager.py`) orchestrates a full training run:
- Wraps a `pl.Trainer`, `Module`, and `DataModule`
- Handles SLURM preemption (SIGTERM → checkpoint → requeue)
- Assigns deterministic run IDs and manages atomic checkpointing
- Resolves logger-specific resume logic (WandB, Trackio, SwanLab, RegistryLogger)

### Minimal working example

```python
import stable_pretraining as spt
import torch
from functools import partial

# 1. Build components
backbone = spt.backbone.resnet18()
projector = torch.nn.Linear(512, 128)

# 2. Wire a forward function into Module
from stable_pretraining.forward import simclr_forward
from stable_pretraining.losses import NTXEntLoss

module = spt.Module(
    forward=simclr_forward,
    backbone=backbone,
    projector=projector,
    simclr_loss=NTXEntLoss(temperature=0.5),
    optim={
        "optimizer": partial(torch.optim.AdamW, lr=1e-3),
        "scheduler": "CosineAnnealingLR",
    },
)

# 3. Add evaluation callbacks
probe = spt.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss=torch.nn.CrossEntropyLoss(),
)

# 4. Run via Manager
import lightning as pl
trainer = pl.Trainer(max_epochs=100, callbacks=[probe])
manager = spt.Manager(trainer=trainer, module=module, data=data_module)
manager()
```

Alternatively, use a pre-wired method class for the same result with fewer lines:
```python
model = spt.SimCLR(backbone=backbone, projector=projector, temperature=0.5, lr=1e-3)
```

## Naming conventions

| Component | Convention | Example |
|-----------|-----------|---------|
| Forward functions | `{method}_forward` (snake_case) | `simclr_forward` |
| Loss classes | `{Method}Loss` (CamelCase) | `NTXEntLoss`, `BYOLLoss` |
| Method classes | `{Method}` (CamelCase) | `SimCLR`, `BYOL`, `DINOv2` |
| Callbacks | Descriptive CamelCase, no suffix | `OnlineProbe`, `RankMe` |
| YAML config keys | Match Python argument names exactly | `max_epochs`, `batch_size` |
| Batch dict keys | snake_case strings | `"image"`, `"label"`, `"embedding"` |

## How to add a new SSL method

1. **Forward function** — add `{method}_forward` to `stable_pretraining/forward.py`:
   - Signature: `def {method}_forward(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:`
   - Full Google-style docstring with `Args:`, `Returns:`, and `Note:` sections
   - Follow the existing pattern: handle multi-view vs single-view, log loss, return dict with `"embedding"` and `"loss"`

2. **Loss class** — add `{Method}Loss` to `stable_pretraining/losses/{method}.py`, export from `stable_pretraining/losses/__init__.py`

3. **Method class** — add `{Method}` LightningModule to `stable_pretraining/methods/{method}.py`

4. **Export from methods** — add to `stable_pretraining/methods/__init__.py`:
   ```python
   from .{method} import {Method}
   ```
   and add `"{Method}"` to `__all__`

5. **Top-level export** — add to `_LAZY_ATTRS` in `stable_pretraining/__init__.py`:
   ```python
   "{Method}": ("stable_pretraining.methods.{method}", "{Method}"),
   ```
   and add `"{Method}"` to `__all__`

6. **Method catalog** — add a row to `METHODS.md` with all columns filled in

7. **Type annotations** — ensure the forward function has full annotations including `-> dict[str, torch.Tensor]`

## How to run examples and tests

**Install:**
```bash
pip install -e ".[dev]"          # includes pytest, ruff, sphinx
pip install -e .                 # core only
```

**Run an example:**
```bash
spt examples/simclr_cifar10_config.yaml                        # YAML config via CLI
spt examples/simclr_cifar10_config.yaml trainer.max_epochs=50  # with overrides
spt examples/simclr_cifar10_slurm.yaml -m                      # SLURM multirun
python examples/supervised_learning.py                         # Python script
```

**Run tests:**
```bash
python -m pytest stable_pretraining/tests -m unit --verbose    # CI default (fast)
python -m pytest stable_pretraining/tests -m integration       # integration tests
python -m pytest -m "not slow"                                 # skip slow tests
# Markers: unit, integration, gpu, slow, download, ddp
```

**Lint:**
```bash
ruff check stable_pretraining --fix
ruff format stable_pretraining
pre-commit run --all-files
```

**Registry CLI:**
```bash
spt registry ls                  # list runs
spt registry best val_acc -n 5   # top 5 by metric
spt registry export sweep.csv    # export to CSV
spt registry scan --full         # rebuild SQLite cache
```

## Callback ordering

Lightning runs `trainer.callbacks` in registration order. Within a single hook, callbacks fire in that order; across hooks, Lightning completes each hook for **every** callback before moving to the next.

Practically: producer/consumer pairs split across **different hooks** (e.g., `OnlineQueue` builds its snapshot in `on_validation_epoch_start`, `OnlineKNN` reads it in `on_validation_batch_end`) are **not** order-sensitive — the producer hook is already done for every callback before any consumer hook runs. Don't worry about ordering those.

Order **does** matter when two callbacks act in the **same** hook and one reads what the other writes:

| Callback | Rule |
|----------|------|
| `TeacherStudentCallback` | After any callback that reads teacher params in `on_train_batch_end` — its EMA update fires there |
| `OnlineProbe` | After callbacks that mutate the batch embedding in `on_train_batch_end` (e.g., normalization probes) |
| `OnlineWriter` | Last among per-batch callbacks — captures all mutations in `on_train_batch_end` |
| `CleanUpCallback` | After callbacks that save artefacts in `on_train_end` / teardown (checkpoint callbacks, `hf_models`, …) |

At runtime, the default `TrainerInfo` callback logs the full callback list with `⚑` markers on order-sensitive ones — check that log first when debugging an ordering issue.

The authoritative registry lives in `stable_pretraining.callbacks.utils.ORDER_SENSITIVE_CALLBACKS`. When adding a new order-sensitive callback, append it there so the runtime log surfaces the constraint.

## What agents must not do

- **Do not** modify `stable_pretraining/__init__.py` lazy-loading machinery without reading it fully first — PEP 562 `__getattr__` is in use and changes break all lazy imports
- **Do not** add public functions without type annotations and Google-style docstrings
- **Do not** add a new method class without a corresponding entry in `METHODS.md` and a forward function in `forward.py` (where applicable)
- **Do not** change loss function implementations without running tests
- **Do not** use deep module imports in example code — use the top-level namespace (`import stable_pretraining as spt; spt.SimCLR`)
- **Do not** add comments that explain *what* code does — only add comments for *why* (hidden constraints, non-obvious invariants, workarounds)
- **Do not** return raw tensors from forward functions — always return a `dict`

## Key design decisions (context for agents)

**Why forward functions are stateless.** Forward functions take `self` as their first argument but are defined as plain functions, not methods. This lets them be specified as dotted import paths in YAML configs (`forward: stable_pretraining.forward.simclr_forward`) and instantiated by Hydra. At runtime `Module` binds the function to `self` using `types.MethodType`, so `self.backbone`, `self.projector`, etc. are all accessible. Keeping them stateless (no internal state, no class inheritance) makes them composable and testable in isolation.

**Why Manager exists alongside Trainer.** `pl.Trainer` handles the training loop; `Manager` handles everything around it. Specifically: detecting SLURM preemption via `SIGTERM`, writing checkpoint-then-requeue, assigning deterministic run IDs (so resumed runs pick up the same WandB/registry run), and resolving which checkpoint to resume from across multiple logger backends. Using `Manager(...)()` instead of `Trainer.fit(...)` is the correct programmatic API.

**Why lazy loading is used.** `import stable_pretraining` is called at CLI startup even for lightweight commands (`spt registry ls`). Eagerly importing Lightning, HuggingFace `datasets`, or timm would add 3–5 seconds to every CLI invocation. PEP 562 `__getattr__` defers those imports until the first heavy attribute access. The deferred init (Lightning manual-optimization patch, atomic checkpoint install) also runs at that point, not at import time.

**How the callback system works.** Callbacks are standard `pl.Callback` subclasses, but they receive the full `batch` dict (which is mutated by `forward`) via `outputs` in `on_train_batch_end` / `on_validation_batch_end`. This means any key written to the return dict in `forward` (e.g., `"embedding"`, `"loss"`, `"swav_queue"`) is automatically available to every callback with no wiring. Adding `OnlineProbe`, `OnlineKNN`, or `RankMe` to a training run requires zero changes to the forward function.

**How multi-optimizer / EMA teacher methods work.** Methods like BYOL and DINO require two optimizers (online and target networks) and an EMA update for the target. `Module` supports manual optimization: if `optim` is a dict of named optimizer configs, Lightning's automatic optimization is disabled and `Module.training_step` calls each optimizer explicitly. The `TeacherStudentCallback` handles the EMA weight update after each step. The `TeacherStudentWrapper` backbone exposes `.forward_student()` and `.forward_teacher()` so forward functions can address each network cleanly.
