# stable-pretraining

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://galilai-group.github.io/stable-pretraining/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-blue.svg)](https://github.com/galilai-group/stable-pretraining/tree/main/benchmarks)
[![Test Status](https://github.com/galilai-group/stable-pretraining/actions/workflows/testing.yml/badge.svg)](https://github.com/galilai-group/stable-pretraining/actions/workflows/testing.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WandB](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/site)



AI is moving beyond labels. Today's models learn through **self-supervision** and **multimodal alignment**, extracting knowledge from raw data to build general-purpose representations that work across tasks. These foundation models are then deployed at scale, often after finetuning, to solve tasks in zero or few shot.

`stable-pretraining` is a PyTorch framework built on top of Lightning for this new paradigm. What sets us apart is **real-time visibility into training quality** through extensive logging and monitoring. Our callback ecosystem (`OnlineProbe`, `OnlineKNN`, `RankMe`, and many more) provides insights into feature collapse, training dynamics, and downstream performance. Data flows as dictionaries through model components, metrics, and callbacks, making any intermediate value accessible and debuggable. With `stable-pretraining`: track everything, debug faster, iterate sooner.

Join our Discord: [https://discord.gg/8M6hT39X](https://discord.gg/adzpqWKM25)

## How?

To reach flexibility, scalability and stability, we rely on battle-tested third party libraries: `PyTorch`, `Lightning`, `HuggingFace`, `TorchMetrics` amongst a few others. Those dependencies allow us to focus on assembling everything into a powerful ML framework. ``stable-pretraining`` adopts a flexible and modular design for seamless integration of components from external libraries, including architectures, loss functions, evaluation metrics, and augmentations.

## Quick setup

```bash
# Clone the repository
git clone https://github.com/galilai-group/stable-pretraining.git

# Install the framework
cd stable-pretraining
pip install -e .
```

For advanced installation options, see [Installation](#installation) below.

## Core Structure

`stable-pretraining` simplifies complex ML workflows into 4 intuitive components:

### 1 - Data
Your dataset must follow a dictionary-structured format where each sample is a dictionary with named fields (e.g., `{"image": ..., "label": ...}`). This ensures consistent behavior across all components. You have multiple options for creating datasets:

- **HuggingFace datasets** (if available on the Hub):
```python
import stable_pretraining as spt
train_dataset = spt.data.HFDataset(
    path="frgfm/imagenette",
    name="160px",
    split="train",
    transform=train_transform,
)
```

- **From PyTorch datasets**:
```python
train_dataset = spt.data.FromTorchDataset(
    torchvision_dataset,
    names=["image", "label"],  # Map tuple outputs to dictionary keys
    transform=train_transform,
)
```

- **Custom datasets**: Any dataset that returns dictionaries

```python
datamodule = spt.data.DataModule(train=train_dataloader, val=val_dataloader)
```

### 2 - Module
The key differentiator from PyTorch Lightning - **you only define the `forward` function**, not `training_step`! This unified approach computes losses and generates useful quantities that can be retrieved for monitoring and analysis:

```python
# Use the pre-built forward functions from stable_pretraining
from stable_pretraining import forward

# Simply use the appropriate forward for your method
module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.simclr_forward,  # Or byol_forward, vicreg_forward, etc.
    simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
    optim={
        "optimizer": {"type": "Adam", "lr": 0.001},
        "scheduler": {"type": "CosineAnnealingLR"},
        "interval": "epoch"
    }
)
```

Or define your own custom forward:
```python
def forward(self, batch, stage):
    out = {}

    if isinstance(batch, list):
        # Multi-view training - batch is a list of view dicts
        embeddings = [self.backbone(view["image"]) for view in batch]
        out["embedding"] = torch.cat(embeddings, dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]
            out["loss"] = self.simclr_loss(projections[0], projections[1])
    else:
        # Single-view validation
        out["embedding"] = self.backbone(batch["image"])

    return out
```

**Key points:**
- The `forward` method defines both the loss and any quantities to monitor
- No need to override `training_step`, `validation_step`, etc.
- Return a dictionary with a `"loss"` key for training
- All model components are passed as kwargs to `spt.Module`

### 3 - Callbacks

Monitor and evaluate your models in real-time during training. Callbacks are key ingredients of `stable-pretraining`, providing rich insights without interrupting your training flow.

#### Evaluation & Monitoring

| Callback | Description |
|----------|-------------|
| `OnlineProbe` | Trains a lightweight linear probe on frozen representations to track downstream task accuracy in real-time. Maintains its own optimizer and training loop. |
| `OnlineKNN` | Non-parametric k-nearest neighbors evaluator using a rolling queue of cached embeddings. Zero training cost. |
| `RankMe` | Tracks effective rank of feature representations via singular values. A rank drop signals dimensional collapse. |
| `LiDAR` | Linear Discriminant Analysis Rank over surrogate classes of augmented views. |
| `CLIPZeroShot` | Zero-shot classification for CLIP-style models. Compares image embeddings against text-encoded class names. |
| `ImageRetrieval` | Image retrieval evaluator following the DINO protocol with query/gallery splits. |
| `LatentViz` | Online 2D visualization of the latent space. Learns a neighborhood-preserving projection and periodically plots it. |
| `EpochMilestones` | Early-stops training if a metric fails to reach a threshold by a given epoch. |

#### Training Utilities

| Callback | Description |
|----------|-------------|
| `TeacherStudentCallback` | Auto-discovers `TeacherStudentWrapper` instances and performs EMA teacher updates at configurable frequency. |
| `WeightDecayUpdater` | Updates weight decay on a per-batch schedule (constant, linear, cosine, or exponential). |
| `EmbeddingCache` | Hooks into named submodules to cache intermediate embeddings for downstream use. |

#### Checkpointing & Export

| Callback | Description |
|----------|-------------|
| `SklearnCheckpoint` | Saves and restores scikit-learn models (probes, classifiers) inside Lightning checkpoints. |
| `WandbCheckpoint` | Uploads checkpoints to Weights & Biases as artifacts with run-resume support. |
| `StrictCheckpointCallback` | Controls strict/non-strict checkpoint loading with detailed mismatch reporting. |
| `HuggingFaceCheckpointCallback` | Exports HuggingFace-compatible checkpoints for any `PreTrainedModel` submodule (zero-knowledge reload). |

#### System & Logging

| Callback | Description |
|----------|-------------|
| `LoggingCallback` | Displays validation metrics in a color-coded formatted table after each epoch. |
| `ModuleSummary` | Logs detailed parameter statistics (trainable, frozen, per-layer) at the start of training. |
| `TrainerInfo` | Links trainer to DataModule and logs trainer configuration. |
| `SLURMInfo` | Extracts and logs SLURM environment information (job ID, partition, resources). |
| `EnvironmentDumpCallback` | Dumps Python version, CUDA info, installed packages, git state, and env vars to `environment.json` for exact reproducibility. |
| `LogUnusedParametersOnce` | Reports parameters that receive no gradient after the first backward pass. Useful for catching wiring bugs. |
| `CleanUpCallback` | Removes selected training artifacts (SLURM logs, Hydra files, checkpoints, etc.) after successful training. Keeps everything on failure for debugging. |
| `ModuleRegistryCallback` | Registers the module for global logging access. Enables `spt.log()` and `spt.log_dict()` from anywhere. |

#### Intelligent Queue System

Callbacks that need rolling feature stores (`OnlineKNN`, `RankMe`, `LiDAR`, `LatentViz`) share memory through an automatic queue management system. If two callbacks monitor the same key with different queue lengths, a single queue is allocated at the maximum length and shared, eliminating redundant computation.

**Why callbacks matter:** Get real-time feedback on representation quality, catch issues like collapse early, and track multiple metrics simultaneously. For detailed usage and practical considerations, see the [Callback guide](stable_pretraining/callbacks/README.md).

**Example:**
```python
# Monitor SSL representations with a linear classifier
linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

# Track representation quality with KNN evaluation
knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    k=10,
)
```

### 4 - Trainer
Orchestrate everything together with PyTorch Lightning's `Trainer`:

```python
trainer = pl.Trainer(
    max_epochs=10,
    num_sanity_val_steps=1,
    callbacks=[linear_probe, knn_probe, rankme],  # Your monitoring callbacks
    precision="16-mixed",
    logger=False,
    enable_checkpointing=False,
)
manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
```

Once configured, the `Manager` connects all components and handles the training loop with precise logging and monitoring (optional).

## Global Configuration

Instead of scattering options across environment variables, callback constructors, and factory functions, `stable-pretraining` provides a single entry-point to configure library-wide behavior:

```python
import stable_pretraining as spt

spt.set(
    verbose="WARNING",                          # Global log level (also controls callback verbosity)
    progress_bar="rich",                        # "auto", "rich", "simple", or "none"
    cleanup={"checkpoints": False, "slurm": False},  # What CleanUpCallback removes
    log_rank="all",                             # Which distributed rank(s) may log (default: 0)
    default_callbacks={"env_dump": False},       # Toggle individual default callbacks on/off
)

# Inspect the current configuration
print(spt.get_config())
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `verbose` | `str` or `int` | `"INFO"` | Loguru log level. Accepts `"DEBUG"`, `"INFO"`, `"WARNING"`, etc., or Python logging ints (`10`, `20`, `30`). Also controls the `verbose` flag on all callbacks when left at their default. |
| `progress_bar` | `str` | `"auto"` | Progress bar style. `"auto"` picks `"rich"` for TTYs and `"simple"` for non-interactive environments. `"none"` disables it. |
| `cleanup` | `dict` | keeps checkpoints & logs | Controls which artifact categories `CleanUpCallback` removes after training. Keys: `"checkpoints"`, `"logs"`, `"hydra"`, `"slurm"`, `"env_dump"`, `"callback_artifacts"`. Values are bools (`True` = keep, `False` = delete). |
| `log_rank` | `int` or `"all"` | `0` | Which distributed rank(s) may produce log output. |
| `default_callbacks` | `dict` | all enabled | Toggle individual default callbacks: `"progress_bar"`, `"registry"`, `"logging"`, `"env_dump"`, `"trainer_info"`, `"sklearn_checkpoint"`, `"wandb_checkpoint"`, `"module_summary"`, `"slurm_info"`, `"unused_params"`, `"hf_checkpoint"`. |

Settings apply immediately and persist for the process lifetime. `spt.set()` can be called multiple times; only the settings you pass are updated.

## Built-in Methods

`stable-pretraining` ships with ready-to-use forward functions and matching loss functions for popular self-supervised learning methods:

| Method | Forward | Loss | Description |
|--------|---------|------|-------------|
| Supervised | `forward.supervised_forward` | any | Standard supervised training with labels |
| SimCLR | `forward.simclr_forward` | `NTXEntLoss` | Contrastive learning with 2 augmented views |
| BYOL | `forward.byol_forward` | `BYOLLoss` | Momentum-based self-distillation without negatives |
| VICReg | `forward.vicreg_forward` | `VICRegLoss` | Variance-invariance-covariance regularization |
| Barlow Twins | `forward.barlow_twins_forward` | `BarlowTwinsLoss` | Cross-correlation matrix alignment to identity |
| SwAV | `forward.swav_forward` | `sinkhorn_knopp` | Online clustering with Sinkhorn-Knopp normalization |
| NNCLR | `forward.nnclr_forward` | `NTXEntLoss` | Nearest-neighbor contrastive learning |
| DINO | `forward.dino_forward` | `DINOv1Loss` | Self-distillation with multi-crop and centering |
| DINOv2 | `forward.dinov2_forward` | `DINOv2Loss` | DINO + iBOT masked patch prediction |

## Backbones

Load architectures from popular libraries or use built-in components:

```python
# From torchvision
backbone = spt.backbone.from_torchvision("resnet50")

# From timm (thousands of pretrained models)
backbone = spt.backbone.from_timm("vit_base_patch16_224")

# From HuggingFace
backbone = spt.backbone.vit_hf("google/vit-base-patch16-224")
```

Additional building blocks: `MLP`, `ConvMixer`, `Resnet9`, `TeacherStudentWrapper` (EMA), `MAEDecoder`, `MaskedEncoder`, `FlexibleTransformer`, `PatchMasking`, `IJEPAMasking`, `MultiBlockMasking`, `LinearProbe`, `AutoLinearClassifier`, `AutoTuneMLP`, and more.

## Optimizers & Schedulers

| Component | Description |
|-----------|-------------|
| `LARS` | Layer-wise Adaptive Rate Scaling - the standard optimizer for SSL |
| `LinearWarmupCosineAnnealing` | Linear warmup followed by cosine decay |
| `LinearWarmupCyclicAnnealing` | Linear warmup followed by cyclic cosine decay |
| `CosineDecayer` | Pure cosine decay schedule |
| `create_optimizer` / `create_scheduler` | Factory functions that accept string names, dicts, or partial objects |

```python
module = spt.Module(
    ...,
    optim={
        "optimizer": {"type": "LARS", "lr": 5, "weight_decay": 1e-6},
        "scheduler": {"type": "LinearWarmupCosineAnnealing"},
        "interval": "epoch",
    },
)
```

## Complete Example

<details>
<summary>SimCLR on CIFAR-10</summary>

This example demonstrates the key features of `stable-pretraining`: dictionary-structured data, unified forward function, and rich monitoring through callbacks.

```python
import lightning as pl
import torch
import torchmetrics
import torchvision
from torch import nn
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms

# Define augmentations for SimCLR (creates 2 views of each image)
simclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        # Second view with slightly different augmentations
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
    ]
)

# Load CIFAR-10 and wrap in dictionary format
cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
cifar_val = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],  # Convert tuple to dictionary
    transform=simclr_transform,
)

val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    ),
)

# Create dataloaders - MultiViewTransform handles the view creation
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=256,
    num_workers=8,
    drop_last=True,
    shuffle=True,  # Simple shuffle, no RepeatedRandomSampler needed
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=10,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

# Build model components
backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
backbone.fc = torch.nn.Identity()  # Remove classification head

projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256),
)

# Create the module using the built-in SimCLR forward function
module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.simclr_forward,  # Use the built-in forward function
    simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
    optim={
        "optimizer": {"type": "LARS", "lr": 5, "weight_decay": 1e-6},
        "scheduler": {"type": "LinearWarmupCosineAnnealing"},
        "interval": "epoch",
    },
)

# Add callbacks for monitoring performance during training
linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
)

# Configure training
trainer = pl.Trainer(
    max_epochs=1000,
    callbacks=[knn_probe, linear_probe],  # Monitor SSL quality in real-time
    precision="16-mixed",
    logger=WandbLogger(project="cifar10-simclr"),
)

# Launch training
manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
```
</details>


## Quick Start with `spt` CLI

The `spt` command launches training from YAML configuration files using Hydra.

**Note:** `spt` requires YAML configs. If you have Python-based configs, you can:
- Convert them to YAML format where each component uses `_target_` to specify the importable class/function
- See `examples/simclr_cifar10_config.yaml` for the structure and syntax

### Local Training

```bash
# Run with a config file
spt examples/simclr_cifar10_config.yaml

# With parameter overrides
spt examples/simclr_cifar10_config.yaml trainer.max_epochs=50 module.optim.lr=0.01

# Run from any directory - supports absolute and relative paths
spt ../configs/my_config.yaml
spt /path/to/config.yaml
```

### SLURM Cluster Training

For training on SLURM clusters, use the `-m` flag to enable multirun mode:

```bash
# Use the provided SLURM template (customize partition/QOS in the file)
spt examples/simclr_cifar10_slurm.yaml -m

# Override SLURM parameters via command line
spt examples/simclr_cifar10_slurm.yaml -m \
    hydra.launcher.partition=gpu \
    hydra.launcher.qos=normal \
    hydra.launcher.timeout_min=720
```

The SLURM template (`examples/simclr_cifar10_slurm.yaml`) includes placeholders for cluster-specific settings. Either modify the file directly or override values via command line.

## Installation

The library is not yet available on PyPI. You can install it from the source code, as follows.

1. <details><summary>conda (optional)</summary>

    First use your favorite environment manager and install your favorite pytorch version, we provide an example with conda
    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
    follow installation instructions... once completed, create your environment
    ```
    conda create -n my_env python=3.11
    ```
    with your environment name (here `my_env`) and your favorite Python version (here, `3.11`). Once completed, make sure to activate your environment (`conda activate my_env`) before proceeding to the next steps!
  </details>

2. Pytorch and our library (we recommend using `uv` for quicker package management):
    ```bash
    pip3 install uv
    uv pip install torch torchvision torchaudio
    uv pip install -e .  # Core dependencies only
    ```

    For optional features (vision models, experiment tracking, cluster support, etc.):
    ```bash
    uv pip install -e ".[vision,tracking]"  # Example: add vision models and wandb
    uv pip install -e ".[all]"  # Or install all optional dependencies
    ```
    See `pyproject.toml` for available dependency groups (`vision`, `tracking`, `cluster`, `visualization`, `datasets`, `extras`, `dev`, `doc`).

    If you do not want to use uv, simply remove it from the above commands.

3. API login (optional)
    ```
    wandb login
    huggingface-cli login
    ```
4. **LaTeX support in Matplotlib** (optional)

    <details>
    <summary>Click to expand setup instructions</summary>

    **Install Computer Modern fonts:**
    ```bash
    mkdir -p ~/.local/share/fonts
    cp assets/cm-unicode-0.7.0\ 2/*.ttf ~/.local/share/fonts/
    fc-cache -f -v
    # verify: fc-list | grep cmu
    ```

    **Clear matplotlib font cache:**
    ```bash
    python -c "import shutil, matplotlib; shutil.rmtree(matplotlib.get_cachedir())"
    ```

    **Install TeX Live (minimal, no sudo):**
    ```bash
    cd /tmp
    wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
    tar xzf install-tl-unx.tar.gz
    cd install-tl-*/
    ./install-tl --texdir ~/texlive --no-interaction --scheme=scheme-basic
    ~/texlive/bin/x86_64-linux/tlmgr install type1cm cm-super dvipng collection-fontsrecommended amsmath amssymb bm underscore xcolor
    ```

    Add to your `~/.bashrc` (or equivalent):
    ```bash
    export PATH="$HOME/texlive/bin/x86_64-linux:$PATH"
    ```

    **Verify:**
    ```bash
    python -c "
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(); plt.title(r'\$\sum_{i=1}^n x_i\$')
    plt.savefig('/tmp/tex_test.png')
    print('Success!')
    "
    ```

    </details>

## Ways You Can Contribute:

- If you'd like to contribute new features, bug fixes, or improvements to the documentation, please refer to our [contributing guide](https://galilai-group.github.io/stable-pretraining/contributing/) for detailed instructions on how to get started.

- You can also contribute by adding new methods, datasets, or configurations that improve the current performance of a method in the [benchmark section](https://github.com/galilai-group/stable-pretraining/tree/main/benchmarks).

## Contributors

Core contributors (in order of joining the project):
- [Randall Balestriero](https://github.com/RandallBalestriero)
- [Hugues Van Assel](https://github.com/huguesva)
- [Sami BuGhanem](https://github.com/sami-bg)
- [Lucas Maes](https://github.com/lucas-maes)
