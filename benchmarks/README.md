# Stable-Pretraining Benchmarks

This directory contains benchmark scripts for various self-supervised learning methods.

## Data Storage Configuration

By default, datasets are stored in `~/.cache/stable-pretraining/data/`. This location can be customized in two ways:

### Option 1: Environment Variable (Recommended)

Set the `STABLE_PRETRAINING_DATA_DIR` environment variable to specify a custom data directory:

```bash
# Set for current session
export STABLE_PRETRAINING_DATA_DIR=/path/to/your/data

# Or set permanently in your shell configuration (~/.bashrc, ~/.zshrc, etc.)
echo 'export STABLE_PRETRAINING_DATA_DIR=/path/to/your/data' >> ~/.bashrc
```

### Option 2: Default Location

If no environment variable is set, data will be stored in:
- `~/.cache/stable-pretraining/data/` (Linux/Mac)
- `C:\Users\<username>\.cache\stable-pretraining\data\` (Windows)

Each dataset will be stored in its own subdirectory (e.g., `cifar10/`, `imagenet/`, etc.).

## Running Benchmarks

### CIFAR-10 Benchmarks

```bash
# SimCLR
python benchmarks/cifar10/simclr-resnet18.py

# BYOL
python benchmarks/cifar10/byol-resnet18.py

# VICReg
python benchmarks/cifar10/vicreg-resnet18.py

# Barlow Twins
python benchmarks/cifar10/barlow-resnet18.py

# NNCLR
python benchmarks/cifar10/nnclr-resnet18.py

# TWIST
python benchmarks/cifar10/twist-resnet18.py

# TWIST on Imagenette (ViT-S/16)
MAX_EPOCHS=200 python benchmarks/imagenet10/twist-vit-small.py
```

## Notes

- The data directory will be created automatically if it doesn't exist
- Downloaded datasets are cached and won't be re-downloaded unless deleted
- Make sure you have sufficient disk space in your chosen data directory
