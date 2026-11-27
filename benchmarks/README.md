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
```


### LeWorldModel (OGBench Cube)

This benchmark evaluates action-conditioned latent prediction
on the OGBench Cube Single Expert dataset using the native
Stable-Pretraining LeWM implementation.

#### Dataset

Download `cube_single_expert.tar.zst` from:
https://huggingface.co/datasets/quentinll/lewm-cube

Extract the dataset:

```bash
tar --use-compress-program=zstd -xf cube_single_expert.tar.zst
```

Install the additional dataset dependencies:

```bash
uv pip install h5py hdf5plugin
```

#### Running the benchmark

```bash
python benchmarks/lewm_cube.py \
    --data /path/to/cube_single_expert.h5 \
    --steps 20000 \
    --batch-size 16
```

#### Training configuration

- Training episodes: 900
- Held-out validation episodes: 100
- Training samples: 163,800
- Validation samples: 18,200
- Training steps: 2,000
- Batch size: 16
- History size: 3
- Prediction horizon: 1
- Frame skip: 5
- Embedding dimension: 192
- SIGReg weight: 0.09
- Optimizer: AdamW
- Learning rate: 5e-5
- Seed: 42


#### Results

| Metric | 2,000 steps | 20,000 steps |
|---|---:|---:|
| Prediction MSE | 0.040693 | 0.012487 |
| Persistence MSE | 0.396340 | 0.729714 |
| Shuffled action MSE | 0.639295 | 0.952279 |
| Embedding STD | 0.671959 | 0.891663 |
| Improvement over persistence | 89.73% | 98.29% |
| Effective rank | Not measured | 11.90 / 192 |

After 20,000 training steps, LeWM achieves a 98.29% reduction in held-out latent prediction error compared with the persistence baseline. Shuffling actions increases prediction error by approximately 76 times, providing evidence of action-conditioned learning.

#### Limitations

The benchmark uses a single seed and a reduced batch size of 16. Planning success was not evaluated. The low effective rank suggests possible partial dimensional collapse, despite strong predictive performance.


## Notes

- The data directory will be created automatically if it doesn't exist
- Downloaded datasets are cached and won't be re-downloaded unless deleted
- Make sure you have sufficient disk space in your chosen data directory
