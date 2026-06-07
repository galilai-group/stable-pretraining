#!/bin/bash
#SBATCH --job-name=jax-simclr-inette
#SBATCH --partition=spot
#SBATCH --qos=debug_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=jax_simclr_imagenette_%j.log
# SimCLR on ImageNette, JAX/Flax-NNX backend, data-parallel over 2 H200 GPUs.
# A single Python process sees both GPUs via CUDA_VISIBLE_DEVICES; the Trainer's
# data_parallel switch shards each batch across them (XLA all-reduces grads).
#
# Launch:  sbatch examples/jax_simclr_imagenette.slurm
# Or interactively:  srun --partition=spot --qos=debug_qos --gres=gpu:2 \
#                         --cpus-per-task=16 --mem=96G --time=00:30:00 --pty bash

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# Let JAX grow GPU memory on demand instead of pre-grabbing 75% per process.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python examples/jax_simclr_imagenette.py \
    --epochs 20 \
    --batch-size 256 \
    --img-size 128 \
    --num-workers 16 \
    --lr 1.0 \
    --temperature 0.5
