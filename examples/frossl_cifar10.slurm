#!/bin/bash
#SBATCH --job-name=frossl-cifar10
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=frossl-cifar10-%j.out
#SBATCH --error=frossl-cifar10-%j.err

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the shared stable-pretraining checkout}"
: "${DATA_ROOT:?Set DATA_ROOT to a shared directory containing CIFAR-10}"
: "${RUN_ROOT:?Set RUN_ROOT to a shared directory for checkpoints and metrics}"
cd "$REPO_ROOT"

"${PYTHON_BIN:-python}" -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot use the allocated GPU"; print("CUDA device:", torch.cuda.get_device_name(0))'

"${PYTHON_BIN:-python}" examples/frossl_cifar10.py \
    --data-root "$DATA_ROOT" \
    --cache-dir "$RUN_ROOT" \
    --epochs "${FROSSL_EPOCHS:-100}" \
    --batch-size "${FROSSL_BATCH_SIZE:-128}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
