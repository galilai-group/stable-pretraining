#!/usr/bin/env python
"""Universal experiment runner for stable-pretraining using Hydra configs.

This script provides a unified entry point for all experiments (training, evaluation, etc.)
via configuration files. It supports both single-file configs and modular Hydra composition.

Usage:
    # Run with a config file
    python -m stable_pretraining.run --config-path ../examples --config-name simclr_cifar10

    # Run with config and override parameters
    python -m stable_pretraining.run --config-path ../examples --config-name simclr_cifar10 \
        module.optimizer.lr=0.01 \
        trainer.max_epochs=200

    # Run hyperparameter sweep
    python -m stable_pretraining.run --multirun \
        --config-path ../examples --config-name simclr_cifar10 \
        module.optimizer.lr=0.001,0.01,0.1
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from .utils.distributed import rank_zero_only

from .config import instantiate_from_config


@rank_zero_only
def print_config(cfg: DictConfig) -> None:
    """Print configuration only on rank 0."""
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)


@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    """Main execution function that instantiates components from config and runs the experiment.

    Args:
        cfg: Hydra configuration dictionary containing all experiment parameters
    """
    # Print configuration for debugging (only on rank 0)
    print_config(cfg)

    # Instantiate and run
    manager = instantiate_from_config(cfg)

    # Check if we got a Manager instance
    if hasattr(manager, "__call__"):
        # It's a Manager, run it
        manager()
    else:
        # It's something else, probably just instantiated components
        print("Warning: Config did not produce a Manager instance.")
        print(f"Got: {type(manager)}")
        return manager


if __name__ == "__main__":
    main()
