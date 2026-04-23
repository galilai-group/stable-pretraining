"""Configuration classes specifying default parameters for stable-SSL."""

from typing import Any, Union

import hydra
import omegaconf
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


def collapse_nested_dict(
    cfg: Union[dict, object],
    level_separator: str = ".",
    _base_name: str = None,
    _flat_cfg: dict = None,
) -> dict:
    """Parse a Hydra config and make it readable for wandb (flatten).

    Args:
        cfg (Union[dict, object]): The original (Hydra) nested dict.
        level_separator (str, optional): The string to separate level names. Defaults to ".".
        _base_name (str, optional): The parent string, used for recursion only, users should ignore.
            Defaults to None.
        _flat_cfg (dict, optional): The flattened config, used for recursion only, users should ignore.
            Defaults to None.

    Returns:
        dict: Flat config.
    """
    # INIT
    if _flat_cfg is None:
        _flat_cfg = {}
    if _base_name is None:
        _base_name = ""
    if isinstance(cfg, list) or isinstance(cfg, tuple):
        for i in range(len(cfg)):
            collapse_nested_dict(
                cfg[i],
                level_separator=level_separator,
                _base_name=_base_name + f"{level_separator}{i}",
                _flat_cfg=_flat_cfg,
            )
    elif isinstance(cfg, dict) or isinstance(cfg, omegaconf.dictconfig.DictConfig):
        for key in cfg:
            collapse_nested_dict(
                cfg[key],
                level_separator=level_separator,
                _base_name=_base_name + f"{level_separator}{key}",
                _flat_cfg=_flat_cfg,
            )
    else:
        if _base_name.startswith(level_separator):
            _base_name = _base_name[len(level_separator) :]
        _flat_cfg[_base_name] = cfg
    return _flat_cfg


def recursive_instantiate(
    cfg: Union[dict, omegaconf.DictConfig], parent_objects: dict = None
) -> dict:
    """Recursively instantiate all components in config with dependency resolution.

    Args:
        cfg: Configuration dictionary or DictConfig with _target_ fields
        parent_objects: Optional dict of already instantiated objects for dependencies

    Returns:
        Dictionary of instantiated components
    """
    if cfg is None:
        return {}

    instantiated = {}
    parent_objects = parent_objects or {}

    # Define instantiation order for proper dependency resolution
    # Later items can depend on earlier ones
    priority_order = ["data", "module", "loss", "callbacks", "logger", "trainer"]

    # First pass: instantiate components in priority order
    for key in priority_order:
        if key in cfg:
            try:
                if (
                    isinstance(cfg[key], (dict, omegaconf.DictConfig))
                    and "_target_" in cfg[key]
                ):
                    # Special handling for Module to resolve forward function
                    if key == "module" and "forward" in cfg[key]:
                        # Resolve interpolations before converting to dict to handle root-level references
                        module_cfg = omegaconf.OmegaConf.to_container(
                            cfg[key], resolve=True
                        )
                        # Import the forward function if it's a string reference
                        if isinstance(module_cfg["forward"], str):
                            parts = module_cfg["forward"].rsplit(".", 1)
                            if len(parts) == 2:
                                import importlib

                                module = importlib.import_module(parts[0])
                                module_cfg["forward"] = getattr(module, parts[1])
                        instantiated[key] = hydra.utils.instantiate(
                            module_cfg, _recursive_=True
                        )
                    else:
                        # Don't use recursive for DataModule as it handles its own instantiation
                        if key == "data":
                            instantiated[key] = hydra.utils.instantiate(
                                cfg[key], _recursive_=False
                            )
                        else:
                            instantiated[key] = hydra.utils.instantiate(
                                cfg[key], _recursive_=True
                            )
                else:
                    instantiated[key] = cfg[key]
            except Exception as e:
                rank_zero_warn(f"Could not instantiate {key}: {e}")
                instantiated[key] = cfg[key]

    # Second pass: instantiate remaining components
    for key, value in cfg.items():
        if key not in instantiated:
            try:
                if (
                    isinstance(value, (dict, omegaconf.DictConfig))
                    and "_target_" in value
                ):
                    instantiated[key] = hydra.utils.instantiate(value, _recursive_=True)
                else:
                    instantiated[key] = value
            except Exception as e:
                rank_zero_warn(f"Could not instantiate {key}: {e}")
                instantiated[key] = value

    return instantiated


def instantiate_from_config(cfg: Union[dict, omegaconf.DictConfig]) -> Any:
    """Main entry point for config-based training.

    This function handles the complete instantiation of a training setup from config:
    - Recursively instantiates all components
    - Creates Manager if trainer/module/data are present
    - Returns appropriate object based on config structure

    Args:
        cfg: Complete configuration dictionary or DictConfig

    Returns:
        Manager instance if config contains trainer/module/data,
        otherwise returns instantiated config dict
    """
    from stable_pretraining.manager import Manager
    import torch

    # Convert to DictConfig if needed
    if isinstance(cfg, dict):
        cfg = omegaconf.OmegaConf.create(cfg)

    # Set matmul precision if specified (must be done before Trainer instantiation)
    if "matmul_precision" in cfg and cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        rank_zero_warn(f"Set float32 matmul precision to: {cfg.matmul_precision}")

    # Instantiate all components
    components = recursive_instantiate(cfg)

    # Check if this is a Manager-based config (has trainer, module, data)
    if all(k in components for k in ["trainer", "module", "data"]):
        # Create Manager for training
        manager = Manager(
            trainer=components["trainer"],
            module=components["module"],
            data=components["data"],
            seed=components.get("seed", None),
            ckpt_path=components.get("ckpt_path", None),
            resume_weights_only=components.get("resume_weights_only", False),
        )
        return manager

    # Otherwise return the instantiated components
    return components
