"""Shared utilities for optimizer and scheduler configuration."""

import inspect
from functools import partial
from typing import Iterable, List, Tuple, Union

import torch
from hydra.utils import instantiate
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

from .. import optim as ssl_optim


def is_bias_or_norm_param(name: str, param: torch.nn.Parameter) -> bool:
    """Check if a parameter is a bias or belongs to a normalization layer.

    Parameters are considered bias/norm if:
    - The parameter name ends with ".bias"
    - The parameter name contains "norm" (e.g., LayerNorm, BatchNorm, GroupNorm)
    - The parameter is 1D (biases are typically 1D tensors)

    Args:
        name: Full parameter name (e.g., "encoder.layer1.bn.weight")
        param: The parameter tensor

    Returns:
        True if the parameter should be excluded from weight decay
    """
    # Check for bias parameters
    if name.endswith(".bias") or name == "bias":
        return True

    # Check for normalization layer parameters
    # Matches: LayerNorm, BatchNorm, GroupNorm, InstanceNorm, RMSNorm, etc.
    name_lower = name.lower()
    if "norm" in name_lower:
        return True

    # 1D parameters are typically biases (though this is a heuristic)
    # Skip this check if the name doesn't suggest it's a bias
    # This catches cases like "bn1.weight" which are norm layer scales
    return False


def split_params_for_weight_decay(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
    weight_decay: float,
) -> List[dict]:
    """Split parameters into groups with and without weight decay.

    Creates two parameter groups:
    - Regular parameters: use the specified weight_decay
    - Bias/norm parameters: use weight_decay=0

    This is a common practice in deep learning that:
    - Prevents biases from being regularized (they have different roles than weights)
    - Prevents normalization parameters from being regularized (they control scale/shift)

    Args:
        named_params: Iterable of (name, parameter) tuples from model.named_parameters()
        weight_decay: Weight decay value for regular parameters

    Returns:
        List of parameter group dicts suitable for optimizer initialization:
        [{"params": regular_params, "weight_decay": weight_decay},
         {"params": bias_norm_params, "weight_decay": 0.0}]

    Example:
        >>> named_params = model.named_parameters()
        >>> param_groups = split_params_for_weight_decay(
        ...     named_params, weight_decay=0.01
        ... )
        >>> optimizer = torch.optim.AdamW(param_groups, lr=1e-3)
    """
    regular_params = []
    bias_norm_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue

        if is_bias_or_norm_param(name, param):
            bias_norm_params.append(param)
        else:
            regular_params.append(param)

    param_groups = []
    if regular_params:
        param_groups.append({"params": regular_params, "weight_decay": weight_decay})
    if bias_norm_params:
        param_groups.append({"params": bias_norm_params, "weight_decay": 0.0})

    logging.debug(
        f"Parameter split: {len(regular_params)} regular params (wd={weight_decay}), "
        f"{len(bias_norm_params)} bias/norm params (wd=0)"
    )

    return param_groups


def create_optimizer(
    params,
    optimizer_config: Union[str, dict, partial, type],
    named_params: Iterable[Tuple[str, torch.nn.Parameter]] = None,
) -> torch.optim.Optimizer:
    """Create an optimizer from flexible configuration.

    This function provides a unified way to create optimizers from various configuration formats,
    used by both Module and OnlineProbe for consistency.

    Args:
        params: Parameters to optimize (e.g., model.parameters()). Used unless
            exclude_bias_norm is True and named_params is provided.
        optimizer_config: Can be:
            - str: optimizer name from torch.optim or stable_pretraining.optim (e.g., "AdamW", "LARS")
            - dict: {"type": "AdamW", "lr": 1e-3, "exclude_bias_norm": True, ...}
            - partial: pre-configured optimizer factory
            - class: optimizer class (e.g., torch.optim.AdamW)
        named_params: Optional iterable of (name, parameter) tuples. Required when
            exclude_bias_norm=True to identify bias and normalization parameters.

    Returns:
        Configured optimizer instance

    Examples:
        >>> # String name (uses default parameters)
        >>> opt = create_optimizer(model.parameters(), "AdamW")

        >>> # Dict with parameters
        >>> opt = create_optimizer(
        ...     model.parameters(), {"type": "SGD", "lr": 0.1, "momentum": 0.9}
        ... )

        >>> # With exclude_bias_norm - excludes bias/norm params from weight decay
        >>> opt = create_optimizer(
        ...     model.parameters(),
        ...     {
        ...         "type": "AdamW",
        ...         "lr": 1e-3,
        ...         "weight_decay": 0.01,
        ...         "exclude_bias_norm": True,
        ...     },
        ...     named_params=model.named_parameters(),
        ... )

        >>> # Using partial
        >>> from functools import partial
        >>> opt = create_optimizer(
        ...     model.parameters(), partial(torch.optim.Adam, lr=1e-3)
        ... )

        >>> # Direct class
        >>> opt = create_optimizer(model.parameters(), torch.optim.RMSprop)
    """
    # Handle Hydra config objects
    if hasattr(optimizer_config, "_target_"):
        return instantiate(optimizer_config, params=params, _convert_="object")

    # partial -> call with params
    if isinstance(optimizer_config, partial):
        return optimizer_config(params)

    # callable (including optimizer factories, but not classes)
    if callable(optimizer_config) and not isinstance(optimizer_config, type):
        return optimizer_config(params)

    # dict -> extract type and kwargs
    if isinstance(optimizer_config, (dict, DictConfig)):
        # Convert DictConfig to dict if needed
        if isinstance(optimizer_config, DictConfig):
            config_copy = OmegaConf.to_container(optimizer_config, resolve=True)
        else:
            config_copy = optimizer_config.copy()
        opt_type = config_copy.pop("type", "AdamW")
        exclude_bias_norm = config_copy.pop("exclude_bias_norm", False)
        kwargs = config_copy
    else:
        opt_type = optimizer_config
        exclude_bias_norm = False
        kwargs = {}

    # resolve class
    if isinstance(opt_type, str):
        if hasattr(torch.optim, opt_type):
            opt_class = getattr(torch.optim, opt_type)
        elif hasattr(ssl_optim, opt_type):
            opt_class = getattr(ssl_optim, opt_type)
        else:
            torch_opts = [n for n in dir(torch.optim) if n[0].isupper()]
            ssl_opts = [n for n in dir(ssl_optim) if n[0].isupper()]
            raise ValueError(
                f"Optimizer '{opt_type}' not found. Available in torch.optim: "
                + ", ".join(torch_opts)
                + ". Available in stable_pretraining.optim: "
                + ", ".join(ssl_opts)
            )
    else:
        opt_class = opt_type

    # Handle exclude_bias_norm: split params into groups with/without weight decay
    if exclude_bias_norm:
        if named_params is None:
            raise ValueError(
                "exclude_bias_norm=True requires named_params to be provided. "
                "Pass named_params=model.named_parameters() to create_optimizer."
            )
        weight_decay = kwargs.pop("weight_decay", 0.0)
        # Convert named_params to list to avoid consuming the iterator
        named_params_list = list(named_params)
        param_groups = split_params_for_weight_decay(named_params_list, weight_decay)

        if not param_groups:
            raise ValueError(
                "No parameters to optimize after splitting for weight decay"
            )

        logging.info(
            f"Creating {opt_class.__name__} with exclude_bias_norm=True: "
            f"{sum(len(g['params']) for g in param_groups)} params in {len(param_groups)} groups"
        )
        params = param_groups

    try:
        return opt_class(params, **kwargs)
    except TypeError as e:
        sig = inspect.signature(opt_class.__init__)
        required = [
            p.name
            for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty and p.name not in ["self", "params"]
        ]
        raise TypeError(
            f"Failed to create {opt_class.__name__}. Required parameters: {required}. "
            f"Provided: {list(kwargs.keys())}. Original error: {e}"
        )
