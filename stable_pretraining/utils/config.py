"""Configuration utilities and model manipulation helpers."""

import functools

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
import torch.distributed as dist
from typing import Any


def is_dist() -> bool:
    """Returns True if torch.distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def load_hparams_from_ckpt(path: str) -> Any:
    """Loads a checkpoint safely in both distributed and non-distributed settings.

    - If torch.distributed is initialized, only src_rank loads from disk,
      then broadcasts to all other ranks.
    - If not, loads directly from disk.

    Args:
        path: Path to checkpoint file.

    Returns:
        The loaded hparams.
    """
    if is_dist():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # If only one process, just load directly
        if world_size == 1:
            return torch.load(
                path, map_location=torch.device("meta"), weights_only=False
            )["hyper_parameters"]
        obj = None
        if rank == 0:
            obj = torch.load(
                path, map_location=torch.device("meta"), weights_only=False
            )["hyper_parameters"]
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=0)
        dist.barrier()
        return obj_list[0]
    else:
        return torch.load(path, map_location=torch.device("meta"))["hyper_parameters"]


def execute_from_config(manager, cfg):
    """Execute a function with support for submitit job submission.

    If submitit configuration is present, submits the job to a cluster.
    Otherwise executes locally.

    Args:
        manager: Function or callable to execute
        cfg: Configuration dictionary

    Returns:
        Result of the executed function
    """
    if "submitit" in cfg:
        assert "hydra" not in cfg
        hydra_conf = HydraConfig.get()
        # force_add ignores nodes in struct mode or Structured Configs nodes
        # and updates anyway, inserting keys as needed.
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            OmegaConf.update(
                cfg, "hydra.sweep.dir", hydra_conf.sweep.dir, force_add=True
            )
        with open_dict(cfg):
            cfg.hydra = {}
            cfg.hydra.job = OmegaConf.create(hydra_conf.job)
            cfg.hydra.sweep = OmegaConf.create(hydra_conf.sweep)
            cfg.hydra.run = OmegaConf.create(hydra_conf.run)

        executor = hydra.utils.instantiate(cfg.submitit.executor, _convert_="object")
        executor.update_parameters(**cfg.submitit.get("update_parameters", {}))
        job = executor.submit(manager)
        return job.result()
    else:
        return manager()


def adapt_resnet_for_lowres(model):
    """Adapt a ResNet model for low resolution images.

    Modifies the first convolution layer to use 3x3 kernels with stride 1
    and removes the max pooling layer.

    Args:
        model: ResNet model to adapt

    Returns:
        Modified model
    """
    model.conv1 = torch.nn.Conv2d(
        3,
        64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False,
    )
    model.maxpool = torch.nn.Identity()
    return model


def rsetattr(obj, attr, val):
    """Recursively set an attribute using dot notation.

    Args:
        obj: Object to set attribute on
        attr: Attribute path (e.g., "module.layer.weight")
        val: Value to set
    """
    pre, _, post = attr.rpartition(".")
    parent = rgetattr(obj, pre) if pre else obj
    if type(parent) is dict:
        parent[post] = val
    else:
        return setattr(parent, post, val)


def _adaptive_getattr(obj, attr):
    """Get attribute that works with both objects and dictionaries."""
    if type(obj) is dict:
        return obj[attr]
    else:
        return getattr(obj, attr)


def rgetattr(obj, attr):
    """Recursively get an attribute using dot notation.

    Args:
        obj: Object to get attribute from
        attr: Attribute path (e.g., "module.layer.weight")

    Returns:
        The requested attribute value
    """
    return functools.reduce(_adaptive_getattr, [obj] + attr.split("."))


def find_module(model: torch.nn.Module, module: torch.nn.Module):
    """Find all instances of a module type in a model.

    Args:
        model: Model to search in
        module: Module class to search for

    Returns:
        Tuple of (names, modules) where names are the module paths
        and modules are the actual module instances
    """
    names = []
    values = []
    for child_name, child in model.named_modules():
        if isinstance(child, module):
            names.append(child_name)
            values.append(child)
    return names, values


def replace_module(model, replacement_mapping):
    """Replace modules in a model based on a mapping function.

    Args:
        model: PyTorch model to modify
        replacement_mapping: Function that takes (name, module) and returns
                           the replacement module

    Returns:
        Modified model
    """
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Torch.nn.Module expected as input.")
    for name, module in model.named_modules():
        if name == "":
            continue
        replacement = replacement_mapping(name, module)
        module_names = name.split(".")
        # we go down the tree up to the parent
        parent = model
        for name in module_names[:-1]:
            parent = getattr(parent, name)
        setattr(parent, module_names[-1], replacement)
    return model
