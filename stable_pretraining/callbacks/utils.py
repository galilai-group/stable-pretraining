import collections.abc
import copy
import dataclasses
import types
from functools import partial
from typing import Any, Dict, Iterable, Optional, Union

import torch
import torchmetrics
from lightning.pytorch import Callback, LightningModule
from loguru import logger as logging

from ..optim import create_optimizer, create_scheduler, LARS

_HEADER_WIDTH = 50


def get_data_from_batch_or_outputs(
    key: Union[Iterable[str], str],
    batch: Dict[str, Any],
    outputs: Optional[Dict[str, Any]] = None,
    caller_name: str = "Callback",
) -> Optional[Any]:
    """Get data from either outputs or batch dictionary.

    In PyTorch Lightning, the outputs parameter in callbacks contains the return
    value from training_step/validation_step, while batch contains the original
    input. Since forward methods may modify batch in-place but Lightning creates
    a copy for outputs, we need to check both.

    Args:
        key: The key(s) to look for in the dictionaries
        batch: The original batch dictionary
        outputs: The outputs dictionary from training/validation step
        caller_name: Name of the calling function/class for logging

    Returns:
        The data associated with the key, or None if not found
    """
    output_as_list = True
    if type(key) is str:
        key = [key]
        output_as_list = False
    out = []
    for k in key:
        # First check outputs (which contains the forward pass results)
        if outputs is not None and k in outputs:
            out.append(outputs[k])
        elif k in batch:
            out.append(batch[k])
        else:
            msg = (
                f"{caller_name}: Key '{k}' not found in batch or outputs. "
                f"Available batch keys: {list(batch.keys())}, "
                f"Available output keys: {list(outputs.keys()) if outputs else 'None'}"
            )
            logging.warning(msg)
            raise ValueError(msg)
    if output_as_list:
        return out
    return out[0]


def detach_tensors(obj: Any) -> Any:
    """Recursively traverse an object and return an equivalent structure with all torch tensors detached.

    - Preserves structure, types, and shared references.
    - Handles cycles and arbitrary Python objects (including __dict__ and __slots__).
    - Does not mutate the input; only rebuilds containers if needed.
    - torch.nn.Parameter is replaced with a detached Tensor (not Parameter).
    - Optionally supports attrs classes if 'attr' is installed.

    Args:
        obj: The input object (can be arbitrarily nested).

    Returns:
        A new object with all torch tensors detached, or the original object if no tensors found.
    Performance notes:
        - Uses memoization to avoid redundant work and preserve shared/cyclic structure.
        - Avoids unnecessary copies: unchanged subtrees are returned as-is (same id).
        - Shallow-copies objects with __dict__ or __slots__ (does not call __init__).
    """
    memo: Dict[int, Any] = {}
    # Feature-detect attrs support
    try:
        import attr

        _HAS_ATTRS = True
    except ImportError:
        _HAS_ATTRS = False

    def _detach_impl(o: Any) -> Any:
        oid = id(o)
        if oid in memo:
            return memo[oid]
        # Tensors (including Parameter)
        if isinstance(o, torch.Tensor):
            result = o.detach()
            memo[oid] = result
            return result
        # defaultdict: must preserve default_factory and handle cycles
        if isinstance(o, collections.defaultdict):
            result = type(o)(o.default_factory)
            memo[oid] = result
            changed = False
            for k, v in o.items():
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                result[k] = new_v
            # Always return the new result, even if not changed, to ensure correct default_factory and keys
            return result
        # dict/OrderedDict/other Mapping (excluding defaultdict)
        if isinstance(o, collections.abc.Mapping):
            # For custom mapping subclasses, try to preserve type
            result = type(o)()
            memo[oid] = result
            changed = False
            for k, v in o.items():
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                result[k] = new_v
            # For plain dict, if nothing changed, return original
            if not changed and type(o) is dict:
                memo[oid] = o
                return o
            return result
        # Dataclasses (handle frozen and init=False fields)
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            # Step 1: create a shallow copy via dataclasses.replace (no field overrides)
            try:
                copy_obj = dataclasses.replace(o)
            except Exception:
                # fallback for dataclasses with no fields
                copy_obj = copy.copy(o)
            memo[oid] = copy_obj
            changed = False
            for f in dataclasses.fields(o):
                v = getattr(o, f.name)
                new_v = _detach_impl(v)
                if new_v is not v:
                    object.__setattr__(copy_obj, f.name, new_v)
                    changed = True
            if not changed:
                memo[oid] = o
                return o
            return copy_obj
        # attrs classes (if available)
        if _HAS_ATTRS and attr.has(o) and not isinstance(o, type):
            # Use attr.evolve to create a shallow copy, then set fields
            copy_obj = attr.evolve(o)
            memo[oid] = copy_obj
            changed = False
            for f in attr.fields(type(o)):
                v = getattr(o, f.name)
                new_v = _detach_impl(v)
                if new_v is not v:
                    object.__setattr__(copy_obj, f.name, new_v)
                    changed = True
            if not changed:
                memo[oid] = o
                return o
            return copy_obj
        # Namedtuple (but not plain tuple)
        if isinstance(o, tuple) and hasattr(o, "_fields"):
            values = []
            changed = False
            for v in o:
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                values.append(new_v)
            if not changed:
                memo[oid] = o
                return o
            result = type(o)(*values)
            memo[oid] = result
            return result
        # List
        if isinstance(o, list):
            result = []
            memo[oid] = result
            changed = False
            for v in o:
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                result.append(new_v)
            if not changed:
                memo[oid] = o
                return o
            return result
        # Tuple (not namedtuple)
        if isinstance(o, tuple):
            values = []
            changed = False
            for v in o:
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                values.append(new_v)
            if not changed:
                memo[oid] = o
                return o
            result = tuple(values)
            memo[oid] = result
            return result
        # Set
        if isinstance(o, set):
            result = set()
            memo[oid] = result
            changed = False
            for v in o:
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                result.add(new_v)
            if not changed:
                memo[oid] = o
                return o
            return result
        # Frozenset
        if isinstance(o, frozenset):
            values = []
            changed = False
            for v in o:
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                values.append(new_v)
            if not changed:
                memo[oid] = o
                return o
            result = frozenset(values)
            memo[oid] = result
            return result
        # Generic objects with __dict__ or __slots__
        if hasattr(o, "__dict__") or hasattr(o, "__slots__"):
            result = copy.copy(o)
            memo[oid] = result
            changed = False
            # __dict__ attributes
            if hasattr(result, "__dict__"):
                for k, v in result.__dict__.items():
                    new_v = _detach_impl(v)
                    if new_v is not v:
                        setattr(result, k, new_v)
                        changed = True
            # __slots__ attributes
            if hasattr(result, "__slots__"):
                for slot in result.__slots__:
                    if hasattr(result, slot):
                        v = getattr(result, slot)
                        new_v = _detach_impl(v)
                        if new_v is not v:
                            setattr(result, slot, new_v)
                            changed = True
            if not changed:
                memo[oid] = o
                return o
            return result
        # All other types: return as is
        memo[oid] = o
        return o

    return _detach_impl(obj)


def log_header(name: str, width: int = _HEADER_WIDTH) -> None:
    """Log a unified section header: ``── Name ────────────``."""
    pad = max(width - len(name) - 4, 2)
    logging.info(f"── {name} " + "─" * pad)


def resolve_verbose(verbose: Optional[bool]) -> bool:
    """Resolve a callback's ``verbose`` flag.

    * ``True`` / ``False`` — honour the explicit choice.
    * ``None`` — derive from the global config: verbose if the global
      log level is INFO or lower (i.e. more detailed).
    """
    if verbose is not None:
        return verbose
    from .._config import get_config, _VALID_LOG_LEVELS

    level = get_config().verbose
    # INFO is index 2; anything <= INFO means "chatty enough for verbose"
    return _VALID_LOG_LEVELS.index(level) <= _VALID_LOG_LEVELS.index("INFO")


class TrainableCallback(Callback):
    """Base callback class with optimizer and scheduler management.

    This base class handles the common logic for callbacks that need their own
    optimizer and scheduler, including automatic inheritance from the main module's
    configuration when not explicitly specified.

    Subclasses should:
    1. Call super().__init__() with appropriate parameters
    2. Store their module configuration in self._module_config
    3. Override configure_model() to create their specific module
    4. Access their module via self.module property after setup
    """

    def __init__(
        self,
        module: LightningModule,
        name: str,
        optimizer: Optional[Union[str, dict, partial, torch.optim.Optimizer]] = None,
        scheduler: Optional[
            Union[str, dict, partial, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = None,
        gradient_clip_algorithm: str = "norm",
    ):
        """Initialize base callback with optimizer/scheduler configuration.

        Args:
            module: spt.Module.
            name: Unique identifier for this callback instance.
            optimizer: Optimizer configuration. If None, uses default LARS.
            scheduler: Scheduler configuration. If None, uses default ConstantLR.
            accumulate_grad_batches: Number of batches to accumulate gradients.
            gradient_clip_val: Value to clip the gradient (default None).
            gradient_clip_algorithm: Algorithm to clip the gradient (default `norm`).
        """
        super().__init__()
        self.name = name
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # Store configurations
        self._optimizer_config = optimizer
        self._scheduler_config = scheduler
        self._pl_module = module
        self.wrap_configure_model(module)
        self.wrap_configure_optimizers(module)

    def wrap_configure_model(self, pl_module):
        fn = pl_module.configure_model

        def new_configure_model(self, callback=self, fn=fn):
            # Initialize module
            fn()
            module = callback.configure_model(self)
            # Store module in pl_module.callbacks_modules
            logging.info("  storing module in callbacks_modules")
            self.callbacks_modules[callback.name] = module
            # Metrics are optional — not all trainable callbacks expose
            # them (e.g. generative/reconstruction heads whose only output
            # is a loss scalar).
            metrics = getattr(callback, "metrics", None)
            if metrics is not None:
                logging.info("  setting up metrics")
                assert callback.name not in self.callbacks_metrics
                self.callbacks_metrics[callback.name] = format_metrics_as_dict(metrics)

        # Bind the new method to the instance
        logging.info("  wrapping configure_model")
        pl_module.configure_model = types.MethodType(new_configure_model, pl_module)

    def configure_model(self, pl_module: LightningModule) -> torch.nn.Module:
        """Initialize the module for this callback.

        Subclasses must override this method to create their specific module.

        Args:
            pl_module: The Lightning module being trained.

        Returns:
            The initialized module.
        """
        raise NotImplementedError("Subclasses must implement configure_model")

    def wrap_configure_optimizers(self, pl_module):
        fn = pl_module.configure_optimizers

        def new_configure_optimizers(self, callback=self, fn=fn):
            outputs = fn()
            if outputs is None:
                optimizers = []
                schedulers = []
            else:
                optimizers, schedulers = outputs
            # assert callback.name not in self._optimizer_name_to_index
            assert callback.name not in self._optimizer_frequencies
            # assert callback.name not in self._optimizer_names
            assert callback.name not in self._optimizer_gradient_clip_val
            assert callback.name not in self._optimizer_gradient_clip_algorithm
            assert len(optimizers) not in self._optimizer_index_to_name
            self._optimizer_index_to_name[len(optimizers)] = callback.name
            # self._optimizer_name_to_index[callback.name] = len(self._optimizer_names)
            # self._optimizer_names.append(callback.name)
            self._optimizer_frequencies[callback.name] = (
                callback.accumulate_grad_batches
            )
            self._optimizer_gradient_clip_val[callback.name] = (
                callback.gradient_clip_val
            )
            self._optimizer_gradient_clip_algorithm[callback.name] = (
                callback.gradient_clip_algorithm
            )
            optimizers.append(callback.setup_optimizer(self))
            schedulers.append(callback.setup_scheduler(optimizers[-1], self))
            return optimizers, schedulers

        # Bind the new method to the instance
        logging.info("  wrapping configure_optimizers")
        pl_module.configure_optimizers = types.MethodType(
            new_configure_optimizers, pl_module
        )

    def setup_optimizer(self, pl_module: LightningModule) -> None:
        """Initialize optimizer with default LARS if not specified."""
        if self._optimizer_config is None:
            # Use default LARS optimizer for SSL linear probes
            logging.info("  no optimizer given, using default LARS")
            return LARS(
                self.module.parameters(),
                lr=0.1,
                clip_lr=True,
                eta=0.02,
                exclude_bias_n_norm=True,
                weight_decay=0,
            )
        # Use explicitly provided optimizer config
        logging.info("  using explicitly provided optimizer")
        return create_optimizer(self.module.parameters(), self._optimizer_config)

    def setup_scheduler(self, optimizer, pl_module: LightningModule) -> None:
        """Initialize scheduler with default ConstantLR if not specified."""
        if self._scheduler_config is None:
            # Use default ConstantLR scheduler
            logging.info("  no scheduler given, using default ConstantLR")
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        logging.info("  using explicitly provided scheduler")
        return create_scheduler(optimizer, self._scheduler_config, module=pl_module)

    @property
    def module(self):
        """Access module from pl_module.callbacks_modules.

        This property is only accessible after setup() has been called.
        The module is stored centrally in pl_module.callbacks_modules
        to avoid duplication in checkpoints.
        """
        if self._pl_module is None:
            raise AttributeError(
                f"{self.name}: module not accessible before setup(). "
                "The module is initialized during the setup phase."
            )
        return self._pl_module.callbacks_modules[self.name]

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"{self.__class__.__name__}[name={self.name}]"


class EarlyStopping(torch.nn.Module):
    """Early stopping mechanism with support for metric milestones and patience.

    This module provides flexible early stopping capabilities that can halt training
    based on metric performance. It supports both milestone-based stopping (stop if
    metric doesn't reach target by specific epochs) and patience-based stopping
    (stop if metric doesn't improve for N epochs).

    Args:
        mode: Optimization direction - 'min' for metrics to minimize (e.g., loss),
            'max' for metrics to maximize (e.g., accuracy).
        milestones: Dict mapping epoch numbers to target metric values. Training
            stops if targets are not met at specified epochs.
        metric_name: Name of the metric to monitor if metric is a dict.
        patience: Number of epochs with no improvement before stopping.

    Example:
        >>> early_stop = EarlyStopping(mode="max", milestones={10: 0.8, 20: 0.9})
        >>> # Stops if accuracy < 0.8 at epoch 10 or < 0.9 at epoch 20
    """

    def __init__(
        self,
        mode: str = "min",
        milestones: dict[int, float] = None,
        metric_name: str = None,
        patience: int = 10,
    ):
        super().__init__()
        self.mode = mode
        self.milestones = milestones or {}
        self.metric_name = metric_name
        self.patience = patience
        self.register_buffer("history", torch.zeros(patience))

    def should_stop(self, metric, step):
        if self.metric_name is None:
            assert type(metric) is not dict
        else:
            assert self.metric_name in metric
            metric = metric[self.metric_name]
        if step in self.milestones:
            if self.mode == "min":
                return metric > self.milestones[step]
            elif self.mode == "max":
                return metric < self.milestones[step]
        return False


def format_metrics_as_dict(metrics):
    """Formats various metric input formats into a standardized dictionary structure.

    This utility function handles multiple input formats for metrics and converts
    them into a consistent ModuleDict structure with separate train and validation
    metrics. This standardization simplifies metric handling across callbacks.

    Args:
        metrics: Can be:
            - None: Returns empty train and val dicts
            - Single torchmetrics.Metric: Applied to validation only
            - Dict with 'train' and 'val' keys: Separated accordingly
            - Dict of metrics: All applied to validation
            - List/tuple of metrics: All applied to validation

    Returns:
        ModuleDict with '_train' and '_val' keys, each containing metric ModuleDicts.

    Raises:
        ValueError: If metrics format is invalid or contains non-torchmetric objects.
    """
    # Handle OmegaConf types
    from omegaconf import ListConfig, DictConfig

    if isinstance(metrics, (ListConfig, DictConfig)):
        import omegaconf

        metrics = omegaconf.OmegaConf.to_container(metrics, resolve=True)

    if metrics is None:
        train = {}
        eval = {}
    elif isinstance(metrics, torchmetrics.Metric):
        train = {}
        eval = torch.nn.ModuleDict({metrics.__class__.__name__: metrics})
    elif type(metrics) is dict and set(metrics.keys()) == set(["train", "val"]):
        train = {}
        eval = {}
        if type(metrics["train"]) in [list, tuple]:
            for m in metrics["train"]:
                if not isinstance(m, torchmetrics.Metric):
                    raise ValueError(f"metric {m} is no a torchmetric")
                train[m.__class__.__name__] = m
        else:
            train[metrics["train"].__class__.__name__] = metrics["train"]
        if type(metrics["val"]) in [list, tuple]:
            for m in metrics["val"]:
                if not isinstance(m, torchmetrics.Metric):
                    raise ValueError(f"metric {m} is no a torchmetric")
                eval[m.__class__.__name__] = m
        else:
            eval[metrics["val"].__class__.__name__] = metrics["val"]
    elif type(metrics) is dict:
        train = {}
        for k, v in metrics.items():
            assert type(k) is str
            assert isinstance(v, torchmetrics.Metric)
        eval = metrics
    elif type(metrics) in [list, tuple]:
        train = {}
        eval = {}
        for m in metrics:
            if not isinstance(m, torchmetrics.Metric):
                raise ValueError(f"metric {m} is no a torchmetric")
            eval[m.__class__.__name__] = m
    else:
        raise ValueError(
            "metrics can only be a torchmetric of list/tuple of torchmetrics"
        )
    return torch.nn.ModuleDict(
        {
            "_train": torch.nn.ModuleDict(train),
            "_val": torch.nn.ModuleDict(eval),
        }
    )
