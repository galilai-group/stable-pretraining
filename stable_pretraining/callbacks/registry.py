from typing import Optional, Dict, Any
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

_MODULE_REGISTRY: Dict[str, LightningModule] = {}


def get_module(name: str = "default") -> Optional[LightningModule]:
    """Retrieve a registered module."""
    return _MODULE_REGISTRY.get(name)


def log(name: str, value: Any, module_name: str = "default", **kwargs) -> None:
    """Log a metric using the registered module."""
    module = _MODULE_REGISTRY.get(module_name)
    if module is not None:
        module.log(name, value, **kwargs)


def log_dict(*args, module_name: str = "default", **kwargs) -> None:
    """Log a metric using the registered module."""
    module = _MODULE_REGISTRY.get(module_name)
    if module is not None:
        module.log(*args, **kwargs)


class ModuleRegistryCallback(Callback):
    """Callback that automatically registers the module for global logging access."""

    def __init__(self, name: str = "default"):
        self.name = name

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Register module at the start of any stage (fit, validate, test, predict)."""
        _MODULE_REGISTRY[self.name] = pl_module

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Clean up registry when done."""
        _MODULE_REGISTRY.pop(self.name, None)
