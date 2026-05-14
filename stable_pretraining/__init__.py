# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Top-level package for stable-pretraining.

The package's public attributes (``Manager``, ``Module``, callbacks,
loggers, sub-packages such as ``data``/``utils``/etc.) are loaded **lazily**
via PEP 562 ``__getattr__``. This keeps ``import stable_pretraining`` cheap
— useful for fast-start CLI commands like ``spt web`` and ``spt registry`` —
while ``stable_pretraining.Manager``, ``import stable_pretraining as spt;
spt.Module``, and similar usage patterns continue to work unchanged.

The first time a heavy attribute is accessed (anything in
``_LAZY_ATTRS`` / ``_LAZY_SUBMODULES``) we run a small one-time
initialisation that applies the Lightning manual-optimisation patch and
adjusts ``datasets`` logging verbosity — both of which used to live at
import time.

Light-weight things (logger config, ``get_config``, version info, optional
dependency probes, OmegaConf resolver registration) stay eager because
they're used everywhere and their cost is negligible.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

os.environ["LOGURU_LEVEL"] = os.environ.get("LOGURU_LEVEL", "INFO")

from loguru import logger
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Optional-dependency probes (cheap; users branch on these flags)
# ---------------------------------------------------------------------------

# ``find_spec`` checks installation without actually importing the package,
# which is ~40 ms total for all four (vs 3 s for eager imports of sklearn /
# wandb / trackio / swanlab).
from importlib.util import find_spec as _find_spec  # noqa: E402

SKLEARN_AVAILABLE = _find_spec("sklearn") is not None
WANDB_AVAILABLE = _find_spec("wandb") is not None
TRACKIO_AVAILABLE = _find_spec("trackio") is not None
SWANLAB_AVAILABLE = _find_spec("swanlab") is not None


# ---------------------------------------------------------------------------
# Eager light-weight imports
# ---------------------------------------------------------------------------

# Global config and version metadata are tiny and used nearly everywhere.
from ._config import get_config, set  # noqa: F401, E402
from .__about__ import (  # noqa: F401, E402
    __author__,
    __license__,
    __summary__,
    __title__,
    __url__,
    __version__,
)

# OmegaConf resolver: register at import time so YAML configs can use ${eval:…}
# without requiring a heavy attribute access first.
OmegaConf.register_new_resolver("eval", eval)


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

# Use richuru for nicer console output if it's available.
try:
    import richuru

    richuru.install()
except ImportError:
    pass


_FILE_COL_WIDTH = 12
_LEVEL_MAP = {"WARNING": "WARN", "SUCCESS": "OK"}


def _log_format(record):
    """Loguru format function — shared with ``_config._apply_verbose``."""
    name = record["file"].name
    if len(name) > _FILE_COL_WIDTH:
        name = name[: _FILE_COL_WIDTH - 1] + "~"
    name = name.ljust(_FILE_COL_WIDTH)
    level = _LEVEL_MAP.get(record["level"].name, record["level"].name)
    level = level.ljust(5)
    return (
        f"<green>{{time:HH:mm:ss}}</green> | <level>{level}</level> | "
        f"<cyan>{name}</cyan>| <level>{{message}}</level>\n{{exception}}"
    )


def _make_log_filter():
    """Build a loguru filter that respects ``get_config().log_rank``."""
    cfg = get_config()

    def _filter(record):
        log_rank = cfg.log_rank
        if log_rank == "all":
            return True
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        return str(rank) == str(log_rank)

    return _filter


logger.remove()
logger.add(
    sys.stdout,
    format=_log_format,
    filter=_make_log_filter(),
    level=os.environ.get("LOGURU_LEVEL", "INFO"),
)


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records into loguru."""

    def emit(self, record):
        logger.log(record.levelname, record.getMessage())


logging.root.handlers = []
logging.basicConfig(handlers=[_InterceptHandler()], level="INFO")


# ---------------------------------------------------------------------------
# Lazy heavy attributes (PEP 562)
# ---------------------------------------------------------------------------

# Mapping of attribute -> (module path, attr name within that module).
# Accessing any of these triggers a one-time submodule import + the deferred
# init below.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Core
    "Manager": ("stable_pretraining.manager", "Manager"),
    "Module": ("stable_pretraining.module", "Module"),
    "TeacherStudentWrapper": (
        "stable_pretraining.backbone.utils",
        "TeacherStudentWrapper",
    ),
    # Callbacks (re-exported from .callbacks)
    "EarlyStopping": ("stable_pretraining.callbacks", "EarlyStopping"),
    "ImageRetrieval": ("stable_pretraining.callbacks", "ImageRetrieval"),
    "LiDAR": ("stable_pretraining.callbacks", "LiDAR"),
    "LoggingCallback": ("stable_pretraining.callbacks", "LoggingCallback"),
    "ModuleSummary": ("stable_pretraining.callbacks", "ModuleSummary"),
    "OnlineKNN": ("stable_pretraining.callbacks", "OnlineKNN"),
    "OnlineProbe": ("stable_pretraining.callbacks", "OnlineProbe"),
    "OnlineWriter": ("stable_pretraining.callbacks", "OnlineWriter"),
    "RankMe": ("stable_pretraining.callbacks", "RankMe"),
    "TeacherStudentCallback": (
        "stable_pretraining.callbacks",
        "TeacherStudentCallback",
    ),
    "TrainerInfo": ("stable_pretraining.callbacks", "TrainerInfo"),
    # Callback registry helpers
    "log": ("stable_pretraining.callbacks.registry", "log"),
    "log_dict": ("stable_pretraining.callbacks.registry", "log_dict"),
    # Loggers
    "TrackioLogger": ("stable_pretraining.loggers", "TrackioLogger"),
    "SwanLabLogger": ("stable_pretraining.loggers", "SwanLabLogger"),
    # Registry
    "RegistryLogger": ("stable_pretraining.registry", "RegistryLogger"),
    "open_registry": ("stable_pretraining.registry", "open_registry"),
}

# Sub-packages exposed as attributes of `stable_pretraining`.
_LAZY_SUBMODULES: set[str] = {
    "backbone",
    "callbacks",
    "data",
    "loggers",
    "losses",
    "module",
    "optim",
    "registry",
    "static",
    "utils",
}


_DEFERRED_INIT_DONE = False


def _do_deferred_init() -> None:
    """Run the one-time deferred setup.

    Used to live at import time but pulls in Lightning or HuggingFace
    ``datasets`` (both expensive). Runs the first time a heavy attribute is
    accessed via ``__getattr__``.
    """
    global _DEFERRED_INIT_DONE
    if _DEFERRED_INIT_DONE:
        return
    _DEFERRED_INIT_DONE = True

    # Apply Lightning's manual-optimisation patch (needs Lightning loaded).
    try:
        from .utils.lightning_patch import apply_manual_optimization_patch

        apply_manual_optimization_patch()
    except Exception:  # pragma: no cover - defensive
        pass

    # Install crash-safe checkpoint saving (writes to ``.<name>.<rand>.tmp``
    # in the target dir, then atomically renames). Replaces Lightning's
    # built-in ``_atomic_save`` which falls back to non-atomic copy across
    # filesystems (target on NFS + temp on /tmp = the common cluster setup).
    try:
        from .utils.atomic_checkpoint import install_atomic_checkpoint_save

        install_atomic_checkpoint_save()
    except Exception:  # pragma: no cover - defensive
        pass

    # Adjust HuggingFace datasets logging if available.
    try:
        import datasets

        datasets.logging.set_verbosity_info()
    except (ModuleNotFoundError, AttributeError):
        # AttributeError can occur with pyarrow version incompatibilities.
        pass


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        _do_deferred_init()
        return mod
    if name in _LAZY_ATTRS:
        modpath, attrname = _LAZY_ATTRS[name]
        mod = importlib.import_module(modpath)
        attr = getattr(mod, attrname)
        globals()[name] = attr
        _do_deferred_init()
        return attr
    if name == "SklearnCheckpoint":
        # Conditional callback — only available when sklearn is installed.
        if not SKLEARN_AVAILABLE:
            globals()["SklearnCheckpoint"] = None
            return None
        from .callbacks import SklearnCheckpoint as _SC

        globals()["SklearnCheckpoint"] = _SC
        _do_deferred_init()
        return _SC
    raise AttributeError(f"module 'stable_pretraining' has no attribute {name!r}")


def __dir__() -> list[str]:
    # Use builtins.set explicitly: ``set`` at module scope is the public
    # ``spt.set(...)`` config helper imported from ``._config`` (it shadows
    # the builtin), so calling ``set(__all__)`` here would invoke that
    # helper and TypeError. Reach for the builtin via ``builtins`` to keep
    # both the runtime helper and this dir() function working.
    import builtins

    return sorted(builtins.set(__all__) | builtins.set(globals().keys()))


__all__ = [
    # Availability flags
    "SKLEARN_AVAILABLE",
    "WANDB_AVAILABLE",
    "TRACKIO_AVAILABLE",
    "SWANLAB_AVAILABLE",
    # Global config
    "set",
    "get_config",
    # Callbacks
    "OnlineProbe",
    "SklearnCheckpoint",
    "OnlineKNN",
    "TrainerInfo",
    "LoggingCallback",
    "ModuleSummary",
    "EarlyStopping",
    "OnlineWriter",
    "RankMe",
    "LiDAR",
    "ImageRetrieval",
    "TeacherStudentCallback",
    # Sub-packages
    "utils",
    "data",
    "module",
    "static",
    "optim",
    "losses",
    "callbacks",
    "backbone",
    # Classes
    "Manager",
    "Module",
    "TeacherStudentWrapper",
    "log",
    "log_dict",
    # Loggers
    "loggers",
    "TrackioLogger",
    "SwanLabLogger",
    # Registry
    "registry",
    "RegistryLogger",
    "open_registry",
    # Package info
    "__author__",
    "__license__",
    "__summary__",
    "__title__",
    "__url__",
    "__version__",
]
