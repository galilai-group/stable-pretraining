# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["LOGURU_LEVEL"] = os.environ.get("LOGURU_LEVEL", "INFO")

import logging
import sys

from loguru import logger
from omegaconf import OmegaConf

# Handle optional dependencies early
try:
    import sklearn.base  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import wandb  # noqa: F401

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import global config first (no heavy deps)
from ._config import get_config, set  # noqa: F401

from . import backbone, callbacks, data, losses, module, optim, static, utils
from .__about__ import (
    __author__,
    __license__,
    __summary__,
    __title__,
    __url__,
    __version__,
)
from .backbone.utils import TeacherStudentWrapper
from .callbacks import (
    EarlyStopping,
    ImageRetrieval,
    LiDAR,
    LoggingCallback,
    ModuleSummary,
    OnlineKNN,
    OnlineProbe,
    OnlineWriter,
    RankMe,
    TeacherStudentCallback,
    TrainerInfo,
)
from .callbacks.registry import log, log_dict
from .manager import Manager
from .module import Module
from .utils.lightning_patch import apply_manual_optimization_patch

# Conditionally import callbacks that depend on optional packages
if SKLEARN_AVAILABLE:
    from .callbacks import SklearnCheckpoint
else:
    SklearnCheckpoint = None

__all__ = [
    # Availability flags
    "SKLEARN_AVAILABLE",
    "WANDB_AVAILABLE",
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
    # Modules
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
    # Package info
    "__author__",
    "__license__",
    "__summary__",
    "__title__",
    "__url__",
    "__version__",
]

# Register OmegaConf resolvers
OmegaConf.register_new_resolver("eval", eval)

# Setup logging

# Try to install richuru for better formatting if available
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


# Redirect standard logging to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger.log(record.levelname, record.getMessage())


# Remove all handlers associated with the root logger object
logging.root.handlers = []
logging.basicConfig(handlers=[InterceptHandler()], level="INFO")

# Try to set datasets logging verbosity if available
try:
    import datasets

    datasets.logging.set_verbosity_info()
except (ModuleNotFoundError, AttributeError):
    # AttributeError can occur with pyarrow version incompatibilities
    pass

# Apply Lightning patch for manual optimization parameter support
apply_manual_optimization_patch()
