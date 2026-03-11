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
    WandbCallback,
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
    "WandbCallback",
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


def rank_zero_only_filter(record):
    """Filter to only log on rank 0 in distributed training."""
    import os

    # Check common environment variables for distributed rank
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return rank == "0" and record["level"].no >= logger.level("INFO").no


logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> (<cyan>{process}, {name}</cyan>) | <level>{message}</level>",
    filter=rank_zero_only_filter,
    level="INFO",
)


# Redirect standard logging to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger.log(record.levelname, record.getMessage())
        # Get corresponding Loguru level if it exists
        # try:
        #     level = logger.level(record.levelname).name
        # except ValueError:
        #     level = "INFO"

        # Find caller from where originated the log message
        # frame, depth = logging.currentframe(), 2
        # while frame.f_code.co_filename == logging.__file__:
        #     frame = frame.f_back
        #     depth += 1
        # logger.opt(depth=depth, exception=record.exc_info).log(
        #     level, record.getMessage()
        # )


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
