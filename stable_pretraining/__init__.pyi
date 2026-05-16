# Type stub for stable_pretraining — resolves PEP 562 lazy-loaded names for
# mypy, pyright, and LSP tools (Pylance, Jedi, etc.).
#
# The runtime __init__.py defers all heavy imports via __getattr__ (PEP 562),
# which static analysers cannot follow. This stub gives them an explicit map
# from every public name to its actual type without triggering real imports.

# ---------------------------------------------------------------------------
# Imports (re-exported names)
# ---------------------------------------------------------------------------
from ._config import get_config as get_config
from ._config import set as set

# Core classes
from .manager import Manager as Manager
from .module import Module as Module
from .backbone.utils import TeacherStudentWrapper as TeacherStudentWrapper

# Callbacks
from .callbacks import EarlyStopping as EarlyStopping
from .callbacks import ImageRetrieval as ImageRetrieval
from .callbacks import LiDAR as LiDAR
from .callbacks import LoggingCallback as LoggingCallback
from .callbacks import ModuleSummary as ModuleSummary
from .callbacks import OnlineKNN as OnlineKNN
from .callbacks import OnlineProbe as OnlineProbe
from .callbacks import OnlineWriter as OnlineWriter
from .callbacks import RankMe as RankMe
from .callbacks import TeacherStudentCallback as TeacherStudentCallback
from .callbacks import TrainerInfo as TrainerInfo
from .callbacks.checkpoint_sklearn import SklearnCheckpoint as SklearnCheckpoint
from .callbacks.registry import log as log
from .callbacks.registry import log_dict as log_dict

# Loggers
from .loggers import TrackioLogger as TrackioLogger
from .loggers import SwanLabLogger as SwanLabLogger

# Registry
from .registry import RegistryLogger as RegistryLogger
from .registry import open_registry as open_registry

# Method classes (most-used; full catalog: stable_pretraining.methods)
from .methods.barlow_twins import BarlowTwins as BarlowTwins
from .methods.byol import BYOL as BYOL
from .methods.dino import DINO as DINO
from .methods.dinov2 import DINOv2 as DINOv2
from .methods.mae import MAE as MAE
from .methods.nnclr import NNCLR as NNCLR
from .methods.simclr import SimCLR as SimCLR
from .methods.swav import SwAV as SwAV
from .methods.vicreg import VICReg as VICReg

# Sub-packages (re-exported as modules)
from . import backbone as backbone
from . import callbacks as callbacks
from . import data as data
from . import loggers as loggers
from . import losses as losses
from . import methods as methods
from . import module as module
from . import optim as optim
from . import registry as registry
from . import utils as utils

# ---------------------------------------------------------------------------
# Availability flags
# ---------------------------------------------------------------------------
SKLEARN_AVAILABLE: bool
WANDB_AVAILABLE: bool
TRACKIO_AVAILABLE: bool
SWANLAB_AVAILABLE: bool

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
__author__: str
__license__: str
__summary__: str
__title__: str
__url__: str
__version__: str
