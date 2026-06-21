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

# Reduce the CUDA-allocator fragmentation that bites memory-hungry workloads
# (e.g. ViT-L two-view at large batch). ``setdefault`` so users can override
# by exporting the env var themselves. Must be set before PyTorch initialises
# the allocator on first CUDA op.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    # Method classes (most-used; full catalog in stable_pretraining.methods)
    "BarlowTwins": ("stable_pretraining.methods.barlow_twins", "BarlowTwins"),
    "BYOL": ("stable_pretraining.methods.byol", "BYOL"),
    "CrossMAE": ("stable_pretraining.methods.cross_mae", "CrossMAE"),
    "DINO": ("stable_pretraining.methods.dino", "DINO"),
    "DINOv2": ("stable_pretraining.methods.dinov2", "DINOv2"),
    "MAE": ("stable_pretraining.methods.mae", "MAE"),
    "NNCLR": ("stable_pretraining.methods.nnclr", "NNCLR"),
    "SimCLR": ("stable_pretraining.methods.simclr", "SimCLR"),
    "SwAV": ("stable_pretraining.methods.swav", "SwAV"),
    "VICReg": ("stable_pretraining.methods.vicreg", "VICReg"),
}

# Sub-packages exposed as attributes of `stable_pretraining`.
_LAZY_SUBMODULES: set[str] = {
    "backbone",
    "callbacks",
    "data",
    # Optional JAX/Flax-NNX backend. Lazy like the rest, so accessing it imports
    # jax/flax only on first use — ``import stable_pretraining`` never does.
    "jax",
    "loggers",
    "losses",
    "methods",
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

    # Install crash-safe checkpoint saving (writes to ``.<name>.<rand>.tmp``
    # in the target dir, then atomically renames). Replaces Lightning's
    # built-in ``_atomic_save`` which falls back to non-atomic copy across
    # filesystems (target on NFS + temp on /tmp = the common cluster setup).
    try:
        from .utils.atomic_checkpoint import install_atomic_checkpoint_save

        install_atomic_checkpoint_save()
    except Exception:  # pragma: no cover - defensive
        pass

    # Register the ``"fsdp2"`` strategy in Lightning's StrategyRegistry so
    # ``Trainer(strategy="fsdp2")`` works without any explicit import. Requires
    # torch>=2.4 (fully_shard); a no-op/guarded fallback on older stacks.
    try:
        from .utils.fsdp2 import register_fsdp2_strategy

        register_fsdp2_strategy()
    except Exception:  # pragma: no cover - defensive (old torch / no distributed)
        pass

    # Adjust HuggingFace datasets logging if available.
    try:
        import datasets

        datasets.logging.set_verbosity_info()
    except (ModuleNotFoundError, AttributeError):
        # AttributeError can occur with pyarrow version incompatibilities.
        pass

    # Auto-prefer the cuDNN SDPA backend on Hopper (H100/H200). cuDNN's
    # attention kernels are FA3-tier on sm_90+ and often beat PyTorch's
    # default backend pick for the same shapes; ``preferred_sdp_backend``
    # is a hint, so torch still falls back automatically when cuDNN can't
    # handle a given shape/dtype.
    #
    # The ``preferred_sdp_backend`` knob was added in a post-2.11 torch
    # release. On older torch (incl. 2.11) all four SDPA backends are
    # already enabled by default and the runtime picks per-call — there
    # is no safe way to express a preference (forcing cuDNN by disabling
    # the others removes the auto-fallback). The ``hasattr`` guard below
    # makes this a no-op on older torch; on newer torch with the knob
    # available, Hopper users get cuDNN-preferred automatically.
    try:
        import torch

        if torch.cuda.is_available() and hasattr(
            torch.backends.cuda, "preferred_sdp_backend"
        ):
            major, _minor = torch.cuda.get_device_capability()
            if major >= 9:
                torch.backends.cuda.preferred_sdp_backend = "cudnn"
    except Exception:  # pragma: no cover - defensive
        pass


def make_it_fast(
    *,
    tf32: bool = True,
    matmul_precision: str = "high",
    cudnn_benchmark: bool = True,
    flash_sdp: bool = True,
    inductor_cache: bool = True,
    verbose: bool = True,
) -> dict:
    """Enable throughput-oriented global PyTorch settings in one call.

    This is **opt-in**: ``import stable_pretraining`` never changes any torch
    default. Call it once at the top of a training script when you want maximum
    speed and don't need bit-exact reproducibility::

        import stable_pretraining as spt

        spt.make_it_fast()

    Each setting is guarded so it's a safe no-op where it doesn't apply (CPU /
    pre-Ampere GPU / older torch):

    - ``torch.set_float32_matmul_precision(matmul_precision)`` — TF32 path for
      fp32 matmuls (``"high"`` = TF32, ``"medium"`` = bf16, ``"highest"`` =
      full fp32).
    - ``cuda.matmul.allow_tf32`` / ``cudnn.allow_tf32`` — TF32 for matmul and
      cuDNN convolutions (CUDA only).
    - ``cudnn.benchmark`` — autotune conv algorithms per input shape. A clear
      win for fixed input shapes (including fixed-size SSL multi-crop).
      Automatically skipped when deterministic algorithms are enabled, because
      the two conflict.
    - flash + memory-efficient SDPA attention backends (CUDA only).
    - ``TORCHINDUCTOR_FX_GRAPH_CACHE=1`` — persist ``torch.compile`` artifacts
      across runs (set via ``setdefault``, so an explicit env var wins).

    Besides the global torch flags above, calling this also flips a
    process-global "fast mode" tag (:mod:`stable_pretraining._fast`) that
    :class:`~stable_pretraining.Manager` and
    :func:`~stable_pretraining.optim.create_optimizer` read later to fill in —
    **only where you didn't configure them yourself** —:

    - ``bf16-mixed`` Trainer precision (when CUDA + bf16 are available),
    - tuned multi-GPU DDP comm (``gradient_as_bucket_view=True``,
      ``find_unused_parameters=False``),
    - the fused CUDA optimizer kernel (AdamW/SGD/Adam/…).

    Because precision and the DDP strategy are fixed when the Trainer is
    constructed, this must be called **before** you run the ``Manager``. If you
    pass an already-built ``pl.Trainer``, the Manager can't retrofit those and
    will warn.

    Trade-offs to be aware of: TF32 and ``cudnn.benchmark`` both break
    bit-exact reproducibility, and ``cudnn.benchmark`` can *hurt* when input
    shapes vary every step (it re-tunes per new shape). Leave those toggles off
    — or simply don't call this — when you need determinism.

    Args:
        tf32: Enable TF32 for matmul and cuDNN conv (CUDA only).
        matmul_precision: Passed to ``torch.set_float32_matmul_precision``.
            ``None`` skips it.
        cudnn_benchmark: Enable the cuDNN autotuner (skipped under deterministic
            mode).
        flash_sdp: Enable flash + mem-efficient SDPA backends (CUDA only).
        inductor_cache: ``setdefault`` the persistent inductor FX-graph cache.
        verbose: Log a one-line summary of what was applied.

    Returns:
        A dict mapping each setting name to the value actually applied (or to a
        short reason string when it was skipped). Handy for logging/asserting.
    """
    import torch

    from . import _fast

    applied: dict = {}

    # Flip the process-global tag that Manager / create_optimizer read later.
    _fast.set_enabled(True)
    applied["fast_mode"] = True

    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
        applied["matmul_precision"] = matmul_precision

    cuda = torch.cuda.is_available()

    if tf32:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            applied["tf32"] = True
        else:
            applied["tf32"] = "skipped: no CUDA"

    if cudnn_benchmark:
        if torch.are_deterministic_algorithms_enabled():
            applied["cudnn_benchmark"] = "skipped: deterministic mode enabled"
        else:
            torch.backends.cudnn.benchmark = True
            applied["cudnn_benchmark"] = True

    if flash_sdp and cuda:
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp"):
            fn = getattr(torch.backends.cuda, name, None)
            if fn is not None:
                try:
                    fn(True)
                    applied[name] = True
                except Exception:  # pragma: no cover - defensive
                    pass

    if inductor_cache:
        os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        applied["TORCHINDUCTOR_FX_GRAPH_CACHE"] = os.environ[
            "TORCHINDUCTOR_FX_GRAPH_CACHE"
        ]

    if verbose:
        logger.info(f"make_it_fast applied: {applied}")
    return applied


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
    # Performance opt-in
    "make_it_fast",
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
    "jax",
    "methods",
    "module",
    "static",
    "optim",
    "losses",
    "callbacks",
    "backbone",
    # Core classes
    "Manager",
    "Module",
    "TeacherStudentWrapper",
    # Method classes (most-used; full catalog: stable_pretraining.methods)
    "BarlowTwins",
    "BYOL",
    "CrossMAE",
    "DINO",
    "DINOv2",
    "MAE",
    "NNCLR",
    "SimCLR",
    "SwAV",
    "VICReg",
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
