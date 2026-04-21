# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Global configuration for stable_pretraining.

Provides a single entry-point — ``stable_pretraining.set(...)`` — for users
to configure library-wide behaviour instead of scattering options across
environment variables, callback constructors, and factory functions.

Example::

    import stable_pretraining as spt

    spt.set(
        verbose="WARNING",
        progress_bar="rich",
        cleanup={"checkpoints": False, "logs": False},
    )
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Literal, Optional, Union

_VALID_LOG_LEVELS = (
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
)

_CLEANUP_KEYS = (
    "checkpoints",
    "logs",
    "hydra",
    "slurm",
    "env_dump",
    "callback_artifacts",
)

# Default cleanup policy mirrors CleanUpCallback defaults
_CLEANUP_DEFAULTS: Dict[str, bool] = {
    "checkpoints": True,
    "logs": True,
    "hydra": False,
    "slurm": False,
    "env_dump": False,
    "callback_artifacts": True,
}


class _GlobalConfig:
    """Singleton holding library-wide configuration.

    Thread-safe: reads/writes are protected by a lock so that ``set()``
    can be called from any thread (e.g. inside a Hydra launcher).
    """

    _instance: Optional["_GlobalConfig"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "_GlobalConfig":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_defaults()
            return cls._instance

    def _init_defaults(self) -> None:
        self._verbose: str = "INFO"
        self._progress_bar: str = "auto"
        self._cleanup: Dict[str, bool] = dict(_CLEANUP_DEFAULTS)
        self._log_rank: Union[int, str] = 0
        self._default_callbacks: Dict[str, bool] = {}
        self._default_loggers: Dict[str, bool] = {}
        _default_cache = str(
            os.path.join(os.path.expanduser("~"), ".cache", "stable-pretraining")
        )
        self._cache_dir: Optional[str] = os.environ.get("SPT_CACHE_DIR", _default_cache)
        self._requeue_checkpoint: bool = True

    # -- verbose ---------------------------------------------------------------

    @property
    def verbose(self) -> str:
        return self._verbose

    @verbose.setter
    def verbose(self, value: Union[str, int]) -> None:
        if isinstance(value, int):
            # Map Python logging-style ints: 10=DEBUG, 20=INFO, 30=WARNING, ...
            _int_map = {
                0: "TRACE",
                10: "DEBUG",
                20: "INFO",
                30: "WARNING",
                40: "ERROR",
                50: "CRITICAL",
            }
            if value not in _int_map:
                raise ValueError(
                    f"Integer verbose level must be one of {list(_int_map.keys())}, got {value}"
                )
            value = _int_map[value]
        value = str(value).upper()
        if value not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"verbose must be one of {_VALID_LOG_LEVELS}, got {value!r}"
            )
        self._verbose = value

    # -- progress_bar ----------------------------------------------------------

    @property
    def progress_bar(self) -> str:
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, value: str) -> None:
        allowed = ("auto", "rich", "simple", "none")
        value = str(value).lower()
        if value not in allowed:
            raise ValueError(f"progress_bar must be one of {allowed}, got {value!r}")
        self._progress_bar = value

    # -- cleanup ---------------------------------------------------------------

    @property
    def cleanup(self) -> Dict[str, bool]:
        return dict(self._cleanup)

    @cleanup.setter
    def cleanup(self, value: Dict[str, bool]) -> None:
        if not isinstance(value, dict):
            raise TypeError(f"cleanup must be a dict, got {type(value).__name__}")
        for k, v in value.items():
            if k not in _CLEANUP_KEYS:
                raise ValueError(
                    f"Unknown cleanup key {k!r}. Valid keys: {_CLEANUP_KEYS}"
                )
            if not isinstance(v, bool):
                raise TypeError(f"cleanup[{k!r}] must be bool, got {type(v).__name__}")
        # Merge — unspecified keys keep their current value
        self._cleanup.update(value)

    # -- log_rank --------------------------------------------------------------

    @property
    def log_rank(self) -> Union[int, str]:
        return self._log_rank

    @log_rank.setter
    def log_rank(self, value: Union[int, str]) -> None:
        if isinstance(value, str):
            if value != "all":
                raise ValueError(f"log_rank string must be 'all', got {value!r}")
        elif isinstance(value, int):
            if value < 0:
                raise ValueError(f"log_rank must be >= 0, got {value}")
        else:
            raise TypeError(
                f"log_rank must be int or 'all', got {type(value).__name__}"
            )
        self._log_rank = value

    # -- default_callbacks -----------------------------------------------------

    @property
    def default_callbacks(self) -> Dict[str, bool]:
        return dict(self._default_callbacks)

    @default_callbacks.setter
    def default_callbacks(self, value: Dict[str, bool]) -> None:
        _VALID_CALLBACK_KEYS = (
            "progress_bar",
            "registry",
            "logging",
            "env_dump",
            "trainer_info",
            "sklearn_checkpoint",
            "wandb_checkpoint",
            "trackio_checkpoint",
            "swanlab_checkpoint",
            "module_summary",
            "slurm_info",
            "unused_params",
            "hf_checkpoint",
        )
        if not isinstance(value, dict):
            raise TypeError(
                f"default_callbacks must be a dict, got {type(value).__name__}"
            )
        for k, v in value.items():
            if k not in _VALID_CALLBACK_KEYS:
                raise ValueError(
                    f"Unknown default_callbacks key {k!r}. Valid keys: {_VALID_CALLBACK_KEYS}"
                )
            if not isinstance(v, bool):
                raise TypeError(
                    f"default_callbacks[{k!r}] must be bool, got {type(v).__name__}"
                )
        self._default_callbacks.update(value)

    # -- default_loggers -------------------------------------------------------

    @property
    def default_loggers(self) -> Dict[str, bool]:
        return dict(self._default_loggers)

    @default_loggers.setter
    def default_loggers(self, value: Dict[str, bool]) -> None:
        _VALID_LOGGER_KEYS = ("registry",)
        if not isinstance(value, dict):
            raise TypeError(
                f"default_loggers must be a dict, got {type(value).__name__}"
            )
        for k, v in value.items():
            if k not in _VALID_LOGGER_KEYS:
                raise ValueError(
                    f"Unknown default_loggers key {k!r}. Valid keys: {_VALID_LOGGER_KEYS}"
                )
            if not isinstance(v, bool):
                raise TypeError(
                    f"default_loggers[{k!r}] must be bool, got {type(v).__name__}"
                )
        self._default_loggers.update(value)

    # -- cache_dir -------------------------------------------------------------

    @property
    def cache_dir(self) -> Optional[str]:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: Optional[str]) -> None:
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(
                    f"cache_dir must be a str or None, got {type(value).__name__}"
                )
            if not value.strip():
                raise ValueError("cache_dir must not be empty")
        self._cache_dir = value

    # -- requeue_checkpoint ----------------------------------------------------

    @property
    def requeue_checkpoint(self) -> bool:
        return self._requeue_checkpoint

    @requeue_checkpoint.setter
    def requeue_checkpoint(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                f"requeue_checkpoint must be a bool, got {type(value).__name__}"
            )
        self._requeue_checkpoint = value

    # -- reset (for testing) ---------------------------------------------------

    def reset(self) -> None:
        """Reset all settings to defaults."""
        self._init_defaults()

    def __repr__(self) -> str:
        return (
            f"GlobalConfig(\n"
            f"  verbose={self._verbose!r},\n"
            f"  progress_bar={self._progress_bar!r},\n"
            f"  cleanup={self._cleanup!r},\n"
            f"  log_rank={self._log_rank!r},\n"
            f"  default_callbacks={self._default_callbacks!r},\n"
            f"  default_loggers={self._default_loggers!r},\n"
            f"  cache_dir={self._cache_dir!r},\n"
            f"  requeue_checkpoint={self._requeue_checkpoint!r},\n"
            f")"
        )


def get_config() -> _GlobalConfig:
    """Return the global configuration singleton."""
    return _GlobalConfig()


def set(
    *,
    verbose: Optional[Union[str, int]] = None,
    progress_bar: Optional[Literal["auto", "rich", "simple", "none"]] = None,
    cleanup: Optional[Dict[str, bool]] = None,
    log_rank: Optional[Union[int, Literal["all"]]] = None,
    default_callbacks: Optional[Dict[str, bool]] = None,
    default_loggers: Optional[Dict[str, bool]] = None,
    cache_dir: Optional[str] = None,
    requeue_checkpoint: Optional[bool] = None,
) -> None:
    """Configure library-wide settings for stable_pretraining.

    All arguments are keyword-only and optional — only the settings you
    pass are updated; the rest keep their current values.

    Args:
        verbose: Global log level.  Accepts loguru level strings
            (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, …) or Python
            ``logging``-style integers (10, 20, 30, …).  Also controls
            the ``verbose`` flag on all callbacks that support it.
        progress_bar: Progress bar style.
            ``"auto"`` (default) picks ``"rich"`` for TTYs and
            ``"simple"`` for non-interactive environments (SLURM, CI).
            ``"none"`` disables the progress bar entirely.
        cleanup: Dict controlling which artifact categories the
            :class:`~stable_pretraining.callbacks.CleanUpCallback` will
            remove after successful training.  Keys:
            ``"checkpoints"``, ``"logs"``, ``"hydra"``, ``"slurm"``,
            ``"env_dump"``, ``"callback_artifacts"``.  Values are bools
            (``True`` = **keep**, ``False`` = **delete**).  Unspecified
            keys keep their current value.
        log_rank: Which distributed rank(s) may log.  ``0`` (default)
            restricts output to rank 0.  ``"all"`` enables logging on
            every rank.
        default_callbacks: Dict toggling individual default callbacks
            on/off.  Keys: ``"progress_bar"``, ``"registry"``,
            ``"logging"``, ``"env_dump"``, ``"trainer_info"``,
            ``"sklearn_checkpoint"``, ``"wandb_checkpoint"``,
            ``"trackio_checkpoint"``, ``"swanlab_checkpoint"``,
            ``"module_summary"``, ``"slurm_info"``, ``"unused_params"``,
            ``"hf_checkpoint"``.
        default_loggers: Dict toggling individual default loggers
            on/off.  Keys: ``"registry"`` (SQLite run registry +
            per-step CSV logger — both are added together as a
            pair).  Enabled by default.
        cache_dir: Root directory for all training outputs.  Each run
            creates a unique subdirectory under
            ``{cache_dir}/runs/{YYYYMMDD}/{HHMMSS}/{run_id}/``.
            The Manager injects this as the Trainer's
            ``default_root_dir`` and auto-configures ``ckpt_path``,
            ensuring no path collisions across parallel sweep jobs.
            Defaults to ``~/.cache/stable-pretraining``.  Can be
            overridden via the ``SPT_CACHE_DIR`` environment variable.
            Set to ``None`` to disable and preserve the standard
            Lightning / Hydra directory behavior.

            .. note::
                SLURM ``.out`` / ``.err`` files are created by the
                scheduler before Python starts and cannot be redirected
                into the run directory.

        requeue_checkpoint: Whether to automatically add a
            ``ModelCheckpoint`` that saves ``last.ckpt`` every epoch for
            SLURM requeue recovery.  ``True`` (default) ensures seamless
            preemption handling.  Set to ``False`` to save time/disk when
            preemption is not a concern.  Only applies when ``cache_dir``
            is set.

    Example::

        import stable_pretraining as spt

        spt.set(verbose="DEBUG")
        spt.set(cleanup={"checkpoints": False, "slurm": False})
        spt.set(progress_bar="simple", log_rank="all")
        spt.set(cache_dir="/scratch/my_runs")
    """
    cfg = get_config()

    if verbose is not None:
        cfg.verbose = verbose
        _apply_verbose(cfg.verbose)

    if progress_bar is not None:
        cfg.progress_bar = progress_bar

    if cleanup is not None:
        cfg.cleanup = cleanup

    if log_rank is not None:
        cfg.log_rank = log_rank
        _apply_log_rank(cfg.log_rank)

    if default_callbacks is not None:
        cfg.default_callbacks = default_callbacks

    if default_loggers is not None:
        cfg.default_loggers = default_loggers

    if cache_dir is not None:
        cfg.cache_dir = cache_dir

    if requeue_checkpoint is not None:
        cfg.requeue_checkpoint = requeue_checkpoint


def _apply_verbose(level: str) -> None:
    """Apply verbose level change to loguru at runtime."""
    import os
    import sys

    os.environ["LOGURU_LEVEL"] = level

    try:
        from loguru import logger

        logger.remove()
        # Re-import the format function and filter from __init__ to reuse them
        from stable_pretraining import _log_format, _make_log_filter

        logger.add(
            sys.stdout,
            format=_log_format,
            filter=_make_log_filter(),
            level=level,
        )
    except Exception:
        # If loguru isn't set up yet (early import), the env var is enough
        pass


def _apply_log_rank(log_rank: Union[int, str]) -> None:
    """Apply log_rank change to loguru at runtime."""
    import sys

    try:
        from loguru import logger

        logger.remove()
        from stable_pretraining import _log_format, _make_log_filter

        cfg = get_config()
        logger.add(
            sys.stdout,
            format=_log_format,
            filter=_make_log_filter(),
            level=cfg.verbose,
        )
    except Exception:
        pass
