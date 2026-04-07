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
            ``"module_summary"``, ``"slurm_info"``, ``"unused_params"``,
            ``"hf_checkpoint"``.

    Example::

        import stable_pretraining as spt

        spt.set(verbose="DEBUG")
        spt.set(cleanup={"checkpoints": False, "slurm": False})
        spt.set(progress_bar="simple", log_rank="all")
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
