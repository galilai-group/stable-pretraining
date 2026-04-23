"""Unit tests for stable_pretraining.set() / get_config() global configuration."""

import pytest

from stable_pretraining._config import (
    _GlobalConfig,
    get_config,
    set as spt_set,
    _VALID_LOG_LEVELS,
    _CLEANUP_DEFAULTS,
    _CLEANUP_KEYS,
)
from stable_pretraining.callbacks.utils import resolve_verbose

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global config before and after every test."""
    cfg = get_config()
    cfg.reset()
    yield
    cfg.reset()


# ============================================================================
# Singleton behaviour
# ============================================================================


def test_singleton():
    a = _GlobalConfig()
    b = _GlobalConfig()
    assert a is b


def test_get_config_returns_singleton():
    assert get_config() is _GlobalConfig()


# ============================================================================
# Defaults
# ============================================================================


def test_defaults():
    cfg = get_config()
    assert cfg.verbose == "INFO"
    assert cfg.progress_bar == "auto"
    assert cfg.cleanup == _CLEANUP_DEFAULTS
    assert cfg.log_rank == 0
    assert cfg.default_callbacks == {}
    assert cfg.cache_dir is not None  # defaults to ~/.cache/stable-pretraining
    assert cfg.requeue_checkpoint is True


def test_reset_restores_defaults():
    cfg = get_config()
    cfg.verbose = "ERROR"
    cfg.progress_bar = "rich"
    cfg.cleanup = {"checkpoints": False}
    cfg.log_rank = "all"
    cfg.default_callbacks = {"logging": False}
    cfg.cache_dir = "/tmp/test"
    cfg.requeue_checkpoint = False
    cfg.reset()
    assert cfg.verbose == "INFO"
    assert cfg.progress_bar == "auto"
    assert cfg.cleanup == _CLEANUP_DEFAULTS
    assert cfg.log_rank == 0
    assert cfg.default_callbacks == {}
    assert cfg.cache_dir is not None  # defaults to ~/.cache/stable-pretraining
    assert cfg.requeue_checkpoint is True


# ============================================================================
# verbose
# ============================================================================


def test_set_verbose_string():
    spt_set(verbose="DEBUG")
    assert get_config().verbose == "DEBUG"


def test_set_verbose_case_insensitive():
    spt_set(verbose="warning")
    assert get_config().verbose == "WARNING"


def test_set_verbose_int():
    spt_set(verbose=10)
    assert get_config().verbose == "DEBUG"

    spt_set(verbose=20)
    assert get_config().verbose == "INFO"

    spt_set(verbose=30)
    assert get_config().verbose == "WARNING"


def test_set_verbose_invalid_string():
    with pytest.raises(ValueError, match="verbose must be one of"):
        spt_set(verbose="BANANA")


def test_set_verbose_invalid_int():
    with pytest.raises(ValueError, match="Integer verbose level"):
        spt_set(verbose=99)


def test_all_valid_log_levels():
    for level in _VALID_LOG_LEVELS:
        spt_set(verbose=level)
        assert get_config().verbose == level


# ============================================================================
# progress_bar
# ============================================================================


def test_set_progress_bar():
    for style in ("auto", "rich", "simple", "none"):
        spt_set(progress_bar=style)
        assert get_config().progress_bar == style


def test_set_progress_bar_case_insensitive():
    spt_set(progress_bar="Rich")
    assert get_config().progress_bar == "rich"


def test_set_progress_bar_invalid():
    with pytest.raises(ValueError, match="progress_bar must be one of"):
        spt_set(progress_bar="fancy")


# ============================================================================
# cleanup
# ============================================================================


def test_set_cleanup_partial_update():
    """Unspecified keys keep their current value."""
    spt_set(cleanup={"checkpoints": False})
    cfg = get_config()
    assert cfg.cleanup["checkpoints"] is False
    # Other keys unchanged
    assert cfg.cleanup["logs"] is True
    assert cfg.cleanup["hydra"] is False


def test_set_cleanup_multiple_keys():
    spt_set(cleanup={"checkpoints": False, "logs": False, "slurm": True})
    c = get_config().cleanup
    assert c["checkpoints"] is False
    assert c["logs"] is False
    assert c["slurm"] is True
    # Unchanged
    assert c["hydra"] is False


def test_set_cleanup_invalid_key():
    with pytest.raises(ValueError, match="Unknown cleanup key"):
        spt_set(cleanup={"nonexistent": True})


def test_set_cleanup_invalid_value_type():
    with pytest.raises(TypeError, match="must be bool"):
        spt_set(cleanup={"checkpoints": "yes"})


def test_set_cleanup_not_dict():
    with pytest.raises(TypeError, match="cleanup must be a dict"):
        spt_set(cleanup=[True, False])


def test_cleanup_returns_copy():
    """Mutating the returned dict should not affect the config."""
    c = get_config().cleanup
    c["checkpoints"] = False
    assert get_config().cleanup["checkpoints"] is True  # unchanged


def test_all_cleanup_keys_settable():
    for key in _CLEANUP_KEYS:
        spt_set(cleanup={key: True})
        assert get_config().cleanup[key] is True
        spt_set(cleanup={key: False})
        assert get_config().cleanup[key] is False


# ============================================================================
# log_rank
# ============================================================================


def test_set_log_rank_int():
    spt_set(log_rank=0)
    assert get_config().log_rank == 0
    spt_set(log_rank=3)
    assert get_config().log_rank == 3


def test_set_log_rank_all():
    spt_set(log_rank="all")
    assert get_config().log_rank == "all"


def test_set_log_rank_invalid_string():
    with pytest.raises(ValueError, match="log_rank string must be 'all'"):
        spt_set(log_rank="none")


def test_set_log_rank_negative():
    with pytest.raises(ValueError, match="log_rank must be >= 0"):
        spt_set(log_rank=-1)


def test_set_log_rank_invalid_type():
    with pytest.raises(TypeError, match="log_rank must be int or 'all'"):
        spt_set(log_rank=3.5)


# ============================================================================
# default_callbacks
# ============================================================================


def test_set_default_callbacks():
    spt_set(default_callbacks={"logging": False, "env_dump": False})
    dc = get_config().default_callbacks
    assert dc["logging"] is False
    assert dc["env_dump"] is False


def test_set_default_callbacks_invalid_key():
    with pytest.raises(ValueError, match="Unknown default_callbacks key"):
        spt_set(default_callbacks={"nonexistent": True})


def test_set_default_callbacks_invalid_value_type():
    with pytest.raises(TypeError, match="must be bool"):
        spt_set(default_callbacks={"logging": "yes"})


def test_set_default_callbacks_not_dict():
    with pytest.raises(TypeError, match="default_callbacks must be a dict"):
        spt_set(default_callbacks=["logging"])


def test_default_callbacks_returns_copy():
    spt_set(default_callbacks={"logging": False})
    dc = get_config().default_callbacks
    dc["logging"] = True
    assert get_config().default_callbacks["logging"] is False


# ============================================================================
# set() — multiple kwargs at once
# ============================================================================


def test_set_multiple_kwargs():
    spt_set(
        verbose="WARNING",
        progress_bar="simple",
        cleanup={"checkpoints": False},
        log_rank="all",
        default_callbacks={"env_dump": False},
    )
    cfg = get_config()
    assert cfg.verbose == "WARNING"
    assert cfg.progress_bar == "simple"
    assert cfg.cleanup["checkpoints"] is False
    assert cfg.log_rank == "all"
    assert cfg.default_callbacks["env_dump"] is False


def test_set_no_args_is_noop():
    cfg = get_config()
    before = repr(cfg)
    spt_set()
    assert repr(cfg) == before


# ============================================================================
# repr
# ============================================================================


def test_repr():
    r = repr(get_config())
    assert "GlobalConfig" in r
    assert "verbose=" in r
    assert "progress_bar=" in r
    assert "cleanup=" in r


# ============================================================================
# resolve_verbose
# ============================================================================


def test_resolve_verbose_explicit_true():
    assert resolve_verbose(True) is True


def test_resolve_verbose_explicit_false():
    assert resolve_verbose(False) is False


def test_resolve_verbose_none_info():
    """Default config (INFO) should resolve to True."""
    assert resolve_verbose(None) is True


def test_resolve_verbose_none_warning():
    spt_set(verbose="WARNING")
    assert resolve_verbose(None) is False


def test_resolve_verbose_none_debug():
    spt_set(verbose="DEBUG")
    assert resolve_verbose(None) is True


def test_resolve_verbose_none_trace():
    spt_set(verbose="TRACE")
    assert resolve_verbose(None) is True


def test_resolve_verbose_none_error():
    spt_set(verbose="ERROR")
    assert resolve_verbose(None) is False


# ============================================================================
# CleanUpCallback integration
# ============================================================================


def test_cleanup_callback_inherits_from_config():
    """CleanUpCallback with no args should use global config defaults."""
    spt_set(cleanup={"checkpoints": False, "slurm": True})

    from stable_pretraining.callbacks.cleanup import CleanUpCallback

    cb = CleanUpCallback()
    assert cb.keep_checkpoints is False
    assert cb.keep_slurm is True
    # Unchanged from global defaults
    assert cb.keep_logs is True
    assert cb.keep_hydra is False


def test_cleanup_callback_explicit_overrides_config():
    """Explicit constructor args take priority over global config."""
    spt_set(cleanup={"checkpoints": False})

    from stable_pretraining.callbacks.cleanup import CleanUpCallback

    cb = CleanUpCallback(keep_checkpoints=True)
    assert cb.keep_checkpoints is True


# ============================================================================
# factories integration
# ============================================================================


def test_factory_progress_bar_none():
    """progress_bar='none' should not include a progress bar callback."""
    spt_set(progress_bar="none")

    from stable_pretraining.callbacks.factories import default

    cbs = default()
    cb_types = [type(cb).__name__ for cb in cbs]
    assert "RichProgressBar" not in cb_types
    assert "PrintProgressBar" not in cb_types


def test_factory_default_callbacks_disable():
    """Disabling a default callback should exclude it from the list."""
    spt_set(default_callbacks={"module_summary": False, "slurm_info": False})

    from stable_pretraining.callbacks.factories import default

    cbs = default()
    cb_types = [type(cb).__name__ for cb in cbs]
    assert "ModuleSummary" not in cb_types
    assert "SLURMInfo" not in cb_types
    # Others still present
    assert "LoggingCallback" in cb_types


# ============================================================================
# Callback verbose integration
# ============================================================================


def test_callback_verbose_inherits_from_config():
    """Callbacks instantiated with default verbose should respect global config."""
    spt_set(verbose="ERROR")

    from stable_pretraining.callbacks.teacher_student import TeacherStudentCallback

    ts = TeacherStudentCallback()
    assert ts.verbose is False


def test_callback_verbose_explicit_overrides_config():
    """Explicit verbose=True should override global config."""
    spt_set(verbose="ERROR")

    from stable_pretraining.callbacks.teacher_student import TeacherStudentCallback

    ts = TeacherStudentCallback(verbose=True)
    assert ts.verbose is True
