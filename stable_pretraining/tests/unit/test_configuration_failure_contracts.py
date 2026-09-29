"""Invalid configuration must fail before partially changing process defaults."""

from unittest.mock import Mock

import pytest

from stable_pretraining import _config

pytestmark = pytest.mark.unit


@pytest.fixture
def isolated_config(monkeypatch):
    config = object.__new__(_config._GlobalConfig)
    config._init_defaults()
    monkeypatch.setattr(_config._GlobalConfig, "_instance", config)
    return config


@pytest.mark.parametrize(
    "value,error",
    [(False, TypeError), ({"wandb": True}, ValueError), ({"registry": 1}, TypeError)],
)
def test_invalid_default_logger_settings_do_not_change_configuration(
    isolated_config, value, error
):
    with pytest.raises(error):
        _config.set(default_loggers=value)
    assert isolated_config.default_loggers == {}


def test_default_logger_and_requeue_configuration_roundtrip(isolated_config):
    _config.set(default_loggers={"registry": False}, requeue_checkpoint_every_n_steps=5)
    assert isolated_config.default_loggers == {"registry": False}
    assert isolated_config.requeue_checkpoint_every_n_steps == 5
    copied = isolated_config.default_loggers
    copied["registry"] = True
    assert isolated_config.default_loggers["registry"] is False


@pytest.mark.parametrize(
    "value,error", [(True, TypeError), (1.5, TypeError), (-1, ValueError)]
)
def test_invalid_requeue_interval_preserves_existing_setting(
    isolated_config, value, error
):
    isolated_config.requeue_checkpoint_every_n_steps = 10
    with pytest.raises(error):
        _config.set(requeue_checkpoint_every_n_steps=value)
    assert isolated_config.requeue_checkpoint_every_n_steps == 10


def test_logging_configuration_survives_early_logger_initialization_failure(
    isolated_config, monkeypatch
):
    from loguru import logger

    monkeypatch.setenv("LOGURU_LEVEL", "INFO")
    monkeypatch.setattr(
        logger, "remove", Mock(side_effect=RuntimeError("initializing"))
    )
    _config.set(verbose="WARNING", log_rank="all")
    assert isolated_config.verbose == "WARNING" and isolated_config.log_rank == "all"
