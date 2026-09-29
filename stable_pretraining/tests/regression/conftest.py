"""Runtime isolation for deterministic component-interaction regressions."""

import pytest
import torch


@pytest.fixture
def interaction_runtime(monkeypatch):
    """Keep interaction tests on CPU without environment-reporting side effects."""
    from stable_pretraining._config import get_config
    from stable_pretraining.callbacks.factories import _DEFAULT_CALLBACK_REGISTRY

    config = get_config()
    monkeypatch.setattr(
        config,
        "_default_callbacks",
        {name: name == "registry" for name in _DEFAULT_CALLBACK_REGISTRY},
    )
    monkeypatch.setattr(config, "_requeue_checkpoint", False)
    monkeypatch.setattr(config, "_exclude_bias_norm", False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)
