"""Pytest configuration and shared fixtures."""

import shutil
from pathlib import Path

import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no GPU required)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (slow, may require GPU)"
    )
    config.addinivalue_line("markers", "gpu: Tests that require GPU")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line(
        "markers", "download: Tests that download data from the internet"
    )
    config.addinivalue_line(
        "markers", "v1: Legacy tests that need updating (auto-skipped)"
    )
    config.addinivalue_line(
        "markers",
        "regression: Regression tests (all methods, fake data, CPU-only, checks registry)",
    )
    config.addinivalue_line(
        "markers",
        "ddp: Multi-GPU DDP tests (requires srun with >=2 GPUs)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on markers."""
    skip_v1 = pytest.mark.skip(reason="v1: legacy test needs updating")
    skip_gpu = pytest.mark.skip(reason="no GPU available")
    skip_ddp = pytest.mark.skip(reason="DDP requires >=2 GPUs (use srun --gpus=N)")
    for item in items:
        if "v1" in item.keywords:
            item.add_marker(skip_v1)
        elif "ddp" in item.keywords and torch.cuda.device_count() < 2:
            item.add_marker(skip_ddp)
        elif "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


@pytest.fixture
def device():
    """Fixture to get appropriate device for tests."""
    import os

    if torch.cuda.is_available() and not os.environ.get("FORCE_CPU"):
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def mock_batch(device):
    """Create a mock batch of data."""
    batch_size = 4
    return {
        "image": torch.randn(batch_size, 3, 224, 224, device=device),
        "label": torch.randint(0, 10, (batch_size,), device=device),
        "index": torch.arange(batch_size, device=device),
    }


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Create a temporary directory for test outputs."""
    return tmp_path_factory.mktemp("test_outputs")


@pytest.fixture(autouse=True)
def _isolate_spt_config(tmp_path):
    """Point spt ``cache_dir`` at a fresh tmp folder for every test.

    No more server lifecycle to manage — the registry is purely
    filesystem-backed, so isolation just means pointing the cache at
    a scratch dir.  The ``_scanner``'s in-process TTL is invalidated
    too so successive tests never see each other's scans.
    """
    from stable_pretraining._config import get_config
    from stable_pretraining.registry import _scanner

    cfg = get_config()
    original = cfg._cache_dir
    cfg._cache_dir = str(tmp_path)
    _scanner.invalidate_ttl()

    yield

    _scanner.invalidate_ttl()
    cfg._cache_dir = original
    # Clean up empty outputs/ dir that Hydra creates when cache_dir is None
    outputs = Path("outputs")
    if outputs.is_dir():
        shutil.rmtree(outputs, ignore_errors=True)
