"""Optional imports, rank discovery, and transient downloads fail predictably."""

import builtins
import importlib.metadata
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import torch

from stable_pretraining.utils import distributed, error_handling

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("kind", ["swanlab", "trackio", "log_reader"])
def test_optional_logger_imports_remain_usable_when_sdks_are_missing(monkeypatch, kind):
    original = builtins.__import__
    blocked = {"swanlab", "trackio", "wandb", "omegaconf"}

    def importing(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ModuleNotFoundError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    subdir = "utils" if kind == "log_reader" else "loggers"
    namespace = runpy.run_path(
        str(ROOT / subdir / f"{kind}.py"),
        run_name=f"stable_pretraining.{subdir}._isolated_{kind}",
    )
    if kind == "swanlab":
        assert not namespace["SWANLAB_AVAILABLE"]
        with pytest.raises(ImportError, match="swanlab"):
            namespace["SwanLabLogger"](project="test")
        assert namespace["find_swanlab_logger"](SimpleNamespace(loggers=[])) is None
    elif kind == "trackio":
        assert not namespace["TRACKIO_AVAILABLE"]
        with pytest.raises(ImportError, match="trackio"):
            namespace["TrackioLogger"](project="test")
    else:
        assert namespace["wandbapi"] is namespace["logging_redirect_tqdm"] is None


@pytest.mark.parametrize("metadata_available", [False, True])
def test_version_falls_back_to_package_metadata_or_reports_failure(
    monkeypatch, metadata_available
):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "_version":
            raise ImportError("source checkout lacks generated version")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    version = (
        Mock(return_value="1.2.3")
        if metadata_available
        else Mock(side_effect=importlib.metadata.PackageNotFoundError())
    )
    monkeypatch.setattr(importlib.metadata, "version", version)
    if metadata_available:
        result = runpy.run_path(
            str(ROOT / "__about__.py"), run_name="stable_pretraining._isolated_about"
        )
        assert result["__version__"] == "1.2.3"
    else:
        with pytest.raises(ImportError, match="Could not determine"):
            runpy.run_path(
                str(ROOT / "__about__.py"),
                run_name="stable_pretraining._isolated_about",
            )


@pytest.mark.parametrize("status", [429, 503])
def test_download_retry_respects_retry_after_and_original_errors(monkeypatch, status):
    response = requests.Response()
    response.status_code = status
    response.headers["Retry-After"] = "7"
    error = requests.HTTPError(f"HTTP {status}", response=response)
    fetch = Mock(side_effect=[error, "data"])
    pause = Mock()
    monkeypatch.setattr(error_handling.time, "sleep", pause)
    if status == 429:
        assert (
            error_handling.with_hf_retry_ratelimit(fetch, "path", max_attempts=2)
            == "data"
        )
        pause.assert_called_once_with(7)
        assert fetch.call_count == 2
    else:
        with pytest.raises(requests.HTTPError) as caught:
            error_handling.with_hf_retry_ratelimit(fetch, "path", max_attempts=2)
        assert caught.value is error
        pause.assert_not_called()


def test_rate_limit_text_retries_are_bounded(monkeypatch):
    error = OSError("Too Many Requests")
    fetch = Mock(side_effect=error)
    pause = Mock()
    monkeypatch.setattr(error_handling.time, "sleep", pause)
    with pytest.raises(OSError) as caught:
        error_handling.with_hf_retry_ratelimit(fetch, max_attempts=3, delay=2)
    assert caught.value is error and fetch.call_count == 3
    assert pause.call_count == 2


def test_malformed_rank_falls_through_and_invalid_seed_is_explicit(monkeypatch):
    monkeypatch.setenv("RANK", "invalid")
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert distributed.get_rank() == 2
    monkeypatch.setenv("PL_GLOBAL_SEED", "invalid")
    with pytest.raises(ValueError, match="Invalid seed"):
        distributed.seed_everything(None)
    monkeypatch.setenv("PL_GLOBAL_SEED", "5")
    assert distributed.seed_everything("3") == 3


def test_real_collectives_preserve_values_and_gradients(tmp_path):
    import torch.distributed as dist

    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        x = torch.randn(3, requires_grad=True)
        gathered = distributed.all_gather(x)
        reduced = distributed.all_reduce(x)
        torch.testing.assert_close(gathered[0], x)
        torch.testing.assert_close(reduced, x)
        (gathered[0].sum() + reduced.sum()).backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2))
        result = distributed.FullGatherLayer.apply(x.detach())
        torch.testing.assert_close(result[0], x.detach())
    finally:
        dist.destroy_process_group()
