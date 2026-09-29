"""Public discovery and optional acceleration settings remain reversible."""

import builtins
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import stable_pretraining as spt
from stable_pretraining import _fast

pytestmark = pytest.mark.unit


def test_public_discovery_and_rank_filter_follow_current_configuration(monkeypatch):
    exported = dir(spt)
    assert (
        "Module" in exported and "SimCLR" in exported and exported == sorted(exported)
    )
    monkeypatch.setattr(spt.get_config(), "_log_rank", "all")
    monkeypatch.setenv("RANK", "5")
    assert spt._make_log_filter()({})
    monkeypatch.setattr(spt.get_config(), "_log_rank", 0)
    assert not spt._make_log_filter()({})


@pytest.mark.parametrize("available", [False, True])
def test_optional_sklearn_callback_is_resolved_and_cached(monkeypatch, available):
    monkeypatch.setattr(spt, "SKLEARN_AVAILABLE", available)
    monkeypatch.setattr(spt, "SklearnCheckpoint", None)
    result = spt.__getattr__("SklearnCheckpoint")
    assert spt.SklearnCheckpoint is result
    if available:
        assert result.__name__ == "SklearnCheckpoint"
    else:
        assert result is None


def test_fast_mode_applies_supported_cuda_knobs_and_reports_them(monkeypatch):
    previous = _fast.enabled()
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    flash, memory = Mock(), Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "enable_flash_sdp", flash)
    monkeypatch.setattr(torch.backends.cuda, "enable_mem_efficient_sdp", memory)
    try:
        result = spt.make_it_fast(
            matmul_precision=None,
            cudnn_benchmark=False,
            inductor_cache=False,
            verbose=True,
        )
        assert result["tf32"] is True and result["enable_flash_sdp"] is True
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        flash.assert_called_once_with(True)
        memory.assert_called_once_with(True)
    finally:
        _fast.set_enabled(previous)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def test_optional_backbone_adapters_report_missing_compatibility_modules(monkeypatch):
    import stable_pretraining.backbone.utils as utils

    original = builtins.__import__

    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "timm.layers.classifier" or (
            name == "transformers" and "TimmWrapperModel" in fromlist
        ):
            raise ImportError("optional adapter unavailable")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", importing)
    namespace = runpy.run_path(
        str(Path(utils.__file__)),
        run_name="stable_pretraining.backbone._isolated_utils",
    )
    assert not namespace["_TIMM_AVAILABLE"] and not namespace["_TRANSFORMERS_AVAILABLE"]
    with pytest.raises(ImportError, match="transformers"):
        namespace["vit_hf"]("tiny")
