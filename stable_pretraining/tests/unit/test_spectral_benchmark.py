"""Regression tests for the matched-architecture CIFAR-10 loss comparison."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import lightning as pl
import pytest
import torch
from torch import nn

import stable_pretraining as spt

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def benchmark():
    path = (
        Path(__file__).resolve().parents[3] / "benchmarks/cifar10/spectral-resnet18.py"
    )
    spec = importlib.util.spec_from_file_location("spectral_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_objectives_have_identical_seeded_models_and_probes(benchmark):
    pl.seed_everything(7, workers=True)
    spectral = benchmark.build_model("spectral", mu=3.0)
    spectral_probe = nn.Linear(spectral.embed_dim, 10)

    pl.seed_everything(7, workers=True)
    simclr = benchmark.build_model("simclr", mu=3.0)
    simclr_probe = nn.Linear(simclr.embed_dim, 10)

    assert spectral.spectral_loss.mu == 3.0
    assert isinstance(simclr.spectral_loss, spt.losses.NTXEntLoss)
    assert repr(spectral.backbone) == repr(simclr.backbone)
    assert repr(spectral.projector) == repr(simclr.projector)
    assert {
        name: parameter.requires_grad for name, parameter in spectral.named_parameters()
    } == {
        name: parameter.requires_grad for name, parameter in simclr.named_parameters()
    }
    assert isinstance(spectral.projector[-1], nn.BatchNorm1d)
    assert spectral.projector[-1].bias.requires_grad
    assert simclr.projector[-1].bias.requires_grad
    for first, second in ((spectral, simclr), (spectral_probe, simclr_probe)):
        assert first.state_dict().keys() == second.state_dict().keys()
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("seed", [None, 23])
def test_main_seeds_before_model_initialization(benchmark, monkeypatch, tmp_path, seed):
    monkeypatch.setenv("NUM_WORKERS", "0")
    if seed is None:
        monkeypatch.delenv("SEED", raising=False)
    else:
        monkeypatch.setenv("SEED", str(seed))
    monkeypatch.setattr(
        benchmark.torchvision.datasets, "CIFAR10", lambda **_: range(512)
    )
    monkeypatch.setattr(benchmark.sys, "path", list(benchmark.sys.path))
    monkeypatch.setitem(
        benchmark.sys.modules,
        "utils",
        SimpleNamespace(get_data_dir=lambda _: tmp_path),
    )
    monkeypatch.setattr(benchmark.spt, "set", lambda **_: None)

    class ModelConstructionReached(Exception):
        pass

    observed_seed = None

    def observe_initialization(*args):
        nonlocal observed_seed
        observed_seed = torch.initial_seed()
        raise ModelConstructionReached

    monkeypatch.setattr(benchmark, "build_model", observe_initialization)
    torch.manual_seed(999)
    with pytest.raises(ModelConstructionReached):
        benchmark.main()
    assert observed_seed == (0 if seed is None else seed)


@pytest.mark.parametrize("loss_name", ["spectral", "simclr"])
def test_control_models_accept_both_dict_view_formats(benchmark, loss_name):
    model = benchmark.build_model(loss_name).train()
    labels = torch.tensor([0, 1])
    views = [{"image": torch.randn(2, 3, 32, 32), "label": labels} for _ in range(2)]
    with torch.no_grad():
        for batch in (
            {"view_0": views[0], "view_1": views[1]},
            {"views": views},
        ):
            output = model(batch, stage="train")
            assert output["loss"].ndim == 0
            assert torch.isfinite(output["loss"])
            assert output["embedding"].shape == (4, 512)
            torch.testing.assert_close(output["label"], labels.repeat(2))


def test_unknown_control_objective_is_rejected(benchmark):
    with pytest.raises(ValueError, match="LOSS"):
        benchmark.build_model("unsupported")
