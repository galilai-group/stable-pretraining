"""Sparsity monitors preserve feature semantics and sample-weighted logging."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import stable_pretraining as spt
from stable_pretraining.callbacks.sparsity import Sparsity
from stable_pretraining.forward import lpjepa

pytestmark = pytest.mark.unit


def _train_batch(callback, trainer, module, batch, outputs, step):
    trainer.global_step = step
    callback.on_train_batch_start(trainer, module, batch, 0)
    trainer.global_step = step + 1
    callback.on_train_batch_end(trainer, module, outputs, batch, 0)


def _assert_log(call, stage, name, l0, l1, count):
    metrics = call.args[0]
    assert set(metrics) == {f"{stage}/l0_{name}", f"{stage}/l1_{name}"}
    assert metrics[f"{stage}/l0_{name}"].item() == pytest.approx(l0, abs=1e-6)
    assert metrics[f"{stage}/l1_{name}"].item() == pytest.approx(l1, abs=1e-6)
    assert call.kwargs == {
        "on_step": stage == "fit",
        "on_epoch": True,
        "sync_dist": False,
        "batch_size": count,
    }
    for value in metrics.values():
        assert value.dtype == torch.float32
        assert not value.requires_grad
        assert value.grad_fn is None


def test_sparsity_is_publicly_available():
    assert spt.callbacks.Sparsity is Sparsity


@pytest.mark.parametrize("interval", [0, -1])
def test_sparsity_rejects_nonpositive_training_interval(interval):
    with pytest.raises(ValueError, match="every_n_steps.*positive"):
        Sparsity("projection", every_n_steps=interval)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sparsity_known_values_prefer_outputs_without_mutating_features(dtype):
    features = torch.tensor(
        [[0, 0, 0, 0], [3, 0, 0, 0], [1, 1, 1, 1], [3, -4, 0, 0]],
        dtype=dtype,
        requires_grad=True,
    )
    before = features.detach().clone()
    batch_features = torch.ones_like(features)
    batch = {"projection": batch_features}
    outputs = {"projection": features, "untouched": object()}
    output_keys = set(outputs)
    callback = Sparsity("sparsity")
    trainer, module = SimpleNamespace(global_step=0), SimpleNamespace(log_dict=Mock())

    _train_batch(callback, trainer, module, batch, outputs, 0)

    module.log_dict.assert_called_once()
    _assert_log(module.log_dict.call_args, "fit", "sparsity", 0.4375, 0.435, 4)
    torch.testing.assert_close(features, before, rtol=0, atol=0)
    assert features.requires_grad and features.grad is None
    assert outputs["projection"] is features
    assert set(outputs) == output_keys
    assert batch["projection"] is batch_features
    torch.testing.assert_close(batch_features, torch.ones_like(batch_features))


@pytest.mark.parametrize("view_format", ["list", "dict"])
@pytest.mark.parametrize("first_view_only", [False, True])
def test_sparsity_first_view_selection_applies_only_to_training(
    view_format, first_view_only
):
    first = {"image": torch.zeros(1, 4)}
    second = {"image": torch.ones(2, 4)}
    views = (
        [first, second]
        if view_format == "list"
        else {"z_first": first, "a_second": second}
    )
    batch = {"views": views, "batch_idx": 0}
    outputs = {"projection": torch.tensor([[3.0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])}
    callback = Sparsity("projector", train_first_view_only=first_view_only)
    trainer, module = SimpleNamespace(global_step=0), SimpleNamespace(log_dict=Mock())

    _train_batch(callback, trainer, module, batch, outputs, 0)

    expected, count = (0.25, 1) if first_view_only else (0.75, 3)
    _assert_log(
        module.log_dict.call_args, "fit", "projector", expected, expected, count
    )
    module.log_dict.reset_mock()
    callback.on_validation_batch_end(trainer, module, outputs, batch, 0)
    _assert_log(module.log_dict.call_args, "validate", "projector", 0.75, 0.75, 3)


def test_sparsity_training_interval_uses_preupdate_step():
    callback = Sparsity("projector", every_n_steps=50)
    trainer, module = SimpleNamespace(global_step=0), SimpleNamespace(log_dict=Mock())
    outputs = {"projection": torch.ones(1, 4)}

    for step in (0, 1, 49, 50, 51, 99, 100):
        before = module.log_dict.call_count
        _train_batch(callback, trainer, module, {}, outputs, step)
        assert module.log_dict.call_count - before == int(step in (0, 50, 100))

    assert module.log_dict.call_count == 3


def test_sparsity_validation_reads_custom_batch_target_and_weights_every_batch():
    callback = Sparsity("encoded", target="hidden", every_n_steps=50)
    trainer, module = SimpleNamespace(global_step=1), SimpleNamespace(log_dict=Mock())

    for batch_idx, count in ((0, 3), (1, 1)):
        batch = {"hidden": torch.ones(count, 4)}
        callback.on_validation_batch_end(trainer, module, {}, batch, batch_idx)
        _assert_log(module.log_dict.call_args, "validate", "encoded", 1.0, 1.0, count)

    assert module.log_dict.call_count == 2


def test_sparsity_distributed_means_use_global_sample_count(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def reduce_totals(totals, op):
        assert op == dist.ReduceOp.SUM
        torch.testing.assert_close(totals, torch.tensor([0.25, 0.25, 1.0]))
        assert not totals.requires_grad
        # The other rank has three dense samples, versus this rank's one sparse row.
        totals.add_(torch.tensor([3.0, 3.0, 3.0]))

    reduce = Mock(side_effect=reduce_totals)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    callback = Sparsity("distributed")
    trainer = SimpleNamespace(global_step=0, global_rank=1)
    module = SimpleNamespace(log_dict=Mock())
    outputs = {"projection": torch.tensor([[3.0, 0, 0, 0]], requires_grad=True)}

    _train_batch(callback, trainer, module, {}, outputs, 0)

    reduce.assert_called_once()
    module.log_dict.assert_called_once()
    _assert_log(module.log_dict.call_args, "fit", "distributed", 0.8125, 0.8125, 4)


def test_sparsity_reports_missing_target_through_shared_lookup():
    callback = Sparsity("encoded", target="hidden")
    with pytest.raises(ValueError, match="Sparsity.*hidden.*not found"):
        callback.on_validation_batch_end(
            None, SimpleNamespace(log_dict=Mock()), {}, {}, 0
        )


def test_sparsity_consumes_shared_lpjepa_outputs_with_batch_metadata(monkeypatch):
    model = spt.Module(
        forward=lpjepa,
        backbone=nn.Identity(),
        projector=nn.Identity(),
        rdmreg=nn.MSELoss(),
        inv_weight=1.0,
        rdmreg_weight=1.0,
    )
    monkeypatch.setattr(model, "log_dict", Mock())
    callback = Sparsity("projector", train_first_view_only=True)
    trainer = SimpleNamespace(global_step=0)
    batch = {
        "views": [
            {"image": torch.tensor([[3.0, 0, 0, 0], [0, 0, 0, 0]], requires_grad=True)},
            {"image": torch.ones(2, 4, requires_grad=True)},
        ],
        "batch_idx": 0,
    }
    callback.on_train_batch_start(trainer, model, batch, 0)
    outputs = model(batch, "fit")
    projection = outputs["projection"]
    keys = set(outputs)
    model.log_dict.reset_mock()
    trainer.global_step = 1

    callback.on_train_batch_end(trainer, model, outputs, batch, 0)

    model.log_dict.assert_called_once()
    _assert_log(model.log_dict.call_args, "fit", "projector", 0.125, 0.125, 2)
    assert set(outputs) == keys
    assert outputs["projection"] is projection
    assert projection.requires_grad
    outputs["loss"].backward()
    assert all(view["image"].grad is not None for view in batch["views"])

    model.eval()
    model.log_dict.reset_mock()
    batch = {"image": torch.ones(3, 4), "batch_idx": 0}
    with torch.no_grad():
        outputs = model(batch, "validate")
    callback.on_validation_batch_end(trainer, model, outputs, batch, 0)
    _assert_log(model.log_dict.call_args, "validate", "projector", 1.0, 1.0, 3)
