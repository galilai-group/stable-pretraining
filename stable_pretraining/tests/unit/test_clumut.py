"""Focused tests for CluMuT assignments, losses, and training integration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stable_pretraining.losses import CluMuTLoss
from stable_pretraining.methods import CluMuT

pytestmark = pytest.mark.unit

_example_path = Path(__file__).resolve().parents[3] / "examples" / "clumut_cifar10.py"
_example_spec = importlib.util.spec_from_file_location(
    "clumut_cifar10_example", _example_path
)
_example = importlib.util.module_from_spec(_example_spec)
_example_spec.loader.exec_module(_example)
CluMuTDiagnosticsCallback = _example.CluMuTDiagnosticsCallback


def _model() -> CluMuT:
    backbone = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 8),
    )
    return CluMuT(
        encoder_name=backbone,
        projector_dims=(8, 4),
        n_prototypes=5,
        temperature=0.2,
    )


def test_assignments_and_loss_gradients_reach_encoder_and_prototypes():
    torch.manual_seed(0)
    model = _model()
    view_a = torch.randn(4, 3, 16, 16)
    view_b = torch.randn(4, 3, 16, 16)

    output = model(view_a, view_b)

    assert output.h_a.shape == (4, 5)
    assert output.h_b.shape == (4, 5)
    assert output.q_a.shape == (4, 5)
    assert output.q_b.shape == (4, 5)
    assert torch.allclose(output.q_a.sum(dim=-1), torch.ones(4))
    assert torch.isfinite(output.loss)
    assert torch.allclose(
        output.loss,
        output.mutual_targeting_loss + output.identitization_loss,
    )
    output.loss.backward()

    assert model.prototypes.grad is not None
    assert model.prototypes.grad.abs().sum() > 0
    backbone_weight = model.backbone[-1].weight
    assert backbone_weight.grad is not None
    assert backbone_weight.grad.abs().sum() > 0


def test_identical_views_have_zero_mutual_targeting_loss():
    model = _model().eval()
    images = torch.randn(4, 3, 16, 16)

    with torch.no_grad():
        output = model(images, images)

    assert output.mutual_targeting_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(output.q_a, output.q_b)


def test_different_views_produce_nonzero_mutual_targeting_loss():
    torch.manual_seed(2)
    model = _model().eval()
    view_a = torch.randn(4, 3, 16, 16)
    view_b = torch.randn(4, 3, 16, 16) + 2

    output = model(view_a, view_b)

    assert output.mutual_targeting_loss.item() > 0
    assert not torch.allclose(output.q_a, output.q_b)


def test_mutual_targeting_loss_sums_per_instance_kl_terms():
    loss = CluMuTLoss()
    q_a = torch.softmax(torch.randn(4, 5), dim=-1)
    q_b = torch.softmax(torch.randn(4, 5), dim=-1)
    log_q_a = q_a.log()
    log_q_b = q_b.log()
    expected = 0.5 * (
        torch.nn.functional.kl_div(log_q_a, q_b.detach(), reduction="sum")
        + torch.nn.functional.kl_div(log_q_b, q_a.detach(), reduction="sum")
    )
    actual = loss.mutual_targeting_loss(q_a, q_b)
    batchmean = 0.5 * (
        torch.nn.functional.kl_div(log_q_a, q_b.detach(), reduction="batchmean")
        + torch.nn.functional.kl_div(log_q_b, q_a.detach(), reduction="batchmean")
    )

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual, batchmean * q_a.shape[0])


def test_identitization_is_finite_for_constant_dimensions_and_rejects_batch_one():
    loss = CluMuTLoss()
    h_a = torch.ones(3, 4, requires_grad=True)
    h_b = torch.ones(3, 4, requires_grad=True)
    result = loss.identitization_loss(h_a, h_b)
    assert torch.isfinite(result)
    result.backward()
    assert torch.isfinite(h_a.grad).all()
    assert torch.isfinite(h_b.grad).all()

    with pytest.raises(ValueError, match="batch size of at least 2"):
        loss.identitization_loss(torch.ones(1, 4), torch.ones(1, 4))


def test_framework_batch_dict_contract_returns_training_state():
    model = _model()
    batch = {"views": [torch.randn(4, 3, 16, 16), torch.randn(4, 3, 16, 16)]}

    state = model(batch, stage="validate")

    assert isinstance(state, dict)
    assert torch.isfinite(state["loss"])
    assert state["embedding"].shape == (8, 8)


def test_diagnostics_report_losses_shapes_assignments_correlation_and_gradients(capsys):
    model = _model()
    callback = CluMuTDiagnosticsCallback(log_every=100, initial_batches=3)
    trainer = SimpleNamespace(current_epoch=0, global_step=1)
    callback.on_fit_start(trainer, model)
    callback.on_train_batch_start(trainer, model, {}, batch_idx=0)
    output = model(torch.randn(4, 3, 16, 16), torch.randn(4, 3, 16, 16))
    output.loss.backward()
    callback.on_train_batch_end(
        trainer,
        model,
        {
            "loss": output.loss,
            "mutual_targeting_loss": output.mutual_targeting_loss,
            "identitization_loss": output.identitization_loss,
            "h_a": output.h_a,
            "h_b": output.h_b,
            "q_a": output.q_a,
            "q_b": output.q_b,
            "correlation": output.correlation,
        },
        {},
        batch_idx=0,
    )
    callback.on_fit_end(trainer, model)
    logged = capsys.readouterr().out

    assert "epoch=1 step=1" in logged
    assert "mutual=" in logged and "identitization=" in logged and "total=" in logged
    assert "h_a=(4, 5)" in logged and "q_b=(4, 5)" in logged
    assert "h_a_std=" in logged and "h_a_mean_prototype_std=" in logged
    assert "corr_diag_mean=" in logged and "corr_diag_minmax=" in logged
    assert "corr_offdiag_abs=" in logged
    assert "corr_offdiag_minmax=" in logged
    assert "grad_norms={" in logged
    assert "'projector':" in logged and "'prototypes':" in logged


def test_diagnostics_warn_on_collapsed_assignments(capsys):
    callback = CluMuTDiagnosticsCallback()
    trainer = SimpleNamespace(current_epoch=2, global_step=101)
    assignments = torch.tensor([[0.9998, 0.0001, 0.0001]]).repeat(4, 1)
    output = {
        "loss": torch.tensor(1.0),
        "mutual_targeting_loss": torch.tensor(0.2),
        "identitization_loss": torch.tensor(0.8),
        "h_a": torch.randn(4, 3),
        "h_b": torch.randn(4, 3),
        "q_a": assignments,
        "q_b": assignments,
        "correlation": torch.eye(3),
    }

    callback.on_train_batch_end(trainer, None, output, {}, batch_idx=0)

    logged = capsys.readouterr().out
    assert "WARNING epoch=3 step=101: assignment entropy is very low" in logged
    assert "most samples assign >=99%" in logged


def test_diagnostics_warn_on_non_finite_losses_assignments_and_correlation(capsys):
    callback = CluMuTDiagnosticsCallback()
    trainer = SimpleNamespace(current_epoch=0, global_step=1)
    assignments = torch.tensor([[float("nan"), 0.5, 0.5]]).repeat(2, 1)
    output = {
        "loss": torch.tensor(float("nan")),
        "mutual_targeting_loss": torch.tensor(0.2),
        "identitization_loss": torch.tensor(0.8),
        "h_a": torch.randn(2, 3),
        "h_b": torch.randn(2, 3),
        "q_a": assignments,
        "q_b": assignments,
        "correlation": torch.full((3, 3), float("nan")),
    }

    callback.on_train_batch_end(trainer, None, output, {}, batch_idx=0)

    logged = capsys.readouterr().out
    assert "WARNING epoch=1 step=1: non-finite CluMuT loss" in logged
    assert "WARNING epoch=1 step=1: non-finite assignments" in logged
    assert "WARNING epoch=1 step=1: non-finite correlation matrix" in logged


class _TwoViewDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(4)
        self.views_a = torch.randn(4, 3, 16, 16, generator=generator)
        self.views_b = torch.randn(4, 3, 16, 16, generator=generator)

    def __len__(self):
        return len(self.views_a)

    def __getitem__(self, index):
        return {"views": (self.views_a[index], self.views_b[index])}


def test_lightning_training_step_runs_with_two_view_loader():
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=0,
    )
    model = CluMuT(
        encoder_name=nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8)
        ),
        projector_dims=(8, 4),
        n_prototypes=5,
    )
    trainer.fit(
        model,
        train_dataloaders=DataLoader(_TwoViewDataset(), batch_size=4),
    )
