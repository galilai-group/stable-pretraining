"""Reference-statistic, gradient, causality, and Lightning checks for Sub-JEPA."""

from unittest.mock import Mock
from datetime import timedelta

import lightning as pl
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining.losses import SubJEPALoss
from stable_pretraining.methods.subjepa import SubJEPAPredictor

pytestmark = pytest.mark.unit


def test_reference_statistic_and_gradient():
    """Check the released per-time statistic against an independent scalar loop."""
    torch.manual_seed(7)
    loss = SubJEPALoss(8, 4, num_projections=5).double()
    x = torch.randn(6, 3, 8, dtype=torch.float64, requires_grad=True)
    generator = torch.Generator().manual_seed(0)
    directions = torch.randn(4, 2, 5, dtype=x.dtype, generator=generator)
    directions /= directions.norm(dim=1, keepdim=True)
    statistics = []
    for k in range(4):
        projected = x @ loss.projection_matrices[k].T
        for t in range(3):
            for m in range(5):
                angles = (projected[:, t] @ directions[k, :, m])[:, None] * loss.knots
                error = (angles.cos().mean(0) - loss.phi).square()
                error += angles.sin().mean(0).square()
                statistics.append((error * loss.weights).sum() * len(x))
    expected = torch.stack(statistics).mean()
    actual = loss.regularization(x)
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected, x, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_frozen_orthogonal_checkpoint():
    loss = SubJEPALoss(12, 4, num_projections=8)
    bases = loss.projection_matrices
    torch.testing.assert_close(
        bases @ bases.transpose(-1, -2), torch.eye(3).expand(4, 3, 3)
    )
    assert not list(loss.parameters())
    x = torch.randn(8, 3, 12)
    loss.regularization(x)
    restored = SubJEPALoss(12, 4, num_projections=8)
    restored.load_state_dict(loss.state_dict())
    torch.testing.assert_close(loss.regularization(x), restored.regularization(x))


def test_gaussian_scores_below_collapsed_embeddings():
    loss = SubJEPALoss(12, 4, num_projections=32)
    torch.manual_seed(100)
    gaussian = loss.regularization(torch.randn(512, 2, 12))
    collapsed = loss.regularization(torch.zeros(512, 2, 12))
    assert gaussian < collapsed / 20


def test_targets_receive_gradients():
    loss = SubJEPALoss(8, 4, num_projections=8)
    prediction = torch.randn(6, 2, 8, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    embedding = torch.randn(6, 3, 8, requires_grad=True)
    result = loss(prediction, target, embedding)
    result["loss"].backward()
    for x in (prediction, target, embedding):
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embed_dim": 0},
        {"num_subspaces": 0},
        {"embed_dim": 7, "num_subspaces": 4},
        {"subspace_dim": 193},
        {"num_projections": 0},
        {"num_knots": 1},
        {"regularization_weight": -1},
        {"regularization_weight": float("nan")},
    ],
)
def test_invalid_loss_configuration(kwargs):
    with pytest.raises(ValueError):
        SubJEPALoss(**kwargs)


def test_predictor_causality_and_action_conditioning():
    torch.manual_seed(3)
    predictor = SubJEPAPredictor(12, 2, depth=1, num_heads=3).eval()
    x, actions = torch.randn(4, 4, 12), torch.randn(4, 4, 2)
    expected = predictor(x, actions)
    changed_x, changed_actions = x.clone(), actions.clone()
    changed_x[:, 2:] += 10
    changed_actions[:, 2:] -= 10
    actual = predictor(changed_x, changed_actions)
    torch.testing.assert_close(actual[:, :2], expected[:, :2])
    assert not torch.allclose(predictor(x, actions + 2), expected)


def _model() -> spt.SubJEPA:
    return spt.SubJEPA(
        nn.Linear(4, 12),
        2,
        embed_dim=12,
        num_subspaces=4,
        num_projections=8,
        predictor_depth=1,
        num_heads=3,
    )


def test_forward_and_public_exports():
    from stable_pretraining.methods import SubJEPA

    assert SubJEPA is spt.SubJEPA
    model = _model()
    model.log_dict = Mock()
    batch = {"image": torch.randn(6, 3, 4), "action": torch.randn(6, 2, 2)}
    output = model(batch, "train")
    assert output["embedding"].shape == (6, 3, 12)
    assert output["prediction"].shape == (6, 2, 12)
    output["loss"].backward()
    assert all(p.grad is not None for p in model.parameters())
    embedded = model({"image": torch.randn(6, 4)}, "val")
    assert embedded["embedding"].shape == (6, 12)
    with pytest.raises(ValueError, match="requires an action"):
        model({"image": torch.randn(6, 4)}, "train")
    with pytest.raises(ValueError, match="action must have shape"):
        model({**batch, "action": torch.randn(6, 1, 2)}, "train")


class _Trajectories(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return {"image": torch.randn(3, 4), "action": torch.randn(2, 2)}


def test_lightning_training_and_validation(tmp_path):
    model = _model()
    before = model.backbone.weight.detach().clone()
    loader = DataLoader(_Trajectories(), batch_size=8)
    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model, loader, loader)
    assert not torch.equal(before, model.backbone.weight)
    assert torch.isfinite(trainer.callback_metrics["validate/loss"])


class _ResumeState(pl.Callback):
    """Capture restored state before a resumed training batch can change it."""

    def on_train_start(self, trainer, pl_module):
        self.global_step = trainer.global_step
        self.state = {
            name: tensor.detach().clone()
            for name, tensor in pl_module.state_dict().items()
        }


class _StopAfterEpoch(pl.Callback):
    """Stop at an epoch boundary without changing the planned training horizon."""

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def test_manager_checkpoint_resume_preserves_frozen_subspaces(tmp_path, monkeypatch):
    """Exercise full Manager resume and observe buffers before and after training."""
    from stable_pretraining._config import get_config
    from stable_pretraining.callbacks.factories import _DEFAULT_CALLBACK_REGISTRY

    config = get_config()
    monkeypatch.setattr(
        config,
        "_default_callbacks",
        {name: name == "registry" for name in _DEFAULT_CALLBACK_REGISTRY},
    )
    monkeypatch.setattr(config, "_requeue_checkpoint", False)
    model = _model()
    model.optim["scheduler"] = {"type": "ConstantLR", "factor": 1.0}
    initial_weight = model.backbone.weight.detach().clone()
    initial_matrices = model.subjepa_loss.projection_matrices.clone()
    loader = DataLoader(_Trajectories(), batch_size=8)
    data = spt.data.DataModule(train=loader)
    trainer_options = {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "limit_val_batches": 0,
    }
    trainer = pl.Trainer(**trainer_options, max_epochs=2, callbacks=[_StopAfterEpoch()])
    spt.Manager(trainer=trainer, module=model, data=data, seed=13)()
    assert trainer.global_step == 2
    assert model.subjepa_loss.projection_step.item() == 2
    assert not torch.equal(initial_weight, model.backbone.weight)
    torch.testing.assert_close(
        model.subjepa_loss.projection_matrices, initial_matrices, rtol=0, atol=0
    )
    checkpoint = tmp_path / "subjepa.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    matrix_key = "subjepa_loss.projection_matrices"
    step_key = "subjepa_loss.projection_step"
    torch.testing.assert_close(
        saved["state_dict"][matrix_key], initial_matrices, rtol=0, atol=0
    )
    assert saved["state_dict"][step_key].item() == saved["global_step"] == 2

    resumed = _model()
    resumed.optim["scheduler"] = {"type": "ConstantLR", "factor": 1.0}
    # Distinct sentinels make a missing checkpoint restore observable.
    resumed.subjepa_loss.projection_matrices.zero_()
    resumed.subjepa_loss.projection_step.fill_(999)
    restored = _ResumeState()
    resumed_trainer = pl.Trainer(**trainer_options, max_epochs=2, callbacks=[restored])
    spt.Manager(
        trainer=resumed_trainer,
        module=resumed,
        data=spt.data.DataModule(train=loader),
        seed=13,
        ckpt_path=str(checkpoint),
        weights_only=False,
    )()
    assert restored.global_step == 2
    for name, expected in saved["state_dict"].items():
        torch.testing.assert_close(restored.state[name], expected, rtol=0, atol=0)
    assert resumed_trainer.global_step == 4
    assert resumed.subjepa_loss.projection_step.item() == 4
    torch.testing.assert_close(
        resumed.subjepa_loss.projection_matrices, initial_matrices, rtol=0, atol=0
    )
    assert not torch.equal(restored.state["backbone.weight"], resumed.backbone.weight)


def _distributed_statistic(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(41)
    reference = SubJEPALoss(8, 4, num_projections=5)
    distributed = SubJEPALoss(8, 4, num_projections=5)
    distributed.load_state_dict(reference.state_dict())
    full = torch.randn(8, 2, 8, requires_grad=True)
    expected = reference.regularization(full)
    expected_gradient = torch.autograd.grad(expected, full)[0]
    start, stop = (0, 3) if rank == 0 else (3, 8)
    local = full.detach()[start:stop].clone().requires_grad_()
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        actual = distributed.regularization(local)
        actual_gradient = torch.autograd.grad(actual, local)[0]
        torch.testing.assert_close(actual, expected)
        # DDP averages parameter gradients after the differentiable sum's backward.
        torch.testing.assert_close(actual_gradient / 2, expected_gradient[start:stop])
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_global_statistics_with_unequal_rank_batches(tmp_path):
    mp.spawn(
        _distributed_statistic, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2
    )


def test_mixed_precision_statistic_is_finite():
    loss = SubJEPALoss(8, 4, num_projections=8)
    x = torch.randn(8, 3, 8, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = loss.regularization(x.to(torch.bfloat16))
    assert result.dtype == torch.float32 and torch.isfinite(result)
    result.backward()
    assert torch.isfinite(x.grad).all()


def test_composable_module_and_trailing_action():
    model = spt.Module(
        forward=spt.forward.subjepa,
        backbone=nn.Linear(4, 12),
        predictor=SubJEPAPredictor(12, 2, depth=1, num_heads=3),
        subjepa_loss=SubJEPALoss(12, 4, num_projections=8),
    )
    model.log_dict = Mock()
    output = model(
        {"image": torch.randn(6, 3, 4), "action": torch.randn(6, 3, 2)}, "fit"
    )
    assert torch.isfinite(output["loss"])
    assert output["prediction"].shape == (6, 2, 12)
