"""Unit tests for MSF (Mean Shift for Self-Supervised Learning)."""

import lightning as pl
import pytest
import torch
import torch.nn as nn

import stable_pretraining as spt
from stable_pretraining.backbone import TeacherStudentWrapper
from stable_pretraining.callbacks.queues import UnsortedQueue
from stable_pretraining.forward import msf
from stable_pretraining.losses import BYOLLoss, MSFLoss
from stable_pretraining.methods.msf import MSF

pytestmark = pytest.mark.unit

B = 4


def encoder():
    """A tiny encoder (8-d features) so the tests run fast on CPU."""
    return nn.Sequential(nn.Conv2d(3, 8, 8, 8), nn.AdaptiveAvgPool2d(1), nn.Flatten())


def make_model():
    return MSF(
        encoder_name=encoder(),
        projector_dims=(16, 4),
        predictor_hidden_dim=16,
        queue_length=32,
        topk=2,
    )


def two_view_batch(labels=True):
    """A training batch as ``MultiViewTransform`` yields it, after collation."""
    views = [
        {"image": torch.randn(B, 3, 32, 32)}
        | ({"label": torch.arange(B)} if labels else {})
        for _ in range(2)
    ]
    return {"views": views}


# --- MSFLoss -------------------------------------------------------------


def test_loss_matches_hand_computation():
    query = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    # 2 nearest neighbours of target: [1, 0] and [0.6, 0.8]
    # distances: 2 - 2 * 1 = 0 and 2 - 2 * 0.6 = 0.8, mean = 0.4
    loss = MSFLoss(topk=2)(query, target, bank)
    torch.testing.assert_close(loss, torch.tensor(0.4))


def test_loss_topk1_is_byol():
    torch.manual_seed(0)
    query, target = torch.randn(4, 8), torch.randn(4, 8)
    bank = torch.cat([target, torch.randn(10, 8)])
    torch.testing.assert_close(
        MSFLoss(topk=1)(query, target, bank), BYOLLoss()(query, target)
    )


def test_loss_topk_larger_than_bank_uses_whole_bank():
    query = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # both entries used: (0 + 2) / 2 = 1
    torch.testing.assert_close(MSFLoss(topk=5)(query, target, bank), torch.tensor(1.0))


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_loss_and_grad_stay_finite_for_extreme_inputs(scale):
    torch.manual_seed(0)
    query = (torch.randn(4, 8) * scale).requires_grad_()
    target = torch.randn(4, 8) * scale
    bank = torch.cat([target, torch.randn(10, 8) * scale])
    loss = MSFLoss(topk=3)(query, target, bank)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(query.grad).all()


# --- MSF module, tensor interface --------------------------------------------


def test_queue_stores_teacher_targets():
    model = make_model().train()
    out = model(torch.randn(B, 3, 32, 32), torch.randn(B, 3, 32, 32))
    torch.testing.assert_close(model.queue.get(), out.target)
    model(torch.randn(B, 3, 32, 32), torch.randn(B, 3, 32, 32))
    assert model.queue.get().shape == (4 * B, 4)


def test_only_student_gets_gradients():
    model = make_model().train()
    model(torch.randn(B, 3, 32, 32), torch.randn(B, 3, 32, 32)).loss.backward()
    assert all(p.grad is None for p in model.backbone.teacher.parameters())
    assert all(p.grad is None for p in model.projector.teacher.parameters())
    assert all(p.grad is not None for p in model.predictor.parameters())


def test_eval_returns_embedding_and_zero_loss():
    model = make_model().eval()
    with torch.no_grad():
        out = model(torch.randn(B, 3, 32, 32))
    assert out.loss.item() == 0
    assert out.embedding.shape == (B, model.embed_dim)


# --- MSF module, batch dict interface (as Module.training_step calls it) ----


def test_batch_training_returns_loss_embedding_and_label():
    out = make_model().train()(two_view_batch(), stage="fit")
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    # Both views are concatenated, as in BYOL
    assert out["embedding"].shape == (2 * B, 8)
    assert out["label"].tolist() == list(range(B)) * 2
    out["loss"].backward()


def test_batch_matches_tensor_interface():
    batch = two_view_batch()
    torch.manual_seed(0)
    from_batch = make_model().train()(batch, stage="fit")["loss"]
    torch.manual_seed(0)
    views = batch["views"]
    from_tensors = make_model().train()(views[0]["image"], views[1]["image"]).loss
    torch.testing.assert_close(from_batch, from_tensors)


def test_batch_without_labels_returns_no_label():
    out = make_model().train()(two_view_batch(labels=False), stage="fit")
    assert "label" not in out and "loss" in out


def test_batch_needs_exactly_two_views():
    batch = {"views": two_view_batch()["views"] * 2}
    with pytest.raises(ValueError, match="exactly 2 views"):
        make_model().train()(batch, stage="fit")


def test_batch_eval_single_image_returns_embedding_without_loss():
    batch = {"image": torch.randn(B, 3, 32, 32), "label": torch.arange(B)}
    with torch.no_grad():
        out = make_model().eval()(batch, stage="validate")
    assert "loss" not in out
    assert out["embedding"].shape == (B, 8) and out["label"].tolist() == [0, 1, 2, 3]


# --- forward function and exports -------------------------------------------


def test_forward_function_with_spt_module():
    module = spt.Module(
        forward=msf,
        backbone=TeacherStudentWrapper(encoder()),
        projector=TeacherStudentWrapper(nn.Linear(8, 4)),
        predictor=nn.Linear(4, 4),
        queue=UnsortedQueue(max_length=32, shape=(4,)),
        msf_loss=MSFLoss(topk=2),
        optim={"optimizer": {"type": "SGD", "lr": 0.1}},
    ).train()
    out = module(two_view_batch(), stage="fit")
    assert torch.isfinite(out["loss"])
    assert module.queue.get().shape == (2 * B, 4)


def test_is_exported():
    assert spt.MSF is MSF
    assert "MSF" in spt.__all__
    assert spt.losses.MSFLoss is MSFLoss


# --- training and checkpoint resume -----------------------------------------


def _fit(model, max_steps, ckpt_path=None):
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=max_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    # batch_size=None yields each prepared dict unchanged
    batches = torch.utils.data.DataLoader(
        [two_view_batch() for _ in range(3)], batch_size=None
    )
    trainer.fit(model, train_dataloaders=batches, ckpt_path=ckpt_path)
    return trainer


def test_trainer_fit_and_resume_from_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = make_model()
    model.optim = {"optimizer": {"type": "SGD", "lr": 0.1}}
    before = [p.detach().clone() for p in model.parameters()]
    trainer = _fit(model, max_steps=2)
    assert trainer.global_step == 2
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))

    ckpt = tmp_path / "msf.ckpt"
    trainer.save_checkpoint(ckpt)
    bank = model.queue.get().clone()

    resumed = make_model()
    resumed.optim = model.optim
    trainer = _fit(resumed, max_steps=3, ckpt_path=ckpt)
    assert trainer.global_step == 3
    torch.testing.assert_close(resumed.queue.get()[: len(bank)], bank)
