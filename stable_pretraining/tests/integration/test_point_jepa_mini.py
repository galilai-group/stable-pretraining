"""Small Lightning integration test for Point-JEPA manual optimization and EMA."""

import lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader

from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.methods.point_jepa import PointJEPA


@pytest.mark.parametrize("epoch_schedule", [False, True])
def test_point_jepa_two_steps_and_checkpoint(tmp_path, epoch_schedule):
    """Two accumulated optimizer steps update EMA and survive a checkpoint."""
    model = PointJEPA(
        num_groups=8,
        group_size=8,
        embed_dim=48,
        encoder_depth=1,
        encoder_heads=3,
        predictor_dim=24,
        predictor_depth=1,
        predictor_heads=3,
        accumulate_steps=2,
    )
    initial = model.encoder.teacher.blocks[0].attn.qkv.weight.detach().clone()
    loader = DataLoader(
        [{"points": torch.randn(64, 3)} for _ in range(4)], batch_size=1
    )
    callback = TeacherStudentCallback()
    learning_rates = []

    class CaptureLR(pl.Callback):
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            learning_rates.append(optimizer.param_groups[0]["lr"])

    trainer = pl.Trainer(
        accelerator="cpu",
        **({"max_epochs": 2} if epoch_schedule else {"max_steps": 2}),
        callbacks=[callback, CaptureLR()],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_val_batches=0,
    )
    trainer.fit(model, loader)
    expected_steps = 4 if epoch_schedule else 2
    assert trainer.global_step == expected_steps
    assert callback._last_global_step == expected_steps
    assert model.encoder.ema_updates.item() == expected_steps
    if epoch_schedule:
        assert learning_rates == pytest.approx([1e-5, 1e-5, 1e-3, 1e-3])
    assert not torch.equal(model.encoder.teacher.blocks[0].attn.qkv.weight, initial)
    path = tmp_path / "point_jepa.ckpt"
    trainer.save_checkpoint(path)
    assert (
        torch.load(path, map_location="cpu", weights_only=False)["global_step"]
        == expected_steps
    )
