"""FLIP trains, validates, and resumes through the normal Manager lifecycle."""

from copy import deepcopy

import lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader

import stable_pretraining as spt
from stable_pretraining.tests.unit.test_flip import TinyTextEncoder, tiny_flip

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]


def test_manager_training_validation_and_checkpoint(tmp_path):
    pl.seed_everything(11, workers=True)
    model = tiny_flip(optim={"optimizer": {"type": "AdamW", "lr": 1e-3}})
    samples = [
        {"image": torch.randn(3, 32, 32), "input_ids": torch.randint(0, 32, (6,))}
        for _ in range(8)
    ]
    data = spt.data.DataModule(
        train=DataLoader(samples, batch_size=4, drop_last=True),
        val=DataLoader(samples, batch_size=4),
    )
    initial = deepcopy(model.state_dict())
    trainer_args = dict(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
    )
    trainer = pl.Trainer(max_epochs=1, **trainer_args)
    spt.Manager(trainer=trainer, module=model, data=data, seed=11)()
    assert trainer.global_step == 2
    assert torch.isfinite(trainer.callback_metrics["fit/loss_epoch"])
    assert torch.isfinite(trainer.callback_metrics["validate/loss"])
    for key in ("image_projector.weight", "text_projector.weight", "logit_scale"):
        assert not torch.equal(model.state_dict()[key], initial[key])

    checkpoint = tmp_path / "flip.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = spt.FLIP.load_from_checkpoint(
        checkpoint,
        text_encoder=TinyTextEncoder(),
        text_dim=16,
        projection_dim=8,
        encoder_name=tiny_flip().image_encoder.vit,
        mask_ratio=0,
        optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
    )
    assert restored.hparams.mask_ratio == 0
    torch.testing.assert_close(restored.logit_scale, model.logit_scale)
    model.eval()
    restored.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        for key, value in model(batch, stage="validate").items():
            torch.testing.assert_close(restored(batch, stage="validate")[key], value)
    resumed = pl.Trainer(max_epochs=2, **trainer_args)
    restored.train()
    spt.Manager(
        trainer=resumed,
        module=restored,
        data=data,
        seed=11,
        ckpt_path=str(checkpoint),
        weights_only=False,
    )()
    assert resumed.global_step == 4
    for key in ("image_projector.weight", "image_encoder.vit.patch_embed.proj.weight"):
        assert not torch.equal(restored.state_dict()[key], model.state_dict()[key])
