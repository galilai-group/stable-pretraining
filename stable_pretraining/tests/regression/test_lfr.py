"""LFR trains, validates, and resumes through the normal Manager lifecycle."""

from copy import deepcopy

import lightning as pl
import pytest
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader

import stable_pretraining as spt
from stable_pretraining.forward import lfr
from stable_pretraining.tests.unit.test_lfr import TinyEncoder, image_samples, tiny_lfr

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]


def test_manager_training_validation_and_checkpoint(tmp_path):
    pl.seed_everything(11, workers=True)
    model = tiny_lfr()
    probe = spt.callbacks.OnlineProbe(
        model,
        name="probe",
        input="embedding",
        target="label",
        probe=nn.Linear(8, 2),
        loss=nn.CrossEntropyLoss(),
        metrics={"accuracy": MulticlassAccuracy(2)},
    )
    samples = image_samples()
    data = spt.data.DataModule(
        train=DataLoader(samples, batch_size=4, drop_last=True),
        val=DataLoader(samples, batch_size=4),
    )
    initial = deepcopy(model.state_dict())
    initial_probe = deepcopy(probe._probe_config.state_dict())
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
    # One cycle: an encoder epoch, then a predictor epoch.
    trainer = pl.Trainer(
        max_epochs=model.max_epochs_for(1), callbacks=[probe], **trainer_args
    )
    spt.Manager(trainer=trainer, module=model, data=data, seed=11)()
    assert trainer.global_step == 4
    assert torch.isfinite(trainer.callback_metrics["fit/loss_epoch"])
    assert torch.isfinite(trainer.callback_metrics["validate/loss"])
    assert "eval/probe_accuracy" in trainer.callback_metrics
    for key in ("backbone.net.0.weight", "predictors.0.0.weight"):
        assert not torch.equal(model.state_dict()[key], initial[key]), key
    trained_probe = model.callbacks_modules["probe"].state_dict()
    assert not torch.equal(trained_probe["weight"], initial_probe["weight"])
    assert torch.equal(
        model.state_dict()["projectors.0.0.weight"], initial["projectors.0.0.weight"]
    )

    checkpoint = tmp_path / "lfr.ckpt"
    trainer.save_checkpoint(checkpoint)
    # The checkpoint also holds the probe; a bare LFR restores everything else.
    restored = spt.LFR.load_from_checkpoint(
        checkpoint, encoder_name=TinyEncoder(), weights_only=False, strict=False
    )
    assert restored.hparams.num_projectors == 2
    assert restored.hparams.predictor_epochs == 1
    restored_state = restored.state_dict()
    for name, value in model.state_dict().items():
        if not name.startswith("callbacks_"):
            # Random projectors come from the checkpoint, not a fresh draw.
            torch.testing.assert_close(restored_state[name], value, msg=name)
    model.eval()
    restored.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        # Call the forward function directly: the probe wraps model.forward.
        expected = lfr(model, batch, "validate")
        actual = restored(batch, stage="validate")
    for key in ("embedding", "loss"):
        torch.testing.assert_close(actual[key], expected[key])

    # Resuming continues the cycle: epoch 2 trains the encoder again.
    encoder_before = restored.backbone.net[0].weight.detach().clone()
    predictor_before = restored.predictors[0][0].weight.detach().clone()
    resumed_probe = spt.callbacks.OnlineProbe(
        restored,
        name="probe",
        input="embedding",
        target="label",
        probe=nn.Linear(8, 2),
        loss=nn.CrossEntropyLoss(),
        metrics={"accuracy": MulticlassAccuracy(2)},
    )
    resumed = pl.Trainer(
        max_epochs=restored.max_epochs_for(1) + 1,
        callbacks=[resumed_probe],
        **trainer_args,
    )
    restored.train()
    spt.Manager(
        trainer=resumed,
        module=restored,
        data=data,
        seed=11,
        ckpt_path=str(checkpoint),
        weights_only=False,
    )()
    assert resumed.global_step == 6
    assert not torch.equal(restored.backbone.net[0].weight, encoder_before)
    assert torch.equal(restored.predictors[0][0].weight, predictor_before)
