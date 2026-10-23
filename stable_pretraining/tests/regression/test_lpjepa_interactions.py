"""CPU Manager training and full-state resume for both LpJEPA entry points."""

from pathlib import Path

import lightning as pl
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import stable_pretraining as spt
from stable_pretraining.methods.lpjepa import RDMReg

from ._interaction_helpers import _Trace, _assert_tree_close, _snapshot

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]

_METRIC_KEYS = (
    "fit/loss_epoch",
    "fit/inv_loss_epoch",
    "fit/rdmreg_loss_epoch",
    "train/encoder_probe_loss_epoch",
    "train/projector_probe_loss_epoch",
    "eval/encoder_probe_top1_epoch",
    "eval/projector_probe_top1_epoch",
    "fit/l0_sparsity_epoch",
    "fit/l1_sparsity_epoch",
    "validate/l0_sparsity",
    "validate/l1_sparsity",
)


@pytest.fixture(params=["shared_forward", "method_class"])
def make_components(request, monkeypatch):
    def create_encoder(*args, **kwargs):
        return VisionTransformer(
            img_size=32,
            patch_size=8,
            embed_dim=24,
            depth=2,
            num_heads=3,
            num_classes=0,
        )

    monkeypatch.setattr("timm.create_model", create_encoder)

    def create_components():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            projector = nn.Sequential(
                nn.Linear(24, 12),
                nn.BatchNorm1d(12),
                nn.ReLU(),
                nn.Linear(12, 8),
            )
            if request.param == "method_class":
                module = spt.LpJEPA(
                    inv_weight=2.0,
                    rdmreg_weight=3.0,
                    encoder_name="tiny",
                    projector=projector,
                    n_slices=16,
                    pretrained=False,
                    drop_path_rate=0.0,
                )
            else:
                module = spt.Module(
                    forward=spt.forward.lpjepa,
                    backbone=create_encoder(),
                    projector=projector,
                    rdmreg=RDMReg(num_slices=16),
                    inv_weight=2.0,
                    rdmreg_weight=3.0,
                )
            module.optim = {
                "optimizer": {"type": "Adam", "lr": 0.001},
                "scheduler": {"type": "ExponentialLR", "gamma": 0.9},
                "interval": "step",
            }
            callbacks = [
                spt.OnlineProbe(
                    module,
                    name=name,
                    input=target,
                    target="label",
                    probe=nn.Linear(dim, 2),
                    loss=nn.CrossEntropyLoss(),
                    optimizer={"type": "Adam", "lr": 0.01},
                    scheduler={"type": "ExponentialLR", "gamma": 0.8},
                    metrics={
                        "top1": MulticlassAccuracy(num_classes=2, average="micro")
                    },
                    verbose=False,
                )
                for name, target, dim in (
                    ("encoder_probe", "embedding", 24),
                    ("projector_probe", "projection", 8),
                )
            ]
        trace = _Trace()
        callbacks.extend(
            [
                spt.callbacks.Sparsity(
                    name="sparsity", every_n_steps=2, train_first_view_only=True
                ),
                trace,
            ]
        )
        return module, callbacks, trace

    return create_components


def _run(module, callbacks, *, max_epochs=3, ckpt_path: Path | None = None):
    generator = torch.Generator().manual_seed(17)
    images = torch.randn(8, 3, 32, 32, generator=generator)
    samples = [
        {"image": image, "label": torch.tensor(i % 2)} for i, image in enumerate(images)
    ]
    paired = [
        {
            "views": [
                sample,
                {"image": sample["image"].flip(-1), "label": sample["label"]},
            ]
        }
        for sample in samples
    ]
    data = spt.data.DataModule(
        train=DataLoader(paired, batch_size=2, shuffle=False),
        val=DataLoader(samples[:4], batch_size=2, shuffle=False),
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        seed=59,
        ckpt_path=ckpt_path,
        weights_only=False,
    )()
    return trainer


class _StopAfterEpoch(pl.Callback):
    """Stop the first leg without changing the planned scheduler horizon."""

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def test_lpjepa_manager_trains_and_validates_with_probes(make_components):
    module, callbacks, trace = make_components()
    trainer = _run(module, callbacks, max_epochs=1)

    assert trainer.global_step == trace.batches_seen == 4
    assert module.rdmreg.global_step.item() == 4
    assert len(trainer.optimizers) == len(trainer.lr_scheduler_configs) == 3
    for key in _METRIC_KEYS:
        assert torch.isfinite(trainer.callback_metrics[key])
    for name in (
        "backbone",
        "projector",
        "callbacks_modules.encoder_probe",
        "callbacks_modules.projector_probe",
    ):
        parameters = dict(module.get_submodule(name).named_parameters())
        assert all(torch.isfinite(parameter).all() for parameter in parameters.values())
        assert any(
            not torch.equal(parameter, trace.start["model"][f"{name}.{key}"])
            for key, parameter in parameters.items()
        )


def test_lpjepa_manager_restores_full_training_state(tmp_path, make_components):
    reference, callbacks, expected = make_components()
    full_trainer = _run(reference, callbacks)

    interrupted, callbacks, first = make_components()
    first_trainer = _run(interrupted, [*callbacks, _StopAfterEpoch()])
    assert first_trainer.global_step == first.batches_seen == 4
    _assert_tree_close(first.epochs[0], expected.epochs[0])
    checkpoint = tmp_path / "lpjepa.ckpt"
    first_trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert len(saved["optimizer_states"]) == len(saved["lr_schedulers"]) == 3
    assert all(state["state"] for state in saved["optimizer_states"])
    assert saved["state_dict"]["rdmreg.global_step"].item() == 4
    assert saved["callbacks"][first.state_key]["batches_seen"] == 4

    resumed, callbacks, second = make_components()
    with torch.no_grad():
        for parameter in resumed.backbone.parameters():
            parameter.add_(10)
        resumed.rdmreg.global_step.fill_(999)
    resumed_trainer = _run(resumed, callbacks, ckpt_path=checkpoint)

    _assert_tree_close(second.start, first.epochs[0])
    assert second.restored_batches == 4
    assert second.batches_seen == expected.batches_seen == 12
    assert len(second.batches) == 8
    for actual, wanted in zip(second.batches, expected.batches[4:]):
        _assert_tree_close(actual, wanted)
    _assert_tree_close(
        _snapshot(resumed_trainer, resumed), _snapshot(full_trainer, reference)
    )
    assert resumed_trainer.current_epoch == full_trainer.current_epoch == 3
    assert resumed_trainer.global_step == full_trainer.global_step == 12
    assert (
        resumed.rdmreg.global_step.item() == reference.rdmreg.global_step.item() == 12
    )
    for key in _METRIC_KEYS:
        _assert_tree_close(
            resumed_trainer.callback_metrics[key], full_trainer.callback_metrics[key]
        )
