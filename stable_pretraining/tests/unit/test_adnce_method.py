"""ADNCE dictionary contracts, callback gradients, and complete training restart."""

from copy import deepcopy
from unittest.mock import Mock

import hydra
import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import stable_pretraining as spt
from stable_pretraining.forward import adnce
from stable_pretraining.methods import ADNCE

pytestmark = pytest.mark.unit


def _model(composable=False):
    torch.manual_seed(17)
    components = {
        "backbone": nn.Sequential(nn.Linear(6, 5), nn.Tanh()),
        "projector": nn.Linear(5, 3),
        "optim": {
            "optimizer": {"type": "Adam", "lr": 0.01},
            "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.8},
        },
    }
    loss_settings = {"temperature": 0.3, "mu": 0.5, "sigma": 0.6}
    if composable:
        return spt.Module(
            forward=adnce,
            adnce_loss=spt.losses.ADNCELoss(**loss_settings),
            **components,
        )
    return ADNCE(**components, **loss_settings)


def _views(labels=True):
    generator = torch.Generator().manual_seed(41)
    views = [{"image": torch.randn(4, 6, generator=generator)} for _ in range(2)]
    if labels:
        for i, view in enumerate(views):
            view["label"] = torch.arange(4) + i
    return views


@pytest.mark.parametrize("composable", [False, True])
@pytest.mark.parametrize("container", ["list", "tuple", "named", "nested_named"])
def test_forward_preserves_view_order_labels_and_gradients(composable, container):
    model = _model(composable)
    model.log = Mock()
    views = _views()
    if container == "list":
        batch = {"views": views}
    elif container == "tuple":
        batch = {"views": tuple(views)}
    elif container == "named":
        batch = {"first": views[0], "second": views[1]}
    else:
        batch = {"views": {"first": views[0], "second": views[1]}}
    batch["batch_idx"] = 3
    batch["metadata"] = {"source": "synthetic"}
    expected = [model.backbone(view["image"]) for view in views]
    output = model(batch, "fit")
    torch.testing.assert_close(output["embedding"], torch.cat(expected))
    torch.testing.assert_close(output["label"], torch.cat([v["label"] for v in views]))
    torch.testing.assert_close(
        output["loss"], model.adnce_loss(*(model.projector(e) for e in expected))
    )
    assert output["loss"].ndim == 0
    output["loss"].backward()
    for parameter in [*model.backbone.parameters(), *model.projector.parameters()]:
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    model.log.assert_called_once()
    assert all("embedding" not in view for view in views)


@pytest.mark.parametrize("stage", ["validate", "val", "test", "predict"])
@pytest.mark.parametrize("multiview", [False, True])
def test_evaluation_skips_projector_and_loss_even_in_train_mode(stage, multiview):
    model = _model()
    model.projector.register_forward_pre_hook(
        lambda *_: pytest.fail("evaluation must not call the projector")
    )
    views = _views(labels=False)
    batch = {"views": views} if multiview else views[0]
    output = model(batch, stage)
    assert set(output) == {"embedding"}
    assert output["embedding"].shape == (8 if multiview else 4, 5)


def test_labels_are_optional_and_never_enter_ssl_loss():
    model = _model()
    model.log = Mock()
    views = _views(labels=False)
    unlabeled = model({"views": views}, "train")
    labeled = model({"views": views, "label": torch.tensor([3, 1, 2, 0])}, "fit")
    torch.testing.assert_close(labeled["loss"], unlabeled["loss"])
    torch.testing.assert_close(labeled["label"], torch.tensor([3, 1, 2, 0] * 2))
    assert "label" not in unlabeled
    model.eval()
    assert "loss" not in model({"views": views}, "fit")
    single = {"image": views[0]["image"], "label": torch.arange(4)}
    torch.testing.assert_close(model(single, "validate")["label"], single["label"])


@pytest.mark.parametrize(
    "batch,stage,message",
    [
        ({}, "unknown", "unsupported stage"),
        (torch.zeros(4, 6), "fit", "dictionary batch"),
        ({"image": torch.zeros(4, 6)}, "fit", "two views"),
        ({"views": []}, "fit", "two view dictionaries"),
        ({"views": "bad"}, "fit", "two view dictionaries"),
        ({"views": [{"image": torch.zeros(4, 6)}, {}]}, "fit", "containing 'image'"),
        ({"views": [{"image": "bad"}] * 2}, "fit", "batched tensors"),
        ({"views": [{"image": torch.zeros(6)}] * 2}, "fit", "batched tensors"),
        (
            {"views": [{"image": torch.zeros(4, 6)}, {"image": torch.zeros(3, 6)}]},
            "fit",
            "equal batch sizes",
        ),
        (
            {
                "views": [
                    {"image": torch.zeros(4, 6), "label": torch.arange(4)},
                    {"image": torch.zeros(4, 6)},
                ]
            },
            "fit",
            "labels must be present",
        ),
        (
            {"views": [{"image": torch.zeros(4, 6)}] * 2, "label": torch.arange(3)},
            "fit",
            "batch-aligned",
        ),
    ],
)
def test_invalid_batches_fail_clearly(batch, stage, message):
    with pytest.raises(ValueError, match=message):
        _model()(batch, stage)


@pytest.mark.parametrize("backbone", [nn.Unflatten(1, (2, 3)), nn.Identity()])
def test_backbone_must_return_flat_batch_features(backbone):
    model = _model()
    model.backbone = backbone
    image = (
        torch.zeros(4, 6)
        if isinstance(backbone, nn.Unflatten)
        else torch.zeros(4, 2, 3)
    )
    with pytest.raises(ValueError, match="backbone must return"):
        model({"image": image}, "predict")


def test_public_exports_and_hydra_configuration():
    assert spt.ADNCE is ADNCE
    assert "ADNCE" in dir(spt)
    model = hydra.utils.instantiate(
        {
            "_target_": "stable_pretraining.ADNCE",
            "backbone": {
                "_target_": "torch.nn.Linear",
                "in_features": 6,
                "out_features": 5,
            },
            "projector": {
                "_target_": "torch.nn.Linear",
                "in_features": 5,
                "out_features": 3,
            },
            "temperature": 0.3,
            "mu": 0.6,
            "sigma": 0.4,
            "hparams": {"experiment": "configuration-test"},
        }
    )
    assert isinstance(model, ADNCE)
    assert model.optim["optimizer"]["type"] == "Adam"
    assert model.hparams["experiment"] == "configuration-test"
    assert model.adnce_loss.temperature == 0.3
    assert model.adnce_loss.mu == 0.6
    assert model.adnce_loss.sigma == 0.4


class _StopAfterFirstEpoch(pl.Callback):
    """Interrupt at an epoch boundary without changing the planned schedule."""

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def _run(tmp_path, stop=False, checkpoint=None, probe=True):
    model = _model()
    before = deepcopy(model.state_dict())
    callbacks = []
    if probe:
        torch.manual_seed(29)
        callbacks.append(
            spt.OnlineProbe(
                model,
                name="probe",
                input="embedding",
                target="label",
                probe=nn.Linear(5, 2),
                loss=nn.CrossEntropyLoss(),
                optimizer={"type": "Adam", "lr": 0.02},
                scheduler={"type": "StepLR", "step_size": 1, "gamma": 0.9},
                metrics={"accuracy": MulticlassAccuracy(num_classes=2)},
                verbose=False,
            )
        )
    if stop:
        callbacks.append(_StopAfterFirstEpoch())
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(8, 6, generator=generator)
    train = [
        {
            "first": {"image": image, "label": torch.tensor(i % 2)},
            "second": {"image": image + 0.1, "label": torch.tensor(i % 2)},
        }
        for i, image in enumerate(images)
    ]
    validation = [
        {"image": image, "label": torch.tensor(i % 2)} for i, image in enumerate(images)
    ]
    data = spt.data.DataModule(
        train=DataLoader(train, batch_size=4),
        val=DataLoader(validation, batch_size=4),
        test=DataLoader(validation, batch_size=4),
        predict=DataLoader(validation, batch_size=4),
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        default_root_dir=str(tmp_path),
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=True,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=model,
        data=data,
        seed=19,
        ckpt_path=str(checkpoint) if checkpoint else None,
        weights_only=False,
    )
    manager()
    return model, trainer, manager, data, before


def _assert_equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_equal_tree(a, b)
    else:
        assert actual == expected


def test_manager_probe_evaluation_and_full_state_resume(tmp_path, monkeypatch):
    from stable_pretraining.callbacks.factories import _DEFAULT_CALLBACK_REGISTRY

    config = spt.get_config()
    monkeypatch.setattr(
        config,
        "_default_callbacks",
        {name: name == "registry" for name in _DEFAULT_CALLBACK_REGISTRY},
    )
    monkeypatch.setattr(config, "_requeue_checkpoint", False)
    monkeypatch.setattr(config, "_exclude_bias_norm", False)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        full, full_trainer, _, data, before = _run(tmp_path / "full")
        assert full_trainer.global_step == 4
        for key in before:
            assert not torch.equal(before[key], full.state_dict()[key])
        assert "train/probe_loss_epoch" in full_trainer.callback_metrics
        assert "eval/probe_accuracy" in full_trainer.callback_metrics

        interrupted, partial_trainer, partial_manager, _, _ = _run(
            tmp_path / "partial", stop=True
        )
        assert partial_trainer.global_step == 2
        checkpoint = tmp_path / "restart.ckpt"
        partial_manager.save_checkpoint(str(checkpoint), verbose=False)
        saved = torch.load(checkpoint, weights_only=False)
        assert len(saved["optimizer_states"]) == len(saved["lr_schedulers"]) == 2
        assert all(state["state"] for state in saved["optimizer_states"])
        _assert_equal_tree(saved["state_dict"], interrupted.state_dict())

        resumed, resumed_trainer, _, _, _ = _run(
            tmp_path / "resumed", checkpoint=checkpoint
        )
        assert resumed_trainer.global_step == full_trainer.global_step
        assert resumed_trainer.current_epoch == full_trainer.current_epoch == 2
        _assert_equal_tree(resumed.state_dict(), full.state_dict())
        for actual, expected in zip(
            resumed_trainer.optimizers, full_trainer.optimizers
        ):
            _assert_equal_tree(actual.state_dict(), expected.state_dict())
        for actual, expected in zip(
            resumed_trainer.lr_scheduler_configs, full_trainer.lr_scheduler_configs
        ):
            _assert_equal_tree(
                actual.scheduler.state_dict(), expected.scheduler.state_dict()
            )

        no_probe, evaluation_trainer, evaluation_manager, _, _ = _run(
            tmp_path / "no_probe", probe=False
        )
        _assert_equal_tree(no_probe.backbone.state_dict(), full.backbone.state_dict())
        _assert_equal_tree(no_probe.projector.state_dict(), full.projector.state_dict())
        export = tmp_path / "encoder.ckpt"
        evaluation_manager.save_checkpoint(str(export), verbose=False)
        restored = ADNCE.load_from_checkpoint(
            export,
            backbone=nn.Sequential(nn.Linear(6, 5), nn.Tanh()),
            projector=nn.Linear(5, 3),
        )
        _assert_equal_tree(restored.state_dict(), no_probe.state_dict())
        for name, expected in {"temperature": 0.3, "mu": 0.5, "sigma": 0.6}.items():
            assert getattr(restored.adnce_loss, name) == expected
            assert restored.hparams[name] == expected
        evaluation_trainer.validate(no_probe, datamodule=data, verbose=False)
        evaluation_trainer.test(no_probe, datamodule=data, verbose=False)
        predictions = evaluation_trainer.predict(no_probe, datamodule=data)
        assert len(predictions) == 2
        assert all(output["embedding"].shape == (4, 5) for output in predictions)
    finally:
        torch.set_num_threads(old_threads)
