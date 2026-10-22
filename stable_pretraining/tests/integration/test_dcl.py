"""Integration tests for DCL functionality."""

from copy import deepcopy
from functools import partial

import lightning as pl
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import stable_pretraining as spt
from stable_pretraining.forward import dcl
from stable_pretraining.losses import DCLLoss
from stable_pretraining.tests.distributed import _dist_harness

pytestmark = pytest.mark.integration


class _CaptureValidation(pl.Callback):
    def __init__(self):
        self.outputs = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.outputs.append(outputs)


def _backbone():
    return torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 8),
    )


def _batch():
    return {
        "views": [
            {"image": torch.randn(4, 3, 8, 8), "label": torch.arange(4)}
            for _ in range(2)
        ]
    }


@pytest.fixture
def disable_default_callbacks(monkeypatch):
    from stable_pretraining.callbacks.factories import _DEFAULT_CALLBACK_REGISTRY

    config = spt.get_config()
    monkeypatch.setattr(
        config,
        "_default_callbacks",
        {name: False for name in _DEFAULT_CALLBACK_REGISTRY},
    )
    monkeypatch.setattr(config, "_requeue_checkpoint", False)


@pytest.mark.parametrize("method_class", [False, True])
@pytest.mark.parametrize("with_probe", [False, True])
def test_cpu_training_and_validation_update_backbone(
    tmp_path, disable_default_callbacks, method_class, with_probe
):
    torch.manual_seed(0)
    if method_class:
        module = spt.DCL(encoder_name=_backbone(), projector_dims=(16, 4))
    else:
        module = spt.Module(
            forward=dcl,
            backbone=_backbone(),
            projector=torch.nn.Linear(8, 4),
            dcl_loss=DCLLoss(),
        )
    before = module.backbone[-1].weight.detach().clone()
    validation = _CaptureValidation()
    callbacks = [validation]
    if with_probe:
        classifier = torch.nn.Linear(8, 4)
        probe_before = classifier.weight.detach().clone()
        callbacks.append(
            spt.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=classifier,
                loss=torch.nn.CrossEntropyLoss(),
                optimizer={"type": "SGD", "lr": 0.1},
                metrics={"accuracy": MulticlassAccuracy(num_classes=4)},
            )
        )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
        callbacks=callbacks,
    )
    batch = _batch()
    train_data = [
        {
            "views": [
                {key: value[i] for key, value in view.items()}
                for view in batch["views"]
            ]
        }
        for i in range(4)
    ]
    train = DataLoader(train_data, batch_size=2)
    val = DataLoader(
        [{"image": torch.randn(3, 8, 8), "label": i} for i in range(2)], batch_size=2
    )
    trainer.fit(module, train_dataloaders=train, val_dataloaders=val)
    assert trainer.global_step == 2
    assert not torch.equal(before, module.backbone[-1].weight)
    if with_probe:
        assert not torch.equal(probe_before, classifier.weight)
        assert torch.isfinite(trainer.callback_metrics["eval/linear_probe_accuracy"])
    assert len(validation.outputs) == 1
    output = validation.outputs[0]
    assert output["embedding"].shape == (2, 8)
    assert "loss" not in output
    if with_probe:
        assert output["linear_probe_preds"].shape == (2, 4)
        torch.testing.assert_close(output["label"], torch.arange(2))


def _ddp_matches_global_reference(rank, world_size, local_batch_size, result_dir):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    encoder = torch.nn.Linear(3, 4).double()
    reference = deepcopy(encoder)
    distributed = DistributedDataParallel(encoder)
    inputs = torch.randn(2, world_size * local_batch_size, 3, dtype=torch.double)

    # Avoid batch normalization here: per-rank statistics would change the objective.
    z_i, z_j = F.normalize(reference(inputs), dim=-1).unbind()
    features = torch.cat([z_i, z_j])
    n = z_i.size(0)
    terms = []
    for anchor in range(2 * n):
        positive = (anchor + n) % (2 * n)
        negatives = [j for j in range(2 * n) if j not in (anchor, positive)]
        terms.append(
            torch.logsumexp(features[anchor] @ features[negatives].T / 0.1, dim=0)
            - features[anchor] @ features[positive] / 0.1
        )
    expected = torch.stack(terms).mean()
    expected.backward()

    start = rank * local_batch_size
    local_inputs = inputs[:, start : start + local_batch_size]
    projections = distributed(local_inputs)
    actual = DCLLoss()(projections[0], projections[1])
    actual.backward()
    mean_loss = actual.detach().clone()
    dist.all_reduce(mean_loss)
    torch.testing.assert_close(mean_loss / world_size, expected)
    for param, ref_param in zip(encoder.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, ref_param.grad)
    torch.optim.SGD(encoder.parameters(), lr=0.01).step()
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    for param, ref_param in zip(encoder.parameters(), reference.parameters()):
        torch.testing.assert_close(param, ref_param)

    # The shared harness does not fail on timeouts; require proof every rank finished.
    (result_dir / f"rank_{rank}").write_text("passed")


@pytest.mark.distributed
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.parametrize("local_batch_size", [1, 3])
def test_ddp_loss_gradients_and_update_match_global_batch(tmp_path, local_batch_size):
    worker = partial(
        _ddp_matches_global_reference,
        local_batch_size=local_batch_size,
        result_dir=tmp_path,
    )
    _dist_harness.run_distributed(worker, world_size=2, backend="gloo")
    for rank in range(2):
        assert (tmp_path / f"rank_{rank}").read_text() == "passed"
