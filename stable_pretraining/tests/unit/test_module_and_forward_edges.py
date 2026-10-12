"""Evaluation and optional runtime paths preserve batch and parameter contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from stable_pretraining import Module, forward
from stable_pretraining.backbone.utils import TeacherStudentWrapper

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name", ["byol", "vicreg", "barlow_twins", "swav", "nnclr", "twist"]
)
@pytest.mark.parametrize("multiple", [False, True])
def test_ssl_evaluation_preserves_labels_and_does_not_require_loss(name, multiple):
    backbone = nn.Linear(3, 4)
    wrapped = TeacherStudentWrapper(backbone) if name == "byol" else backbone
    model = SimpleNamespace(backbone=wrapped, training=False)
    views = [
        {"image": torch.randn(2, 3), "label": torch.tensor([0, 1])} for _ in range(2)
    ]
    batch = {"views": views} if multiple else views[0]
    result = getattr(forward, name)(model, batch, "validate")
    images = (
        torch.cat([view["image"] for view in views]) if multiple else views[0]["image"]
    )
    torch.testing.assert_close(result["embedding"], backbone(images))
    assert result["label"].tolist() == ([0, 1] * (2 if multiple else 1))
    assert "loss" not in result


@pytest.mark.parametrize("name", ["byol", "vicreg", "barlow_twins", "nnclr", "twist"])
def test_two_view_methods_reject_wrong_number_of_views(name):
    with pytest.raises(ValueError, match="exactly 2"):
        getattr(forward, name)(
            SimpleNamespace(), {"views": [{"image": torch.zeros(2, 3)}]}, "fit"
        )


@pytest.mark.parametrize("epoch", [0, 2])
def test_swav_queue_only_supplies_detached_features_after_warmup(monkeypatch, epoch):
    features = torch.randn(3, 4, requires_grad=True)
    queue = SimpleNamespace(get=lambda: features)
    monkeypatch.setattr(forward.OnlineQueue, "_shared_queues", {"swav_queue": queue})
    factory = Mock(return_value=SimpleNamespace(key="swav_queue"))
    monkeypatch.setattr(forward, "find_or_create_queue_callback", factory)
    loss = Mock(return_value=torch.tensor(1.0, requires_grad=True))
    model = SimpleNamespace(
        training=True,
        backbone=nn.Linear(3, 4),
        projector=nn.Identity(),
        prototypes=nn.Linear(4, 2),
        swav_loss=loss,
        use_queue=True,
        queue_length=3,
        projection_dim=4,
        start_queue_at_epoch=1,
        trainer=SimpleNamespace(current_epoch=epoch),
        log=Mock(),
    )
    batch = {"views": [{"image": torch.randn(2, 3)} for _ in range(2)]}
    result = forward.swav(model, batch, "fit")
    supplied = loss.call_args.args[-1]
    if epoch == 0:
        assert supplied is None
    else:
        torch.testing.assert_close(supplied, features)
        assert not supplied.requires_grad
        assert supplied.data_ptr() != features.data_ptr()
    assert result["swav_queue"].shape == (4, 4)
    assert not result["swav_queue"].requires_grad
    forward.swav(model, batch, "fit")
    factory.assert_called_once()


class _Forward:
    def __call__(self, module, batch, stage):
        return {
            "embedding": module.backbone(batch["image"]),
            "stage": stage,
            "index": batch["batch_idx"],
        }


@pytest.mark.parametrize(
    "method,stage",
    [
        ("validation_step", "validate"),
        ("test_step", "test"),
        ("predict_step", "predict"),
    ],
)
def test_callable_forward_adapter_receives_evaluation_stage_and_batch_index(
    method, stage
):
    model = Module(forward=_Forward(), backbone=nn.Linear(3, 2))
    batch = {"image": torch.ones(2, 3)}
    result = getattr(model, method)(batch, 7)
    assert result["stage"] == stage and result["index"] == 7
    torch.testing.assert_close(result["embedding"], model.backbone(batch["image"]))


class _Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(3) * 2)

    def forward(self, batch):
        return {"image": batch["image"] @ self.weight}


@pytest.mark.parametrize("stage", ["train", "val", "test", "predict"])
def test_parameterized_data_transform_resolves_for_every_stage(stage):
    transform = _Transform()
    model = Module(forward=_Forward(), backbone=nn.Linear(3, 2))
    model._trainer = SimpleNamespace(
        training=stage == "train",
        validating=stage == "val",
        sanity_checking=False,
        testing=stage == "test",
        predicting=stage == "predict",
        datamodule=SimpleNamespace(gpu_transform={stage: transform}),
    )
    result = model.on_after_batch_transfer({"image": torch.ones(2, 3)}, 0)
    torch.testing.assert_close(result["image"], torch.full((2, 3), 2.0))
    result["image"].sum().backward()
    assert transform.weight.grad is not None


@pytest.mark.parametrize("custom", [False, True])
def test_configure_model_passes_mesh_and_precision_policy(monkeypatch, custom):
    from stable_pretraining.utils import fsdp2

    shard = Mock()
    monkeypatch.setattr(fsdp2, "default_parallelize_fn", shard)
    model = Module(forward=_Forward(), parallelize_fn=shard if custom else None)
    mesh, policy = object(), object()
    model._device_mesh = mesh
    model._trainer = SimpleNamespace(strategy=SimpleNamespace(_spt_mp_policy=policy))
    model.configure_model()
    if custom:
        shard.assert_called_once_with(model, mesh)
    else:
        shard.assert_called_once_with(model, mesh, mp_policy=policy)


@pytest.mark.parametrize("configuration", [{"forward": 3}, {"forward": None}])
def test_missing_forward_configuration_is_explicit(configuration):
    if configuration["forward"] is None:
        with pytest.raises(NotImplementedError):
            Module(**configuration)({})
    else:
        with pytest.raises(ValueError, match="not callable"):
            Module(**configuration)


@pytest.mark.parametrize("state", ["buffer", "none"])
def test_data_transform_without_parameters_resolves_buffers_or_empty_state(state):
    transform = _Transform() if state == "buffer" else nn.Identity()
    if state == "buffer":
        weight = transform.weight.detach()
        del transform.weight
        transform.register_buffer("weight", weight)
    model = Module(forward=_Forward(), backbone=nn.Linear(3, 2))
    model._trainer = SimpleNamespace(
        training=True,
        validating=False,
        sanity_checking=False,
        testing=False,
        predicting=False,
        datamodule=SimpleNamespace(gpu_transform=transform),
    )
    result = model.on_after_batch_transfer({"image": torch.ones(2, 3)}, 0)
    torch.testing.assert_close(
        result["image"], torch.full((2, 3), 2.0 if state == "buffer" else 1.0)
    )
