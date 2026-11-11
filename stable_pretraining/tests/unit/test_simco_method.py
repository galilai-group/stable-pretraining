"""SimCo's composable Module contract and label-free representation learning."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

import stable_pretraining as spt

pytestmark = pytest.mark.unit


def _model() -> spt.Module:
    model = spt.SimCo(
        backbone=nn.Linear(4, 6),
        projector=nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3)),
        optim={"optimizer": {"type": "SGD", "lr": 0.01}},
    )
    model.log_dict = Mock()
    return model


def test_training_contract_and_gradients_do_not_depend_on_labels() -> None:
    torch.manual_seed(17)
    model = _model().train()
    views = [{"image": torch.randn(3, 4), "label": torch.arange(3)} for _ in range(2)]
    output = model({"views": views}, stage="fit")
    assert output["embedding"].shape == (6, 6)
    assert output["label"].tolist() == [0, 1, 2, 0, 1, 2]
    parameters = list(model.parameters())
    gradients = torch.autograd.grad(output["loss"], parameters)
    for view in views:
        view["label"] = torch.full((3,), 99)
    changed = model({"views": views}, stage="fit")
    changed_gradients = torch.autograd.grad(changed["loss"], parameters)
    torch.testing.assert_close(output["loss"], changed["loss"])
    for original, replacement in zip(gradients, changed_gradients):
        assert torch.isfinite(original).all()
        torch.testing.assert_close(original, replacement)
    assert any(gradient.norm() > 0 for gradient in gradients)


def test_evaluation_bypasses_projector_and_preserves_sample_metadata() -> None:
    model = _model().eval()
    model.projector.forward = Mock(
        side_effect=AssertionError("Projector used at evaluation")
    )
    image = torch.randn(3, 4)
    indices = torch.tensor([2, 0, 1])
    output = model(
        {"image": image, "label": torch.arange(3), "sample_idx": indices},
        stage="validate",
    )
    torch.testing.assert_close(output["embedding"], model.backbone(image))
    torch.testing.assert_close(output["sample_idx"], indices)
    assert "loss" not in output
    model.projector.forward.assert_not_called()


@pytest.mark.parametrize("key", ["label", "sample_idx"])
@pytest.mark.parametrize("both_none", [False, True])
def test_missing_or_none_metadata_is_omitted(key: str, both_none: bool) -> None:
    model = _model().train()
    views = [{"image": torch.randn(3, 4)} for _ in range(2)]
    views[0][key] = None
    if both_none:
        views[1][key] = None
    output = model({"views": views}, stage="fit")
    assert key not in output
    assert torch.isfinite(output["loss"])
    output["loss"].backward()


def test_evaluation_omits_none_metadata() -> None:
    output = _model().eval()(
        {"image": torch.randn(3, 4), "label": None, "sample_idx": None},
        stage="validate",
    )
    assert set(output) == {"embedding"}


def test_training_preserves_sample_indices_for_both_views() -> None:
    views = [
        {"image": torch.randn(3, 4), "sample_idx": torch.tensor([4, 2, 9])}
        for _ in range(2)
    ]
    output = _model().train()({"views": views}, stage="fit")
    assert output["sample_idx"].tolist() == [4, 2, 9, 4, 2, 9]


@pytest.mark.parametrize("key", ["label", "sample_idx"])
@pytest.mark.parametrize("populated_view", [0, 1])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_rejects_partially_populated_metadata(
    key: str, populated_view: int, explicit_none: bool
) -> None:
    views = [{"image": torch.randn(3, 4)} for _ in range(2)]
    views[populated_view][key] = torch.arange(3)
    if explicit_none:
        views[1 - populated_view][key] = None
    with pytest.raises(ValueError, match=key):
        _model()({"views": views}, stage="fit")


@pytest.mark.parametrize("key", ["label", "sample_idx"])
@pytest.mark.parametrize(
    "invalid",
    [torch.arange(2), torch.tensor(0), torch.zeros(3, 2), [0, 1, 2]],
    ids=["wrong_batch_size", "scalar", "mismatched_shape", "not_tensor"],
)
def test_rejects_inconsistent_metadata(key: str, invalid: object) -> None:
    views = [
        {"image": torch.randn(3, 4), key: torch.arange(3)},
        {"image": torch.randn(3, 4), key: invalid},
    ]
    with pytest.raises(ValueError, match=key):
        _model()({"views": views}, stage="fit")


def test_forward_function_and_method_class_agree() -> None:
    model = _model().train()
    batch = {"views": [{"image": torch.randn(3, 4)} for _ in range(2)]}
    actual = model(batch, stage="fit")
    expected = spt.forward.simco(model, batch, stage="fit")
    torch.testing.assert_close(actual["loss"], expected["loss"])


@pytest.mark.parametrize("view_count", [1, 3])
def test_rejects_wrong_view_count(view_count: int) -> None:
    with pytest.raises(ValueError, match="exactly 2 views"):
        _model()(
            {"views": [{"image": torch.randn(3, 4)} for _ in range(view_count)]},
            stage="fit",
        )


def test_backbone_and_projector_both_receive_default_optimizer_updates() -> None:
    torch.manual_seed(9)
    model = spt.SimCo(backbone=nn.Linear(4, 6), projector=nn.Linear(6, 5))
    model.log_dict = Mock()
    model._trainer = Mock(estimated_stepping_batches=2)
    optimizers, _ = model.configure_optimizers()
    optimizer = optimizers[0]
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {id(parameter) for parameter in model.parameters()}
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer.zero_grad()
    output = model(
        {"views": [{"image": torch.randn(3, 4)} for _ in range(2)]}, stage="fit"
    )
    output["loss"].backward()
    for component in (model.backbone, model.projector):
        gradients = [parameter.grad for parameter in component.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
        assert any(gradient.norm() > 0 for gradient in gradients)
    optimizer.step()
    for prefix, component in (
        ("backbone", model.backbone),
        ("projector", model.projector),
    ):
        assert any(
            not torch.equal(parameter.detach(), before[f"{prefix}.{name}"])
            for name, parameter in component.named_parameters()
        )
    assert not hasattr(model, "momentum_encoder")
    assert not hasattr(model, "queue")


def test_public_exports_and_temperature_metadata() -> None:
    from stable_pretraining.losses import SimCoLoss
    from stable_pretraining.methods import SimCo

    assert spt.SimCo is SimCo
    model = SimCo(
        backbone=nn.Linear(4, 6),
        projector=nn.Linear(6, 5),
        intra_temperature=0.2,
        inter_temperature=0.7,
        hparams={"seed": 3},
    )
    assert isinstance(model.simco_loss, SimCoLoss)
    assert model.simco_loss.intra_temperature == 0.2
    assert model.simco_loss.inter_temperature == 0.7
    assert model.hparams["method"] == "simco"
    assert model.hparams["seed"] == 3
