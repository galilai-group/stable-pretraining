"""CorInfoMax numerical and interface contracts; no downloaded models or data."""

import copy
import importlib.util
import io
import math
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import stable_pretraining as spt
from stable_pretraining.forward import corinfomax
from stable_pretraining.losses import CorInfoMaxLoss


pytestmark = pytest.mark.unit


def _reference(views, history, normalize=True, sim_coeff=250.0):
    normalized = [
        z / z.norm(dim=1, keepdim=True).clamp_min(1e-12) if normalize else z
        for z in views
    ]
    next_history = []
    penalty = 0
    for z, (old_mean, old_covariance) in zip(normalized, history):
        mean = 0.01 * old_mean + 0.99 * z.mean(0).double()
        deviations = z.double() - mean
        moment = sum(torch.outer(row, row) for row in deviations) / len(z)
        covariance = 0.01 * old_covariance + 0.99 * moment
        sign, logabsdet = torch.linalg.slogdet(
            covariance + 1e-8 * torch.eye(z.shape[1], dtype=torch.float64)
        )
        assert sign == 1
        penalty -= logabsdet / z.shape[1]
        next_history.append((mean.detach(), covariance.detach()))
    agreement = (normalized[0] - normalized[1]).square().mean()
    return sim_coeff * agreement + penalty, next_history


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalize", [False, True])
def test_reference_values_gradients_and_history(
    dtype: torch.dtype, normalize: bool
) -> None:
    """Match an outer-product reference over multiple independent batch graphs."""
    generator = torch.Generator().manual_seed(12)
    objective = CorInfoMaxLoss(4, normalize=normalize)
    history = [(torch.zeros(4).double(), torch.eye(4).double()) for _ in range(2)]
    for _ in range(3):
        views = [
            torch.randn(7, 4, generator=generator, dtype=dtype).requires_grad_()
            for _ in range(2)
        ]
        reference_views = [z.detach().clone().requires_grad_() for z in views]
        expected, history = _reference(reference_views, history, normalize)
        actual = objective(*views)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        gradients = torch.autograd.grad(actual, views)
        expected_gradients = torch.autograd.grad(expected, reference_views)
        for gradient, expected_gradient in zip(gradients, expected_gradients):
            torch.testing.assert_close(
                gradient, expected_gradient, rtol=2e-5, atol=2e-6
            )
        for index, (mean, covariance) in enumerate(history, start=1):
            torch.testing.assert_close(getattr(objective, f"mean_{index}"), mean)
            torch.testing.assert_close(
                getattr(objective, f"covariance_{index}"), covariance
            )
        assert all(buffer.grad_fn is None for buffer in objective.buffers())


def test_covariance_term_alone_reaches_both_views() -> None:
    """The state update must not detach the covariance learning signal."""
    objective = CorInfoMaxLoss(3, sim_coeff=0, normalize=False)
    views = [torch.randn(8, 3, requires_grad=True) for _ in range(2)]
    objective(*views).backward()
    for view in views:
        assert torch.isfinite(view.grad).all()
        assert view.grad.abs().sum() > 0


def test_checkpoint_continuation_and_eval_state() -> None:
    """Serialization preserves the next loss and evaluation does not write history."""
    objective = CorInfoMaxLoss(3)
    objective(torch.randn(5, 3), torch.randn(5, 3))
    stream = io.BytesIO()
    torch.save(objective.state_dict(), stream)
    stream.seek(0)
    restored = CorInfoMaxLoss(3)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    views = [torch.randn(5, 3) for _ in range(2)]
    torch.testing.assert_close(restored(*views), objective(*views))
    before = copy.deepcopy(restored.state_dict())
    restored.eval()
    restored(*views)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"projection_dim": 0},
        {"epsilon": 0},
        {"epsilon": float("nan")},
        {"initial_covariance": -1},
        {"mean_momentum": 1},
        {"covariance_momentum": -0.1},
        {"sim_coeff": -1},
        {"statistics_dtype": torch.float16},
    ],
)
def test_invalid_parameters(kwargs: dict) -> None:
    """Invalid objective parameters fail before allocating training state."""
    with pytest.raises(ValueError):
        CorInfoMaxLoss(**{"projection_dim": 3, **kwargs})


@pytest.mark.parametrize(
    "shapes",
    [((0, 3), (0, 3)), ((4, 2), (4, 2)), ((4, 3), (5, 3)), ((4, 3, 1), (4, 3, 1))],
)
def test_invalid_projection_shapes(shapes: tuple) -> None:
    """Malformed batches cannot modify history."""
    objective = CorInfoMaxLoss(3)
    before = copy.deepcopy(objective.state_dict())
    with pytest.raises(ValueError):
        objective(*(torch.randn(shape) for shape in shapes))
    for key, value in objective.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_precision_and_normalization_contract() -> None:
    """AMP preserves float64 state, and explicit float/half conversions are clear."""
    views = [torch.randn(8, 4) for _ in range(2)]
    reference = CorInfoMaxLoss(4)
    mixed = copy.deepcopy(reference)
    expected = reference(*views)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = mixed(*views)
    torch.testing.assert_close(actual, expected)
    assert mixed.covariance_1.dtype == torch.float64
    low_precision = [view.bfloat16() for view in views]
    torch.testing.assert_close(
        CorInfoMaxLoss(4)(*low_precision),
        CorInfoMaxLoss(4)(*(view.float() for view in low_precision)),
    )
    reduced = CorInfoMaxLoss(4).float()
    assert reduced(*views).dtype == torch.float32
    with pytest.raises(ValueError, match="history"):
        CorInfoMaxLoss(4).half()(*(view.half() for view in views))
    explicit = CorInfoMaxLoss(4, normalize=False)
    normalized = [nn.functional.normalize(view, dim=-1) for view in views]
    torch.testing.assert_close(explicit(*normalized), CorInfoMaxLoss(4)(*views))


def _model(width=5, projection=3):
    return spt.CorInfoMax(
        encoder_name=nn.Sequential(nn.Linear(6, width), nn.BatchNorm1d(width)),
        embed_dim=width,
        projector_dims=(9, projection),
    )


def test_constructor_does_not_execute_backbone() -> None:
    """Metadata/explicit dimensions avoid mutating BatchNorm during construction."""
    backbone = nn.Sequential(nn.Linear(6, 5), nn.BatchNorm1d(5))
    before = copy.deepcopy(backbone.state_dict())
    spt.CorInfoMax(encoder_name=backbone, embed_dim=5, projector_dims=(3,))
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="embed_dim"):
        spt.CorInfoMax(encoder_name=backbone)


@pytest.mark.parametrize("layout", ["list", "tuple", "nested_named", "named"])
def test_tensor_dict_and_composable_apis_agree(layout: str) -> None:
    """All supported view layouts preserve both the objective and label order."""
    tensor_model = _model()
    dict_model = copy.deepcopy(tensor_model)
    composed = spt.Module(
        forward=corinfomax,
        backbone=copy.deepcopy(tensor_model.backbone),
        projector=copy.deepcopy(tensor_model.projector),
        corinfomax_loss=copy.deepcopy(tensor_model.corinfomax_loss),
    )
    images = [torch.randn(4, 6) for _ in range(2)]
    labels = [torch.arange(4), torch.arange(4) + 4]
    views = [dict(image=image, label=label) for image, label in zip(images, labels)]
    named = {"first": views[0], "second": views[1]}
    batch = {
        "list": {"views": views},
        "tuple": {"views": tuple(views)},
        "nested_named": {"views": named},
        "named": {**named, "batch_idx": 0},
    }[layout]
    expected = tensor_model(*images)
    actual = dict_model(batch, "fit")
    reusable = composed(batch, stage="fit")
    assert isinstance(actual, dict)
    for key in ("loss", "embedding", "projection"):
        torch.testing.assert_close(actual[key], expected[key])
        torch.testing.assert_close(reusable[key], expected[key])
    torch.testing.assert_close(actual.label, torch.cat(labels))


def test_evaluation_bypasses_projector_and_preserves_history() -> None:
    """Paired evaluation is feature-only and cannot advance running statistics."""
    model = _model()
    images = [torch.randn(4, 6) for _ in range(2)]
    model(*images)
    before = copy.deepcopy(model.corinfomax_loss.state_dict())
    model.eval()
    handle = model.projector.register_forward_hook(
        lambda *args: pytest.fail("Evaluation called the projector")
    )
    try:
        assert model(*images).embedding.shape == (8, 5)
        assert model(images[0]).embedding.shape == (4, 5)
        assert model(images[0]).loss == 0
    finally:
        handle.remove()
    for key, value in model.corinfomax_loss.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("width,projection,size", [(7, 2, 16), (11, 5, 28)])
def test_other_feature_widths_and_image_sizes(
    width: int, projection: int, size: int
) -> None:
    """The method depends on feature dimensions, not CIFAR or ResNet conventions."""
    backbone = nn.Sequential(
        nn.Conv2d(1, width, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )
    model = spt.CorInfoMax(
        encoder_name=backbone, embed_dim=width, projector_dims=(projection,)
    )
    out = model(torch.randn(3, 1, size, size), torch.randn(3, 1, size, size))
    assert out.embedding.shape == (6, width)
    assert out.projection.shape == (6, projection)
    out.loss.backward()
    assert backbone[0].weight.grad.abs().sum() > 0


def test_online_probe_positional_stage_and_gradient_separation() -> None:
    """OnlineProbe can wrap the method and train its classifier on detached features."""
    model = _model()
    model.log_dict = Mock()
    probe = spt.OnlineProbe(
        model,
        name="probe",
        input="embedding",
        target="label",
        probe=nn.Linear(5, 2),
        loss=nn.CrossEntropyLoss(),
        metrics={},
    )
    model.configure_model()
    batch = {
        "views": [
            {"image": torch.randn(4, 6), "label": torch.tensor([0, 1, 0, 1])}
            for _ in range(2)
        ]
    }
    out = model(batch, stage="fit")
    probe_loss = nn.functional.cross_entropy(out["probe_preds"], out["label"])
    encoder_gradients = torch.autograd.grad(
        probe_loss,
        tuple(model.backbone.parameters()),
        allow_unused=True,
        retain_graph=True,
    )
    assert all(gradient is None for gradient in encoder_gradients)
    out["loss"].backward()
    assert out["probe_preds"].shape == (8, 2)
    assert probe.module.weight.grad.abs().sum() > 0
    assert model.backbone[0].weight.grad.abs().sum() > 0


def test_bad_training_views_and_labels_fail() -> None:
    """Invalid batches fail before silently producing an embedding-only loss."""
    model = _model()
    image = torch.randn(4, 6)
    with pytest.raises(ValueError, match="two views"):
        model({"image": image}, stage="fit")
    with pytest.raises(ValueError, match="Labels"):
        model({"views": [{"image": image, "label": torch.zeros(4)}, {"image": image}]})


@pytest.fixture(scope="module")
def recipe_module() -> ModuleType:
    """Load recipe helpers without invoking main or accessing datasets."""
    path = Path(__file__).resolve().parents[3] / "examples" / "corinfomax_cifar10.py"
    spec = importlib.util.spec_from_file_location("corinfomax_recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_epoch_schedule_boundaries_and_resume(recipe_module: ModuleType) -> None:
    """LR stays constant within epochs, preserves warmup boundaries, and resumes."""
    recipe = dict(
        steps_per_epoch=3,
        epochs=1000,
        peak_lr=0.5,
        min_lr=1e-6,
        warmup_epochs=10,
        warmup_start_lr=0.003,
    )
    factor = recipe_module._epoch_lr_factor
    assert factor(0, **recipe) * 0.5 == pytest.approx(0.003)
    assert factor(2, **recipe) == factor(0, **recipe)
    assert factor(27, **recipe) == pytest.approx(1)
    assert factor(30, **recipe) == pytest.approx(1)
    assert factor(3000, **recipe) * 0.5 == pytest.approx(1e-6)
    assert factor(33, **recipe) * 0.5 == pytest.approx(
        1e-6 + 0.5 * (0.5 - 1e-6) * (1 + math.cos(math.pi / 990))
    )
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(()))], lr=0.5)
    scheduler = recipe_module._scheduler(recipe)(optimizer)
    for _ in range(32):
        optimizer.step()
        scheduler.step()
    restored_optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(()))], lr=0.5)
    restored_scheduler = recipe_module._scheduler(recipe)(restored_optimizer)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())
    for opt, sched in (
        (optimizer, scheduler),
        (restored_optimizer, restored_scheduler),
    ):
        opt.step()
        sched.step()
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_linear_checkpoint_component_loading(recipe_module: ModuleType) -> None:
    """Loading the probe before callback registration preserves its saved weights."""
    backbone = nn.Linear(6, 5)
    classifier = nn.Linear(5, 2)
    checkpoint = {
        "state_dict": {
            **{
                f"backbone.backbone.{key}": value
                for key, value in backbone.state_dict().items()
            },
            **{
                f"callbacks_modules.linear_probe.{key}": value
                for key, value in classifier.state_dict().items()
            },
        }
    }
    loaded_backbone, loaded_classifier = nn.Linear(6, 5), nn.Linear(5, 2)
    recipe_module._load_component(loaded_backbone, checkpoint, "backbone.backbone.")
    recipe_module._load_component(
        loaded_classifier, checkpoint, "callbacks_modules.linear_probe."
    )
    model = spt.Module(backbone=spt.backbone.EvalOnly(loaded_backbone), optim=None)
    probe = spt.OnlineProbe(
        model,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=loaded_classifier,
        metrics={},
    )
    model.configure_model()
    model.train()
    assert not model.backbone.training
    assert not loaded_backbone.training
    assert not any(
        parameter.requires_grad for parameter in loaded_backbone.parameters()
    )
    torch.testing.assert_close(probe.module.weight, classifier.weight)
