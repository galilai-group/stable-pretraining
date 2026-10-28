"""Formula, gradient, pooling and API tests for LeVLJEPA (offline, CPU)."""

import copy
from types import SimpleNamespace

import pytest
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from stable_pretraining.losses import LeVLJEPALoss
from torch import nn

pytestmark = pytest.mark.unit


class TinyText(nn.Module):
    """Token encoder exposing the HuggingFace output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 5)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def make_model(**kwargs) -> spt.Module:
    """Build an injectable encoder configuration without model downloads."""
    return spt.LeVLJEPA(
        vision_encoder=nn.Linear(4, 6),
        text_encoder=TinyText(),
        vision_dim=6,
        text_dim=5,
        embedding_dim=8,
        pre_projection_dims=(),
        predictor_dims=(),
        num_projections=16,
        **kwargs,
    )


def make_batch() -> dict[str, torch.Tensor]:
    """Build nonidentical image/text samples with right and left padding."""
    return {
        "image": torch.randn(4, 4),
        "input_ids": torch.tensor(
            [[1, 2, 3, 0], [0, 1, 2, 3], [1, 0, 0, 0], [2, 3, 4, 5]]
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1]]
        ),
    }


def reference_sigreg(x: torch.Tensor, seed: int, projections: int = 16) -> torch.Tensor:
    """Independent literal reference calculation of the sliced EP objective."""
    g = torch.Generator().manual_seed(seed)
    directions = F.normalize(torch.randn(x.shape[1], projections, generator=g), dim=0)
    t = torch.linspace(0, 3, 17)
    phi = (-t.square() / 2).exp()
    phase = (x @ directions).unsqueeze(-1) * t
    error = (phase.cos().mean(0) - phi).square() + phase.sin().mean(0).square()
    # Trapezoidal integral includes the two-sided characteristic-function factor.
    return 2 * torch.trapezoid(error * phi, t, dim=-1).mean() * x.shape[0]


@pytest.mark.parametrize("alignment", ["mse", "cosine"])
def test_reference_formula_and_gradients(alignment):
    torch.manual_seed(8)
    tensors = [torch.randn(7, d, requires_grad=True) for d in (4, 5, 5, 4)]
    image, text, ip, tp = tensors
    criterion = LeVLJEPALoss(
        lambda_vision=0.1,
        lambda_text=0.2,
        alignment=alignment,
        num_projections=16,
        seed=42,
    )
    result = criterion(*tensors)
    if alignment == "mse":
        prediction = (
            F.mse_loss(ip, text.detach()) + F.mse_loss(tp, image.detach())
        ) / 2
    else:
        prediction = (
            (F.normalize(ip, dim=-1) - F.normalize(text.detach(), dim=-1))
            .square()
            .sum(-1)
            .mean()
            + (F.normalize(tp, dim=-1) - F.normalize(image.detach(), dim=-1))
            .square()
            .sum(-1)
            .mean()
        ) / 2
    expected = (
        0.7 * prediction
        + 0.1 * reference_sigreg(image, 42)
        + 0.2 * reference_sigreg(text, 43)
    )
    torch.testing.assert_close(result["loss"], expected)
    actual_grad = torch.autograd.grad(result["loss"], tensors, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, tensors)
    for actual, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected_value)


def test_stop_gradient_targets_but_regularizer_trains_encoders():
    tensors = [torch.randn(4, 8, requires_grad=True) for _ in range(4)]
    result = LeVLJEPALoss(num_projections=16)(*tensors)
    result["prediction_loss"].backward(retain_graph=True)
    assert tensors[0].grad is None and tensors[1].grad is None
    assert tensors[2].grad.abs().sum() > 0 and tensors[3].grad.abs().sum() > 0
    (result["sigreg_vision"] + result["sigreg_text"]).backward()
    assert tensors[0].grad.abs().sum() > 0 and tensors[1].grad.abs().sum() > 0


def test_dict_api_pooling_and_all_modules_receive_gradients():
    model, batch = make_model(), make_batch()
    result = model(batch, stage="fit")
    assert "loss" not in batch
    assert result["embedding"].shape == (4, 6)
    assert result["image_prediction"].shape == (4, 8)
    expected_raw = model.text_encoder.embedding(torch.tensor([3, 3, 1, 5]))
    torch.testing.assert_close(
        result["text_embedding"], model.text_pre_proj(expected_raw)
    )
    result["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_checkpoint_reproduces_next_random_slices_and_eval_does_not_advance():
    model, batch = make_model(), make_batch()
    model(batch, stage="fit")
    saved = copy.deepcopy(model.state_dict())
    expected = model(batch, stage="fit")["loss"]
    restored = make_model()
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored(batch, stage="fit")["loss"], expected)
    restored.eval()
    before = restored.levljepa_loss.vision_regularizer.step.clone()
    torch.testing.assert_close(
        restored(batch, stage="validate")["loss"], restored(batch, stage="test")["loss"]
    )
    assert torch.equal(before, restored.levljepa_loss.vision_regularizer.step)
    assert "loss" not in restored(batch, stage="predict")
    assert restored({"image": batch["image"]}, stage="predict")["embedding"].shape == (
        4,
        6,
    )


@pytest.mark.parametrize("value", [0.0, 100.0])
def test_collapsed_bfloat16_embeddings_remain_finite(value):
    tensors = [
        torch.full((8, 16), value, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(4)
    ]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = LeVLJEPALoss(num_projections=16)(*tensors)
    assert result["loss"].dtype == torch.float32
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(torch.isfinite(t.grad).all() for t in tensors)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_vision": -0.1},
        {"lambda_text": float("nan")},
        {"lambda_vision": 0.8, "lambda_text": 0.2},
        {"alignment": "other"},
        {"num_projections": 0},
        {"num_knots": 2},
    ],
)
def test_invalid_loss_configuration(kwargs):
    with pytest.raises(ValueError):
        LeVLJEPALoss(**kwargs)


def test_bad_stage_shapes_and_empty_text_fail_clearly():
    model, batch = make_model(), make_batch()
    with pytest.raises(ValueError, match="stage"):
        model(batch, stage="typo")
    with pytest.raises(ValueError, match="paired"):
        model({"image": batch["image"]}, stage="fit")
    batch["attention_mask"][0] = 0
    with pytest.raises(ValueError, match="unmasked"):
        model(batch, stage="fit")
    with pytest.raises(ValueError, match="shapes"):
        LeVLJEPALoss(num_projections=16)(
            torch.randn(4, 3), torch.randn(4, 5), torch.randn(4, 3), torch.randn(4, 3)
        )


def test_composable_forward_matches_method():
    model, batch = make_model().eval(), make_batch()
    components = {
        name: copy.deepcopy(getattr(model, name))
        for name in (
            "vision_encoder",
            "text_encoder",
            "vision_pre_proj",
            "text_pre_proj",
            "projector_vision",
            "projector_text",
            "levljepa_loss",
        )
    }
    composed = spt.Module(forward=spt.forward.levljepa, **components).eval()
    torch.testing.assert_close(
        model(batch, stage="validate")["loss"],
        composed(batch, stage="validate")["loss"],
    )


def test_normal_distribution_has_lower_sigreg_than_collapse():
    torch.manual_seed(73)
    normal = torch.randn(512, 32)
    collapsed = torch.zeros_like(normal)
    criterion = LeVLJEPALoss(num_projections=64).eval()
    normal_stat = criterion.vision_regularizer(normal)
    collapsed_stat = criterion.vision_regularizer(collapsed)
    assert normal_stat < collapsed_stat / 10


def test_batchnorm_heads_under_bfloat16_autocast():
    model = spt.LeVLJEPA(
        vision_encoder=nn.Linear(4, 6),
        text_encoder=TinyText(),
        vision_dim=6,
        text_dim=5,
        embedding_dim=8,
        pre_projection_dims=(16,),
        predictor_dims=(16, 16),
        num_projections=16,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(make_batch(), stage="fit")
    assert output["loss"].dtype == torch.float32
    output["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_example_evaluation_restores_training_mode():
    from examples.levljepa_flickr8k import evaluate

    class Rows(list):
        captions = ["one", "two", "three", "four"]

    batch = make_batch()
    rows = Rows([{key: value[i] for key, value in batch.items()} for i in range(4)])
    loader = torch.utils.data.DataLoader(rows, batch_size=4)
    model = make_model().train()
    evaluate(model, loader, torch.device("cpu"))
    assert model.training and model.levljepa_loss.vision_regularizer.training
    model(batch, stage="fit")
    assert int(model.levljepa_loss.vision_regularizer.step) == 1
    model.eval()
    evaluate(model, loader, torch.device("cpu"))
    assert not model.training


@pytest.mark.parametrize(
    "hidden_dims, linear_indices, batchnorm_indices",
    [((), [0], []), ((7,), [0, 3], [1]), ((7, 9), [0, 3, 6], [1, 4])],
)
def test_framework_mlp_preserves_legacy_checkpoint_keys(
    hidden_dims, linear_indices, batchnorm_indices
):
    """Shared MLP construction must keep already-trained head state names."""
    from stable_pretraining.methods.levljepa import _head

    head = _head(4, hidden_dims, 8)
    expected = {
        f"{index}.{name}" for index in linear_indices for name in ("weight", "bias")
    }
    expected.update(
        f"{index}.{name}"
        for index in batchnorm_indices
        for name in (
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        )
    )
    assert set(head.state_dict()) == expected


@pytest.mark.parametrize("saved_warmup", [False, True])
def test_example_rejects_incompatible_resume_scheduler(
    tmp_path, monkeypatch, capsys, saved_warmup
):
    """Reject a scheduler-family mismatch before any dataset/tokenizer downloads."""
    import sys

    from examples import levljepa_flickr8k as example

    checkpoint = tmp_path / "resume.ckpt"
    torch.save(
        {"lr_schedulers": [{"warmup_steps": 2} if saved_warmup else {"T_max": 6}]},
        checkpoint,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "levljepa_flickr8k.py",
            "--resume",
            str(checkpoint),
            "--warmup-fraction",
            "0" if saved_warmup else "0.05",
        ],
    )

    def unexpected_download(*args, **kwargs):
        pytest.fail("scheduler mismatch must be rejected before tokenizer loading")

    monkeypatch.setattr(example.AutoTokenizer, "from_pretrained", unexpected_download)
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2
    assert "checkpoint's scheduler family" in capsys.readouterr().err


def test_sigreg_pairwise_kernel_value_and_analytic_gradient():
    """Check the EP expansion without forming empirical characteristic means."""
    from stable_pretraining.losses.levljepa import _SlicedGaussianRegularizer

    values = torch.tensor([-2.0, -0.4, 0.25, 1.6], requires_grad=True)
    regularizer = _SlicedGaussianRegularizer(7, 17, False, 19).eval()
    result = regularizer(values[:, None])
    # In one dimension each unit direction is +/-1; the statistic is even.
    x = values.detach().double()
    t = torch.arange(17, dtype=torch.float64) * (3 / 16)
    phi = torch.exp(-t.square() / 2)
    weights = torch.full_like(t, 2 * 3 / 16)
    weights[[0, -1]] /= 2
    weights *= phi
    pair_phase = (x[:, None, None] - x[None, :, None]) * t
    phase = x[:, None] * t
    expected = (
        weights
        * (
            pair_phase.cos().sum((0, 1)) / len(x)
            - 2 * phi * phase.cos().sum(0)
            + len(x) * phi.square()
        )
    ).sum()
    derivative = (
        weights
        * (-2 / len(x) * t * pair_phase.sin().sum(1) + 2 * phi * t * phase.sin())
    ).sum(-1)
    torch.testing.assert_close(result.double(), expected, rtol=2e-6, atol=2e-6)
    result.backward()
    torch.testing.assert_close(values.grad.double(), derivative, rtol=2e-6, atol=2e-6)


def test_example_identical_captions_do_not_merge_distinct_image_positives():
    from examples.levljepa_flickr8k import evaluate

    class Rows(list):
        captions = ["same caption", "same caption"]

    class SwappedModel(nn.Module):
        retrieval_mode = "embedding"

        def forward(self, batch, stage):
            image = batch["image"]
            text = image.flip(-1)
            return {
                "image_embedding": image,
                "text_embedding": text,
                "image_prediction": image,
                "text_prediction": text,
            }

    loader = torch.utils.data.DataLoader(
        Rows(
            [{"image": torch.tensor([1.0, 0.0])}, {"image": torch.tensor([0.0, 1.0])}]
        ),
        batch_size=2,
    )
    metrics = evaluate(SwappedModel(), loader, torch.device("cpu"))
    assert metrics["image_to_text_r1"] == metrics["text_to_image_r1"] == 0
    assert metrics["random_r1"] == 0.5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_explicit_low_precision_module_keeps_fp32_quadrature(dtype):
    """Upcasting already-rounded phi/weights does not restore fp32 accuracy."""
    torch.manual_seed(57)
    tensors = [torch.randn(9, 5, requires_grad=True) for _ in range(4)]
    criterion = LeVLJEPALoss(num_projections=16).eval()
    converted = copy.deepcopy(criterion).to(dtype=dtype)
    reference = criterion(*tensors)["loss"]
    actual = converted(*tensors)["loss"]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for observed, expected in zip(
        torch.autograd.grad(actual, tensors), torch.autograd.grad(reference, tensors)
    ):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [3, 7])
def test_prediction_reduction_is_elementwise_mean_and_directional_average(width):
    """Fix the executable convention explicitly; do not silently use summed MSE."""
    image = torch.zeros(4, width)
    text = torch.zeros_like(image)
    image_prediction = torch.ones_like(image, requires_grad=True)
    text_prediction = torch.zeros_like(image, requires_grad=True)
    result = LeVLJEPALoss(num_projections=4)(
        image, text, image_prediction, text_prediction
    )
    assert result["prediction_loss"].item() == 0.5
    result["prediction_loss"].backward()
    torch.testing.assert_close(
        image_prediction.grad, torch.full_like(image, 1 / (4 * width))
    )
    torch.testing.assert_close(text_prediction.grad, torch.zeros_like(image))


@pytest.mark.parametrize("probability", [-0.1, 1.0, float("nan")])
def test_predictor_dropout_rejects_invalid_probability(probability):
    with pytest.raises(ValueError, match="predictor_dropout"):
        make_model(predictor_dropout=probability)


def test_predictor_dropout_matches_reference_and_preserves_checkpoint_keys():
    from stable_pretraining.methods.levljepa import _head
    from torchvision.ops import MLP

    torch.manual_seed(42)
    baseline = _head(8, (16, 16), 8)
    actual = _head(8, (16, 16), 8, dropout=0.1)
    actual.load_state_dict(baseline.state_dict(), strict=True)
    assert list(actual.state_dict()) == list(baseline.state_dict())
    assert not any(isinstance(layer, nn.Dropout) for layer in baseline)
    reference = MLP(
        8, [16, 16, 8], norm_layer=nn.BatchNorm1d, activation_layer=nn.GELU, dropout=0.1
    )
    for target, source in zip(reference.parameters(), actual.parameters(), strict=True):
        target.data.copy_(source.data)
    x = torch.randn(12, 8)
    for training in [True, False]:
        actual.train(training)
        reference.train(training)
        u = x.clone().requires_grad_()
        v = x.clone().requires_grad_()
        torch.manual_seed(7)
        y = actual(u)
        torch.manual_seed(7)
        expected = reference(v)
        torch.testing.assert_close(y, expected)
        y.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(u.grad, v.grad)


def test_predictor_dropout_only_changes_predictors():
    model = make_model(predictor_dropout=0.1)
    assert not any(
        isinstance(layer, nn.Dropout) for layer in model.vision_pre_proj.modules()
    )
    assert not any(
        isinstance(layer, nn.Dropout) for layer in model.text_pre_proj.modules()
    )
    for head in [model.projector_vision, model.projector_text]:
        assert any(
            isinstance(layer, nn.Dropout) and layer.p == 0.1 for layer in head.modules()
        )
    model.eval()
    model(make_batch(), "predict")["image_prediction"].sum().backward()
    assert model.vision_encoder.weight.grad is not None
