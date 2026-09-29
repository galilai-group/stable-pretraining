"""Numerical contracts for the invertible Jet backbone."""

from copy import deepcopy
import math

import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.backbone.jet import parameterize_scale

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny(**kwargs):
    return spt.backbone.Jet(
        **(
            dict(
                image_size=4,
                patch_size=2,
                in_channels=1,
                coupling_layers=2,
                hidden_dim=8,
                depth=1,
                num_heads=2,
            )
            | kwargs
        )
    )


def perturb(model):
    with torch.no_grad():
        for block in model.blocks:
            block.conditioner.final_proj.weight.normal_(std=0.05)
            block.conditioner.final_proj.bias.normal_(std=0.05)


@pytest.mark.parametrize("eps", [1e-4, 0.01, 0.5])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
def test_scale_identity_floor_growth(eps, dtype):
    raw = torch.tensor([-100.0, 0.0, 1.0, 2.0, 5.0], dtype=dtype)
    scale, log_scale = parameterize_scale(raw, eps)
    assert (
        scale.dtype
        == log_scale.dtype
        == (torch.float64 if dtype == torch.float64 else torch.float32)
    )
    assert torch.isfinite(scale).all() and torch.isfinite(log_scale).all()
    assert (scale >= eps).all()
    assert scale[1].item() == pytest.approx(1, abs=1e-7)
    assert log_scale[1].item() == pytest.approx(0, abs=1e-7)
    assert log_scale[0].item() == pytest.approx(math.log(eps), abs=1e-6)
    assert scale[2] < scale[3] < scale[4] and scale[4] > 2
    reference = eps + (1 - eps) * raw.double().exp()
    torch.testing.assert_close(scale.double(), reference, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        log_scale.double(), reference.log(), atol=1e-7, rtol=1e-6
    )


@pytest.mark.parametrize("mode", ["exp_floor", "jet_sigmoid"])
def test_scale_gradients(mode):
    raw = torch.linspace(-5, 5, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: parameterize_scale(x, parameterization=mode), (raw,)
    )


def test_warning_and_nonfinite():
    with pytest.warns(RuntimeWarning, match="no clamping"):
        scale, _ = parameterize_scale(torch.tensor([20.0]))
    assert scale > 1e8
    for value in (100.0, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(FloatingPointError):
            parameterize_scale(torch.tensor([value]))
    with pytest.raises(FloatingPointError, match="underflowed"):
        parameterize_scale(torch.tensor([-1000.0]), parameterization="jet_sigmoid")


@pytest.mark.parametrize("eps", [0, 1, -1, float("nan")])
def test_invalid_eps(eps):
    with pytest.raises(ValueError, match="scale_eps"):
        parameterize_scale(torch.zeros(1), eps)


@pytest.mark.parametrize("mode", ["exp_floor", "jet_sigmoid"])
@pytest.mark.parametrize("nonidentity", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_identity_inverse_and_diagnostics(mode, nonidentity, autocast):
    torch.manual_seed(4)
    model = tiny(scale_parameterization=mode, capture_stats=True)
    if nonidentity:
        perturb(model)
    x = torch.randn(2, 1, 4, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        tokens, ld = model(x)
        diagnostics = model.diagnostics
        reconstructed, inv_ld = model.inverse(tokens)
    assert tokens.dtype == ld.dtype == torch.float32
    torch.testing.assert_close(reconstructed, x, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(ld + inv_ld, torch.zeros_like(ld), atol=2e-5, rtol=0)
    if not nonidentity:
        torch.testing.assert_close(tokens, model.patchify(x), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(ld, torch.zeros_like(ld), atol=1e-6, rtol=0)
    for i in range(2):
        for key in (
            "log_scale_mean",
            "log_scale_std",
            "log_scale_min",
            "log_scale_max",
            "scale_mean",
            "scale_max",
        ):
            assert torch.isfinite(diagnostics[f"flow/layer_{i}/{key}"])
            assert not diagnostics[f"flow/layer_{i}/{key}"].requires_grad
    assert diagnostics["flow/logdet_per_dim_mean"] == (ld.detach() / 16).mean()


@pytest.mark.parametrize("mode", ["exp_floor", "jet_sigmoid"])
@pytest.mark.parametrize("kind", ["channel", "spatial"])
def test_autograd_jacobian(mode, kind):
    torch.manual_seed(5)
    model = tiny(
        coupling_types=(kind,), coupling_layers=1, scale_parameterization=mode
    ).double()
    perturb(model)
    x = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    _, ld = model(x)
    jac = torch.autograd.functional.jacobian(
        lambda flat: model(flat.reshape_as(x))[0].flatten(), x.flatten()
    )
    reference = torch.linalg.slogdet(jac).logabsdet
    torch.testing.assert_close(ld.squeeze(), reference, atol=1e-9, rtol=1e-9)


def test_checkpoint_recomputation_gradients_and_scale_metadata(tmp_path):
    torch.manual_seed(6)
    model = tiny().train()
    perturb(model)
    recomputed = tiny(checkpoint_conditioner=True)
    recomputed.load_state_dict(model.state_dict())
    x = torch.randn(2, 1, 4, 4)
    for encoder in (model, recomputed):
        z, ld = encoder(x)
        (z.square().mean() - ld.mean() / 16).backward()
    for a, b in zip(model.parameters(), recomputed.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
        assert torch.isfinite(a.grad).all()
    assert model.blocks[0].conditioner.final_proj.weight.grad.abs().sum() > 0
    checkpoint = tmp_path / "jet.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = tiny()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for a, b in zip(model(x), restored(x)):
        torch.testing.assert_close(a, b)
    for incompatible in (
        tiny(scale_parameterization="jet_sigmoid"),
        tiny(scale_eps=0.01),
    ):
        with pytest.raises(RuntimeError, match="configuration differs"):
            incompatible.load_state_dict(model.state_dict())


@pytest.mark.parametrize(
    "options",
    [
        {"image_size": 5},
        {"patch_size": 0},
        {"hidden_dim": 7},
        {"image_size": 6},
        {"coupling_types": ()},
        {"coupling_types": ("unknown",)},
        {"scale_parameterization": "bad"},
        {"scale_eps": 0},
        {"depth": 0},
        {"patch_size": 1},
        {"image_size": (4, 4, 4)},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        tiny(**options)


def test_shapes_and_nonfinite_transform():
    model = tiny()
    for x in (
        torch.ones(1, 4, 4, 1),
        torch.ones(1, 1, 4, 4, dtype=torch.int64),
        torch.ones(0, 1, 4, 4),
    ):
        with pytest.raises(ValueError):
            model(x)
    with pytest.raises(ValueError):
        model.inverse(torch.zeros(1, 2, 4))
    with pytest.raises(ValueError):
        model.unpatchify(torch.zeros(1, 2, 4))
    with pytest.raises(ValueError):
        parameterize_scale(torch.ones(1, dtype=torch.int64))
    with pytest.raises(ValueError):
        parameterize_scale(torch.zeros(1), parameterization="bad")
    with pytest.raises(FloatingPointError):
        model(torch.full((1, 1, 4, 4), float("nan")))
    broken = deepcopy(model)
    with torch.no_grad():
        broken.blocks[0].conditioner.final_proj.bias[0] = float("inf")
    with pytest.raises(FloatingPointError):
        broken(torch.ones(1, 1, 4, 4))


def test_rectangular_geometry_and_single_sample_stats():
    model = tiny(image_size=(4, 8), capture_stats=True, coupling_layers=4)
    x = torch.randn(1, 1, 4, 8)
    z, ld = model(x)
    reconstructed, inv = model.inverse(z)
    torch.testing.assert_close(reconstructed, x)
    torch.testing.assert_close(ld, -inv)
    assert model.diagnostics["flow/logdet_per_dim_std"] == 0
