"""Single-process unit tests for VIbCReg.

Covers :class:`IterNorm` (whitening quality, running statistics, eval-mode
behavior, dtype handling, gradients), :class:`VIbCRegLoss` (closed-form
values on collapsed and duplicated features, scale invariance of the
decorrelation term, coefficient weighting), and the :class:`VIbCReg` method
class and ``vibcreg`` forward function. Under a single process the cross-GPU
gather in the loss is a no-op.
"""

import math

import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.backbone import IterNorm
from stable_pretraining.losses import VIbCRegLoss


def _max_group_whitening_error(y: torch.Tensor, num_groups: int) -> float:
    """Largest deviation of any per-group covariance of ``y`` from identity."""
    n, d = y.shape
    yg = y.T.reshape(num_groups, d // num_groups, n)
    yc = yg - yg.mean(dim=-1, keepdim=True)
    cov = yc @ yc.transpose(1, 2) / n
    eye = torch.eye(d // num_groups, dtype=y.dtype)
    return (cov - eye).abs().max().item()


def _correlated(n: int, d: int, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return (torch.randn(n, d, generator=g) @ torch.randn(d, d, generator=g) + 3).to(
        dtype
    )


@pytest.mark.unit
class TestIterNorm:
    """Unit tests for the IterNorm whitening layer."""

    def test_whitens_each_group(self):
        """With enough iterations every group covariance converges to identity."""
        x = _correlated(512, 64, torch.float64)
        layer = IterNorm(64, num_groups=4, T=10, affine=False).double().train()

        assert _max_group_whitening_error(x, 4) > 10
        assert _max_group_whitening_error(layer(x), 4) < 1e-4

    def test_default_iterations_reduce_correlation(self):
        """The default T=5 is an approximate whitening, far better than the input."""
        x = _correlated(512, 64)
        y = IterNorm(64, num_groups=4, affine=False).train()(x)

        assert _max_group_whitening_error(y, 4) < 0.5
        assert y.mean(dim=0).abs().max() < 1e-4

    def test_running_statistics_update_only_in_train(self):
        """Train mode updates running buffers; eval mode leaves them untouched."""
        layer = IterNorm(16, num_groups=2)
        x = _correlated(64, 16)

        layer.train()
        layer(x)
        assert not torch.equal(layer.running_wm, torch.eye(8).repeat(2, 1, 1)), (
            "running whitening matrix should move away from identity"
        )

        mean, wm = layer.running_mean.clone(), layer.running_wm.clone()
        layer.eval()
        layer(x)
        assert torch.equal(layer.running_mean, mean)
        assert torch.equal(layer.running_wm, wm)

    def test_eval_is_per_sample(self):
        """In eval mode each output depends only on its own input row."""
        layer = IterNorm(16, num_groups=2).train()
        x = _correlated(64, 16)
        layer(x)
        layer.eval()

        assert torch.allclose(layer(x)[:5], layer(x[:5]), atol=1e-6)

    def test_gradcheck(self):
        """Autograd gradients through the Newton-Schulz iterations are correct."""
        torch.manual_seed(0)
        layer = IterNorm(8, num_groups=2, T=3).double().train()
        x = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(layer, (x,))

    def test_affine_parameters_receive_gradients(self):
        """The learnable scale and shift are part of the computation graph."""
        layer = IterNorm(16, num_groups=2).train()
        layer(_correlated(32, 16)).pow(2).sum().backward()

        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
        assert layer.bias.grad is not None and layer.bias.grad.abs().sum() > 0

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_low_precision_input_is_whitened_in_float32(self, dtype):
        """Half-precision inputs are upcast so the iterations stay finite."""
        y = IterNorm(16, num_groups=2).train()(_correlated(64, 16, dtype))

        assert y.dtype == torch.float32
        assert torch.isfinite(y).all()

    def test_float64_module(self):
        """A float64 module keeps float64 buffers and outputs."""
        layer = IterNorm(16, num_groups=2).double().train()
        y = layer(_correlated(64, 16, torch.float64))

        assert y.dtype == torch.float64
        assert layer.running_wm.dtype == torch.float64

    def test_rejects_indivisible_groups(self):
        with pytest.raises(ValueError, match="divisible"):
            IterNorm(10, num_groups=3)

    def test_rejects_wrong_input_shape(self):
        layer = IterNorm(16, num_groups=2)
        with pytest.raises(ValueError, match=r"\[N, 16\]"):
            layer(torch.randn(4, 8))
        with pytest.raises(ValueError, match=r"\[N, 16\]"):
            layer(torch.randn(4, 16, 1))


@pytest.mark.unit
class TestVIbCRegLoss:
    """Unit tests for the VIbCRegLoss function (single process)."""

    def test_collapsed_embeddings(self):
        """Constant embeddings only pay the variance hinge, once per view."""
        z = torch.ones(8, 4)

        loss = VIbCRegLoss()(z, z)

        expected = 25 * 2 * (1 - math.sqrt(1e-4))
        assert torch.allclose(loss, torch.tensor(expected))

    def test_duplicated_feature_decorrelation(self):
        """Two copies of one feature have correlation 1 in both off-diagonals."""
        a = torch.randn(32, 1)
        z = torch.cat([a, a], dim=1)

        # 2 off-diagonal entries equal to 1, averaged over 2 * 2 entries
        assert torch.allclose(VIbCRegLoss._decorrelation(z), torch.tensor(0.5))

    def test_decorrelation_is_scale_invariant(self):
        """Unlike VICReg's covariance term, rescaling features has no effect."""
        z = _correlated(256, 32)

        base = VIbCRegLoss._decorrelation(z)
        assert torch.allclose(VIbCRegLoss._decorrelation(z * 1e3 - 7), base, atol=1e-6)
        assert torch.allclose(VIbCRegLoss._decorrelation(z * 1e-3), base, atol=1e-6)

    def test_decorrelation_lower_for_independent_features(self):
        torch.manual_seed(0)
        independent = VIbCRegLoss._decorrelation(torch.randn(1024, 32))
        correlated = VIbCRegLoss._decorrelation(_correlated(1024, 32))

        assert independent < 0.01 < correlated

    def test_coefficients_weight_each_term(self):
        """The loss is the weighted sum of invariance, variance and decorrelation."""
        torch.manual_seed(0)
        z_i, z_j = torch.randn(64, 16), torch.randn(64, 16) * 0.5

        sim = VIbCRegLoss(sim_coeff=1, std_coeff=0, cov_coeff=0)(z_i, z_j)
        std = VIbCRegLoss(sim_coeff=0, std_coeff=1, cov_coeff=0)(z_i, z_j)
        cov = VIbCRegLoss(sim_coeff=0, std_coeff=0, cov_coeff=1)(z_i, z_j)

        assert torch.allclose(sim, torch.nn.functional.mse_loss(z_i, z_j))
        assert std > 0 and cov > 0
        assert torch.allclose(
            VIbCRegLoss()(z_i, z_j), 25 * sim + 25 * std + 200 * cov, rtol=1e-5
        )

    def test_gradients_flow_to_both_views(self):
        torch.manual_seed(0)
        z_i = torch.randn(32, 16, requires_grad=True)
        z_j = torch.randn(32, 16, requires_grad=True)

        loss = VIbCRegLoss()(z_i, z_j)
        loss.backward()

        assert loss.ndim == 0 and torch.isfinite(loss)
        assert z_i.grad.abs().sum() > 0 and z_j.grad.abs().sum() > 0

    def test_projector_with_iternorm_trains(self):
        """End-to-end: a projector ending in IterNorm reduces the loss."""
        torch.manual_seed(0)
        projector = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            IterNorm(64, num_groups=8),
        )
        loss_fn = VIbCRegLoss()
        opt = torch.optim.Adam(projector.parameters(), lr=1e-2)
        x = torch.randn(128, 32)

        losses = []
        for _ in range(30):
            loss = loss_fn(projector(x + 0.1 * torch.randn_like(x)), projector(x))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert all(math.isfinite(v) for v in losses)
        assert losses[-1] < losses[0]


class _TinyEncoder(torch.nn.Module):
    """Pooled-conv encoder so method tests run in milliseconds on CPU."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(2, 3))


@pytest.mark.unit
class TestVIbCRegMethod:
    """Unit tests for the VIbCReg method class and forward function."""

    def test_default_projector_matches_reference_recipe(self):
        model = spt.VIbCReg(encoder_name=_TinyEncoder())
        layers = list(model.projector)

        assert [type(m) for m in layers] == [
            torch.nn.Linear,
            torch.nn.BatchNorm1d,
            torch.nn.GELU,
            torch.nn.Linear,
            torch.nn.BatchNorm1d,
            torch.nn.GELU,
            torch.nn.Linear,
            IterNorm,
        ]
        assert layers[-1].num_features == 2048
        assert layers[-1].num_groups == 64 and layers[-1].T == 5
        assert (
            model.vibcreg_loss.sim_coeff,
            model.vibcreg_loss.std_coeff,
            model.vibcreg_loss.cov_coeff,
        ) == (25.0, 25.0, 200.0)

    def test_iternorm_can_be_disabled(self):
        model = spt.VIbCReg(
            encoder_name=_TinyEncoder(), projector_dims=(64, 64), iternorm=False
        )

        assert not any(isinstance(m, IterNorm) for m in model.projector)
        assert isinstance(model.projector[-1], torch.nn.Linear)

    @pytest.mark.parametrize("layout", ["views_list", "named_views"])
    def test_dict_forward_in_training(self, layout):
        """``forward(batch, stage)`` accepts both multi-view batch layouts."""
        torch.manual_seed(0)
        model = spt.VIbCReg(
            encoder_name=_TinyEncoder(), projector_dims=(64, 64), iternorm_groups=8
        )
        model.log = lambda *args, **kwargs: None
        model.train()
        views = [
            {"image": torch.randn(8, 3, 16, 16), "label": torch.arange(8)}
            for _ in range(2)
        ]
        if layout == "views_list":
            batch = {"views": views}
        else:
            batch = {"view_0": views[0], "view_1": views[1]}

        out = model(batch, stage="fit")
        out["loss"].backward()

        assert isinstance(out, dict)
        assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
        assert out["embedding"].shape == (16, 32)
        assert torch.equal(out["label"], torch.cat([torch.arange(8)] * 2))
        assert model.projector[-1].weight.grad is not None

    def test_dict_forward_in_eval(self):
        """Single-image eval batches return embeddings and labels, no loss."""
        model = spt.VIbCReg(
            encoder_name=_TinyEncoder(), projector_dims=(64, 64), iternorm_groups=8
        )
        model.eval()
        batch = {"image": torch.randn(8, 3, 16, 16), "label": torch.arange(8)}

        with torch.no_grad():
            out = model(batch, stage="validate")

        assert "loss" not in out
        assert out["embedding"].shape == (8, 32)
        assert torch.equal(out["label"], torch.arange(8))

    def test_forward_function_with_module(self):
        """The ``vibcreg`` forward fn trains end-to-end inside ``spt.Module``."""
        torch.manual_seed(0)
        module = spt.Module(
            forward=spt.forward.vibcreg,
            backbone=_TinyEncoder(),
            projector=torch.nn.Sequential(
                torch.nn.Linear(32, 64), IterNorm(64, num_groups=8)
            ),
            vibcreg_loss=VIbCRegLoss(),
        )
        module.log = lambda *args, **kwargs: None
        module.train()
        batch = {
            "views": [
                {"image": torch.randn(8, 3, 16, 16), "label": torch.arange(8)},
                {"image": torch.randn(8, 3, 16, 16), "label": torch.arange(8)},
            ]
        }

        out = module.forward(batch, "fit")
        out["loss"].backward()

        assert out["embedding"].shape == (16, 32)
        assert torch.equal(out["label"], torch.cat([torch.arange(8)] * 2))
        assert module.projector[1].weight.grad is not None

    def test_forward_function_rejects_wrong_view_count(self):
        module = spt.Module(
            forward=spt.forward.vibcreg,
            backbone=_TinyEncoder(),
            projector=torch.nn.Linear(32, 64),
            vibcreg_loss=VIbCRegLoss(),
        )
        batch = {"views": [{"image": torch.randn(4, 3, 16, 16)} for _ in range(3)]}

        with pytest.raises(ValueError, match="exactly 2 views"):
            module.forward(batch, "fit")
