import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses import MECLoss


def _reference_trace(p, z, lambda_inv, order=4):
    """Trace term from the official MEC implementation (xinliu20/MEC)."""
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    c = p @ z.T / lambda_inv
    power = c
    sum_matrix = torch.zeros_like(c)
    for k in range(order):
        if k > 0:
            power = power @ c
        sum_matrix = sum_matrix + (-1) ** k / (k + 1) * power
    return torch.trace(sum_matrix)


@pytest.mark.unit
class TestMECLoss:
    """Unit tests for MECLoss."""

    def test_initialization(self):
        loss_fn = MECLoss(eps=32.0, order=3)
        assert loss_fn.eps == 32.0
        assert loss_fn.order == 3
        assert loss_fn.symmetric is False

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            MECLoss(order=0)

    def test_lambda_inv(self):
        assert MECLoss(eps=64.0).lambda_inv(batch_size=1024, dim=2048) == 32.0

    def test_matches_reference_implementation(self):
        torch.manual_seed(0)
        p1, p2, z1, z2 = (torch.randn(16, 8) for _ in range(4))
        loss_fn = MECLoss()
        lambda_inv = loss_fn.lambda_inv(16, 8)

        ours = (loss_fn(p1, z2) + loss_fn(p2, z1)) / 2
        mec = (
            (
                _reference_trace(p1, z2, lambda_inv)
                + _reference_trace(p2, z1, lambda_inv)
            )
            * 0.5
            / 16
        )
        assert torch.allclose(ours, -mec * lambda_inv)

    def test_approximates_log_det(self):
        """With enough terms the series converges to the exact log-determinant."""
        torch.manual_seed(0)
        m, d = 16, 8
        z = torch.randn(m, d)
        loss_fn = MECLoss(order=10)
        lambda_inv = loss_fn.lambda_inv(m, d)

        zn = F.normalize(z, dim=1)
        exact = -lambda_inv * torch.logdet(torch.eye(m) + zn @ zn.T / lambda_inv) / m
        assert torch.allclose(loss_fn(z, z), exact, atol=1e-4)

    def test_symmetric_in_arguments(self):
        torch.manual_seed(0)
        p, z = torch.randn(8, 4), torch.randn(8, 4)
        loss_fn = MECLoss()
        assert torch.allclose(loss_fn(p, z), loss_fn(z, p))

    def test_invariant_to_input_scale(self):
        torch.manual_seed(0)
        p, z = torch.randn(8, 4), torch.randn(8, 4)
        loss_fn = MECLoss()
        assert torch.allclose(loss_fn(p, z), loss_fn(3.0 * p, 0.5 * z))

    def test_lambda_inv_scale(self):
        """Scaling 1/λ is the same as running with a proportionally larger ε."""
        torch.manual_seed(0)
        p, z = torch.randn(8, 4), torch.randn(8, 4)
        assert torch.allclose(
            MECLoss(eps=64.0)(p, z, lambda_inv_scale=8.0), MECLoss(eps=512.0)(p, z)
        )

    def test_spread_beats_collapsed(self):
        """Diverse embeddings have higher entropy, hence lower loss."""
        torch.manual_seed(0)
        m, d = 16, 8
        spread = torch.randn(m, d)
        collapsed = torch.randn(1, d).repeat(m, 1)
        loss_fn = MECLoss()
        assert loss_fn(spread, spread) < loss_fn(collapsed, collapsed)

    def test_gradient_flows_to_prediction(self):
        torch.manual_seed(0)
        p = torch.randn(8, 4, requires_grad=True)
        z = torch.randn(8, 4)
        loss = MECLoss()(p, z)

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert p.grad is not None
        assert p.grad.abs().sum() > 0

    def test_low_precision_inputs_computed_in_fp32(self):
        torch.manual_seed(0)
        p, z = torch.randn(8, 4), torch.randn(8, 4)
        loss_fn = MECLoss()
        loss_bf16 = loss_fn(p.bfloat16(), z.bfloat16())

        assert loss_bf16.dtype == torch.float32
        assert torch.allclose(loss_bf16, loss_fn(p, z), atol=1e-2)

    def test_autocast_does_not_downcast_series(self):
        """Mixed-precision training must not run the Taylor series in bf16."""
        torch.manual_seed(0)
        p = torch.randn(64, 32)
        z = p + 0.5 * torch.randn_like(p)
        loss_fn = MECLoss()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss_autocast = loss_fn(p, z)

        assert loss_autocast.dtype == torch.float32
        torch.testing.assert_close(loss_autocast, loss_fn(p, z))

    def test_symmetric_matches_plain_when_c_is_symmetric(self):
        torch.manual_seed(0)
        z = torch.randn(32, 16)
        torch.testing.assert_close(
            MECLoss(symmetric=True)(z, z), MECLoss(symmetric=False)(z, z)
        )

    def test_symmetric_uses_symmetrized_correlation(self):
        torch.manual_seed(0)
        p, z = torch.randn(32, 16), torch.randn(32, 16)
        loss_fn = MECLoss(symmetric=True)
        lambda_inv = loss_fn.lambda_inv(32, 16)

        c = F.normalize(p, dim=1) @ F.normalize(z, dim=1).T
        c = (c + c.T) / 2 / lambda_inv
        expected = (
            -lambda_inv
            * torch.trace(c - c @ c / 2 + c @ c @ c / 3 - c @ c @ c @ c / 4)
            / 32
        )
        torch.testing.assert_close(loss_fn(p, z), expected)

    @pytest.mark.parametrize("angle", [torch.pi / 8, torch.pi / 4])
    def test_symmetric_resists_rotation_exploit(self, angle):
        """Collapsed features plus a rotating predictor game the plain loss.

        Targets use only 2 dimensions and predictions are those targets
        rotated in that plane, so ``C`` has large complex eigenvalues on which
        the truncated series rewards collapse past the bound that holds for
        real eigenvalues (each order-4 term is at most 7/12). The symmetrized
        loss respects the bound and penalizes the same configuration.
        """
        m, d = 64, 16
        z = torch.zeros(m, d)
        z[::2, 0] = 1
        z[1::2, 1] = 1
        cos, sin = torch.tensor(angle).cos(), torch.tensor(angle).sin()
        rotation = torch.eye(d)
        rotation[:2, :2] = torch.stack(
            [torch.stack([cos, -sin]), torch.stack([sin, cos])]
        )
        p = z @ rotation.T

        plain, symmetric = MECLoss(eps=1.0), MECLoss(eps=1.0, symmetric=True)
        bound = -plain.lambda_inv(m, d) * (7 / 12) * min(m, d) / m

        assert plain(p, z) < bound
        assert symmetric(p, z) > 0


def _extreme_inputs(name, m=32, d=16):
    """Prediction/target pairs at the edges of what the loss may receive."""
    g = torch.Generator().manual_seed(0)

    def randn():
        return torch.randn(m, d, generator=g)

    if name == "huge":
        return randn() * 1e30, randn() * 1e30
    if name == "tiny":
        return randn() * 1e-30, randn() * 1e-30
    if name == "mixed_row_scales":
        scales = torch.logspace(-30, 30, m)[:, None]
        return randn() * scales, randn() * scales.flip(0)
    if name == "all_zero":
        return torch.zeros(m, d), torch.zeros(m, d)
    if name == "some_zero_rows":
        p, z = randn(), randn()
        p[::4], z[1::4] = 0, 0
        return p, z
    if name == "collapsed":
        row = torch.randn(1, d, generator=g)
        return row.repeat(m, 1), row.repeat(m, 1)
    if name == "identical":
        z = randn()
        return z.clone(), z
    if name == "batch_of_one":
        return torch.randn(1, d, generator=g), torch.randn(1, d, generator=g)
    if name == "dim_of_one":
        return torch.randn(m, 1, generator=g), torch.randn(m, 1, generator=g)
    if name == "fp16_max":
        big = torch.finfo(torch.float16).max
        return (randn().clamp(-1, 1) * big).half(), (randn().clamp(-1, 1) * big).half()
    raise ValueError(name)


EXTREME_CASES = [
    "huge",
    "tiny",
    "mixed_row_scales",
    "all_zero",
    "some_zero_rows",
    "collapsed",
    "identical",
    "batch_of_one",
    "dim_of_one",
    "fp16_max",
]


@pytest.mark.unit
class TestMECLossExtremeInputs:
    """The loss and its gradient must stay finite on extreme inputs."""

    @pytest.mark.parametrize("name", EXTREME_CASES)
    @pytest.mark.parametrize("symmetric", [False, True])
    @pytest.mark.parametrize("lambda_inv_scale", [1e-3, 1.0, 8.0])
    # ``eps`` far below 1 puts ``λC`` far outside the Taylor series' radius
    # of convergence, where ``(λC)^order`` overflows fp32 by construction.
    @pytest.mark.parametrize("eps", [1.0, 64.0, 1e6])
    def test_loss_and_gradients_are_finite(
        self, name, symmetric, lambda_inv_scale, eps
    ):
        p, z = _extreme_inputs(name)
        p = p.requires_grad_()
        z = z.requires_grad_()
        loss = MECLoss(eps=eps, symmetric=symmetric)(p, z, lambda_inv_scale)

        assert loss.ndim == 0
        assert torch.isfinite(loss), f"loss is {loss.item()}"
        loss.backward()
        assert torch.isfinite(p.grad).all()
        assert torch.isfinite(z.grad).all()

    @pytest.mark.parametrize("scale", [1e-30, 1e-20, 1e20, 1e30])
    def test_scale_invariance_holds_at_extreme_magnitudes(self, scale):
        """Rows far from unit norm must not overflow or underflow to zero."""
        torch.manual_seed(0)
        p, z = torch.randn(32, 16), torch.randn(32, 16)
        loss_fn = MECLoss()
        torch.testing.assert_close(loss_fn(scale * p, scale * z), loss_fn(p, z))

    def test_gradient_matches_unscaled_input(self):
        """Gradients of huge inputs are the unit-scale gradients divided by the scale."""
        torch.manual_seed(0)
        p, z = torch.randn(16, 8), torch.randn(16, 8)
        scale = 1e25
        p_unit = p.clone().requires_grad_()
        p_huge = (p * scale).requires_grad_()
        MECLoss()(p_unit, z).backward()
        MECLoss()(p_huge, z).backward()
        torch.testing.assert_close(p_huge.grad * scale, p_unit.grad)
