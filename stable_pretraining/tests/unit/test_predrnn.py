"""Unit tests for PredRNN-v2 recurrent video encoder.

Run with: ``pytest stable_pretraining/tests/unit/test_predrnn.py -v -m unit``
"""

import pytest
import torch

from stable_pretraining.backbone.video import (
    GHU,
    PredRNNv2,
    PredRNNv2Output,
    STLSTMCell,
    predrnn_v2_tiny,
    predrnn_v2_small,
    predrnn_v2_base,
)


# --- STLSTMCell --------------------------------------------------------------


@pytest.mark.unit
class TestSTLSTMCell:
    """Tests for the :class:`STLSTMCell` (PredRNN family cell)."""

    def test_output_shapes(self):
        cell = STLSTMCell(in_channels=8, hidden_channels=16, kernel_size=3)
        x = torch.randn(2, 8, 10, 10)
        h = torch.zeros(2, 16, 10, 10)
        c = torch.zeros(2, 16, 10, 10)
        m = torch.zeros(2, 16, 10, 10)
        h_new, c_new, m_new, dc, dm = cell(x, h, c, m)
        assert h_new.shape == h.shape
        assert c_new.shape == c.shape
        assert m_new.shape == m.shape
        assert dc.shape == c.shape
        assert dm.shape == m.shape

    def test_even_kernel_rejected(self):
        with pytest.raises(ValueError, match="odd"):
            STLSTMCell(4, 4, kernel_size=4)

    def test_grad_flow(self):
        cell = STLSTMCell(4, 8, kernel_size=3)
        x = torch.randn(1, 4, 6, 6, requires_grad=True)
        h, c, m = (torch.zeros(1, 8, 6, 6) for _ in range(3))
        h_new, _, _, _, _ = cell(x, h, c, m)
        h_new.sum().backward()
        assert x.grad is not None
        for p in cell.parameters():
            assert p.grad is not None


@pytest.mark.unit
class TestGHU:
    """Tests for the :class:`GHU` (Gradient Highway Unit)."""

    def test_shape(self):
        ghu = GHU(channels=8, kernel_size=3)
        x = torch.randn(2, 8, 10, 10)
        z = torch.zeros(2, 8, 10, 10)
        out = ghu(x, z)
        assert out.shape == x.shape

    def test_zero_z_passthrough_is_well_defined(self):
        # When z=0, output = s * tanh(x_p) (no NaN, no Inf).
        ghu = GHU(8, 3)
        x = torch.randn(1, 8, 4, 4)
        z = torch.zeros_like(x)
        out = ghu(x, z)
        assert torch.isfinite(out).all()


# --- PredRNNv2 ---------------------------------------------------------------


@pytest.mark.unit
class TestPredRNNv2:
    """Tests for the full :class:`PredRNNv2` encoder."""

    @pytest.fixture(scope="class")
    def small_model(self):
        torch.manual_seed(0)
        return PredRNNv2(
            in_channels=3,
            hidden_channels=16,
            num_layers=3,
            kernel_size=3,
            num_frames=6,
            patch_size=2,  # halves spatial
            use_ghu=True,
        )

    def test_output_shape(self, small_model):
        x = torch.randn(2, 3, 6, 16, 16)
        out = small_model(x)
        assert isinstance(out, PredRNNv2Output)
        # patch_size=2 → spatial 8x8; T preserved; hidden=16
        assert out.feature_map.shape == (2, 16, 6, 8, 8)
        assert out.pooled.shape == (2, 16)
        assert out.decouple_loss is None  # not requested

    def test_no_pool(self):
        m = PredRNNv2(hidden_channels=8, num_layers=2, num_frames=4, global_pool="")
        out = m(torch.randn(1, 3, 4, 8, 8))
        assert out.pooled is None
        assert out.feature_map.ndim == 5

    def test_grad_flow(self, small_model):
        x = torch.randn(1, 3, 6, 16, 16, requires_grad=True)
        out = small_model(x)
        out.feature_map.sum().backward()
        assert x.grad is not None
        for p in small_model.parameters():
            assert p.requires_grad and p.grad is not None

    def test_no_future_leakage(self, small_model):
        """Recurrent model → naturally causal.

        Perturbing input frames at ``t >= k+1`` must leave output frames at
        ``t <= k`` untouched, bit-identical.
        """
        torch.manual_seed(0)
        small_model.eval()
        x_a = torch.randn(1, 3, 6, 16, 16)
        x_b = x_a.clone()
        x_b[:, :, 3:] = torch.randn_like(x_b[:, :, 3:])

        with torch.no_grad():
            y_a = small_model(x_a).feature_map
            y_b = small_model(x_b).feature_map

        assert torch.allclose(y_a[:, :, :3], y_b[:, :, :3], atol=1e-6)
        # Sanity: outputs at t >= 3 should actually differ.
        assert not torch.allclose(y_a[:, :, 3:], y_b[:, :, 3:], atol=1e-6)

    def test_determinism(self, small_model):
        small_model.eval()
        x = torch.randn(1, 3, 6, 16, 16)
        with torch.no_grad():
            a = small_model(x).feature_map
            b = small_model(x).feature_map
        assert torch.allclose(a, b)

    def test_decouple_loss_positive_and_backprop(self):
        m = PredRNNv2(
            hidden_channels=8,
            num_layers=2,
            num_frames=4,
            return_decouple_loss=True,
            use_ghu=False,
        )
        out = m(torch.randn(2, 3, 4, 8, 8))
        assert out.decouple_loss is not None
        assert out.decouple_loss.ndim == 0
        assert out.decouple_loss.item() >= 0.0
        # Should be backprop-friendly.
        out.decouple_loss.backward()

    def test_ghu_optional(self):
        m_no_ghu = PredRNNv2(
            hidden_channels=8, num_layers=3, num_frames=4, use_ghu=False
        )
        x = torch.randn(1, 3, 4, 8, 8)
        out = m_no_ghu(x)
        assert out.feature_map.shape == (1, 8, 4, 8, 8)
        assert m_no_ghu.ghu is None

    def test_ghu_requires_two_layers(self):
        with pytest.raises(ValueError, match="num_layers >= 2"):
            PredRNNv2(hidden_channels=8, num_layers=1, num_frames=4, use_ghu=True)

    def test_checkpoint_parity(self):
        """Activation checkpointing must not change the forward output."""
        torch.manual_seed(0)
        m_ref = PredRNNv2(hidden_channels=8, num_layers=2, num_frames=4, use_ghu=False)
        m_ckpt = PredRNNv2(
            hidden_channels=8,
            num_layers=2,
            num_frames=4,
            use_ghu=False,
            use_checkpoint=True,
        )
        m_ckpt.load_state_dict(m_ref.state_dict())
        m_ref.train()
        m_ckpt.train()
        x = torch.randn(1, 3, 4, 8, 8)
        y_ref = m_ref(x).feature_map
        y_ckpt = m_ckpt(x).feature_map
        assert torch.allclose(y_ref, y_ckpt, atol=1e-5)


# --- Factories ---------------------------------------------------------------


@pytest.mark.unit
class TestFactories:
    """Smoke tests for the named PredRNN-v2 factory presets."""

    @pytest.mark.parametrize(
        "factory,min_params,max_params",
        [
            (predrnn_v2_tiny, 200_000, 1_500_000),
            (predrnn_v2_small, 1_500_000, 6_000_000),
            (predrnn_v2_base, 6_000_000, 20_000_000),
        ],
    )
    def test_param_count_in_range(self, factory, min_params, max_params):
        m = factory(num_frames=4)
        n = sum(p.numel() for p in m.parameters())
        assert min_params < n < max_params, (
            f"{factory.__name__}: got {n / 1e6:.2f}M params, "
            f"expected ({min_params / 1e6:.2f}M, {max_params / 1e6:.2f}M)"
        )

    def test_tiny_forward(self):
        m = predrnn_v2_tiny(num_frames=4)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, 3, 4, 16, 16))
        assert out.feature_map.shape == (1, 32, 4, 16, 16)
        assert out.pooled.shape == (1, 32)
