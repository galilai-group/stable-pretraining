"""Unit tests for MAGVIT-v2 causal video encoder.

Run with: ``pytest stable_pretraining/tests/unit/test_magvit2.py -v -m unit``
"""

import pytest
import torch

from stable_pretraining.backbone.video import (
    CausalConv3d,
    MAGVIT2Encoder,
    MAGVIT2Output,
    magvit2_tiny,
    magvit2_small,
    magvit2_base,
)


# --- CausalConv3d ------------------------------------------------------------


@pytest.mark.unit
class TestCausalConv3d:
    """Tests for the :class:`CausalConv3d` primitive."""

    def test_output_shape_identity_stride(self):
        conv = CausalConv3d(8, 16, kernel_size=3)
        x = torch.randn(2, 8, 7, 12, 12)
        y = conv(x)
        assert y.shape == (2, 16, 7, 12, 12)

    def test_output_shape_temporal_stride(self):
        conv = CausalConv3d(4, 4, kernel_size=3, stride=(2, 2, 2))
        x = torch.randn(1, 4, 8, 16, 16)
        y = conv(x)
        # Temporal: pad-left = 2, kernel 3, stride 2 → ceil(8 / 2) = 4
        # Spatial:  same-pad with stride 2 → 8
        assert y.shape == (1, 4, 4, 8, 8)

    def test_no_future_leakage(self):
        """Verify the defining property of a causal conv.

        Perturbing frame ``t = k+1`` and onward must not change output at
        ``t <= k``.
        """
        torch.manual_seed(0)
        conv = CausalConv3d(3, 5, kernel_size=3)
        T = 10
        k = 4
        x_a = torch.randn(2, 3, T, 8, 8)
        x_b = x_a.clone()
        x_b[:, :, k + 1 :] = torch.randn_like(x_b[:, :, k + 1 :])

        y_a = conv(x_a)
        y_b = conv(x_b)

        assert torch.allclose(y_a[:, :, : k + 1], y_b[:, :, : k + 1], atol=1e-6)
        # Sanity: the suffix MUST differ (otherwise the conv is a no-op).
        assert not torch.allclose(y_a[:, :, k + 1 :], y_b[:, :, k + 1 :], atol=1e-6)

    def test_grad_flow(self):
        conv = CausalConv3d(4, 4, kernel_size=3)
        x = torch.randn(1, 4, 5, 8, 8, requires_grad=True)
        conv(x).sum().backward()
        assert x.grad is not None
        assert conv.weight.grad is not None


# --- MAGVIT2Encoder ----------------------------------------------------------


@pytest.mark.unit
class TestMAGVIT2Encoder:
    """Tests for the full :class:`MAGVIT2Encoder` (shape, causality, parity)."""

    @pytest.fixture(scope="class")
    def small_model(self):
        torch.manual_seed(0)
        # An intentionally tiny config so the whole test class runs quickly
        # on CPU. Real presets are exercised in test_factories.
        return MAGVIT2Encoder(
            base_channels=16,
            channel_multipliers=(1, 2, 2, 4),
            n_res_blocks=1,
            latent_dim=8,
            temporal_downsample_stages=(1, 2),
            groups=8,
        )

    def test_output_shape(self, small_model):
        x = torch.randn(2, 3, 8, 64, 64)
        out = small_model(x)
        assert isinstance(out, MAGVIT2Output)
        # 4 stages, 3 with downsample → spatial / 8, temporal / 4
        assert out.feature_map.shape == (2, 8, 2, 8, 8)
        assert out.pooled.shape == (2, 8)

    def test_no_global_pool(self):
        m = MAGVIT2Encoder(
            base_channels=16, n_res_blocks=1, latent_dim=8, groups=8, global_pool=""
        )
        out = m(torch.randn(1, 3, 8, 32, 32))
        assert out.pooled is None
        assert out.feature_map.ndim == 5

    def test_grad_flow(self, small_model):
        x = torch.randn(1, 3, 8, 32, 32, requires_grad=True)
        out = small_model(x)
        out.feature_map.sum().backward()
        assert x.grad is not None
        # All trainable params should have non-zero gradients.
        n_zero = sum(
            1 for p in small_model.parameters() if p.requires_grad and p.grad is None
        )
        assert n_zero == 0

    def test_no_future_leakage(self, small_model):
        """End-to-end causality test.

        Perturbing the suffix of input frames leaves the prefix of the
        output feature map unchanged.

        With temporal downsampling at stages 1 and 2 the time stride is 4×,
        so input frame index ``ti`` maps to output frame ``ti // 4``. If we
        perturb input from frame ``ti = 4`` onward, output frames ``[0]`` (the
        first temporal cell, fed by input frames 0–3) must be untouched.
        """
        torch.manual_seed(0)
        small_model.eval()
        x_a = torch.randn(1, 3, 8, 32, 32)
        x_b = x_a.clone()
        x_b[:, :, 4:] = torch.randn_like(x_b[:, :, 4:])

        with torch.no_grad():
            y_a = small_model(x_a).feature_map
            y_b = small_model(x_b).feature_map

        # Output time index 0 receives input frames 0..3 only → must match.
        assert torch.allclose(y_a[:, :, 0], y_b[:, :, 0], atol=1e-5)
        # Index 1 receives input frames 4..7 → must differ (sanity).
        assert not torch.allclose(y_a[:, :, 1], y_b[:, :, 1], atol=1e-5)

    def test_determinism(self, small_model):
        small_model.eval()
        x = torch.randn(1, 3, 8, 32, 32)
        with torch.no_grad():
            a = small_model(x).feature_map
            b = small_model(x).feature_map
        assert torch.allclose(a, b)

    def test_checkpoint_parity(self):
        """Checkpointed forward must match the non-checkpointed one.

        Activation checkpointing only changes the memory profile, not the
        numerical output.
        """
        torch.manual_seed(0)
        m_ref = MAGVIT2Encoder(base_channels=16, n_res_blocks=1, latent_dim=8, groups=8)
        m_ckpt = MAGVIT2Encoder(
            base_channels=16,
            n_res_blocks=1,
            latent_dim=8,
            groups=8,
            use_checkpoint=True,
        )
        m_ckpt.load_state_dict(m_ref.state_dict())
        m_ref.train()
        m_ckpt.train()
        x = torch.randn(1, 3, 8, 32, 32)
        y_ref = m_ref(x).feature_map
        y_ckpt = m_ckpt(x).feature_map
        assert torch.allclose(y_ref, y_ckpt, atol=1e-5)


# --- Factories ---------------------------------------------------------------


@pytest.mark.unit
class TestFactories:
    """Smoke tests for the named MAGVIT-v2 factory presets."""

    @pytest.mark.parametrize(
        "factory,min_params,max_params",
        [
            (magvit2_tiny, 1_000_000, 15_000_000),
            (magvit2_small, 5_000_000, 50_000_000),
            (magvit2_base, 30_000_000, 200_000_000),
        ],
    )
    def test_param_count_in_range(self, factory, min_params, max_params):
        m = factory()
        n = sum(p.numel() for p in m.parameters())
        assert min_params < n < max_params, (
            f"{factory.__name__}: got {n / 1e6:.1f}M params, "
            f"expected ({min_params / 1e6:.0f}M, {max_params / 1e6:.0f}M)"
        )

    def test_tiny_forward(self):
        m = magvit2_tiny()
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, 3, 8, 64, 64))
        assert out.feature_map.shape[0] == 1
        assert out.feature_map.shape[1] == m.latent_dim
        assert out.pooled.shape == (1, m.latent_dim)
