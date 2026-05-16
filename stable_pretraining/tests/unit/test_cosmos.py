"""Unit tests for Cosmos Tokenizer (NVIDIA) causal video encoder.

Run with: ``pytest stable_pretraining/tests/unit/test_cosmos.py -v -m unit``
"""

import pytest
import torch

from stable_pretraining.backbone.video import (
    CosmosCausalTemporalAttention,
    CosmosEncoder,
    CosmosOutput,
    CosmosSpatialAttention,
    cosmos_tiny,
    cosmos_small,
    cosmos_base,
)


# --- Attention blocks --------------------------------------------------------


@pytest.mark.unit
class TestCosmosSpatialAttention:
    """Tests for the per-frame spatial self-attention block."""

    def test_shape_preserved(self):
        attn = CosmosSpatialAttention(channels=16, num_heads=4)
        x = torch.randn(2, 16, 4, 8, 8)
        assert attn(x).shape == x.shape

    def test_per_frame_isolation(self):
        """Spatial attention is strictly per-frame.

        Perturbing frame ``t`` should not change the output at any other
        frame ``t' != t``.
        """
        torch.manual_seed(0)
        attn = CosmosSpatialAttention(channels=16, num_heads=4).eval()
        x_a = torch.randn(1, 16, 6, 4, 4)
        x_b = x_a.clone()
        x_b[:, :, 3] = torch.randn_like(x_b[:, :, 3])
        with torch.no_grad():
            y_a = attn(x_a)
            y_b = attn(x_b)
        # Frames 0,1,2,4,5 should match.
        for t in (0, 1, 2, 4, 5):
            assert torch.allclose(y_a[:, :, t], y_b[:, :, t], atol=1e-5), (
                f"frame {t} differs"
            )

    def test_invalid_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            CosmosSpatialAttention(channels=15, num_heads=4)


@pytest.mark.unit
class TestCosmosCausalTemporalAttention:
    """Tests for the causal temporal self-attention block."""

    def test_shape_preserved(self):
        attn = CosmosCausalTemporalAttention(channels=16, num_heads=4)
        x = torch.randn(2, 16, 4, 8, 8)
        assert attn(x).shape == x.shape

    def test_no_future_leakage(self):
        """Defining property of causal temporal attention.

        Perturbing frame ``t > k`` cannot change the output at frame
        ``<= k``.
        """
        torch.manual_seed(0)
        attn = CosmosCausalTemporalAttention(channels=16, num_heads=4).eval()
        x_a = torch.randn(1, 16, 8, 4, 4)
        x_b = x_a.clone()
        k = 3
        x_b[:, :, k + 1 :] = torch.randn_like(x_b[:, :, k + 1 :])
        with torch.no_grad():
            y_a = attn(x_a)
            y_b = attn(x_b)
        assert torch.allclose(y_a[:, :, : k + 1], y_b[:, :, : k + 1], atol=1e-5)
        assert not torch.allclose(y_a[:, :, k + 1 :], y_b[:, :, k + 1 :], atol=1e-5)


# --- CosmosEncoder -----------------------------------------------------------


@pytest.mark.unit
class TestCosmosEncoder:
    """Tests for the full :class:`CosmosEncoder` (shape, causality, parity)."""

    @pytest.fixture(scope="class")
    def small_model(self):
        torch.manual_seed(0)
        return CosmosEncoder(
            base_channels=16,
            channel_multipliers=(1, 2, 2, 4),
            n_res_blocks=1,
            latent_dim=8,
            attn_stages=(2, 3),
            num_heads=4,
            temporal_downsample_stages=(1, 2),
            groups=8,
        )

    def test_output_shape(self, small_model):
        x = torch.randn(2, 3, 8, 64, 64)
        out = small_model(x)
        assert isinstance(out, CosmosOutput)
        # 4 stages → 3 spatial halvings, 2 temporal halvings.
        assert out.feature_map.shape == (2, 8, 2, 8, 8)
        assert out.pooled.shape == (2, 8)

    def test_grad_flow(self, small_model):
        x = torch.randn(1, 3, 8, 32, 32, requires_grad=True)
        out = small_model(x)
        out.feature_map.sum().backward()
        assert x.grad is not None
        for p in small_model.parameters():
            assert p.grad is not None

    def test_no_future_leakage(self, small_model):
        """End-to-end causality including attention layers.

        With 2 temporal-downsample stages, output time 0 receives input
        frames {0,1,2,3} and output time 1 receives {4,5,6,7}. Perturbing
        input from frame 4 must leave output[0] bit-identical.
        """
        torch.manual_seed(0)
        small_model.eval()
        x_a = torch.randn(1, 3, 8, 32, 32)
        x_b = x_a.clone()
        x_b[:, :, 4:] = torch.randn_like(x_b[:, :, 4:])
        with torch.no_grad():
            y_a = small_model(x_a).feature_map
            y_b = small_model(x_b).feature_map
        assert torch.allclose(y_a[:, :, 0], y_b[:, :, 0], atol=1e-5)
        assert not torch.allclose(y_a[:, :, 1], y_b[:, :, 1], atol=1e-5)

    def test_determinism(self, small_model):
        small_model.eval()
        x = torch.randn(1, 3, 8, 32, 32)
        with torch.no_grad():
            a = small_model(x).feature_map
            b = small_model(x).feature_map
        assert torch.allclose(a, b)

    def test_no_pool(self):
        m = CosmosEncoder(
            base_channels=16,
            n_res_blocks=1,
            latent_dim=8,
            groups=8,
            attn_stages=(),
            global_pool="",
        )
        out = m(torch.randn(1, 3, 8, 32, 32))
        assert out.pooled is None
        assert out.feature_map.ndim == 5

    def test_checkpoint_parity(self):
        torch.manual_seed(0)
        m_ref = CosmosEncoder(
            base_channels=16,
            n_res_blocks=1,
            latent_dim=8,
            groups=8,
            attn_stages=(2,),
            num_heads=4,
        )
        m_ckpt = CosmosEncoder(
            base_channels=16,
            n_res_blocks=1,
            latent_dim=8,
            groups=8,
            attn_stages=(2,),
            num_heads=4,
            use_checkpoint=True,
        )
        m_ckpt.load_state_dict(m_ref.state_dict())
        m_ref.train()
        m_ckpt.train()
        x = torch.randn(1, 3, 8, 32, 32)
        y_ref = m_ref(x).feature_map
        y_ckpt = m_ckpt(x).feature_map
        assert torch.allclose(y_ref, y_ckpt, atol=1e-5)

    def test_attn_stage_index_validation(self):
        with pytest.raises(ValueError, match="out of range"):
            CosmosEncoder(
                base_channels=16,
                channel_multipliers=(1, 2),
                attn_stages=(5,),
                groups=8,
            )

    def test_head_divisibility_check(self):
        with pytest.raises(ValueError, match="not divisible"):
            CosmosEncoder(
                base_channels=10,
                channel_multipliers=(1, 1),
                attn_stages=(0,),
                num_heads=4,  # 10 % 4 != 0
                groups=2,
            )


# --- Factories ---------------------------------------------------------------


@pytest.mark.unit
class TestFactories:
    """Smoke tests for the named Cosmos factory presets."""

    @pytest.mark.parametrize(
        "factory,min_params,max_params",
        [
            (cosmos_tiny, 5_000_000, 30_000_000),
            (cosmos_small, 15_000_000, 60_000_000),
            (cosmos_base, 30_000_000, 100_000_000),
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
        m = cosmos_tiny()
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, 3, 8, 64, 64))
        assert out.feature_map.shape[0] == 1
        assert out.feature_map.shape[1] == m.latent_dim
        assert out.pooled.shape == (1, m.latent_dim)
