"""Unit tests for VideoMamba.

Run with: ``pytest stable_pretraining/tests/unit/test_videomamba.py -v -m unit``

These tests exercise the pure-PyTorch reference Mamba scan path. They use
intentionally short sequences (small ``num_frames``, ``img_size``, ``depth``)
because the reference scan is sequential.
"""

import pytest
import torch

from stable_pretraining.backbone.video import (
    BiMambaBlock,
    CausalMambaBlock,
    MambaSSMBlock,
    VideoMamba,
    VideoMambaOutput,
    videomamba_tiny,
    videomamba_small,
)


# --- MambaSSMBlock -----------------------------------------------------------


@pytest.mark.unit
class TestMambaSSMBlock:
    """Tests for the pure-PyTorch Mamba S6 block."""

    def test_shape(self):
        m = MambaSSMBlock(d_model=16, d_state=8, d_conv=4, expand=2)
        x = torch.randn(2, 12, 16)
        assert m(x).shape == x.shape

    def test_strict_causality(self):
        """S6 with forward scan + causal 1D conv must be strictly causal.

        Perturbing token ``i > k`` cannot affect token ``<= k``.
        """
        torch.manual_seed(0)
        m = MambaSSMBlock(d_model=16, d_state=8, d_conv=4, expand=2).eval()
        x_a = torch.randn(1, 16, 16)
        x_b = x_a.clone()
        k = 6
        x_b[:, k + 1 :] = torch.randn_like(x_b[:, k + 1 :])
        with torch.no_grad():
            y_a = m(x_a)
            y_b = m(x_b)
        assert torch.allclose(y_a[:, : k + 1], y_b[:, : k + 1], atol=1e-5)
        # Sanity: suffix must actually differ.
        assert not torch.allclose(y_a[:, k + 1 :], y_b[:, k + 1 :], atol=1e-5)

    def test_grad_flow(self):
        m = MambaSSMBlock(d_model=8, d_state=4, d_conv=4, expand=2)
        x = torch.randn(1, 6, 8, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        for p in m.parameters():
            assert p.grad is not None


@pytest.mark.unit
class TestVideoMambaBlocks:
    """Tests for the video-level Mamba block wrappers (causal / bidirectional)."""

    def test_causal_block_shape_and_causality(self):
        torch.manual_seed(0)
        blk = CausalMambaBlock(d_model=16).eval()
        x = torch.randn(1, 10, 16)
        assert blk(x).shape == x.shape
        # Causal
        x2 = x.clone()
        x2[:, 5:] = torch.randn_like(x2[:, 5:])
        with torch.no_grad():
            y = blk(x)
            y2 = blk(x2)
        assert torch.allclose(y[:, :5], y2[:, :5], atol=1e-5)

    def test_bi_block_not_causal(self):
        """BiMambaBlock fuses forward + backward scans.

        Outputs at the prefix depend on the suffix, so it is NOT causal
        by design.
        """
        torch.manual_seed(0)
        blk = BiMambaBlock(d_model=16).eval()
        x = torch.randn(1, 10, 16)
        x2 = x.clone()
        x2[:, 5:] = torch.randn_like(x2[:, 5:])
        with torch.no_grad():
            y = blk(x)
            y2 = blk(x2)
        # Prefix should differ — bi block leaks information backward.
        assert not torch.allclose(y[:, :5], y2[:, :5], atol=1e-5)


# --- VideoMamba (encoder) ----------------------------------------------------


@pytest.mark.unit
class TestVideoMamba:
    """Tests for the full :class:`VideoMamba` encoder."""

    @pytest.fixture(scope="class")
    def small_model(self):
        torch.manual_seed(0)
        return VideoMamba(
            img_size=16,
            num_frames=4,
            patch_size=(1, 8, 8),
            embed_dim=16,
            depth=2,
            d_state=4,
            causal=True,
            class_token=True,
            num_classes=0,
            global_pool="token",
        )

    def test_output_shape(self, small_model):
        x = torch.randn(2, 3, 4, 16, 16)
        out = small_model(x)
        assert isinstance(out, VideoMambaOutput)
        # 4 frames × (16/8)² spatial patches = (T'=4, H'=2, W'=2) feature map.
        assert out.feature_map.shape == (2, 16, 4, 2, 2)
        # Token sequence still available: 4*2*2 = 16 patch tokens + 1 CLS.
        assert out.tokens.shape == (2, 17, 16)
        assert out.pooled.shape == (2, 16)

    def test_grad_flow(self, small_model):
        x = torch.randn(1, 3, 4, 16, 16, requires_grad=True)
        out = small_model(x)
        out.feature_map.sum().backward()
        assert x.grad is not None
        for p in small_model.parameters():
            assert p.grad is not None

    def test_no_future_leakage_causal(self, small_model):
        """Causal VideoMamba leaves the prefix bit-identical under suffix perturbation.

        Perturbing input frame ``t > k`` must leave the output feature-map
        slice at frames ``[0, k]`` unchanged.

        Using the 5D ``feature_map`` view, this becomes a clean check on
        the temporal axis (axis 2) — independent of the underlying token
        ordering. The ``tokens`` view (which includes the CLS) is also
        spot-checked for the patch-token slice.
        """
        torch.manual_seed(0)
        small_model.eval()
        x_a = torch.randn(1, 3, 4, 16, 16)
        x_b = x_a.clone()
        k = 1  # perturb frames 2 and 3
        x_b[:, :, k + 1 :] = torch.randn_like(x_b[:, :, k + 1 :])

        with torch.no_grad():
            out_a = small_model(x_a)
            out_b = small_model(x_b)

        # 5D feature map: clean frames must match exactly.
        assert torch.allclose(
            out_a.feature_map[:, :, : k + 1],
            out_b.feature_map[:, :, : k + 1],
            atol=1e-5,
        )
        # Perturbed frames must actually differ (sanity).
        assert not torch.allclose(
            out_a.feature_map[:, :, k + 1 :],
            out_b.feature_map[:, :, k + 1 :],
            atol=1e-5,
        )

    def test_determinism(self, small_model):
        small_model.eval()
        x = torch.randn(1, 3, 4, 16, 16)
        with torch.no_grad():
            a = small_model(x).feature_map
            b = small_model(x).feature_map
        assert torch.allclose(a, b)

    def test_avg_pool(self):
        m = VideoMamba(
            img_size=16,
            num_frames=2,
            patch_size=(1, 8, 8),
            embed_dim=16,
            depth=2,
            d_state=4,
            class_token=False,
            global_pool="avg",
        )
        out = m(torch.randn(1, 3, 2, 16, 16))
        assert out.pooled.shape == (1, 16)

    def test_no_pool(self):
        m = VideoMamba(
            img_size=16,
            num_frames=2,
            patch_size=(1, 8, 8),
            embed_dim=16,
            depth=2,
            d_state=4,
            global_pool="",
        )
        out = m(torch.randn(1, 3, 2, 16, 16))
        assert out.pooled is None

    def test_classification_head(self):
        m = VideoMamba(
            img_size=16,
            num_frames=2,
            patch_size=(1, 8, 8),
            embed_dim=16,
            depth=2,
            d_state=4,
            num_classes=5,
            global_pool="token",
        )
        out = m(torch.randn(1, 3, 2, 16, 16))
        assert out.pooled.shape == (1, 5)

    def test_patch_divisibility_error(self):
        with pytest.raises(ValueError, match="must divide"):
            VideoMamba(
                img_size=16, num_frames=3, patch_size=(2, 8, 8), embed_dim=8, depth=1
            )

    def test_checkpoint_parity(self):
        torch.manual_seed(0)
        m_ref = VideoMamba(
            img_size=16,
            num_frames=2,
            patch_size=(1, 8, 8),
            embed_dim=8,
            depth=2,
            d_state=4,
            causal=True,
        )
        m_ckpt = VideoMamba(
            img_size=16,
            num_frames=2,
            patch_size=(1, 8, 8),
            embed_dim=8,
            depth=2,
            d_state=4,
            causal=True,
            use_checkpoint=True,
        )
        m_ckpt.load_state_dict(m_ref.state_dict())
        m_ref.train()
        m_ckpt.train()
        x = torch.randn(1, 3, 2, 16, 16)
        y_ref = m_ref(x).feature_map
        y_ckpt = m_ckpt(x).feature_map
        assert torch.allclose(y_ref, y_ckpt, atol=1e-5)


# --- Factories ---------------------------------------------------------------


@pytest.mark.unit
class TestFactories:
    """Smoke tests for the named VideoMamba factory presets."""

    @pytest.mark.parametrize(
        "factory,causal,min_params,max_params",
        [
            (videomamba_tiny, True, 4_000_000, 20_000_000),
            (videomamba_small, True, 15_000_000, 60_000_000),
        ],
    )
    def test_param_count_in_range(self, factory, causal, min_params, max_params):
        # Use a small grid so factory call is cheap; depth matters for params.
        m = factory(img_size=32, num_frames=4, patch_size=(1, 16, 16), causal=causal)
        n = sum(p.numel() for p in m.parameters())
        assert min_params < n < max_params, (
            f"{factory.__name__}: got {n / 1e6:.1f}M params, "
            f"expected ({min_params / 1e6:.0f}M, {max_params / 1e6:.0f}M)"
        )
