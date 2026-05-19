"""Unit tests for the RecurrentViT video encoder.

Run with: ``pytest stable_pretraining/tests/unit/test_recurrent_vit.py -v -m unit``
"""

import pytest
import torch

from stable_pretraining.backbone.video import (
    RecurrentViT,
    RecurrentViTOutput,
    recurrent_vit_tiny,
    recurrent_vit_small,
    recurrent_vit_base,
    recurrent_vit_large,
    recurrent_vit_huge,
)


@pytest.mark.unit
class TestRecurrentViTShapes:
    """Output shape and dataclass contract."""

    def test_recurrent_vit_shapes(self):
        """Default tiny produces the documented output shapes.

        Input ``(2, 3, 8, 64, 64)`` → ``feature_map=(2, 192, 8, 8, 8)``,
        ``pooled=(2, 192)``, ``tokens=(2, 8, 192)``.
        """
        torch.manual_seed(0)
        enc = recurrent_vit_tiny()
        enc.eval()
        x = torch.randn(2, 3, 8, 64, 64)
        with torch.no_grad():
            out = enc(x)
        assert isinstance(out, RecurrentViTOutput)
        # patch_size=8, img_size=64 -> grid=8.
        assert out.feature_map.shape == (2, 192, 8, 8, 8)
        assert out.pooled.shape == (2, 192)
        assert out.tokens.shape == (2, 8, 192)

    def test_no_pool(self):
        """``global_pool=''`` drops the pooled slot but keeps feature_map and tokens."""
        m = RecurrentViT(
            img_size=32,
            patch_size=8,
            embed_dim=64,
            spatial_depth=2,
            num_heads=4,
            gru_layers=1,
            global_pool="",
        )
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, 3, 4, 32, 32))
        assert out.pooled is None
        assert out.feature_map.shape == (1, 64, 4, 4, 4)
        assert out.tokens.shape == (1, 4, 64)

    def test_rejects_wrong_spatial_size(self):
        m = RecurrentViT(
            img_size=32,
            patch_size=8,
            embed_dim=32,
            spatial_depth=1,
            num_heads=4,
            gru_layers=1,
        )
        with pytest.raises(ValueError, match="img_size"):
            m(torch.randn(1, 3, 4, 24, 24))


@pytest.mark.unit
class TestNoFutureLeakage:
    """Causality of feature_map and tokens.

    The GRU on the pooled CLS sequence makes ``tokens`` causal in time;
    the per-frame ViT can't leak future info into ``feature_map`` either,
    because frames are encoded independently.
    """

    def test_no_future_leakage(self):
        torch.manual_seed(0)
        enc = recurrent_vit_tiny()
        enc.eval()
        x_a = torch.randn(1, 3, 8, 64, 64)
        x_b = x_a.clone()
        x_b[:, :, 4:] = torch.randn_like(x_b[:, :, 4:])
        with torch.no_grad():
            out_a = enc(x_a)
            out_b = enc(x_b)
        # feature_map slot t depends only on input frame t.
        assert torch.allclose(
            out_a.feature_map[:, :, :4], out_b.feature_map[:, :, :4], atol=1e-5
        )
        assert not torch.allclose(
            out_a.feature_map[:, :, 4:], out_b.feature_map[:, :, 4:], atol=1e-5
        )
        # tokens are GRU output: causal by construction.
        assert torch.allclose(out_a.tokens[:, :4], out_b.tokens[:, :4], atol=1e-5)
        assert not torch.allclose(out_a.tokens[:, 4:], out_b.tokens[:, 4:], atol=1e-5)


@pytest.mark.unit
class TestFactoryParamCounts:
    """Each preset lands in the documented ViT-family param budget.

    Tolerance is ±20%: the targets are standard-ViT-shaped approximations,
    and the GRU + patch embed + QK-norm each shift the count by a few
    percent, so a single fixed target won't sit at the center of every
    preset.
    """

    @pytest.mark.parametrize(
        "factory,target",
        [
            (recurrent_vit_tiny, 5_000_000),
            (recurrent_vit_small, 20_000_000),
            (recurrent_vit_base, 80_000_000),
            (recurrent_vit_large, 290_000_000),
            (recurrent_vit_huge, 600_000_000),
        ],
    )
    def test_factory_param_counts(self, factory, target):
        # Meta device so the huge preset doesn't allocate ~2GB.
        with torch.device("meta"):
            m = factory()
        n = sum(p.numel() for p in m.parameters())
        lo, hi = int(target * 0.8), int(target * 1.2)
        assert lo <= n <= hi, (
            f"{factory.__name__}: got {n / 1e6:.1f}M params, "
            f"expected {target / 1e6:.0f}M ±20% = "
            f"[{lo / 1e6:.1f}M, {hi / 1e6:.1f}M]"
        )


@pytest.mark.unit
class TestSmallerPresetsForward:
    """Real-tensor smoke forward for the smaller presets.

    Exercises the GRU path on a real tensor without paying the large
    preset's memory cost.
    """

    @pytest.mark.parametrize("factory", [recurrent_vit_tiny, recurrent_vit_small])
    def test_forward_runs(self, factory):
        m = factory()
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, 3, 4, 64, 64))
        assert out.feature_map.shape[0] == 1
        assert out.feature_map.shape[1] == m.embed_dim
        assert out.feature_map.shape[2] == 4
        assert out.tokens.shape == (1, 4, m.embed_dim)
        assert out.pooled.shape == (1, m.embed_dim)


@pytest.mark.unit
class TestGradFlow:
    """Verify gradient flow through both the spatial ViT and the GRU."""

    def test_grad_flow(self):
        torch.manual_seed(0)
        m = RecurrentViT(
            img_size=32,
            patch_size=8,
            embed_dim=64,
            spatial_depth=2,
            num_heads=4,
            gru_layers=1,
        )
        x = torch.randn(1, 3, 4, 32, 32, requires_grad=True)
        out = m(x)
        (out.feature_map.sum() + out.tokens.sum()).backward()
        assert x.grad is not None
        for p in m.parameters():
            assert p.requires_grad and p.grad is not None
