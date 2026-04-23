"""Unit tests for positional embedding utilities."""

import math

import pytest
import torch

from stable_pretraining.backbone.pos_embed import (
    RotaryPositionEmbedding1D,
    RotaryPositionEmbedding2D,
    RotaryPositionEmbedding3D,
    apply_rotary_emb,
    build_rotary_pos_embed,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
    get_sincos_pos_embed,
    get_timestep_embed,
    interpolate_pos_embed,
)


@pytest.mark.unit
class TestGet1DSincosEmbed:
    """Test 1d sincos embed."""

    def test_output_shape(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=100)
        assert pe.shape == (100, 64)

    def test_output_shape_with_cls_token(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=100, cls_token=True)
        assert pe.shape == (101, 64)

    def test_cls_token_is_zeros(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=100, cls_token=True)
        assert torch.allclose(pe[0], torch.zeros(64))

    def test_dtype_is_float32(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=100)
        assert pe.dtype == torch.float32

    def test_values_bounded(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=100)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_different_positions_have_different_embeddings(self):
        pe = get_1d_sincos_pos_embed(embed_dim=64, length=10)
        # All rows should be unique
        for i in range(10):
            for j in range(i + 1, 10):
                assert not torch.allclose(pe[i], pe[j])

    def test_deterministic(self):
        pe1 = get_1d_sincos_pos_embed(embed_dim=64, length=50)
        pe2 = get_1d_sincos_pos_embed(embed_dim=64, length=50)
        assert torch.allclose(pe1, pe2)


@pytest.mark.unit
class TestGet2DSincosEmbed:
    """Test 2d sincos embed."""

    def test_output_shape(self):
        pe = get_2d_sincos_pos_embed(embed_dim=64, grid_size=14)
        assert pe.shape == (196, 64)  # 14*14 = 196

    def test_output_shape_with_cls_token(self):
        pe = get_2d_sincos_pos_embed(embed_dim=64, grid_size=14, cls_token=True)
        assert pe.shape == (197, 64)

    def test_cls_token_is_zeros(self):
        pe = get_2d_sincos_pos_embed(embed_dim=64, grid_size=7, cls_token=True)
        assert torch.allclose(pe[0], torch.zeros(64))

    def test_requires_divisible_by_4(self):
        with pytest.raises(ValueError):
            get_2d_sincos_pos_embed(embed_dim=65, grid_size=7)

    def test_values_bounded(self):
        pe = get_2d_sincos_pos_embed(embed_dim=64, grid_size=7)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_deterministic(self):
        pe1 = get_2d_sincos_pos_embed(embed_dim=64, grid_size=7)
        pe2 = get_2d_sincos_pos_embed(embed_dim=64, grid_size=7)
        assert torch.allclose(pe1, pe2)


@pytest.mark.unit
class TestGet3DSincosEmbed:
    """Test 3d sincos embed."""

    def test_output_shape_int_grid(self):
        pe = get_3d_sincos_pos_embed(embed_dim=48, grid_size=4)
        assert pe.shape == (64, 48)  # 4*4*4 = 64

    def test_output_shape_tuple_grid(self):
        pe = get_3d_sincos_pos_embed(embed_dim=48, grid_size=(2, 5, 7))
        assert pe.shape == (70, 48)  # 2*5*7 = 70

    def test_output_shape_with_cls_token(self):
        pe = get_3d_sincos_pos_embed(embed_dim=48, grid_size=(2, 4, 4), cls_token=True)
        assert pe.shape == (33, 48)  # 1 + 2*4*4

    def test_cls_token_is_zeros(self):
        pe = get_3d_sincos_pos_embed(embed_dim=48, grid_size=(2, 3, 3), cls_token=True)
        assert torch.allclose(pe[0], torch.zeros(48))

    def test_requires_divisible_by_6(self):
        with pytest.raises(ValueError):
            get_3d_sincos_pos_embed(embed_dim=49, grid_size=4)

    def test_rejects_non_positive_embed_dim(self):
        with pytest.raises(ValueError):
            get_3d_sincos_pos_embed(embed_dim=0, grid_size=4)

    def test_rejects_non_positive_grid(self):
        with pytest.raises(ValueError):
            get_3d_sincos_pos_embed(embed_dim=48, grid_size=(0, 4, 4))

    def test_values_bounded(self):
        pe = get_3d_sincos_pos_embed(embed_dim=48, grid_size=(2, 4, 4))
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_deterministic(self):
        pe1 = get_3d_sincos_pos_embed(embed_dim=48, grid_size=(2, 3, 3))
        pe2 = get_3d_sincos_pos_embed(embed_dim=48, grid_size=(2, 3, 3))
        assert torch.allclose(pe1, pe2)

    def test_different_positions_distinct(self):
        pe = get_3d_sincos_pos_embed(embed_dim=48, grid_size=(2, 3, 3))
        # (0,0,0) and (1,0,0) should differ along temporal block only
        assert not torch.allclose(pe[0], pe[9])  # different t
        assert not torch.allclose(pe[0], pe[3])  # different h
        assert not torch.allclose(pe[0], pe[1])  # different w

    def test_flatten_order_is_t_h_w(self):
        # With T=2, H=3, W=3 → first 9 rows share the same temporal block.
        pe = get_3d_sincos_pos_embed(embed_dim=48, grid_size=(2, 3, 3))
        # Temporal block = first 2 * (48//6) = 16 dims
        t_dims = 2 * (48 // 6)
        assert torch.allclose(pe[0, :t_dims], pe[8, :t_dims])
        assert not torch.allclose(pe[0, :t_dims], pe[9, :t_dims])


@pytest.mark.unit
class TestGetSincosEmbed:
    """Test get sincos embed."""

    def test_1d_mode(self):
        pe = get_sincos_pos_embed(embed_dim=64, num_patches=100, mode="1d")
        expected = get_1d_sincos_pos_embed(embed_dim=64, length=100)
        assert torch.allclose(pe, expected)

    def test_2d_mode(self):
        pe = get_sincos_pos_embed(embed_dim=64, num_patches=49, mode="2d", grid_size=7)
        expected = get_2d_sincos_pos_embed(embed_dim=64, grid_size=7)
        assert torch.allclose(pe, expected)

    def test_3d_mode(self):
        pe = get_sincos_pos_embed(
            embed_dim=48, num_patches=32, mode="3d", grid_size=(2, 4, 4)
        )
        expected = get_3d_sincos_pos_embed(embed_dim=48, grid_size=(2, 4, 4))
        assert torch.allclose(pe, expected)

    def test_2d_mode_requires_grid_size(self):
        with pytest.raises(ValueError):
            get_sincos_pos_embed(embed_dim=64, num_patches=49, mode="2d")

    def test_3d_mode_requires_grid_size(self):
        with pytest.raises(ValueError):
            get_sincos_pos_embed(embed_dim=48, num_patches=32, mode="3d")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            get_sincos_pos_embed(embed_dim=64, num_patches=100, mode="4d")

    def test_cls_token_passthrough(self):
        pe = get_sincos_pos_embed(
            embed_dim=64, num_patches=100, mode="1d", cls_token=True
        )
        assert pe.shape == (101, 64)


@pytest.mark.unit
class TestGetTimestepEmbed:
    """Test continuous timestep sinusoidal embeddings."""

    def test_output_shape_even_dim(self):
        t = torch.linspace(0.0, 1.0, 8)
        emb = get_timestep_embed(t, dim=64)
        assert emb.shape == (8, 64)

    def test_output_shape_odd_dim(self):
        t = torch.linspace(0.0, 1.0, 4)
        emb = get_timestep_embed(t, dim=7)
        assert emb.shape == (4, 7)

    def test_handles_column_input(self):
        t = torch.linspace(0.0, 1.0, 5).unsqueeze(1)
        emb = get_timestep_embed(t, dim=32)
        assert emb.shape == (5, 32)

    def test_values_bounded(self):
        emb = get_timestep_embed(torch.linspace(0.0, 1.0, 16), dim=64)
        assert emb.min() >= -1.0
        assert emb.max() <= 1.0

    def test_different_timesteps_distinct(self):
        emb = get_timestep_embed(torch.tensor([0.1, 0.5, 0.9]), dim=32)
        assert not torch.allclose(emb[0], emb[1])
        assert not torch.allclose(emb[1], emb[2])

    def test_zero_timestep_matches_closed_form(self):
        emb = get_timestep_embed(torch.zeros(1), dim=8)
        # cos(0)=1 for first half, sin(0)=0 for second half
        expected = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(emb, expected)

    def test_deterministic(self):
        t = torch.linspace(0.0, 1.0, 6)
        assert torch.allclose(
            get_timestep_embed(t, dim=32), get_timestep_embed(t, dim=32)
        )


@pytest.mark.unit
class TestInterpolatePosEmbed:
    """Test bilinear/bicubic positional embedding interpolation."""

    def test_identity_no_change(self):
        pe = torch.randn(1, 196, 64)
        out = interpolate_pos_embed(pe, src_size=(14, 14), tgt_size=(14, 14))
        assert torch.equal(out, pe)

    def test_upsample_shape(self):
        pe = torch.randn(1, 196, 64)
        out = interpolate_pos_embed(pe, src_size=(14, 14), tgt_size=(16, 16))
        assert out.shape == (1, 256, 64)

    def test_downsample_shape(self):
        pe = torch.randn(1, 256, 64)
        out = interpolate_pos_embed(pe, src_size=(16, 16), tgt_size=(8, 8))
        assert out.shape == (1, 64, 64)

    def test_with_cls_prefix_preserved(self):
        pe = torch.randn(1, 197, 64)  # 1 + 14*14
        cls = pe[:, :1].clone()
        out = interpolate_pos_embed(
            pe, src_size=(14, 14), tgt_size=(16, 16), num_prefix_tokens=1
        )
        assert out.shape == (1, 257, 64)
        assert torch.equal(out[:, :1], cls)

    def test_with_multiple_prefix_tokens(self):
        pe = torch.randn(1, 5 + 49, 32)  # 5 registers + 7*7
        prefix = pe[:, :5].clone()
        out = interpolate_pos_embed(
            pe, src_size=(7, 7), tgt_size=(14, 14), num_prefix_tokens=5
        )
        assert out.shape == (1, 5 + 196, 32)
        assert torch.equal(out[:, :5], prefix)

    def test_2d_input_round_trip(self):
        pe = torch.randn(196, 64)
        out = interpolate_pos_embed(pe, src_size=(14, 14), tgt_size=(7, 7))
        assert out.dim() == 2
        assert out.shape == (49, 64)

    def test_identity_2d_input(self):
        pe = torch.randn(49, 32)
        out = interpolate_pos_embed(pe, src_size=(7, 7), tgt_size=(7, 7))
        assert out.dim() == 2
        assert torch.equal(out, pe)

    def test_length_mismatch_raises(self):
        pe = torch.randn(1, 100, 64)
        with pytest.raises(ValueError):
            interpolate_pos_embed(pe, src_size=(14, 14), tgt_size=(16, 16))

    def test_bad_dims_raise(self):
        with pytest.raises(ValueError):
            interpolate_pos_embed(
                torch.randn(196, 64), src_size=(14, 0), tgt_size=(16, 16)
            )

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            interpolate_pos_embed(
                torch.randn(1, 1, 196, 64), src_size=(14, 14), tgt_size=(16, 16)
            )

    @pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic", "area"])
    def test_modes_work(self, mode):
        pe = torch.randn(1, 196, 64)
        out = interpolate_pos_embed(pe, src_size=(14, 14), tgt_size=(16, 16), mode=mode)
        assert out.shape == (1, 256, 64)


@pytest.mark.unit
class TestApplyRotaryEmb:
    """Test the standalone apply_rotary_emb helper."""

    def _freqs(self, seq_len, dim):
        # Random frequencies, then build matching cos/sin tables.
        theta = torch.randn(seq_len, dim)
        return theta.cos(), theta.sin()

    def test_shape_preserved(self):
        x = torch.randn(2, 4, 10, 16)
        cos, sin = self._freqs(10, 16)
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_identity_when_theta_zero(self):
        x = torch.randn(2, 3, 7, 8)
        cos = torch.ones(7, 8)
        sin = torch.zeros(7, 8)
        assert torch.allclose(apply_rotary_emb(x, cos, sin), x)

    def test_quarter_turn_rotation(self):
        # theta = pi/2 → (x1, x2) -> (-x2, x1) on each pair
        x = torch.randn(1, 1, 4, 8)
        theta = torch.full((4, 8), math.pi / 2)
        cos, sin = theta.cos(), theta.sin()
        out = apply_rotary_emb(x, cos, sin)
        x1, x2 = x[..., ::2], x[..., 1::2]
        expected = torch.stack([-x2, x1], dim=-1).flatten(-2)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_preserves_norm(self):
        x = torch.randn(2, 2, 6, 16)
        cos, sin = self._freqs(6, 16)
        out = apply_rotary_emb(x, cos, sin)
        # Rotation is norm-preserving on each (x1, x2) pair.
        assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


@pytest.mark.unit
class TestRotaryPositionEmbedding1D:
    """Test the 1D RoPE module."""

    def test_buffer_shapes(self):
        rope = RotaryPositionEmbedding1D(head_dim=64)
        assert rope.inv_freq.shape == (32,)  # head_dim // 2

    def test_head_dim_too_small_raises(self):
        with pytest.raises(ValueError):
            RotaryPositionEmbedding1D(head_dim=0)

    def test_head_dim_odd_raises(self):
        with pytest.raises(ValueError):
            RotaryPositionEmbedding1D(head_dim=7)

    def test_get_freqs_shape(self):
        rope = RotaryPositionEmbedding1D(head_dim=32)
        cos, sin = rope.get_freqs(seq_len=10, device=torch.device("cpu"))
        assert cos.shape == (10, 32)
        assert sin.shape == (10, 32)

    def test_get_freqs_recomputes_on_seq_change(self):
        rope = RotaryPositionEmbedding1D(head_dim=32)
        cos1, _ = rope.get_freqs(8, torch.device("cpu"))
        cos2, _ = rope.get_freqs(16, torch.device("cpu"))
        assert cos1.shape != cos2.shape
        assert rope._cached_seq_len == 16

    def test_get_freqs_deterministic(self):
        rope = RotaryPositionEmbedding1D(head_dim=32)
        a = rope.get_freqs(8, torch.device("cpu"))
        b = rope.get_freqs(8, torch.device("cpu"))
        assert torch.allclose(a[0], b[0])
        assert torch.allclose(a[1], b[1])

    def test_forward_shapes_preserved(self):
        rope = RotaryPositionEmbedding1D(head_dim=64)
        q = torch.randn(2, 4, 32, 64)
        k = torch.randn(2, 4, 32, 64)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_forward_preserves_norm(self):
        rope = RotaryPositionEmbedding1D(head_dim=64)
        q = torch.randn(2, 4, 32, 64)
        k = torch.randn(2, 4, 32, 64)
        q_rot, k_rot = rope(q, k)
        assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
        assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4)

    def test_zero_position_is_identity(self):
        # At seq_len=1, the only position is 0 → theta=0 → no rotation.
        rope = RotaryPositionEmbedding1D(head_dim=32)
        x = torch.randn(1, 1, 1, 32)
        cos, sin = rope.get_freqs(1, torch.device("cpu"))
        assert torch.allclose(apply_rotary_emb(x, cos, sin), x)

    def test_relative_position_invariance(self):
        # ⟨R_m q, R_n k⟩ depends only on (n-m). Shifting both positions
        # by the same delta must leave all pairwise dot products unchanged.
        rope = RotaryPositionEmbedding1D(head_dim=64)
        seq = 16
        shift = 5
        q = torch.randn(1, 1, seq, 64)
        k = torch.randn(1, 1, seq, 64)
        cos_full, sin_full = rope.get_freqs(seq + shift, torch.device("cpu"))
        # Rotate q,k at positions [0, seq) vs [shift, shift+seq).
        q0 = apply_rotary_emb(q, cos_full[:seq], sin_full[:seq])
        k0 = apply_rotary_emb(k, cos_full[:seq], sin_full[:seq])
        q1 = apply_rotary_emb(
            q, cos_full[shift : shift + seq], sin_full[shift : shift + seq]
        )
        k1 = apply_rotary_emb(
            k, cos_full[shift : shift + seq], sin_full[shift : shift + seq]
        )
        attn0 = (q0 @ k0.transpose(-1, -2)).squeeze()
        attn1 = (q1 @ k1.transpose(-1, -2)).squeeze()
        assert torch.allclose(attn0, attn1, atol=1e-4)

    def test_cache_buffers_not_persistent(self):
        rope = RotaryPositionEmbedding1D(head_dim=32)
        rope.get_freqs(8, torch.device("cpu"))
        state = rope.state_dict()
        assert "inv_freq" in state
        assert "cos_cached" not in state
        assert "sin_cached" not in state

    def test_dtype_propagates(self):
        rope = RotaryPositionEmbedding1D(head_dim=32)
        cos, sin = rope.get_freqs(8, torch.device("cpu"), dtype=torch.float64)
        assert cos.dtype == torch.float64
        assert sin.dtype == torch.float64


@pytest.mark.unit
class TestRotaryPositionEmbedding2D:
    """Test the learnable-buffer RoPE module."""

    def test_buffer_shapes(self):
        rope = RotaryPositionEmbedding2D(head_dim=64)
        assert rope.inv_freq.shape == (16,)  # head_dim // 4

    def test_head_dim_too_small_raises(self):
        with pytest.raises(ValueError):
            RotaryPositionEmbedding2D(head_dim=2)

    def test_get_freqs_shape(self):
        rope = RotaryPositionEmbedding2D(head_dim=32)
        cos, sin = rope.get_freqs(grid_h=4, grid_w=5, device=torch.device("cpu"))
        assert cos.shape == (20, 32)
        assert sin.shape == (20, 32)

    def test_get_freqs_recomputes_on_grid_change(self):
        rope = RotaryPositionEmbedding2D(head_dim=32)
        cos1, _ = rope.get_freqs(4, 4, torch.device("cpu"))
        cos2, _ = rope.get_freqs(6, 6, torch.device("cpu"))
        assert cos1.shape != cos2.shape
        assert rope._cached_grid_h == 6 and rope._cached_grid_w == 6

    def test_get_freqs_deterministic(self):
        rope = RotaryPositionEmbedding2D(head_dim=32)
        a = rope.get_freqs(4, 4, torch.device("cpu"))
        b = rope.get_freqs(4, 4, torch.device("cpu"))
        assert torch.allclose(a[0], b[0])
        assert torch.allclose(a[1], b[1])

    def test_forward_shapes_preserved(self):
        rope = RotaryPositionEmbedding2D(head_dim=64)
        q = torch.randn(2, 4, 49, 64)
        k = torch.randn(2, 4, 49, 64)
        q_rot, k_rot = rope(q, k, grid_h=7, grid_w=7)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_forward_preserves_norm(self):
        rope = RotaryPositionEmbedding2D(head_dim=64)
        q = torch.randn(2, 4, 49, 64)
        k = torch.randn(2, 4, 49, 64)
        q_rot, k_rot = rope(q, k, grid_h=7, grid_w=7)
        assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
        assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4)

    def test_cache_buffers_not_persistent(self):
        rope = RotaryPositionEmbedding2D(head_dim=32)
        rope.get_freqs(4, 4, torch.device("cpu"))
        state = rope.state_dict()
        assert "inv_freq" in state
        assert "cos_cached" not in state
        assert "sin_cached" not in state

    def test_dtype_propagates(self):
        rope = RotaryPositionEmbedding2D(head_dim=32)
        cos, sin = rope.get_freqs(4, 4, torch.device("cpu"), dtype=torch.float64)
        assert cos.dtype == torch.float64
        assert sin.dtype == torch.float64


@pytest.mark.unit
class TestRotaryPositionEmbedding3D:
    """Test the 3D RoPE module."""

    def test_buffer_shapes(self):
        rope = RotaryPositionEmbedding3D(head_dim=48)
        assert rope.inv_freq.shape == (8,)  # head_dim // 6

    def test_head_dim_not_divisible_by_6_raises(self):
        with pytest.raises(ValueError):
            RotaryPositionEmbedding3D(head_dim=32)

    def test_head_dim_too_small_raises(self):
        with pytest.raises(ValueError):
            RotaryPositionEmbedding3D(head_dim=0)

    def test_get_freqs_shape(self):
        rope = RotaryPositionEmbedding3D(head_dim=48)
        cos, sin = rope.get_freqs(2, 3, 4, device=torch.device("cpu"))
        assert cos.shape == (24, 48)
        assert sin.shape == (24, 48)

    def test_get_freqs_recomputes_on_grid_change(self):
        rope = RotaryPositionEmbedding3D(head_dim=48)
        cos1, _ = rope.get_freqs(2, 3, 3, torch.device("cpu"))
        cos2, _ = rope.get_freqs(3, 3, 3, torch.device("cpu"))
        assert cos1.shape != cos2.shape
        assert rope._cached_grid_t == 3

    def test_forward_shapes_preserved(self):
        rope = RotaryPositionEmbedding3D(head_dim=48)
        q = torch.randn(1, 2, 24, 48)
        k = torch.randn(1, 2, 24, 48)
        q_rot, k_rot = rope(q, k, grid_t=2, grid_h=3, grid_w=4)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_forward_preserves_norm(self):
        rope = RotaryPositionEmbedding3D(head_dim=48)
        q = torch.randn(1, 2, 24, 48)
        k = torch.randn(1, 2, 24, 48)
        q_rot, k_rot = rope(q, k, grid_t=2, grid_h=3, grid_w=4)
        assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
        assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4)

    def test_cache_buffers_not_persistent(self):
        rope = RotaryPositionEmbedding3D(head_dim=48)
        rope.get_freqs(2, 3, 4, torch.device("cpu"))
        state = rope.state_dict()
        assert "inv_freq" in state
        assert "cos_cached" not in state
        assert "sin_cached" not in state

    def test_dtype_propagates(self):
        rope = RotaryPositionEmbedding3D(head_dim=48)
        cos, sin = rope.get_freqs(2, 3, 4, torch.device("cpu"), dtype=torch.float64)
        assert cos.dtype == torch.float64
        assert sin.dtype == torch.float64


@pytest.mark.unit
class TestBuildRotaryPosEmbed:
    """Test the factory that picks a RoPE module by mode string."""

    def test_none_returns_none(self):
        assert build_rotary_pos_embed(None, head_dim=32) is None

    def test_1d_returns_1d_module(self):
        m = build_rotary_pos_embed("1d", head_dim=32)
        assert isinstance(m, RotaryPositionEmbedding1D)

    def test_2d_returns_2d_module(self):
        m = build_rotary_pos_embed("2d", head_dim=32)
        assert isinstance(m, RotaryPositionEmbedding2D)

    def test_3d_returns_3d_module(self):
        m = build_rotary_pos_embed("3d", head_dim=48)
        assert isinstance(m, RotaryPositionEmbedding3D)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            build_rotary_pos_embed("4d", head_dim=32)

    def test_forwards_params(self):
        m = build_rotary_pos_embed("2d", head_dim=32, max_grid_size=16, base=20000.0)
        assert m.max_grid_size == 16
        assert m.base == 20000.0


@pytest.mark.unit
class TestAttentionRopeModes:
    """Test that Attention wires the RoPE mode string end-to-end."""

    def test_string_1d_runs(self):
        from stable_pretraining.backbone.vit import Attention

        attn = Attention(dim=64, num_heads=4, use_rope="1d")
        assert attn.rope_mode == "1d"
        assert isinstance(attn.rope, RotaryPositionEmbedding1D)
        out = attn(torch.randn(1, 10, 64))
        assert out.shape == (1, 10, 64)

    def test_string_2d_runs(self):
        from stable_pretraining.backbone.vit import Attention

        attn = Attention(dim=64, num_heads=4, use_rope="2d")
        assert attn.rope_mode == "2d"
        out = attn(torch.randn(1, 16, 64), grid_size=(4, 4))
        assert out.shape == (1, 16, 64)

    def test_string_3d_runs(self):
        from stable_pretraining.backbone.vit import Attention

        attn = Attention(dim=72, num_heads=4, use_rope="3d")
        assert attn.rope_mode == "3d"
        out = attn(torch.randn(1, 24, 72), grid_size=(2, 3, 4))
        assert out.shape == (1, 24, 72)

    def test_bool_true_aliases_2d(self):
        from stable_pretraining.backbone.vit import Attention

        attn = Attention(dim=64, num_heads=4, use_rope=True)
        assert attn.rope_mode == "2d"

    def test_bool_false_disables(self):
        from stable_pretraining.backbone.vit import Attention

        attn = Attention(dim=64, num_heads=4, use_rope=False)
        assert attn.rope_mode is None
        assert attn.rope is None

    def test_none_disables(self):
        from stable_pretraining.backbone.vit import Attention

        attn = Attention(dim=64, num_heads=4, use_rope=None)
        assert attn.rope_mode is None

    def test_invalid_mode_raises(self):
        from stable_pretraining.backbone.vit import Attention

        with pytest.raises(ValueError):
            Attention(dim=64, num_heads=4, use_rope="4d")

    def test_3d_requires_triple_grid(self):
        from stable_pretraining.backbone.vit import Attention

        attn = Attention(dim=72, num_heads=4, use_rope="3d")
        with pytest.raises(ValueError):
            attn(torch.randn(1, 24, 72), grid_size=(3, 8))
