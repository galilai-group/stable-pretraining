"""Standalone pytest file for FlexibleTransformer and related modules.

Run with: pytest test_flexible_transformer.py -v -m unit
"""

import pytest
import torch
import torch.nn as nn

# Assuming the module is saved as flexible_transformer.py
# Adjust the import path as needed
from stable_pretraining.backbone import (
    Attention,
    CrossAttention,
    TransformerBlock,
    FlexibleTransformer,
    modulate,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def context_len():
    return 32


@pytest.fixture
def dim():
    return 64


@pytest.fixture
def num_heads():
    return 4


@pytest.fixture
def num_patches():
    return 49  # 7x7 grid


@pytest.mark.unit
class TestModulate:
    """Tests for the modulate helper function."""

    def test_modulate_identity(self, batch_size, seq_len, dim):
        """Test modulate with zero shift/scale is identity."""
        x = torch.randn(batch_size, seq_len, dim)
        shift = torch.zeros(batch_size, 1, dim)
        scale = torch.zeros(batch_size, 1, dim)
        out = modulate(x, shift, scale)
        assert torch.allclose(out, x)

    def test_modulate_scale_only(self, batch_size, seq_len, dim):
        """Test modulate with scale doubles values."""
        x = torch.randn(batch_size, seq_len, dim)
        shift = torch.zeros(batch_size, 1, dim)
        scale = torch.ones(batch_size, 1, dim)  # scale = 1 means multiply by 2
        out = modulate(x, shift, scale)
        assert torch.allclose(out, x * 2)

    def test_modulate_shift_only(self, batch_size, seq_len, dim):
        """Test modulate with shift adds offset."""
        x = torch.randn(batch_size, seq_len, dim)
        shift = torch.ones(batch_size, 1, dim) * 5
        scale = torch.zeros(batch_size, 1, dim)
        out = modulate(x, shift, scale)
        assert torch.allclose(out, x + 5)


# =============================================================================
# Attention Module Tests
# =============================================================================


@pytest.mark.unit
class TestAttention:
    """Tests for the Attention (self-attention) module."""

    def test_output_shape(self, batch_size, seq_len, dim, num_heads):
        """Test attention output has same shape as input."""
        attn = Attention(dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_deterministic(self, batch_size, seq_len, dim, num_heads):
        """Test attention is deterministic in eval mode."""
        attn = Attention(dim, num_heads=num_heads, attn_drop=0.0, proj_drop=0.0)
        attn.eval()
        x = torch.randn(batch_size, seq_len, dim)
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2)

    def test_gradient_flow(self, batch_size, seq_len, dim, num_heads):
        """Test gradients flow through attention."""
        attn = Attention(dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nans_in_output(self, batch_size, seq_len, dim, num_heads):
        """Test no NaN values in attention output."""
        attn = Attention(dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim)
        out = attn(x)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_various_head_counts(self, batch_size, seq_len, num_heads):
        """Test attention works with various head counts."""
        dim = 64  # Must be divisible by all head counts
        attn = Attention(dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim)
        out = attn(x)
        assert out.shape == x.shape


@pytest.mark.unit
class TestCrossAttention:
    """Tests for the CrossAttention module."""

    def test_output_shape(self, batch_size, seq_len, context_len, dim, num_heads):
        """Test cross-attention output has query shape."""
        cross_attn = CrossAttention(dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim)
        context = torch.randn(batch_size, context_len, dim)
        out = cross_attn(x, context)
        assert out.shape == x.shape

    def test_different_context_dim(
        self, batch_size, seq_len, context_len, dim, num_heads
    ):
        """Test cross-attention with different context dimension."""
        context_dim = dim * 2
        cross_attn = CrossAttention(dim, context_dim=context_dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim)
        context = torch.randn(batch_size, context_len, context_dim)
        out = cross_attn(x, context)
        assert out.shape == x.shape

    def test_gradient_flow_to_both_inputs(
        self, batch_size, seq_len, context_len, dim, num_heads
    ):
        """Test gradients flow to both query and context."""
        cross_attn = CrossAttention(dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        context = torch.randn(batch_size, context_len, dim, requires_grad=True)
        out = cross_attn(x, context)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert context.grad is not None

    def test_deterministic(self, batch_size, seq_len, context_len, dim, num_heads):
        """Test cross-attention is deterministic in eval mode."""
        cross_attn = CrossAttention(
            dim, num_heads=num_heads, attn_drop=0.0, proj_drop=0.0
        )
        cross_attn.eval()
        x = torch.randn(batch_size, seq_len, dim)
        context = torch.randn(batch_size, context_len, dim)
        out1 = cross_attn(x, context)
        out2 = cross_attn(x, context)
        assert torch.allclose(out1, out2)


# =============================================================================
# TransformerBlock Tests
# =============================================================================


@pytest.mark.unit
class TestTransformerBlockModes:
    """Tests for TransformerBlock attention modes."""

    def test_self_attn_only_mode(self, batch_size, seq_len, dim, num_heads):
        """Test Mode 3: self-attention only (joint attention)."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=False
        )
        x = torch.randn(batch_size, seq_len, dim)
        out = block(x)
        assert out.shape == x.shape

    def test_cross_attn_only_mode(
        self, batch_size, seq_len, context_len, dim, num_heads
    ):
        """Test Mode 1: cross-attention only."""
        block = TransformerBlock(
            dim, num_heads, self_attn=False, cross_attn=True, use_adaln=False
        )
        x = torch.randn(batch_size, seq_len, dim)
        context = torch.randn(batch_size, context_len, dim)
        out = block(x, context=context)
        assert out.shape == x.shape

    def test_both_attn_mode(self, batch_size, seq_len, context_len, dim, num_heads):
        """Test Mode 2: self-attention + cross-attention (decoder style)."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=True, use_adaln=False
        )
        x = torch.randn(batch_size, seq_len, dim)
        context = torch.randn(batch_size, context_len, dim)
        out = block(x, context=context)
        assert out.shape == x.shape

    def test_no_attn_raises_error(self, dim, num_heads):
        """Test that disabling both attention types raises error."""
        with pytest.raises(ValueError, match="At least one of"):
            TransformerBlock(dim, num_heads, self_attn=False, cross_attn=False)

    def test_cross_attn_requires_context(self, batch_size, seq_len, dim, num_heads):
        """Test cross-attention mode raises error without context."""
        block = TransformerBlock(
            dim, num_heads, self_attn=False, cross_attn=True, use_adaln=False
        )
        x = torch.randn(batch_size, seq_len, dim)
        with pytest.raises(ValueError, match="context required"):
            block(x)


@pytest.mark.unit
class TestTransformerBlockAdaLN:
    """Tests for TransformerBlock AdaLN conditioning."""

    def test_adaln_enabled(self, batch_size, seq_len, dim, num_heads):
        """Test block with AdaLN enabled."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=True
        )
        x = torch.randn(batch_size, seq_len, dim)
        cond = torch.randn(batch_size, dim)
        out = block(x, cond=cond)
        assert out.shape == x.shape

    def test_adaln_disabled(self, batch_size, seq_len, dim, num_heads):
        """Test block with AdaLN disabled (standard transformer)."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=False
        )
        x = torch.randn(batch_size, seq_len, dim)
        out = block(x)  # No cond needed
        assert out.shape == x.shape

    def test_adaln_requires_cond(self, batch_size, seq_len, dim, num_heads):
        """Test AdaLN mode raises error without conditioning."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=True
        )
        x = torch.randn(batch_size, seq_len, dim)
        with pytest.raises(ValueError, match="cond required"):
            block(x)

    def test_adaln_zero_init(self, dim, num_heads):
        """Test AdaLN MLP is zero-initialized (identity at init)."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=True
        )
        # Check the final linear layer is zero-initialized
        assert torch.allclose(
            block.adaLN_mlp[1].weight, torch.zeros_like(block.adaLN_mlp[1].weight)
        )
        assert torch.allclose(
            block.adaLN_mlp[1].bias, torch.zeros_like(block.adaLN_mlp[1].bias)
        )

    def test_adaln_gate_starts_at_zero(self, batch_size, seq_len, dim, num_heads):
        """Test that at initialization, AdaLN block is approximately identity."""
        torch.manual_seed(42)
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=True
        )
        block.eval()

        x = torch.randn(batch_size, seq_len, dim)
        cond = torch.randn(batch_size, dim)
        out = block(x, cond=cond)

        # With zero-init, the residual contributions should be zero,
        # so output should equal input
        assert torch.allclose(out, x, atol=1e-5)


@pytest.mark.unit
class TestTransformerBlockDropPath:
    """Tests for TransformerBlock drop path (stochastic depth)."""

    def test_drop_path_zero(self, batch_size, seq_len, dim, num_heads):
        """Test drop_path=0 is deterministic."""
        block = TransformerBlock(
            dim,
            num_heads,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
            drop_path=0.0,
        )
        block.eval()
        x = torch.randn(batch_size, seq_len, dim)
        out1 = block(x)
        out2 = block(x)
        assert torch.allclose(out1, out2)

    def test_drop_path_nonzero_train(self, batch_size, seq_len, dim, num_heads):
        """Test drop_path > 0 creates stochasticity in training."""
        block = TransformerBlock(
            dim,
            num_heads,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
            drop_path=0.5,
        )
        block.train()
        x = torch.randn(batch_size, seq_len, dim)

        # Run many times and check for variation
        outputs = [block(x.clone()) for _ in range(10)]
        # At least some outputs should differ (with high probability)
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        # Note: This could theoretically fail with very low probability
        assert not all_same, "Drop path should create variation in training mode"

    def test_drop_path_deterministic_eval(self, batch_size, seq_len, dim, num_heads):
        """Test drop_path is deterministic in eval mode."""
        block = TransformerBlock(
            dim,
            num_heads,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
            drop_path=0.5,
        )
        block.eval()
        x = torch.randn(batch_size, seq_len, dim)
        out1 = block(x)
        out2 = block(x)
        assert torch.allclose(out1, out2)


@pytest.mark.unit
class TestTransformerBlockGradients:
    """Tests for gradient flow through TransformerBlock."""

    @pytest.mark.parametrize("use_adaln", [True, False])
    @pytest.mark.parametrize(
        "self_attn,cross_attn", [(True, False), (False, True), (True, True)]
    )
    def test_gradient_flow_all_configs(
        self,
        batch_size,
        seq_len,
        context_len,
        dim,
        num_heads,
        use_adaln,
        self_attn,
        cross_attn,
    ):
        """Test gradients flow for all block configurations."""
        block = TransformerBlock(
            dim,
            num_heads,
            self_attn=self_attn,
            cross_attn=cross_attn,
            use_adaln=use_adaln,
        )

        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        context = (
            torch.randn(batch_size, context_len, dim, requires_grad=True)
            if cross_attn
            else None
        )
        cond = torch.randn(batch_size, dim, requires_grad=True) if use_adaln else None

        out = block(x, context=context, cond=cond)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        if cross_attn:
            assert context.grad is not None
        if use_adaln:
            assert cond.grad is not None


# =============================================================================
# FlexibleTransformer Tests
# =============================================================================


@pytest.mark.unit
class TestFlexibleTransformerInit:
    """Tests for FlexibleTransformer initialization."""

    def test_basic_init(self, num_patches):
        """Test basic initialization."""
        model = FlexibleTransformer(
            input_dim=768,
            hidden_dim=64,
            output_dim=768,
            num_patches=num_patches,
            depth=2,
            num_heads=4,
        )
        assert model is not None

    def test_hidden_dim_divisibility_check(self, num_patches):
        """Test that hidden_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible by"):
            FlexibleTransformer(
                input_dim=768,
                hidden_dim=65,  # Not divisible by 4
                output_dim=768,
                num_patches=num_patches,
                depth=2,
                num_heads=4,
            )

    def test_sincos_2d_requires_square(self):
        """Test sincos_2d with non-square patches raises error."""
        with pytest.raises(ValueError, match="perfect square"):
            FlexibleTransformer(
                input_dim=768,
                hidden_dim=64,
                output_dim=768,
                num_patches=50,  # Not a perfect square
                depth=2,
                num_heads=4,
                pos_embed_type="sincos_2d",
            )

    @pytest.mark.parametrize("pos_embed_type", ["sincos_1d", "sincos_2d", "learned"])
    def test_pos_embed_types(self, num_patches, pos_embed_type):
        """Test all positional embedding types initialize correctly."""
        model = FlexibleTransformer(
            input_dim=768,
            hidden_dim=64,
            output_dim=768,
            num_patches=num_patches,
            depth=2,
            num_heads=4,
            pos_embed_type=pos_embed_type,
        )
        assert hasattr(model, "pos_embed")

    def test_zero_init_output(self, num_patches):
        """Test output projection is zero-initialized when requested."""
        model = FlexibleTransformer(
            input_dim=768,
            hidden_dim=64,
            output_dim=768,
            num_patches=num_patches,
            depth=2,
            num_heads=4,
            zero_init_output=True,
        )
        assert torch.allclose(
            model.output_proj.weight, torch.zeros_like(model.output_proj.weight)
        )
        assert torch.allclose(
            model.output_proj.bias, torch.zeros_like(model.output_proj.bias)
        )


@pytest.mark.unit
class TestFlexibleTransformerForward:
    """Tests for FlexibleTransformer forward pass."""

    @pytest.fixture
    def model_config(self, num_patches):
        """Common model configuration."""
        return {
            "input_dim": 128,
            "hidden_dim": 64,
            "output_dim": 128,
            "num_patches": num_patches,
            "depth": 2,
            "num_heads": 4,
            "num_prefix_tokens": 1,
        }

    def test_output_shape(self, batch_size, model_config):
        """Test output has correct shape."""
        model = FlexibleTransformer(**model_config, use_adaln=True)

        n_ctx, n_qry = 20, 10
        context = torch.randn(batch_size, n_ctx, model_config["input_dim"])
        queries = torch.randn(batch_size, n_qry, model_config["input_dim"])
        context_idx = torch.randint(0, model_config["num_patches"], (batch_size, n_ctx))
        query_idx = torch.randint(0, model_config["num_patches"], (batch_size, n_qry))
        t = torch.rand(batch_size)

        # Set prefix token indices (first position)
        context_idx[:, 0] = 0

        out = model(context, queries, context_idx, query_idx, t=t)
        assert out.shape == (batch_size, n_qry, model_config["output_dim"])

    def test_mode_mae_decoder(self, batch_size, model_config):
        """Test MAE decoder mode (self_attn only, no adaln)."""
        model = FlexibleTransformer(
            **model_config,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
        )

        n_ctx, n_qry = 20, 10
        context = torch.randn(batch_size, n_ctx, model_config["input_dim"])
        queries = torch.randn(batch_size, n_qry, model_config["input_dim"])
        context_idx = torch.randint(0, model_config["num_patches"], (batch_size, n_ctx))
        query_idx = torch.randint(0, model_config["num_patches"], (batch_size, n_qry))
        context_idx[:, 0] = 0

        out = model(context, queries, context_idx, query_idx)  # No timestep needed
        assert out.shape == (batch_size, n_qry, model_config["output_dim"])

    def test_mode_ijepa_predictor(self, batch_size, model_config):
        """Test IJEPA predictor mode (self + cross attn, no adaln)."""
        model = FlexibleTransformer(
            **model_config,
            self_attn=True,
            cross_attn=True,
            use_adaln=False,
        )

        n_ctx, n_qry = 20, 10
        context = torch.randn(batch_size, n_ctx, model_config["input_dim"])
        queries = torch.randn(batch_size, n_qry, model_config["input_dim"])
        context_idx = torch.randint(0, model_config["num_patches"], (batch_size, n_ctx))
        query_idx = torch.randint(0, model_config["num_patches"], (batch_size, n_qry))
        context_idx[:, 0] = 0

        out = model(context, queries, context_idx, query_idx)
        assert out.shape == (batch_size, n_qry, model_config["output_dim"])

    def test_mode_dit_flow(self, batch_size, model_config):
        """Test DiT/Flow mode (joint attn with adaln)."""
        model = FlexibleTransformer(
            **model_config,
            self_attn=True,
            cross_attn=False,
            use_adaln=True,
        )

        n_ctx, n_qry = 20, 10
        context = torch.randn(batch_size, n_ctx, model_config["input_dim"])
        queries = torch.randn(batch_size, n_qry, model_config["input_dim"])
        context_idx = torch.randint(0, model_config["num_patches"], (batch_size, n_ctx))
        query_idx = torch.randint(0, model_config["num_patches"], (batch_size, n_qry))
        t = torch.rand(batch_size)
        context_idx[:, 0] = 0

        out = model(context, queries, context_idx, query_idx, t=t)
        assert out.shape == (batch_size, n_qry, model_config["output_dim"])

    def test_adaln_requires_timestep(self, batch_size, model_config):
        """Test AdaLN mode raises error without timestep."""
        model = FlexibleTransformer(**model_config, use_adaln=True)

        n_ctx, n_qry = 20, 10
        context = torch.randn(batch_size, n_ctx, model_config["input_dim"])
        queries = torch.randn(batch_size, n_qry, model_config["input_dim"])
        context_idx = torch.randint(0, model_config["num_patches"], (batch_size, n_ctx))
        query_idx = torch.randint(0, model_config["num_patches"], (batch_size, n_qry))
        context_idx[:, 0] = 0

        with pytest.raises(ValueError, match="Timestep t required"):
            model(context, queries, context_idx, query_idx)  # Missing t

    def test_no_prefix_tokens(self, batch_size, num_patches):
        """Test model with no prefix tokens."""
        model = FlexibleTransformer(
            input_dim=128,
            hidden_dim=64,
            output_dim=128,
            num_patches=num_patches,
            depth=2,
            num_heads=4,
            num_prefix_tokens=0,
            use_adaln=False,
        )

        n_ctx, n_qry = 20, 10
        context = torch.randn(batch_size, n_ctx, 128)
        queries = torch.randn(batch_size, n_qry, 128)
        context_idx = torch.randint(0, num_patches, (batch_size, n_ctx))
        query_idx = torch.randint(0, num_patches, (batch_size, n_qry))

        out = model(context, queries, context_idx, query_idx, num_prefix=0)
        assert out.shape == (batch_size, n_qry, 128)


@pytest.mark.unit
class TestFlexibleTransformerGradients:
    """Tests for gradient flow through FlexibleTransformer."""

    @pytest.mark.parametrize("use_adaln", [True, False])
    @pytest.mark.parametrize("cross_attn", [True, False])
    def test_gradient_flow(self, batch_size, num_patches, use_adaln, cross_attn):
        """Test gradients flow for different configurations."""
        model = FlexibleTransformer(
            input_dim=128,
            hidden_dim=64,
            output_dim=128,
            num_patches=num_patches,
            depth=2,
            num_heads=4,
            self_attn=True,
            cross_attn=cross_attn,
            use_adaln=use_adaln,
            num_prefix_tokens=1,
        )

        n_ctx, n_qry = 20, 10
        context = torch.randn(batch_size, n_ctx, 128, requires_grad=True)
        queries = torch.randn(batch_size, n_qry, 128, requires_grad=True)
        context_idx = torch.randint(0, num_patches, (batch_size, n_ctx))
        query_idx = torch.randint(0, num_patches, (batch_size, n_qry))
        context_idx[:, 0] = 0
        t = torch.rand(batch_size) if use_adaln else None

        out = model(context, queries, context_idx, query_idx, t=t)
        loss = out.sum()
        loss.backward()

        assert context.grad is not None
        assert queries.grad is not None
        assert not torch.isnan(context.grad).any()
        assert not torch.isnan(queries.grad).any()


@pytest.mark.unit
class TestFlexibleTransformerDropPath:
    """Tests for drop path in FlexibleTransformer."""

    def test_drop_path_increases_linearly(self, num_patches):
        """Test drop path rate increases linearly through layers."""
        model = FlexibleTransformer(
            input_dim=128,
            hidden_dim=64,
            output_dim=128,
            num_patches=num_patches,
            depth=4,
            num_heads=4,
            drop_path_rate=0.1,
            use_adaln=False,
        )

        # Check that drop path rates increase
        drop_rates = []
        for block in model.blocks:
            if hasattr(block.drop_path3, "drop_prob"):
                drop_rates.append(block.drop_path3.drop_prob)
            else:
                drop_rates.append(0.0)

        # Should be [0, 0.033, 0.066, 0.1] approximately
        assert drop_rates[-1] == pytest.approx(0.1, abs=0.01)
        assert drop_rates[0] == pytest.approx(0.0, abs=0.01)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_training_step_simulation(self, batch_size, num_patches):
        """Simulate a complete training step."""
        model = FlexibleTransformer(
            input_dim=128,
            hidden_dim=64,
            output_dim=128,
            num_patches=num_patches,
            depth=2,
            num_heads=4,
            use_adaln=True,
            drop_path_rate=0.1,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create fake batch
        n_ctx, n_qry = 30, 19
        context = torch.randn(batch_size, n_ctx, 128)
        queries = torch.randn(batch_size, n_qry, 128)
        targets = torch.randn(batch_size, n_qry, 128)
        context_idx = torch.randint(0, num_patches, (batch_size, n_ctx))
        query_idx = torch.randint(0, num_patches, (batch_size, n_qry))
        context_idx[:, 0] = 0
        t = torch.rand(batch_size)

        # Training step
        model.train()
        optimizer.zero_grad()
        out = model(context, queries, context_idx, query_idx, t=t)
        loss = nn.functional.mse_loss(out, targets)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)

    def test_eval_mode_deterministic(self, batch_size, num_patches):
        """Test model is deterministic in eval mode."""
        model = FlexibleTransformer(
            input_dim=128,
            hidden_dim=64,
            output_dim=128,
            num_patches=num_patches,
            depth=2,
            num_heads=4,
            use_adaln=True,
            drop_path_rate=0.1,
        )
        model.eval()

        n_ctx, n_qry = 30, 19
        context = torch.randn(batch_size, n_ctx, 128)
        queries = torch.randn(batch_size, n_qry, 128)
        context_idx = torch.randint(0, num_patches, (batch_size, n_ctx))
        query_idx = torch.randint(0, num_patches, (batch_size, n_qry))
        context_idx[:, 0] = 0
        t = torch.rand(batch_size)

        out1 = model(context, queries, context_idx, query_idx, t=t)
        out2 = model(context, queries, context_idx, query_idx, t=t)

        assert torch.allclose(out1, out2)

    def test_different_batch_sizes(self, num_patches):
        """Test model works with different batch sizes."""
        model = FlexibleTransformer(
            input_dim=128,
            hidden_dim=64,
            output_dim=128,
            num_patches=num_patches,
            depth=2,
            num_heads=4,
            use_adaln=False,
        )
        model.eval()

        for bs in [1, 2, 8, 16]:
            n_ctx, n_qry = 20, 10
            context = torch.randn(bs, n_ctx, 128)
            queries = torch.randn(bs, n_qry, 128)
            context_idx = torch.randint(0, num_patches, (bs, n_ctx))
            query_idx = torch.randint(0, num_patches, (bs, n_qry))
            context_idx[:, 0] = 0

            out = model(context, queries, context_idx, query_idx)
            assert out.shape == (bs, n_qry, 128)

    def test_different_sequence_lengths(self, batch_size, num_patches):
        """Test model works with different context/query lengths."""
        model = FlexibleTransformer(
            input_dim=128,
            hidden_dim=64,
            output_dim=128,
            num_patches=num_patches,
            depth=2,
            num_heads=4,
            use_adaln=False,
        )
        model.eval()

        for n_ctx, n_qry in [(10, 5), (30, 19), (40, 9), (5, 44)]:
            context = torch.randn(batch_size, n_ctx, 128)
            queries = torch.randn(batch_size, n_qry, 128)
            context_idx = torch.randint(0, num_patches, (batch_size, n_ctx))
            query_idx = torch.randint(0, num_patches, (batch_size, n_qry))
            context_idx[:, 0] = 0

            out = model(context, queries, context_idx, query_idx)
            assert out.shape == (batch_size, n_qry, 128)


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_single_token_sequence(self, batch_size, dim, num_heads):
        """Test with single token sequences."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=False
        )
        x = torch.randn(batch_size, 1, dim)
        out = block(x)
        assert out.shape == x.shape

    def test_single_token_cross_attn(self, batch_size, dim, num_heads):
        """Test cross-attention with single query token."""
        block = TransformerBlock(
            dim, num_heads, self_attn=False, cross_attn=True, use_adaln=False
        )
        x = torch.randn(batch_size, 1, dim)
        context = torch.randn(batch_size, 10, dim)
        out = block(x, context=context)
        assert out.shape == x.shape

    def test_large_sequence(self, batch_size, dim, num_heads):
        """Test with larger sequence length."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=False
        )
        x = torch.randn(batch_size, 256, dim)
        out = block(x)
        assert out.shape == x.shape

    def test_batch_size_one(self, dim, num_heads):
        """Test with batch size of 1."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=False
        )
        x = torch.randn(1, 16, dim)
        out = block(x)
        assert out.shape == x.shape

    def test_zeros_input(self, batch_size, seq_len, dim, num_heads):
        """Test with zero inputs."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=False
        )
        x = torch.zeros(batch_size, seq_len, dim)
        out = block(x)
        assert not torch.isnan(out).any()

    def test_large_values(self, batch_size, seq_len, dim, num_heads):
        """Test with large input values."""
        block = TransformerBlock(
            dim, num_heads, self_attn=True, cross_attn=False, use_adaln=False
        )
        x = torch.randn(batch_size, seq_len, dim) * 100
        out = block(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
