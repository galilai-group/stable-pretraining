"""Unit tests for MAE decoder."""

import pytest
import torch

from stable_pretraining.backbone import MAEDecoder


@pytest.mark.unit
class TestMAEDecoderInit:
    """Test MAE decoder initialization."""

    def test_default_initialization(self):
        decoder = MAEDecoder(num_patches=196)
        assert decoder is not None

    def test_custom_dimensions(self):
        decoder = MAEDecoder(
            embed_dim=512,
            decoder_embed_dim=256,
            output_dim=768,
            num_patches=64,
            depth=4,
            num_heads=8,
        )
        # Check transformer internal projections
        assert decoder.transformer.context_proj.in_features == 512
        assert decoder.transformer.context_proj.out_features == 256
        assert decoder.transformer.output_proj.in_features == 256
        assert decoder.transformer.output_proj.out_features == 768
        assert len(decoder.transformer.blocks) == 4

    def test_mask_token_is_learnable(self):
        decoder = MAEDecoder(embed_dim=512, num_patches=196)
        assert isinstance(decoder.mask_token, torch.nn.Parameter)
        assert decoder.mask_token.requires_grad
        assert decoder.mask_token.shape == (
            1,
            1,
            512,
        )  # embed_dim, not decoder_embed_dim

    def test_sincos_1d_pos_embed_is_buffer(self):
        decoder = MAEDecoder(num_patches=100, pos_embed_type="sincos_1d")
        assert "pos_embed" in dict(decoder.transformer.named_buffers())
        assert not decoder.transformer.pos_embed.requires_grad

    def test_sincos_2d_pos_embed_is_buffer(self):
        decoder = MAEDecoder(num_patches=49, pos_embed_type="sincos_2d", grid_size=7)
        assert "pos_embed" in dict(decoder.transformer.named_buffers())
        assert not decoder.transformer.pos_embed.requires_grad
        assert decoder.transformer.pos_embed.shape == (1, 49, 512)  # decoder_embed_dim

    def test_learned_pos_embed_is_parameter(self):
        decoder = MAEDecoder(num_patches=100, pos_embed_type="learned")
        assert "pos_embed" in dict(decoder.transformer.named_parameters())
        assert decoder.transformer.pos_embed.requires_grad


@pytest.mark.unit
class TestMAEDecoderForward:
    """Test MAE decoder forward pass."""

    @pytest.fixture
    def decoder(self):
        return MAEDecoder(
            embed_dim=64,
            decoder_embed_dim=32,
            output_dim=64,
            num_patches=16,
            depth=2,
            num_heads=4,
            pos_embed_type="sincos_1d",
        )

    def test_output_shape_masked_only(self, decoder):
        """Default: output only masked positions."""
        B, T, D = 2, 16, 64
        N_vis = 4
        N_mask = T - N_vis

        x = torch.randn(B, N_vis, D)
        mask = torch.ones(B, T)
        mask[:, :N_vis] = 0

        out = decoder(x, mask, output_masked_only=True)
        assert out.shape == (B, N_mask, D)

    def test_output_shape_full_sequence(self, decoder):
        """output_masked_only=False returns full sequence."""
        B, T, D = 2, 16, 64
        N_vis = 4

        x = torch.randn(B, N_vis, D)
        mask = torch.ones(B, T)
        mask[:, :N_vis] = 0

        out = decoder(x, mask, output_masked_only=False)
        assert out.shape == (B, T, D)

    def test_output_dim_differs_from_embed_dim(self):
        decoder = MAEDecoder(
            embed_dim=64,
            decoder_embed_dim=32,
            output_dim=128,
            num_patches=16,
            depth=2,
            num_heads=4,
            pos_embed_type="sincos_1d",
        )

        x = torch.randn(2, 4, 64)
        mask = torch.zeros(2, 16)
        mask[:, 4:] = 1

        # Masked only
        out = decoder(x, mask, output_masked_only=True)
        assert out.shape == (2, 12, 128)

        # Full sequence
        out = decoder(x, mask, output_masked_only=False)
        assert out.shape == (2, 16, 128)

    def test_output_dtype(self, decoder):
        x = torch.randn(2, 4, 64)
        mask = torch.zeros(2, 16)
        mask[:, 4:] = 1

        out = decoder(x, mask)
        assert out.dtype == torch.float32

    def test_different_mask_ratios(self, decoder):
        B, T, D = 2, 16, 64

        for num_visible in [1, 4, 8, 15]:
            num_masked = T - num_visible
            x = torch.randn(B, num_visible, D)
            mask = torch.ones(B, T)
            mask[:, :num_visible] = 0

            out = decoder(x, mask, output_masked_only=True)
            assert out.shape == (B, num_masked, D)

            out = decoder(x, mask, output_masked_only=False)
            assert out.shape == (B, T, D)

    def test_batch_size_one(self, decoder):
        x = torch.randn(1, 4, 64)
        mask = torch.zeros(1, 16)
        mask[:, 4:] = 1

        out = decoder(x, mask, output_masked_only=True)
        assert out.shape == (1, 12, 64)

    def test_full_sequence_input(self, decoder):
        """Test with full sequence input (not just visible tokens)."""
        B, T, D = 2, 16, 64
        x = torch.randn(B, T, D)  # Full sequence
        mask = torch.zeros(B, T)
        mask[:, 4:] = 1  # Mask last 12

        out = decoder(x, mask, output_masked_only=True)
        assert out.shape == (B, 12, D)

        out = decoder(x, mask, output_masked_only=False)
        assert out.shape == (B, T, D)

    def test_no_masking(self, decoder):
        """All tokens visible (mask all zeros) - edge case."""
        B, T, D = 2, 16, 64
        x = torch.randn(B, T, D)
        mask = torch.zeros(B, T)

        # With no masked tokens, output_masked_only=True returns empty
        out = decoder(x, mask, output_masked_only=True)
        assert out.shape == (B, 0, D)

        out = decoder(x, mask, output_masked_only=False)
        assert out.shape == (B, T, D)

    def test_gradient_flow(self, decoder):
        x = torch.randn(2, 4, 64, requires_grad=True)
        mask = torch.zeros(2, 16)
        mask[:, 4:] = 1

        out = decoder(x, mask)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert decoder.mask_token.grad is not None

    def test_mask_token_affects_output(self, decoder):
        x = torch.randn(1, 4, 64)
        mask = torch.zeros(1, 16)
        mask[:, 4:] = 1

        out1 = decoder(x, mask).detach().clone()

        with torch.no_grad():
            decoder.mask_token.add_(10.0)

        out2 = decoder(x, mask)

        assert not torch.allclose(out1, out2)


@pytest.mark.unit
class TestMAEDecoderPositionalEmbeddings:
    """Test positional embedding configurations."""

    def test_sincos_1d(self):
        decoder = MAEDecoder(
            num_patches=100,
            decoder_embed_dim=512,
            pos_embed_type="sincos_1d",
        )
        assert decoder.transformer.pos_embed.shape == (1, 100, 512)

    def test_sincos_2d(self):
        decoder = MAEDecoder(
            num_patches=49,
            decoder_embed_dim=512,
            pos_embed_type="sincos_2d",
            grid_size=7,
        )
        assert decoder.transformer.pos_embed.shape == (1, 49, 512)

    def test_learned(self):
        decoder = MAEDecoder(
            num_patches=100,
            decoder_embed_dim=512,
            pos_embed_type="learned",
        )
        assert decoder.transformer.pos_embed.shape == (1, 100, 512)
        assert decoder.transformer.pos_embed.requires_grad


@pytest.mark.unit
class TestMAEDecoderDropPath:
    """Test drop path (stochastic depth)."""

    def test_drop_path_zero_deterministic(self):
        decoder = MAEDecoder(
            num_patches=16,
            depth=2,
            drop_path_rate=0.0,
            pos_embed_type="sincos_1d",
        )
        decoder.eval()

        x = torch.randn(1, 4, 768)
        mask = torch.zeros(1, 16)
        mask[:, 4:] = 1

        out1 = decoder(x, mask)
        out2 = decoder(x, mask)
        assert torch.allclose(out1, out2)

    def test_drop_path_nonzero_stochastic_train(self):
        decoder = MAEDecoder(
            num_patches=16,
            depth=4,
            drop_path_rate=0.5,
            pos_embed_type="sincos_1d",
        )
        decoder.train()

        x = torch.randn(1, 4, 768)
        mask = torch.zeros(1, 16)
        mask[:, 4:] = 1

        outputs = [decoder(x.clone(), mask.clone()) for _ in range(10)]
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Drop path should create variation in training"

    def test_drop_path_deterministic_eval(self):
        decoder = MAEDecoder(
            num_patches=16,
            depth=4,
            drop_path_rate=0.5,
            pos_embed_type="sincos_1d",
        )
        decoder.eval()

        x = torch.randn(1, 4, 768)
        mask = torch.zeros(1, 16)
        mask[:, 4:] = 1

        out1 = decoder(x, mask)
        out2 = decoder(x, mask)
        assert torch.allclose(out1, out2)
