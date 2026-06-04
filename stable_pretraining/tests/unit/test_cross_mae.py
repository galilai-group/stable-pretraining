"""Unit tests for CrossMAE-specific behavior."""

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.backbone import patchify
from stable_pretraining.methods.cross_mae import CrossMAE, CrossMAEDecoder

pytestmark = pytest.mark.unit


def _decoder(self_attn: bool = False, weight_feature_maps: bool = False):
    return CrossMAEDecoder(
        encoder_dim=32,
        decoder_dim=16,
        output_dim=12,
        num_patches=16,
        grid_size=(4, 4),
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        self_attn=self_attn,
        weight_feature_maps=weight_feature_maps,
        num_feature_maps=3,
    )


def _tiny_vit(img_size: int = 64):
    import timm

    return timm.create_model(
        "vit_tiny_patch16_224",
        num_classes=0,
        pretrained=False,
        img_size=img_size,
    )


@pytest.mark.parametrize("self_attn", [False, True])
def test_cross_decoder_context_path_shapes_and_gradients(self_attn):
    decoder = _decoder(self_attn=self_attn, weight_feature_maps=False)
    context = torch.randn(2, 5, 32, requires_grad=True)
    query_idx = torch.tensor([[0, 2, 5], [1, 3, 4]])

    output = decoder(context, query_idx)
    assert output.shape == (2, 3, 12)

    output.sum().backward()
    assert context.grad is not None
    assert context.grad.abs().sum() > 0
    assert decoder.mask_token.grad is not None
    assert decoder.mask_token.grad.abs().sum() > 0


def test_cross_decoder_feature_map_mixer_path_uses_feature_maps():
    decoder = _decoder(weight_feature_maps=True)
    feature_maps = [torch.randn(2, 5, 32, requires_grad=True) for _ in range(3)]
    query_idx = torch.tensor([[0, 2, 5], [1, 3, 4]])

    output = decoder(feature_maps, query_idx)
    assert output.shape == (2, 3, 12)

    output.sum().backward()
    assert decoder.feature_mixer.linear.weight.grad is not None
    assert decoder.feature_mixer.linear.weight.grad.abs().sum() > 0
    assert all(f.grad is not None and f.grad.abs().sum() > 0 for f in feature_maps)


def test_crossmae_reconstructs_only_kept_mask_subset_and_targets_match():
    torch.manual_seed(0)
    model = CrossMAE(
        encoder_name=_tiny_vit(),
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
        mask_ratio=0.75,
        kept_mask_ratio=0.25,
        norm_pix_loss=False,
        weight_feature_maps=False,
    )
    model.train()
    images = torch.randn(2, 3, 64, 64)

    output = model(images)
    num_patches = (64 // model.patch_size) ** 2
    num_predicted = int(output.mask.sum(dim=1)[0].item())
    patch_dim = 3 * model.patch_size * model.patch_size

    assert output.mask.shape == (2, num_patches)
    assert output.mask.dtype == torch.bool
    assert output.mask.sum(dim=1).tolist() == [4, 4]
    assert output.predictions.shape == (2, num_predicted, patch_dim)
    assert output.targets.shape == output.predictions.shape
    assert torch.allclose(output.loss, F.mse_loss(output.predictions, output.targets))

    query_idx = torch.argsort(output.mask.int(), dim=1, stable=True)[:, -4:]
    expected_targets = torch.gather(
        patchify(images, (3, model.patch_size, model.patch_size)),
        dim=1,
        index=query_idx.unsqueeze(-1).expand(-1, -1, patch_dim),
    )
    assert torch.allclose(output.targets, expected_targets)


def test_crossmae_default_feature_map_path_backpropagates():
    torch.manual_seed(1)
    model = CrossMAE(
        encoder_name=_tiny_vit(),
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
        mask_ratio=0.5,
        kept_mask_ratio=0.5,
        use_feature_maps=(-1,),
    )
    model.train()
    output = model(torch.randn(2, 3, 64, 64))

    assert output.predictions.shape == output.targets.shape
    output.loss.backward()
    assert model.decoder.feature_mixer.linear.weight.grad is not None
    assert model.decoder.feature_mixer.linear.weight.grad.abs().sum() > 0


def test_crossmae_eval_returns_embedding_without_reconstruction():
    model = CrossMAE(
        encoder_name=_tiny_vit(),
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
    )
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))

    assert output.loss.item() == 0.0
    assert output.embedding.shape == (2, 192)
    assert output.predictions is None
    assert output.targets is None
    assert output.mask is None


@pytest.mark.parametrize(
    "mask_ratio,kept_mask_ratio",
    [(0.75, 0.0), (0.5, 0.75), (1.0, 0.5)],
)
def test_crossmae_rejects_invalid_mask_ratios(mask_ratio, kept_mask_ratio):
    with pytest.raises(ValueError, match="kept_mask_ratio and mask_ratio"):
        CrossMAE(mask_ratio=mask_ratio, kept_mask_ratio=kept_mask_ratio)
