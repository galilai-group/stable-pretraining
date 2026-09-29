"""Mask semantics, spatial resizing, and conditioning contracts for transformers."""

from unittest.mock import Mock

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from stable_pretraining.backbone.vit import (
    Attention,
    CrossAttention,
    FlexibleTransformer,
    MaskedEncoder,
)

pytestmark = pytest.mark.unit


def transformer(**kwargs):
    config = dict(
        input_dim=8,
        hidden_dim=24,
        output_dim=8,
        num_patches=4,
        depth=1,
        num_heads=4,
        num_prefix_tokens=0,
        use_adaln=False,
        cross_attn=False,
        zero_init_output=False,
    )
    return FlexibleTransformer(**(config | kwargs))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"hidden_dim": 23}, "divisible"),
        ({"use_rope": "2d"}, "requires grid_size"),
        ({"use_rope": "2d", "grid_size": (2, 2, 2)}, "needs grid_size"),
        ({"use_rope": "3d", "grid_size": 2}, "needs grid_size"),
        ({"num_patches": 5}, "perfect square"),
        ({"pos_embed_type": "sincos_3d"}, "requires grid_size"),
        ({"pos_embed_type": "sincos_3d", "grid_size": (2, 2, 2)}, "elements"),
        ({"pos_embed_type": "unknown"}, "pos_embed_type"),
    ],
)
def test_invalid_transformer_geometry_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        transformer(**kwargs)


@pytest.mark.parametrize(
    "mode", ["none", "learned", "sincos_1d", "sincos_2d", "sincos_3d"]
)
def test_position_modes_support_real_forward_and_backward(mode):
    kwargs = dict(num_patches=8, grid_size=(2, 2, 2)) if mode == "sincos_3d" else {}
    model = transformer(pos_embed_type=mode, **kwargs)
    x = torch.randn(2, kwargs.get("num_patches", 4), 8, requires_grad=True)
    result = model(x, return_all=True)
    assert result.shape == x.shape
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    if mode == "learned":
        assert model.pos_embed.grad.abs().sum() > 0


@pytest.mark.parametrize("cross_attn", [False, True])
def test_masked_content_cannot_leak_into_predictions(cross_attn):
    model = transformer(
        add_mask_token=True, cross_attn=cross_attn, num_registers=2
    ).eval()
    context, queries = torch.randn(2, 3, 8), torch.randn(2, 1, 8)
    cm = torch.tensor([[True, False, False], [False, True, False]])
    qm = torch.ones(2, 1, dtype=torch.bool)
    kwargs = dict(context_mask=cm, query_mask=qm, return_registers=True)
    out, registers = model(context, queries, **kwargs)
    changed = context.clone()
    changed[cm] += 100
    again, again_registers = model(changed, queries + 100, **kwargs)
    torch.testing.assert_close(again, out)
    torch.testing.assert_close(again_registers, registers)
    assert out.shape == (2, 1, 8) and registers.shape == (2, 2, 8)


def test_joint_attention_mask_broadcast_and_query_unshuffle():
    model = transformer(num_registers=2).eval()
    context, queries = torch.randn(2, 2, 8), torch.randn(2, 2, 8)
    ci = torch.tensor([[2, 0], [2, 0]])
    qi = torch.tensor([[3, 1], [3, 1]])
    mask = torch.eye(2, dtype=torch.bool)
    kwargs = dict(context_idx=ci, query_idx=qi, return_registers=True)
    out, registers = model(context, queries, attn_mask=mask, return_all=True, **kwargs)
    batched, regs = model(
        context, queries, attn_mask=mask.expand(2, -1, -1), return_all=True, **kwargs
    )
    query_out, _ = model(context, queries, attn_mask=mask, **kwargs)
    torch.testing.assert_close(batched, out)
    torch.testing.assert_close(regs, registers)
    torch.testing.assert_close(out[:, [3, 1]], query_out)
    assert model(context, return_registers=True)[0].shape == (2, 0, 8)


def test_conditioning_and_mask_require_explicit_configuration():
    x = torch.randn(2, 4, 8)
    with pytest.raises(ValueError, match="add_mask_token=False"):
        transformer()(x, context_mask=torch.zeros(2, 4, dtype=torch.bool))
    model = transformer(use_adaln=True)
    with pytest.raises(ValueError, match="Timestep"):
        model(x)
    out = model(x, t=torch.tensor([0.0, 1.0]), return_all=True)
    assert out.shape == x.shape and torch.isfinite(out).all()


@pytest.mark.parametrize("kind", ["self", "cross"])
@pytest.mark.parametrize("rank", [2, 3, 4])
def test_boolean_and_additive_attention_masks_agree_and_block_information(kind, rank):
    model = Attention(8, 2) if kind == "self" else CrossAttention(8, num_heads=2)
    x = torch.randn(2, 3, 8)
    context = torch.randn(2, 4, 8)
    mask = torch.zeros(3, 3 if kind == "self" else 4, dtype=torch.bool)
    mask[:, -1] = True
    if rank >= 3:
        mask = mask.expand(2, -1, -1)
    if rank == 4:
        mask = mask[:, None].expand(-1, 2, -1, -1)
    additive = torch.zeros_like(mask, dtype=torch.float).masked_fill(mask, -torch.inf)
    args = (x,) if kind == "self" else (x, context)
    torch.testing.assert_close(
        model(*args, attn_mask=mask), model(*args, attn_mask=additive)
    )
    if kind == "cross":
        changed = context.clone()
        changed[:, -1] += 1000
        torch.testing.assert_close(model(x, context, mask), model(x, changed, mask))


def tiny_vit(**kwargs):
    return VisionTransformer(
        img_size=8,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        num_classes=0,
        **kwargs,
    )


@pytest.mark.parametrize("no_embed_class", [False, True])
def test_changing_patch_size_resizes_positions_without_destroying_encoder(
    no_embed_class,
):
    vit = tiny_vit(no_embed_class=no_embed_class)
    original = vit.blocks[0].attn.qkv.weight.detach().clone()
    encoder = MaskedEncoder(vit, patch_size=2)
    result = encoder(torch.randn(2, 3, 8, 8))
    assert result.encoded.shape == (2, 17, 16)
    assert result.grid_size == (4, 4)
    assert not result.mask.any()
    torch.testing.assert_close(result.ids_keep, torch.arange(16).expand(2, -1))
    torch.testing.assert_close(vit.blocks[0].attn.qkv.weight, original)


def test_named_encoder_passes_factory_options_without_download(monkeypatch):
    import timm

    factory = Mock(return_value=tiny_vit())
    monkeypatch.setattr(timm, "create_model", factory)
    encoder = MaskedEncoder(
        "tiny-test",
        pretrained=False,
        img_size=8,
        patch_size=4,
        norm_layer=torch.nn.LayerNorm,
    )
    assert encoder(torch.randn(1, 3, 8, 8)).encoded.shape == (1, 5, 16)
    assert factory.call_args.args == ("tiny-test",)
    assert factory.call_args.kwargs["pretrained"] is False
    assert factory.call_args.kwargs["norm_layer"] is torch.nn.LayerNorm
