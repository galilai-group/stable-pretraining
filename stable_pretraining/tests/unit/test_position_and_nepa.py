"""Position encodings and causal prediction preserve their geometric contracts."""

import pytest
import torch
from torch import nn

from stable_pretraining.backbone.vit import PositionalEncoding2D, TransformerBlock
from stable_pretraining.methods.nepa import NEPA

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["none", "learnable", "sinusoidal", "rope"])
@pytest.mark.parametrize("grid", [(2, 3), (3, 4), (1, 2)])
def test_position_encoding_preserves_shape_gradients_and_prefixes(kind, grid):
    encoding = PositionalEncoding2D(16, (2, 3), pos_type=kind, num_prefix_tokens=2)
    x = torch.randn(2, 2 + grid[0] * grid[1], 16, requires_grad=True)
    out = encoding(x, grid)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    if kind == "none":
        assert out is x
    elif kind == "rope":
        torch.testing.assert_close(out[:, :2], x[:, :2])
        torch.testing.assert_close(out.square().sum(-1), x.square().sum(-1))
    else:
        torch.testing.assert_close(
            out[:, :2] - x[:, :2], encoding.pos_embed[:, :2].expand(2, -1, -1)
        )
    out.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    if kind == "learnable":
        assert encoding.pos_embed.grad is not None


@pytest.mark.parametrize("learnable", [False, True])
def test_sinusoidal_positions_match_independent_formula(learnable):
    encoding = PositionalEncoding2D(
        8, (2, 3), pos_type="sinusoidal", num_prefix_tokens=1, learnable=learnable
    )
    assert isinstance(encoding.pos_embed, nn.Parameter) is learnable
    expected = torch.tensor(
        [
            torch.sin(torch.tensor(1.0)),
            torch.cos(torch.tensor(1.0)),
            torch.sin(torch.tensor(0.01)),
            torch.cos(torch.tensor(0.01)),
            torch.sin(torch.tensor(2.0)),
            torch.cos(torch.tensor(2.0)),
            torch.sin(torch.tensor(0.02)),
            torch.cos(torch.tensor(0.02)),
        ]
    )
    torch.testing.assert_close(encoding.pos_embed[0, -1], expected)
    torch.testing.assert_close(encoding.pos_embed[0, 0], torch.zeros(8))


def test_unknown_position_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown pos_type"):
        PositionalEncoding2D(8, (2, 2), pos_type="typo")


@pytest.mark.parametrize("self_attention", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_cross_attention_block_is_differentiable_and_adaln_starts_as_identity(
    self_attention, adaptive
):
    block = TransformerBlock(
        16,
        4,
        self_attn=self_attention,
        cross_attn=True,
        use_adaln=adaptive,
        use_layer_scale=True,
        drop_path=0.1,
        mlp_type="swiglu",
    )
    x = torch.randn(2, 3, 16, requires_grad=True)
    context = torch.randn(2, 5, 16, requires_grad=True)
    out = block(x, context=context, cond=torch.randn(2, 16))
    if adaptive:
        torch.testing.assert_close(out, x)
    out.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()


@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("swiglu", [False, True])
def test_nepa_loss_matches_detached_next_patch_targets(rope, swiglu):
    model = NEPA(
        img_size=8,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        use_rope=rope,
        use_swiglu=swiglu,
    )
    x = torch.randn(2, 3, 8, 8)
    out = model(x)
    targets = model.patch_embed(x)
    if model.pos_embed is not None:
        targets = targets + model.pos_embed
    expected = -torch.nn.functional.cosine_similarity(
        out.embeddings[:, :-1], targets.detach()[:, 1:], dim=-1
    ).mean()
    torch.testing.assert_close(out.loss, expected)
    assert out.grid_size == (2, 2)
    out.loss.backward()
    assert model.patch_embed.proj.weight.grad is not None
    assert torch.isfinite(model.patch_embed.proj.weight.grad).all()
    model.eval()
    evaluation = model(x)
    assert evaluation.loss.item() == 0.0
    torch.testing.assert_close(model.get_dense_features(x), evaluation.embeddings)
    torch.testing.assert_close(
        model.get_classifier_features(x), evaluation.embeddings[:, -1]
    )
    model.freeze_patch_embed()
    assert all(not p.requires_grad for p in model.patch_embed.parameters())


@pytest.mark.parametrize("rope", [False, True])
def test_nepa_causal_features_do_not_see_future_patches(rope):
    model = NEPA(
        img_size=8, patch_size=4, embed_dim=16, depth=2, num_heads=4, use_rope=rope
    ).eval()
    x = torch.randn(1, 3, 8, 8)
    changed = x.clone()
    changed[:, :, 4:, 4:] += 20
    torch.testing.assert_close(
        model.forward_features(x, causal=True)[:, :-1],
        model.forward_features(changed, causal=True)[:, :-1],
    )
