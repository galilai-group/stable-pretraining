"""Legacy ViT wrappers preserve token identity across positional layouts."""

import pytest
import torch
from torch import nn

from stable_pretraining.backbone.utils import EfficientMaskedTimmViT, EvalOnly

pytestmark = pytest.mark.unit


def wrapper(cls, dist, reg, positioned):
    vit = nn.Module()
    vit.patch_embed = nn.Linear(2, 2)
    vit.blocks = nn.Identity()
    for name, number, value in [
        ("cls_token", cls, -1.0),
        ("dist_token", dist, -2.0),
        ("reg_token", reg, -3.0),
    ]:
        if number:
            setattr(vit, name, nn.Parameter(torch.full((1, number, 2), value)))
    vit.pos_embed = nn.Parameter(
        torch.arange((4 + positioned) * 2).reshape(1, -1, 2).float()
    )
    return EfficientMaskedTimmViT(vit)


@pytest.mark.parametrize("cls,dist,reg", [(0, 0, 2), (1, 0, 2), (1, 1, 2), (0, 1, 0)])
def test_extra_token_order_matches_cls_dist_register_patch_contract(cls, dist, reg):
    model = wrapper(cls, dist, reg, cls + dist)
    patches = torch.ones(2, 4, 2)
    result = model._add_extra_tokens(patches)
    expected = [-1.0] * cls + [-2.0] * dist + [-3.0] * reg + [1.0] * 4
    torch.testing.assert_close(
        result, torch.tensor(expected).reshape(1, -1, 1).expand(2, -1, 2)
    )
    assert model._get_num_extra_tokens() == cls + dist + reg


@pytest.mark.parametrize(
    "cls,reg,positioned", [(0, 0, 0), (1, 0, 1), (1, 2, 1), (1, 2, 3), (1, 2, 0)]
)
def test_per_sample_position_selection_keeps_prefixes(cls, reg, positioned):
    model = wrapper(cls, 0, reg, positioned)
    indices = torch.tensor([[3, 0], [1, 2]])
    result = model._subsample_pos_embed_different_patterns(indices, 2, 4, 2)
    prefix = model.vit.pos_embed[:, :positioned]
    patches = model.vit.pos_embed[:, positioned:]
    expected = torch.cat([torch.cat([prefix, patches[:, i]], 1) for i in indices], 0)
    torch.testing.assert_close(result, expected)
    same = model._subsample_pos_embed_same_pattern(indices[0], 2, 4)
    torch.testing.assert_close(same[0], expected[0])
    torch.testing.assert_close(same[1], expected[0])


def test_eval_only_remains_frozen_when_parent_trains_and_delegates_attributes():
    backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.9))
    backbone.feature_size = 4
    frozen = EvalOnly(backbone)
    parent = nn.Sequential(frozen)
    parent.train()
    x = torch.randn(2, 4)
    torch.testing.assert_close(parent(x), backbone[0](x))
    assert frozen.feature_size == 4
    assert not frozen.training and not backbone.training
    assert all(not p.requires_grad for p in frozen.parameters())
    with pytest.raises(AttributeError):
        _ = frozen.missing_attribute
    backbone.train()
    with pytest.raises(RuntimeError, match="training mode"):
        frozen(x)


@pytest.mark.parametrize("cls,reg", [(0, 0), (1, 0), (1, 2)])
def test_spatial_interpolation_preserves_constant_patches_and_special_positions(
    cls, reg
):
    positioned = cls + reg
    model = wrapper(cls, 0, reg, positioned)
    source = model.vit.pos_embed.detach().clone()
    source[:, positioned:] = 7
    result = model._interpolate_pos_embed(source, 9)
    assert result.shape == (1, 9 + positioned, 2)
    torch.testing.assert_close(result[:, :positioned], source[:, :positioned])
    torch.testing.assert_close(result[:, positioned:], torch.full((1, 9, 2), 7.0))
    with pytest.raises(RuntimeError, match="Target number"):
        model._interpolate_pos_embed(source, 10)
    with pytest.raises(RuntimeError, match="Original positional"):
        model._interpolate_pos_embed(source[:, :-1], 9)
