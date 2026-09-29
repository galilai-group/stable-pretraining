"""Generated views preserve sample identity and the advertised similarity targets."""

from multiprocessing.dummy import Pool

import numpy as np
import pytest
import torch
from PIL import Image

from stable_pretraining.utils import data_generation as generation
from stable_pretraining.backbone.resnet9 import MLP, ResidualBlock, Resnet9

pytestmark = pytest.mark.unit


@pytest.fixture
def images(monkeypatch):
    monkeypatch.setattr(generation, "Pool", Pool)
    return [
        Image.fromarray(np.full((32, 32, 3), v, dtype=np.uint8))
        for v in [32, 64, 96, 128, 160]
    ]


def test_dae_zero_noise_has_exact_clean_gram_matrix(images):
    views, gram = generation.generate_dae_samples(images[:2], n=2, eps=0, num_workers=2)
    assert views.shape == (4, 3, 224, 224)
    torch.testing.assert_close(views[0], views[1])
    torch.testing.assert_close(views[2], views[3])
    torch.testing.assert_close(gram, views.flatten(1) @ views.flatten(1).T)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_diffusion_clean_targets_follow_selected_noise_schedule(images, as_tensor):
    betas = torch.zeros(3) if as_tensor else [0.0, 0.0, 0.0]
    views, gram = generation.generate_dm_samples(
        images[:2], 2, betas, [0, 2], num_workers=2
    )
    assert views.shape == (8, 3, 224, 224)
    torch.testing.assert_close(gram, views.flatten(1) @ views.flatten(1).T)
    torch.testing.assert_close(views[:4], views[:1].expand(4, -1, -1, -1))


def test_supervised_similarity_filters_rare_classes_and_sorts_labels(images):
    views, gram = generation.generate_sup_samples(
        images, np.array([2, 0, 2, 1, 0]), 2, num_workers=2
    )
    expected = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
    assert views.shape == (4, 3, 224, 224)
    torch.testing.assert_close(gram, expected)
    assert views[0].mean() < views[1].mean()


def test_ssl_views_have_block_diagonal_identity_targets(images):
    views, gram = generation.generate_ssl_samples(images[:2], 3, num_workers=2)
    assert len(views) == 6
    assert all(x.shape == (3, 224, 224) and torch.isfinite(x).all() for x in views)
    assert torch.equal(gram[:3, :3], torch.ones(3, 3))
    assert torch.equal(gram[3:, 3:], torch.ones(3, 3))
    assert not gram[:3, 3:].any()


@pytest.mark.parametrize("stride", [1, 2])
def test_residual_block_zero_residual_preserves_skip_path(stride):
    block = ResidualBlock(4, 4, 3, 1, stride).eval()
    torch.nn.init.zeros_(block.conv_res2.weight)
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    expected = x if stride == 1 else block.downsample(x)
    torch.testing.assert_close(block(x), expected)
    block(x).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


@pytest.mark.parametrize("lazy,norm", [(False, None), (True, "batch_norm")])
def test_mlp_lazy_and_explicit_inputs_support_training(lazy, norm):
    model = MLP(None if lazy else 4, [8, 3], norm_layer=norm, inplace=True)
    x = torch.randn(2, 4, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 3)
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_resnet9_trains_on_non_square_grayscale_images():
    model = Resnet9(num_classes=3, num_channels=1)
    out = model(torch.randn(2, 1, 24, 32))
    torch.nn.functional.cross_entropy(out, torch.tensor([0, 2])).backward()
    assert out.shape == (2, 3)
    assert torch.isfinite(model.fc.weight.grad).all()
