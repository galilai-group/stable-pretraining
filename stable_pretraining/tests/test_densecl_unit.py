import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def feature_maps():
    """Small synthetic DenseCL-style feature maps: [B, D, H, W]."""
    torch.manual_seed(0)
    view1 = torch.randn(4, 8, 4, 4, requires_grad=True)
    view2 = torch.randn(4, 8, 4, 4, requires_grad=True)
    return view1, view2


@pytest.mark.unit
def test_feature_maps_have_expected_shape(feature_maps):
    view1, view2 = feature_maps

    assert view1.shape == (4, 8, 4, 4)
    assert view2.shape == (4, 8, 4, 4)


@pytest.mark.unit
def test_feature_maps_are_finite(feature_maps):
    view1, view2 = feature_maps

    assert torch.isfinite(view1).all()
    assert torch.isfinite(view2).all()


@pytest.mark.unit
def test_flattened_spatial_tokens_have_expected_shape(feature_maps):
    view1, _ = feature_maps

    # [B, D, H, W] -> [B, H*W, D]
    tokens = view1.flatten(2).transpose(1, 2)

    assert tokens.shape == (4, 16, 8)


@pytest.mark.unit
def test_flattened_tokens_preserve_spatial_feature_values(feature_maps):
    view1, _ = feature_maps
    tokens = view1.flatten(2).transpose(1, 2)

    # Token 0 represents spatial position (0, 0).
    assert torch.equal(tokens[:, 0, :], view1[:, :, 0, 0])

    # Token 5 represents spatial position (1, 1).
    assert torch.equal(tokens[:, 5, :], view1[:, :, 1, 1])


@pytest.mark.unit
def test_normalized_tokens_have_unit_norm(feature_maps):
    view1, _ = feature_maps
    tokens = view1.flatten(2).transpose(1, 2)

    normalized = F.normalize(tokens, dim=-1)
    norms = normalized.norm(dim=-1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.unit
def test_identical_tokens_have_cosine_similarity_one(feature_maps):
    view1, _ = feature_maps
    tokens = view1.flatten(2).transpose(1, 2)

    similarity = F.cosine_similarity(tokens, tokens, dim=-1)

    assert torch.allclose(
        similarity,
        torch.ones_like(similarity),
        atol=1e-5,
    )


@pytest.mark.unit
def test_dense_similarity_has_expected_shape(feature_maps):
    view1, view2 = feature_maps

    tokens1 = F.normalize(view1.flatten(2).transpose(1, 2), dim=-1)
    tokens2 = F.normalize(view2.flatten(2).transpose(1, 2), dim=-1)

    # Compare each spatial location in view 1 with every location in view 2.
    similarities = tokens1 @ tokens2.transpose(1, 2)

    assert similarities.shape == (4, 16, 16)
    assert torch.isfinite(similarities).all()
    assert (similarities >= -1.0001).all()
    assert (similarities <= 1.0001).all()


@pytest.mark.unit
def test_dense_similarity_supports_backpropagation(feature_maps):
    view1, view2 = feature_maps

    tokens1 = view1.flatten(2).transpose(1, 2)
    tokens2 = view2.flatten(2).transpose(1, 2)

    similarities = tokens1 @ tokens2.transpose(1, 2)
    loss = similarities.mean()
    loss.backward()

    assert view1.grad is not None
    assert view2.grad is not None
    assert torch.isfinite(view1.grad).all()
    assert torch.isfinite(view2.grad).all()


@pytest.mark.unit
def test_feature_maps_support_batch_size_one():
    features = torch.randn(1, 8, 4, 4)

    tokens = features.flatten(2).transpose(1, 2)

    assert tokens.shape == (1, 16, 8)


@pytest.mark.unit
def test_feature_maps_support_different_spatial_sizes():
    features = torch.randn(4, 8, 8, 8)

    tokens = features.flatten(2).transpose(1, 2)

    assert tokens.shape == (4, 64, 8)
