"""Contract tests for DIET's sample-index classification objective."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stable_pretraining.data import FromTorchDataset
from stable_pretraining.methods import DIET

pytestmark = pytest.mark.unit


def _model(num_samples=5):
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 8))
    return DIET(
        encoder_name=encoder,
        num_samples=num_samples,
        label_smoothing=0.0,
        image_size=4,
    )


def test_training_uses_stable_sample_indices_and_backpropagates():
    model = _model()
    model.train()
    images = torch.randn(3, 3, 4, 4)
    sample_idx = torch.tensor([4, 1, 3])
    labels = torch.tensor([0, 0, 2])

    output = model(
        {"image": images, "sample_idx": sample_idx, "label": labels}, stage="fit"
    )

    assert output["logits"].shape == (3, 5)
    assert output["embedding"].shape == (3, 8)
    assert torch.equal(output["label"], labels)
    assert model.index_classifier.bias is None
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.index_classifier.weight.grad.abs().sum() > 0
    assert model.backbone[1].weight.grad.abs().sum() > 0


def test_index_targets_not_semantic_labels_define_the_loss():
    model = _model()
    model.train()
    images = torch.randn(2, 3, 4, 4)
    sample_idx = torch.tensor([2, 0])
    first = model(
        {"image": images, "sample_idx": sample_idx, "label": torch.tensor([0, 1])},
        stage="fit",
    )
    second = model(
        {"image": images, "sample_idx": sample_idx, "label": torch.tensor([1, 0])},
        stage="fit",
    )
    torch.testing.assert_close(first["loss"], second["loss"])


def test_reordering_batch_keeps_per_example_index_loss():
    model = _model()
    model.train()
    images = torch.randn(3, 3, 4, 4)
    sample_idx = torch.tensor([3, 1, 4])
    loss = model(images, sample_idx)["loss"]
    permutation = torch.tensor([2, 0, 1])
    reordered_loss = model(images[permutation], sample_idx[permutation])["loss"]
    torch.testing.assert_close(loss, reordered_loss)


def test_from_torch_dataset_keeps_sample_indices_when_loader_shuffles():
    source = TensorDataset(torch.arange(5), torch.zeros(5, dtype=torch.long))
    dataset = FromTorchDataset(source, names=["image", "label"])
    batch = next(
        iter(DataLoader(dataset, batch_size=5, sampler=[3, 1, 4, 0, 2]))
    )
    assert batch["sample_idx"].tolist() == [3, 1, 4, 0, 2]


def test_evaluation_returns_embedding_without_index_classifier_loss():
    model = _model().eval()
    images = torch.randn(2, 3, 4, 4)

    with torch.no_grad():
        output = model(images)

    assert output["embedding"].shape == (2, 8)
    assert output["logits"] is None
    assert output["loss"].item() == 0
    assert not output["embedding"].requires_grad


def test_validation_dict_preserves_semantic_labels_without_training_loss():
    model = _model()
    model.train()
    labels = torch.tensor([2, 4])
    output = model(
        {
            "image": torch.randn(2, 3, 4, 4),
            "sample_idx": torch.tensor([0, 1]),
            "label": labels,
        },
        stage="validate",
    )
    assert torch.equal(output["label"], labels)
    assert output["loss"].item() == 0
    assert output["logits"] is None


def test_training_requires_valid_sample_indices():
    model = _model().train()
    images = torch.randn(2, 3, 4, 4)
    with pytest.raises(ValueError, match="requires stable sample_idx"):
        model(images)
    with pytest.raises(ValueError, match="sample_idx must have shape"):
        model(images, torch.tensor([[0, 1]]))
    with pytest.raises(ValueError, match="sample_idx values must be"):
        model(images, torch.tensor([0, 5]))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_samples": 0}, "num_samples must be positive"),
        ({"label_smoothing": 1.0}, "label_smoothing must be"),
    ],
)
def test_constructor_rejects_invalid_objective_configuration(kwargs, message):
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 8))
    with pytest.raises(ValueError, match=message):
        DIET(encoder_name=encoder, image_size=4, **kwargs)