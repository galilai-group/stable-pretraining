"""Integration tests for transforms with actual datasets."""

import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

import stable_pretraining as spt
import stable_pretraining.data.transforms as transforms


@pytest.mark.v1
@pytest.mark.integration
@pytest.mark.download
@pytest.mark.parametrize(
    "our_transform,true_transform",
    [
        (transforms.GaussianBlur(3), v2.GaussianBlur(3)),
        (transforms.RandomChannelPermutation(), v2.RandomChannelPermutation()),
        (transforms.RandomHorizontalFlip(0.5), v2.RandomHorizontalFlip(0.5)),
        (transforms.RandomGrayscale(0.5), v2.RandomGrayscale(0.5)),
        (
            transforms.ColorJitter(0.8, 0.4, 0.4, 0.4),
            v2.ColorJitter(0.8, 0.4, 0.4, 0.4),
        ),
        (transforms.RandomResizedCrop((32, 32)), v2.RandomResizedCrop((32, 32))),
        (transforms.RandomSolarize(0.5, 0.2), v2.RandomSolarize(0.5, 0.2)),
        (transforms.RandomRotation(90), v2.RandomRotation(90)),
    ],
)
def test_controlled_transforms(our_transform, true_transform):
    """Test controlled transforms match torchvision behavior."""
    transform = transforms.Compose(
        transforms.ControlledTransform(transform=our_transform, seed_offset=0),
        transforms.ToImage(),
    )
    our_dataset = spt.data.module.DictFormat(CIFAR10("~/data", download=True))
    our_dataset = spt.data.datasets.AddTransform(our_dataset, transform)
    t = v2.Compose(
        [true_transform, v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    true_dataset = CIFAR10("~/data", transform=t)
    for _ in range(3):
        for i in range(10):
            ours = our_dataset[i]
            torch.manual_seed(i)
            truth = true_dataset[i]
            assert torch.allclose(ours["image"], truth[0], atol=1e-5)


@pytest.mark.v1
@pytest.mark.integration
@pytest.mark.slow
def test_transforms_performance():
    """Test transform performance with data loading."""
    import itertools
    import time

    transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=(5, 5), p=1.0),
        transforms.ToImage(),
    )
    dataset = spt.data.dataset.DictFormat(CIFAR10("~/data", download=True))
    dataset = spt.data.dataset.AddTransform(dataset, transform)
    dataset = Subset(dataset, list(range(256)))
    loader = DataLoader(dataset, batch_size=64, num_workers=0)
    s = time.time()
    for _ in itertools.islice(loader, 10):
        pass
    e = time.time()
    assert e - s < 20, "Transform time should be reasonable"
