"""Dataset, frozen-evaluation, and export contracts for the All4One example."""

import pytest
import torch
from PIL import Image
from torch import nn

import stable_pretraining as spt
from stable_pretraining.tests._all4one_helpers import load_all4one_example


pytestmark = pytest.mark.unit


@pytest.fixture
def example():
    return load_all4one_example()


def test_imagenet_class_mapping_and_crop_shapes(example, tmp_path, monkeypatch):
    for split in ("train", "val"):
        for index in range(100):
            folder = tmp_path / split / f"n{index:08d}"
            folder.mkdir(parents=True)
            Image.new("RGB", (40, 48), (index, 50, 70)).save(folder / "image.png")
    args = example.parse_args(
        [
            "--dataset",
            "imagenet100",
            "--data-root",
            str(tmp_path),
            "--batch-size",
            "2",
            "--epochs",
            "1",
        ]
    )
    # Exercise real ImageFolder/transforms without allocating a full ResNet.
    monkeypatch.setattr(
        example,
        "build_encoder",
        lambda _: (nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()), 3),
    )
    _, data, _, config = example.build_experiment(args)
    data.setup("fit")
    train = next(iter(data.train_dataloader()))
    val = next(iter(data.val_dataloader()))
    assert [tuple(view["image"].shape) for view in train["views"]] == [
        (2, 3, 224, 224)
    ] * 2
    assert val["image"].shape == (2, 3, 224, 224)
    assert config["held_out_split"] == "validation"
    assert len(config["class_to_idx"]) == 100
    folder = tmp_path / "val" / "n00000099"
    folder.rename(tmp_path / "val" / "different_class")
    with pytest.raises(ValueError, match="same 100 class"):
        example.load_datasets(args)


def test_frozen_forward_keeps_batchnorm_state(example):
    encoder = nn.Sequential(nn.BatchNorm2d(3), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    encoder.requires_grad_(False)
    model = spt.Module(backbone=encoder, forward=example.frozen_forward, optim=None)
    before = {key: value.clone() for key, value in encoder.state_dict().items()}
    model.train()
    output = model(
        {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}, "fit"
    )
    assert not output["embedding"].requires_grad
    assert not encoder.training
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_export_rejects_changed_dataset_labels(example, tmp_path):
    args = example.parse_args(["--epochs", "1", "--batch-size", "2"])
    model, _, _, config = example.build_experiment(args)
    path = tmp_path / "encoder.pt"
    example.export_encoder(model, config, path)
    payload = torch.load(path, weights_only=True)
    payload["class_to_idx"]["0"], payload["class_to_idx"]["1"] = 1, 0
    torch.save(payload, path)
    args = example.parse_args(["--stage", "linear", "--encoder-path", str(path)])
    with pytest.raises(ValueError, match="dataset/classes"):
        example.build_experiment(args)


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100", "imagenet100"])
def test_backbone_initialization_matches_solo_learn(example, dataset):
    from torchvision.models import resnet18

    with torch.random.fork_rng():
        torch.manual_seed(17)
        expected = resnet18(weights=None, zero_init_residual=False)
        expected.fc = nn.Identity()
        if dataset != "imagenet100":
            expected.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            expected.maxpool = nn.Identity()
        torch.manual_seed(17)
        actual, width = example.build_encoder(dataset)
    assert width == 512
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dataset,epochs", [("cifar10", 1000), ("cifar100", 1000), ("imagenet100", 400)]
)
@pytest.mark.parametrize("batch_size", [128, 256, 512])
def test_released_training_recipe(example, monkeypatch, dataset, epochs, batch_size):
    from torchvision.datasets import FakeData

    count = 10 if dataset == "cifar10" else 100
    labels = {str(i): i for i in range(count)}
    monkeypatch.setattr(
        example,
        "load_datasets",
        lambda _: (FakeData(1024), FakeData(16), labels),
    )
    encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    encoder.conv1 = nn.Identity()
    monkeypatch.setattr(example, "build_encoder", lambda _: (encoder, 3))
    args = example.parse_args(["--dataset", dataset, "--batch-size", str(batch_size)])
    model, _, callbacks, config = example.build_experiment(args)
    assert args.epochs == epochs
    assert args.cifar_stem_padding == 2
    representation = model.optim["representation"]["optimizer"]
    attention = model.optim["attention"]["optimizer"]
    assert representation["type"] == attention["type"] == "LARS"
    assert representation["lr"] == pytest.approx(batch_size / 256)
    assert attention["lr"] == pytest.approx(0.1)
    assert config["effective_representation_lr"] == representation["lr"]
    assert config["lr_scale"] == batch_size / 256
    assert (
        config["optim"]["representation"]["scheduler"]["warmup_steps"]
        == (1024 // batch_size) * 10
    )
    expected_mean = {
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4865, 0.4409),
        "imagenet100": (0.485, 0.456, 0.406),
    }[dataset]
    assert config["normalization"]["mean"] == expected_mean
    assert config["reference_commit"] == "00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08"
