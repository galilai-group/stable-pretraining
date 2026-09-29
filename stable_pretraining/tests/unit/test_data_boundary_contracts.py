"""Data wrappers preserve batching, routing, and streaming metadata."""

from unittest.mock import Mock

import datasets
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from stable_pretraining.data import datasets as data
from stable_pretraining.data import gpu_transforms as gpu
from stable_pretraining.data import transforms as t

pytestmark = pytest.mark.unit


def test_hf_streaming_factory_preserves_renamed_columns_and_values(monkeypatch):
    source = datasets.Dataset.from_dict(
        {"x": [1, 2], "unused": [3, 4]}
    ).to_iterable_dataset()
    load = Mock(return_value=source)
    monkeypatch.setattr(datasets, "load_dataset", load)
    result = data.HFDataset(
        "local",
        streaming=True,
        rename_columns={"x": "value"},
        remove_columns=["unused"],
    )
    assert isinstance(result, data.HFIterableDataset)
    assert result.column_names == result.dataset.column_names
    assert list(result) == [
        {"value": 1, "sample_idx": 0},
        {"value": 2, "sample_idx": 1},
    ]
    assert load.call_args.kwargs["streaming"] is True


def test_dataset_factory_staggers_distributed_cache_access(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    pause = Mock()
    monkeypatch.setattr(data.time, "sleep", pause)
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **kw: datasets.Dataset.from_dict({"value": [7]}),
    )
    assert data.HFDataset("local")[0]["value"] == 7
    pause.assert_called_once_with(2)


def test_subset_list_indexing_and_tuple_dataset_length():
    source = datasets.Dataset.from_dict({"value": [0, 1, 2, 3]})
    subset = data.Subset(source, [3, 1])
    assert subset[[1, 0]] == {"value": [1, 3]}
    wrapped = data.FromTorchDataset(TensorDataset(torch.arange(3)), names=["value"])
    assert len(wrapped) == 3


def test_nested_numeric_target_and_shared_geometric_transforms():
    image = torch.arange(3 * 8 * 8).reshape(3, 8, 8).float()
    sample = [dict(image=image)]
    t.Transform().single_nested_set(sample, image + 1, "0.image")
    torch.testing.assert_close(sample[0]["image"], image + 1)
    flipped = t.RandomHorizontalFlip(p=1)({"image": [image, image + 1]})["image"]
    torch.testing.assert_close(flipped[0], image.flip(-1))
    torch.testing.assert_close(flipped[1], (image + 1).flip(-1))
    cropped = t.RandomResizedCrop(4, scale=(1, 1), ratio=(1, 1))(
        {"image": [image, image + 1]}
    )["image"]
    assert cropped[0].shape == (3, 4, 4)
    torch.testing.assert_close(
        cropped[1] - cropped[0], torch.ones_like(cropped[0]), atol=2e-5, rtol=1e-5
    )


def test_color_jitter_bypass_preserves_target_and_records_zero_parameters():
    image = torch.rand(3, 4, 4)
    result = t.ColorJitter(brightness=0.5, p=0, target="augmented")({"image": image})
    assert result["augmented"] is image
    assert torch.equal(result["ColorJitter"], torch.zeros(8))


@pytest.mark.parametrize("kwargs", [{"drop_ratio": -1}, {"patch_size": 0}])
def test_patch_mask_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        t.PatchMasking(**kwargs)


def test_patch_mask_accepts_grayscale_and_rejects_unsupported_input():
    transform = t.PatchMasking(patch_size=2, drop_ratio=0)
    image = torch.ones(4, 4)
    assert transform._to_tensor(image).shape == (1, 4, 4)
    with pytest.raises(TypeError, match="Unsupported"):
        transform({"image": object()})


@pytest.mark.parametrize("kind", ["RandomMask", "ContextTargetsMultiBlockMask"])
def test_patch_index_masks_support_pil_and_reject_invalid_source(kind):
    kwargs = {"patch_size": 4}
    if kind == "ContextTargetsMultiBlockMask":
        kwargs.update(
            min_keep=1, target_scales=((0.1, 0.2),), target_aspect_ratios=((0.75, 1.5),)
        )
    transform = getattr(t, kind)(**kwargs)
    result = transform({"image": Image.new("RGB", (32, 32))})
    assert len(result) > 1
    with pytest.raises(ValueError, match="Source must"):
        transform({"image": "bad"})
    with pytest.raises(ValueError, match="associated aspect ratio"):
        t.ContextTargetsMultiBlockMask(target_scales=((0.1, 0.2),))


def test_device_transfer_preserves_nested_tuple_structure():
    source = (torch.ones(2, 3), {"x": torch.ones(4)}, "label")
    result = gpu.ToDevice("meta")(source)
    assert isinstance(result, tuple) and result[2] == "label"
    assert result[0].device.type == result[1]["x"].device.type == "meta"
    assert result[0].shape == (2, 3) and source[0].device.type == "cpu"


def test_repeated_gpu_transforms_keep_unique_record_names():
    batch = {"image": torch.rand(2, 3, 8, 8)}
    transform = gpu.GPURandomHorizontalFlip(p=0)
    for _ in range(3):
        transform(batch)
    assert "GPURandomHorizontalFlip_1" in batch
    transform._last_call_params = []
    transform._record_params(batch, 0)


def test_randaugment_cpu_preserves_batch_shape_and_finite_values():
    transform = gpu.GPURandAugment(n=1, m=1)
    batch = {"image": torch.rand(2, 3, 8, 8)}
    result = transform(batch)
    assert result["image"].shape == (2, 3, 8, 8)
    assert torch.isfinite(result["image"]).all()
