"""Local dataset wrappers and augmentation labels preserve sample identity."""

import pickle
from types import SimpleNamespace

import datasets
import pytest
import torch

from stable_pretraining.data.datasets import (
    HFMapDataset,
    HFIterableDataset,
    FromTorchDataset,
    Subset,
)
from stable_pretraining.data.gpu_transforms import RandomMixupCutmix

pytestmark = pytest.mark.unit


def local_data():
    return datasets.Dataset.from_dict(
        {"value": list(range(8)), "target": [0, 1] * 4, "unused": [0] * 8}
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_hf_wrappers_rename_remove_shuffle_and_preserve_sample_indices(streaming):
    data = local_data()
    cls = HFMapDataset
    if streaming:
        data = data.to_iterable_dataset()
        cls = HFIterableDataset
    wrapped = cls(data, rename_columns={"target": "label"}, remove_columns=["unused"])
    trainer = SimpleNamespace(global_step=3, current_epoch=2)
    wrapped.set_pl_trainer(trainer)
    if not streaming:
        assert set(wrapped.column_names) == {"value", "label", "sample_idx"}
    assert wrapped.shuffle(seed=17) is wrapped
    rows = list(wrapped) if streaming else [wrapped[i] for i in range(len(wrapped))]
    assert len(rows) == 8
    assert set(rows[0]) == {
        "value",
        "label",
        "sample_idx",
        "global_step",
        "current_epoch",
    }
    assert sorted(row["sample_idx"] for row in rows) == list(range(8))
    assert all(
        row["value"] == row["sample_idx"]
        and row["global_step"] == 3
        and row["current_epoch"] == 2
        for row in rows
    )
    if not streaming:
        assert wrapped[(0, 2)]["view_idx"] == 2
    state = pickle.loads(pickle.dumps(wrapped))
    assert state._trainer is None


@pytest.mark.parametrize("reserved", ["global_step", "current_epoch"])
def test_dataset_rejects_overwriting_reserved_trainer_fields(reserved):
    wrapped = HFMapDataset(local_data())
    wrapped.set_pl_trainer(SimpleNamespace(global_step=0, current_epoch=0))
    with pytest.raises(ValueError, match="reserved key"):
        wrapped.process_sample({reserved: 10})


@pytest.mark.parametrize("indices", [[3, 0], [1, 2]])
def test_torch_wrapper_and_subset_batch_fetch_agree(indices):
    raw = torch.utils.data.TensorDataset(torch.arange(4), torch.arange(4) % 2)
    wrapped = FromTorchDataset(raw, ["value", "label"])
    subset = Subset(wrapped, indices)
    rows = subset.__getitems__([0, 1])
    assert [row["value"].item() for row in rows] == indices
    assert [row["sample_idx"] for row in rows] == indices
    assert subset.column_names == ["value", "label", "sample_idx"]
    assert len(subset) == 2
    restored = pickle.loads(pickle.dumps(subset))
    assert restored[0]["value"] == indices[0]


@pytest.mark.parametrize("mode", ["disabled", "mixup", "cutmix"])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_mixup_targets_match_actual_fraction_of_pixels_from_each_class(mode, smoothing):
    transform = RandomMixupCutmix(
        2,
        mixup_alpha=0.8 if mode != "cutmix" else 0.0,
        cutmix_alpha=1.0 if mode == "cutmix" else 0.0,
        prob=0.0 if mode == "disabled" else 1.0,
        label_smoothing=smoothing,
    )
    labels = torch.tensor([0, 1, 0, 1])
    for seed in range(5):
        torch.manual_seed(seed)
        images = labels.float().reshape(4, 1, 1, 1).expand(4, 3, 8, 8).clone()
        mixed, targets = transform(images, labels)
        expected_class1 = mixed.mean((1, 2, 3)) * (1 - smoothing) + smoothing / 2
        torch.testing.assert_close(targets[:, 1], expected_class1)
        torch.testing.assert_close(targets.sum(1), torch.ones(4))
        assert targets.min() >= 0 and targets.max() <= 1
