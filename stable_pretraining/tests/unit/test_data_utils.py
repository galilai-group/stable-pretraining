"""Data integrity and resource-allocation checks for the data utilities."""

import pytest
import torch

from stable_pretraining.data import utils

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "cpus,world_size,expected", [(16, 1, 16), (16, 4, 4), (2, 8, 1), (0, 1, 1)]
)
def test_workers_respect_affinity_and_distributed_world(
    monkeypatch, cpus, world_size, expected
):
    monkeypatch.setattr(
        utils.os, "sched_getaffinity", lambda _: set(range(cpus)), raising=False
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    assert utils.get_num_workers() == expected


@pytest.mark.parametrize("cpus,expected", [(None, 1), (8, 8)])
@pytest.mark.parametrize("distributed_available", [True, False])
def test_workers_without_affinity_or_distributed_initialization(
    monkeypatch, cpus, expected, distributed_available
):
    monkeypatch.delattr(utils.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(utils.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(
        torch.distributed, "is_available", lambda: distributed_available
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert utils.get_num_workers() == expected


@pytest.mark.parametrize(
    "lengths,expected", [([3, 5], [3, 5]), ([0.3, 0.3, 0.4], [3, 2, 3])]
)
def test_split_is_reproducible_disjoint_and_exhaustive(lengths, expected):
    dataset = list(range(8))
    splits = utils.random_split(dataset, lengths, torch.Generator().manual_seed(7))
    repeated = utils.random_split(dataset, lengths, torch.Generator().manual_seed(7))
    assert [len(split) for split in splits] == expected
    assert [split.indices for split in splits] == [split.indices for split in repeated]
    samples = [sample for split in splits for sample in split]
    assert sorted(samples) == dataset
    assert len(set(samples)) == len(dataset)


def test_split_warns_about_empty_fractional_subset():
    with pytest.warns(UserWarning, match="Length of split at index 1 is 0"):
        splits = utils.random_split([1], [0.5, 0.5])
    assert [len(split) for split in splits] == [1, 0]


@pytest.mark.parametrize(
    "lengths,match",
    [([2, 2], "Sum of input lengths"), ([-0.1, 1.1], "not between 0 and 1")],
)
def test_split_rejects_invalid_lengths(lengths, match):
    with pytest.raises(ValueError, match=match):
        utils.random_split(list(range(8)), lengths)


@pytest.mark.parametrize("lengths", [[-1, 9], [2.5, 5.5]])
def test_split_rejects_invalid_absolute_sizes(lengths):
    with pytest.raises(ValueError, match="non-negative integers"):
        utils.random_split(list(range(8)), lengths)


def test_fold_views_keeps_sample_alignment_and_gradients():
    values = torch.arange(12.0).reshape(6, 2).requires_grad_()
    indices = torch.tensor([3, 1, 3, 1, 2, 2])
    first, second = utils.fold_views(values, indices)
    torch.testing.assert_close(first, values[[1, 4, 0]])
    torch.testing.assert_close(second, values[[3, 5, 2]])
    (first.sum() + second.sum()).backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_fold_views_rejects_missing_view():
    with pytest.raises(RuntimeError, match="counts are not the same"):
        utils.fold_views(torch.zeros(3, 2), torch.tensor([0, 0, 1]))


def test_apply_masks_requires_a_mask():
    with pytest.raises(ValueError, match="At least one mask"):
        utils.apply_masks(torch.zeros(2, 4, 3))
