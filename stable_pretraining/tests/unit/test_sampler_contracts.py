"""Batch samplers must produce complete batches with deterministic rank partitioning."""

import numpy as np
import pytest
import torch

from stable_pretraining.data import sampler as sampling

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def restore_numpy_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.mark.parametrize(
    "source", [17, torch.utils.data.TensorDataset(torch.arange(17))]
)
def test_random_batch_sampler_covers_full_batches_without_duplicates(source):
    sampler = sampling.RandomBatchSampler(batch_size=4, length_or_dataset=source)
    batches = list(sampler)
    assert len(sampler) == len(batches) == 4
    flat = np.concatenate(batches)
    assert len(set(flat)) == 16
    assert min(flat) >= 0 and max(flat) < 17
    assert all(len(batch) == 4 for batch in batches)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_random_batch_sampler_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        sampling.RandomBatchSampler(batch_size, 10)


@pytest.mark.parametrize("as_dataset", [False, True])
def test_supervised_batches_group_distinct_examples_from_same_class(as_dataset):
    labels = np.repeat(np.arange(3), 8)
    if as_dataset:
        source = torch.utils.data.TensorDataset(torch.arange(len(labels)))
        source.targets = labels.tolist()
    else:
        source = labels.tolist()
    sampler = sampling.SupervisedBatchSampler(6, 2, source)
    batches = list(sampler)
    assert len(sampler) == len(batches) == 4
    for batch in batches:
        assert len(batch) == 6
        for pair in np.asarray(batch).reshape(-1, 2):
            assert pair[0] != pair[1]
            assert labels[pair[0]] == labels[pair[1]]


@pytest.mark.parametrize(
    "batch_size,views,labels",
    [(3, 2, [0] * 8), (4, 2, [0, 1, 1]), (4, 0, [0] * 8), (True, 1, [0] * 8)],
)
def test_supervised_invalid_grouping_fails_before_iteration(batch_size, views, labels):
    with pytest.raises(ValueError):
        sampling.SupervisedBatchSampler(batch_size, views, labels)


@pytest.mark.parametrize("views", [1, 3])
def test_repeated_sampler_ranks_are_disjoint_and_epoch_is_reproducible(
    monkeypatch, views
):
    monkeypatch.setattr(sampling.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sampling.dist, "get_world_size", lambda: 3)
    rank_samples = []
    for rank in range(3):
        monkeypatch.setattr(sampling.dist, "get_rank", lambda: rank)
        sampler = sampling.RepeatedRandomSampler(
            20, n_views=views, pass_view_idx=True, seed=12
        )
        first = list(sampler)
        assert len(first) == len(sampler) == 6 * views
        assert list(sampler) == first
        sampler.set_epoch(1)
        assert list(sampler) != first
        for i in range(0, len(first), views):
            assert first[i : i + views] == [
                (first[i][0], view) for view in range(views)
            ]
        rank_samples.append({idx for idx, _ in first})
    assert len(set.union(*rank_samples)) == 18
    assert not rank_samples[0] & rank_samples[1]
