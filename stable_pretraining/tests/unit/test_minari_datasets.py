"""Minari subsets and short episodes must never create cross-episode windows."""

from types import SimpleNamespace

import numpy as np
import pytest
from minari.dataset.episode_data import EpisodeData

from stable_pretraining.data.synthetic_data import (
    MinariEpisodeDataset,
    MinariStepsDataset,
)

pytestmark = pytest.mark.unit


class _Subset:
    def __init__(self, lengths):
        self.episode_indices = np.arange(len(lengths)) * 3 + 5
        self.episodes = []
        for episode_id, length in zip(self.episode_indices, lengths):
            rewards = episode_id * 100 + np.arange(length)
            observations = np.concatenate([rewards, [episode_id * 100 + length]])
            self.episodes.append(
                EpisodeData(
                    id=episode_id,
                    observations={"state": (observations[:, None], observations)},
                    actions=rewards[:, None],
                    rewards=rewards,
                    terminations=np.zeros(length, dtype=bool),
                    truncations=np.zeros(length, dtype=bool),
                    infos={},
                )
            )
        self.total_episodes = len(lengths)
        self.total_steps = sum(lengths)
        self.reads = 0
        self.storage = SimpleNamespace(get_episode_metadata=self._metadata)

    def _metadata(self, indices):
        assert np.array_equal(indices, self.episode_indices)
        return [{"total_steps": len(episode)} for episode in self.episodes]

    def __getitem__(self, position):
        self.reads += 1
        return self.episodes[position]


@pytest.mark.parametrize("lengths", [(1, 4, 2), (0, 3, 0, 2), (), (1, 1)])
@pytest.mark.parametrize("window", [1, 2, 3])
def test_windows_match_explicit_per_episode_slices(lengths, window):
    source = _Subset(lengths)
    data = MinariStepsDataset(source, num_steps=window)
    assert source.reads == 0
    expected = [
        episode.rewards[start : start + window]
        for episode in source.episodes
        for start in range(max(0, len(episode) - window + 1))
    ]
    assert len(data) == len(expected)
    assert data.column_names == data.NAMES
    for idx, rewards in enumerate(expected):
        sample = data[idx]
        np.testing.assert_array_equal(sample["rewards"], rewards)
        np.testing.assert_array_equal(sample["observations"]["state"][0][:, 0], rewards)
        np.testing.assert_array_equal(sample["observations"]["state"][1], rewards)
    if expected:
        np.testing.assert_array_equal(data[-1]["rewards"], expected[-1])
    for idx in (len(data), -len(data) - 1):
        with pytest.raises(IndexError):
            data[idx]


@pytest.mark.parametrize("lengths", [(1, 4, 2), (0, 3, 0, 2), ()])
def test_flattened_steps_use_episode_lengths_and_preserve_nested_observations(
    lengths, capsys
):
    source = _Subset(lengths)
    data = MinariEpisodeDataset(source)
    assert source.reads == 0
    assert len(data) == sum(lengths)
    assert data.column_names == data.NAMES
    expected = [reward for episode in source.episodes for reward in episode.rewards]
    for idx, reward in enumerate(expected):
        sample = data[idx]
        assert sample["rewards"] == reward
        assert sample["observations"]["state"][0].item() == reward
        assert sample["observations"]["state"][1] == reward
    data.set_pl_trainer(SimpleNamespace(global_step=8, current_epoch=3))
    if expected:
        assert data[-1]["rewards"] == expected[-1]
        assert data[0]["global_step"] == 8 and data[0]["current_epoch"] == 3
    for idx in (len(data), -len(data) - 1):
        with pytest.raises(IndexError):
            data[idx]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("window", [0, -1, 1.5])
def test_invalid_window_length_is_rejected(window):
    with pytest.raises(ValueError, match="num_steps"):
        MinariStepsDataset(_Subset([3]), num_steps=window)
