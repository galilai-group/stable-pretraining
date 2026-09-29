"""Synthetic and simulated data generators.

This module provides various synthetic data generators including manifold datasets,
noise generators, statistical models, and simulated environments for testing and
experimentation purposes.
"""

from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from loguru import logger as logging

from .datasets import Dataset


# ============================================================================
# MANIFOLD DATASETS
# ============================================================================


def swiss_roll(
    N,
    margin=1,
    sampler_time=torch.distributions.uniform.Uniform(0.1, 3),
    sampler_width=torch.distributions.uniform.Uniform(0, 1),
):
    """Generate Swiss Roll dataset points.

    Args:
        N: Number of points to generate
        margin: Margin parameter for the roll
        sampler_time: Distribution for sampling time parameter
        sampler_width: Distribution for sampling width parameter

    Returns:
        Tensor of shape (N, 3) containing Swiss Roll points
    """
    t0 = sampler_time.sample(sample_shape=(N,)) * 2 * np.pi
    radius = margin * t0 / np.pi + 0.1
    x = radius * torch.cos(t0)
    z = radius * torch.sin(t0)
    y = sampler_width.sample(sample_shape=(N,))
    xyz = torch.stack([x, y, z], 1)
    return xyz


# ============================================================================
# PERLIN NOISE GENERATORS
# ============================================================================


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a, b, x):
    return a + x * (b - a)


def _grad(hash, x, y):
    h = hash & 7
    u = torch.where(h < 4, x, y)
    v = torch.where(h < 4, y, x)
    return torch.where((h & 1) == 0, u, -u) + torch.where((h & 2) == 0, v, -v)


def _perlin(x, y, permutation):
    xi = x.to(torch.int32) & 255
    yi = y.to(torch.int32) & 255
    xf = x - x.to(torch.int32)
    yf = y - y.to(torch.int32)
    u = _fade(xf)
    v = _fade(yf)
    aa = permutation[permutation[xi] + yi]
    ab = permutation[permutation[xi] + yi + 1]
    ba = permutation[permutation[xi + 1] + yi]
    bb = permutation[permutation[xi + 1] + yi + 1]
    x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1, yf), u)
    x2 = _lerp(_grad(ab, xf, yf - 1), _grad(bb, xf - 1, yf - 1), u)
    return _lerp(x1, x2, v)


def generate_perlin_noise_2d(shape, res, octaves=1, persistence=0.5, lacunarity=2.0):
    """Generate 2D Perlin noise.

    Args:
        shape: Output shape (height, width)
        res: Resolution tuple
        octaves: Number of octaves for fractal noise
        persistence: Amplitude multiplier for each octave
        lacunarity: Frequency multiplier for each octave

    Returns:
        2D tensor of Perlin noise
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0]),
                torch.arange(0, res[1], delta[1]),
                indexing="ij",
            ),
            dim=-1,
        )
        % 256
    )
    permutation = torch.arange(256, dtype=torch.int32)
    permutation = permutation[torch.randperm(256)]
    permutation = torch.cat([permutation, permutation])
    noise = torch.zeros(shape)
    frequency = 1.0
    amplitude = 1.0
    max_amplitude = 0.0
    for _ in range(octaves):
        for i in range(d[0]):
            for j in range(d[1]):
                noise[i :: d[0], j :: d[1]] += amplitude * _perlin(
                    grid[i :: d[0], j :: d[1], 0] * frequency,
                    grid[i :: d[0], j :: d[1], 1] * frequency,
                    permutation,
                )
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    noise /= max_amplitude
    return noise


def perlin_noise_3d(x, y, z):
    """Generate 3D Perlin noise at given coordinates.

    Args:
        x: X coordinate for noise generation
        y: Y coordinate for noise generation
        z: Z coordinate for noise generation

    Returns:
        Perlin noise value at the given coordinates
    """

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(a, b, x):
        return a + x * (b - a)

    def grad(hash, x, y, z):
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h in (12, 14) else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    # Generate a permutation table
    perm = np.arange(256, dtype=int)
    np.random.shuffle(perm)
    perm = np.concatenate([perm, perm])
    xi = np.floor(x).astype(int) & 255
    yi = np.floor(y).astype(int) & 255
    zi = np.floor(z).astype(int) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    zf = z - np.floor(z)
    u = fade(xf)
    v = fade(yf)
    w = fade(zf)
    aaa = perm[perm[perm[xi] + yi] + zi]
    aba = perm[perm[perm[xi] + yi + 1] + zi]
    aab = perm[perm[perm[xi] + yi] + zi + 1]
    abb = perm[perm[perm[xi] + yi + 1] + zi + 1]
    baa = perm[perm[perm[xi + 1] + yi] + zi]
    bba = perm[perm[perm[xi + 1] + yi + 1] + zi]
    bab = perm[perm[perm[xi + 1] + yi] + zi + 1]
    bbb = perm[perm[perm[xi + 1] + yi + 1] + zi + 1]
    x1 = lerp(grad(aaa, xf, yf, zf), grad(baa, xf - 1, yf, zf), u)
    x2 = lerp(grad(aba, xf, yf - 1, zf), grad(bba, xf - 1, yf - 1, zf), u)
    y1 = lerp(x1, x2, v)
    x1 = lerp(grad(aab, xf, yf, zf - 1), grad(bab, xf - 1, yf, zf - 1), u)
    x2 = lerp(grad(abb, xf, yf - 1, zf - 1), grad(bbb, xf - 1, yf - 1, zf - 1), u)
    y2 = lerp(x1, x2, v)
    return (lerp(y1, y2, w) + 1) / 2


# ============================================================================
# STATISTICAL MODEL DATASETS
# ============================================================================


class GMM(Dataset):
    """Gaussian Mixture Model dataset for synthetic data generation."""

    def __init__(self, num_components=5, num_samples=100, dim=2):
        super().__init__()
        # Define the means for each component
        means = torch.rand(num_components, dim) * 10
        # Define the covariance matrices for each component
        # For simplicity, we'll use diagonal covariance matrices
        covariances = torch.stack(
            [torch.eye(dim) * torch.rand(1) for _ in range(num_components)]
        )
        # Define the mixing coefficients (weights) for each component
        weights = torch.distributions.Dirichlet(torch.ones(num_components)).sample()
        # Create a categorical distribution for the mixture components
        mix = dist.Categorical(weights)
        # Create a multivariate normal distribution for each component
        components = dist.MultivariateNormal(means, covariance_matrix=covariances)
        # Create the Gaussian Mixture Model
        self.model = dist.MixtureSameFamily(mix, components)
        self.samples = self.model.sample((num_samples,))
        # Calculate the log-likelihoods of all samples
        self.log_likelihoods = self.model.log_prob(self.samples)

    def score(self, samples):
        return self.model.log_prob(samples)

    def __getitem__(self, idx):
        sample = dict(
            sample=self.samples[idx], log_likelihood=self.log_likelihoods[idx]
        )
        return self.process_sample(sample)

    def __len__(self):
        return len(self.samples)


# ============================================================================
# SIMULATED ENVIRONMENT DATASETS
# ============================================================================


class MinariStepsDataset(Dataset):
    """Dataset for Minari reinforcement learning data with step-based access."""

    NAMES = ["observations", "actions", "rewards", "terminations", "truncations"]

    def __init__(self, dataset, num_steps=2, transform=None):
        super().__init__(transform)
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        self.num_steps = num_steps
        self.dataset = dataset

        # Metadata avoids reading every observation just to locate boundaries.
        # Minari episode IDs can be noncontiguous within a dataset subset.
        windows = [
            max(0, metadata["total_steps"] - num_steps + 1)
            for metadata in dataset.storage.get_episode_metadata(
                dataset.episode_indices
            )
        ]
        self.bounds = np.cumsum([0] + windows)
        self._length = int(self.bounds[-1])
        logging.info("Minari Dataset setup")
        logging.info(f"\t- {self.dataset.total_episodes} episodes")
        logging.info(f"\t- {len(self)} steps")

    def nested_step(self, value, idx):
        if isinstance(value, dict):
            return {k: self.nested_step(v, idx) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self.nested_step(v, idx) for v in value)
        return value[idx : idx + self.num_steps]

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        ep_idx = np.searchsorted(self.bounds, idx, side="right") - 1
        frame_idx = idx - self.bounds[ep_idx]
        episode = self.dataset[ep_idx]
        sample = {
            name: self.nested_step(getattr(episode, name), frame_idx)
            for name in self.NAMES
        }
        return self.process_sample(sample)

    def __len__(self):
        return self._length

    @property
    def column_names(self):
        return self.NAMES


class MinariEpisodeDataset(torch.utils.data.Dataset):
    """Flatten Minari episodes into individual training steps."""

    NAMES = ["observations", "actions", "rewards", "terminations", "truncations"]

    def __init__(self, dataset):
        self.dataset = dataset
        self.bounds = np.cumsum(
            [0]
            + [
                metadata["total_steps"]
                for metadata in dataset.storage.get_episode_metadata(
                    dataset.episode_indices
                )
            ]
        )
        self._trainer = None

        logging.info("Minari Dataset setup")
        logging.info(f"\t- {self.dataset.total_episodes} episodes")
        logging.info(f"\t- {len(self)} steps")

    def set_pl_trainer(self, trainer):
        self._trainer = trainer

    def nested_step(self, value, idx):
        if isinstance(value, dict):
            return {k: self.nested_step(v, idx) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self.nested_step(v, idx) for v in value)
        return value[idx]

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        ep_idx = np.searchsorted(self.bounds, idx, side="right") - 1
        frame_idx = idx - self.bounds[ep_idx]
        episode = self.dataset[ep_idx]
        sample = {
            name: self.nested_step(getattr(episode, name), frame_idx)
            for name in self.NAMES
        }
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("Can't use that keywords")
            if "current_epoch" in sample:
                raise ValueError("Can't use that keywords")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        return sample

    def __len__(self):
        return int(self.bounds[-1])

    @property
    def column_names(self):
        return self.NAMES


# ============================================================================
# NOISE MODELS FOR AUGMENTATION
# ============================================================================


class Categorical(torch.nn.Module):
    """Categorical distribution for sampling discrete values with given probabilities."""

    def __init__(
        self,
        values: Union[list, torch.Tensor],
        probabilities: Union[list, torch.Tensor],
    ):
        super().__init__()
        self.mix = torch.distributions.Categorical(torch.Tensor(probabilities))
        self.values = torch.Tensor(values)
        print(self.mix, self.values)

    def __call__(self):
        return self.values[self.mix.sample()]

    def sample(self, *args, **kwargs):
        return self.values[self.mix.sample(*args, **kwargs)]


class ExponentialMixtureNoiseModel(torch.nn.Module):
    """Exponential mixture noise model for data augmentation or sampling."""

    def __init__(self, rates, prior, upper_bound=torch.inf):
        super().__init__()
        mix = torch.distributions.Categorical(torch.Tensor(prior))
        comp = torch.distributions.Exponential(torch.Tensor(rates))
        self.mm = torch.distributions.MixtureSameFamily(mix, comp)
        self.upper_bound = upper_bound

    def __call__(self):
        return self.mm.sample().clip_(min=0, max=self.upper_bound)

    def sample(self, *args, **kwargs):
        return self.mm.sample(*args, **kwargs).clip_(min=0, max=self.upper_bound)


class ExponentialNormalNoiseModel(torch.nn.Module):
    """Exponential-normal noise model combining exponential and normal distributions."""

    def __init__(self, rate, mean, std, prior, upper_bound=torch.inf):
        super().__init__()
        self.mix = torch.distributions.Categorical(torch.Tensor(prior))
        self.exp = torch.distributions.Exponential(rate)
        self.gauss = torch.distributions.Normal(mean, std)
        self.upper_bound = upper_bound

    def __call__(self):
        mix = self.mix.sample()
        if mix == 0:
            return self.exp.sample().clip_(min=0, max=self.upper_bound)
        return self.gauss.sample().clip_(min=0, max=self.upper_bound)

    def sample(self, *args, **kwargs):
        mix = self.mix.sample(*args, **kwargs)
        exp = self.exp.sample(*args, **kwargs)
        gauss = self.gauss.sample(*args, **kwargs)
        return torch.where(mix.bool(), gauss, exp).clip_(min=0, max=self.upper_bound)
