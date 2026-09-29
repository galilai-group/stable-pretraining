"""Reproducibility, geometry, and distribution checks for synthetic data."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stable_pretraining.data import synthetic_data as synthetic

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_rng():
    state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            np.random.set_state(state)


def test_swiss_roll_geometry_and_seed_reproducibility():
    kwargs = dict(
        N=32,
        margin=2,
        sampler_time=torch.distributions.Uniform(0.1, 0.2),
        sampler_width=torch.distributions.Uniform(5.0, 6.0),
    )
    torch.manual_seed(17)
    points = synthetic.swiss_roll(**kwargs)
    torch.manual_seed(17)
    torch.testing.assert_close(points, synthetic.swiss_roll(**kwargs))
    assert points.shape == (32, 3)
    radius = points[:, [0, 2]].norm(dim=1)
    assert torch.all((radius >= 0.5) & (radius <= 0.9))
    assert torch.all((points[:, 1] >= 5) & (points[:, 1] <= 6))


@pytest.mark.parametrize("octaves", [1, 3])
def test_perlin_image_is_finite_nonconstant_and_reproducible(octaves):
    torch.manual_seed(19)
    noise = synthetic.generate_perlin_noise_2d((12, 16), (3, 4), octaves=octaves)
    torch.manual_seed(19)
    repeated = synthetic.generate_perlin_noise_2d((12, 16), (3, 4), octaves=octaves)
    torch.testing.assert_close(noise, repeated)
    assert noise.shape == (12, 16)
    assert torch.isfinite(noise).all()
    assert noise.std() > 0.01


def test_perlin_values_do_not_depend_on_other_points_in_batch():
    permutation = torch.arange(256).repeat(2)
    x = torch.tensor([0.2, 1.3, 2.4, 3.5, 4.6, 5.7, 6.8, 7.9])
    y = torch.tensor([0.7, 1.6, 2.5, 3.4, 4.3, 5.2, 6.1, 7.2])
    batched = synthetic._perlin(x, y, permutation)
    individual = torch.stack(
        [synthetic._perlin(a, b, permutation) for a, b in zip(x, y)]
    )
    torch.testing.assert_close(batched, individual)


@pytest.mark.parametrize("point", [(0.0, 0.0, 0.0), (-2.0, 3.0, 1.0)])
def test_perlin_3d_is_neutral_at_lattice_points(point):
    assert synthetic.perlin_noise_3d(*point) == pytest.approx(0.5)


def test_perlin_3d_seed_reproducibility():
    np.random.seed(3)
    value = synthetic.perlin_noise_3d(0.2, -0.4, 0.7)
    np.random.seed(3)
    assert synthetic.perlin_noise_3d(0.2, -0.4, 0.7) == value
    assert 0 <= value <= 1


def test_gaussian_dataset_matches_analytic_density_and_batches():
    torch.manual_seed(5)
    dataset = synthetic.GMM(num_components=1, num_samples=16, dim=3)
    component = dataset.model.component_distribution
    expected = (
        torch.distributions.Normal(component.loc[0], component.stddev[0])
        .log_prob(dataset.samples)
        .sum(-1)
    )
    torch.testing.assert_close(dataset.score(dataset.samples), expected)
    torch.testing.assert_close(dataset.log_likelihoods, expected)
    dataset.set_pl_trainer(SimpleNamespace(global_step=12, current_epoch=2))
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))
    assert len(dataset) == 16
    assert batch["sample"].shape == (4, 3)
    torch.testing.assert_close(batch["log_likelihood"], expected[:4])
    assert batch["global_step"].tolist() == [12] * 4
    assert batch["current_epoch"].tolist() == [2] * 4


def test_gaussian_dataset_reproducibility():
    torch.manual_seed(5)
    first = synthetic.GMM(num_components=3, num_samples=8, dim=2)
    torch.manual_seed(5)
    second = synthetic.GMM(num_components=3, num_samples=8, dim=2)
    torch.testing.assert_close(first.samples, second.samples)
    torch.testing.assert_close(first.log_likelihoods, second.log_likelihoods)


def test_categorical_sampler_respects_zero_probability():
    model = synthetic.Categorical(values=[-5, 7, 20], probabilities=[0, 1, 0])
    assert model().item() == 7
    assert torch.equal(model.sample((2, 3)), torch.full((2, 3), 7.0))


def test_exponential_mixture_samples_are_bounded_and_reproducible():
    model = synthetic.ExponentialMixtureNoiseModel(
        rates=[1, 2], prior=[0.3, 0.7], upper_bound=0.2
    )
    assert 0 <= model() <= 0.2
    torch.manual_seed(6)
    samples = model.sample((16, 4))
    torch.manual_seed(6)
    torch.testing.assert_close(samples, model.sample((16, 4)))
    assert samples.shape == (16, 4)
    assert torch.all((samples >= 0) & (samples <= 0.2))
    assert (samples == 0.2).any()


@pytest.mark.parametrize(
    "prior,mean,expected",
    [([0, 1], -100.0, 0.0), ([0, 1], 100.0, 0.5), ([1, 0], 0.0, None)],
)
def test_exponential_normal_selection_and_clipping(prior, mean, expected):
    model = synthetic.ExponentialNormalNoiseModel(
        rate=1.0, mean=mean, std=0.01, prior=prior, upper_bound=0.5
    )
    scalar = model()
    samples = model.sample((8, 3))
    assert 0 <= scalar <= 0.5
    assert samples.shape == (8, 3)
    assert torch.all((samples >= 0) & (samples <= 0.5))
    if expected is not None:
        assert scalar.item() == expected
        torch.testing.assert_close(samples, torch.full_like(samples, expected))
