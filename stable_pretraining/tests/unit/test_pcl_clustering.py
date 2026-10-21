"""Unit tests for PCL clustering."""

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.methods.pcl import (
    estimate_concentrations,
    kmeans,
)


@pytest.mark.unit
def test_kmeans_two_clusters():
    features = torch.tensor(
        [
            [-2.0, 0.0],
            [-2.1, 0.1],
            [-1.9, -0.1],
            [2.0, 0.0],
            [2.1, 0.1],
            [1.9, -0.1],
        ]
    )

    assignments, centroids, concentrations = kmeans(features, num_clusters=2, seed=3)

    assert assignments.shape == (6,)
    assert centroids.shape == (2, 2)

    assert torch.unique(assignments[:3]).numel() == 1
    assert torch.unique(assignments[3:]).numel() == 1
    assert assignments[0] != assignments[3]

    torch.testing.assert_close(
        centroids.norm(dim=1),
        torch.ones(2),
    )


@pytest.mark.unit
def test_kmeans_reproducibility():
    generator = torch.Generator().manual_seed(7)
    features = F.normalize(torch.randn(30, 8, generator=generator), dim=1)

    a1, c1, phi1 = kmeans(features, num_clusters=4, seed=42)
    a2, c2, phi2 = kmeans(features, num_clusters=4, seed=42)

    # Cluster assignments should be reproducible.
    torch.testing.assert_close(a1, a2)

    # Cluster prototypes should be reproducible.
    torch.testing.assert_close(c1, c2)

    # Concentrations should also be reproducible.
    torch.testing.assert_close(phi1, phi2)


@pytest.mark.unit
def test_concentrations_have_correct_mean():
    features = torch.tensor(
        [
            [-2.0, 0.0],
            [-2.1, 0.1],
            [-1.9, -0.1],
            [2.0, 0.0],
            [2.1, 0.1],
            [1.9, -0.1],
        ]
    )

    assignments = torch.tensor([0, 0, 0, 1, 1, 1])
    raw_centroids = torch.tensor(
        [
            [-2.0, 0.0],
            [2.0, 0.0],
        ]
    )

    phi = estimate_concentrations(
        features,
        assignments,
        raw_centroids,
        temperature=0.2,
    )

    assert phi.shape == (2,)
    assert torch.isfinite(phi).all()
    assert (phi > 0).all()
    torch.testing.assert_close(
        phi.mean(),
        torch.tensor(0.2),
    )


@pytest.mark.unit
def test_singleton_cluster_uses_fallback():
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [2.0, 0.0],
        ]
    )

    assignments = torch.tensor([0, 0, 1])
    raw_centroids = torch.tensor(
        [
            [0.1, 0.0],
            [2.0, 0.0],
        ]
    )

    phi = estimate_concentrations(
        features,
        assignments,
        raw_centroids,
    )

    assert torch.isfinite(phi).all()
    assert (phi > 0).all()
    torch.testing.assert_close(
        phi.mean(),
        torch.tensor(0.2),
    )


@pytest.mark.unit
def test_all_singleton_clusters_rejected():
    features = torch.eye(3)
    assignments = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError, match="At least one cluster"):
        estimate_concentrations(
            features,
            assignments,
            features.clone(),
        )


@pytest.mark.unit
def test_kmeans_configurable_temperature():
    features = torch.tensor(
        [
            [-2.0, 0.0],
            [-2.1, 0.1],
            [-1.9, -0.1],
            [2.0, 0.0],
            [2.1, 0.1],
            [1.9, -0.1],
        ]
    )

    _, _, phi = kmeans(
        features,
        num_clusters=2,
        seed=3,
        temperature=0.5,
    )

    torch.testing.assert_close(
        phi.mean(),
        torch.tensor(0.5),
    )


@pytest.mark.unit
def test_kmeans_repairs_empty_clusters():
    # Two clusters begin with a chance of receiving competing
    # initial centers. Verify the returned assignment is nonempty.
    features = torch.tensor(
        [
            [-5.0, 0.0],
            [-4.9, 0.0],
            [-4.8, 0.0],
            [0.0, 0.0],
            [0.2, 0.1],
            [5.0, 0.0],
            [5.1, 0.1],
            [5.2, 0.0],
        ]
    )

    assignments, prototypes, phi = kmeans(
        features,
        num_clusters=3,
        seed=7,
    )

    counts = torch.bincount(assignments, minlength=3)

    assert (counts > 0).all()
    assert prototypes.shape == (3, 2)
    assert torch.isfinite(phi).all()
    assert (phi > 0).all()


@pytest.mark.unit
def test_kmeans_centroids_match_assignments():
    features = torch.tensor(
        [
            [-3.0, 0.0],
            [-2.8, 0.2],
            [-2.9, -0.2],
            [3.0, 0.0],
            [3.2, 0.2],
            [2.8, -0.2],
        ]
    )

    assignments, prototypes, phi = kmeans(
        features,
        num_clusters=2,
        seed=3,
    )

    for cluster_id in range(2):
        members = features[assignments == cluster_id]
        assert members.shape[0] > 0

        expected = F.normalize(members.mean(dim=0), dim=0)

        torch.testing.assert_close(
            prototypes[cluster_id],
            expected,
        )

    torch.testing.assert_close(
        phi.mean(),
        torch.tensor(0.2),
    )
