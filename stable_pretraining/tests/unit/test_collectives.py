"""Single-process behavior lock for ``utils.all_gather`` / ``all_reduce``.

In a single process (no initialized process group) these must be identity-like:
``all_gather`` returns ``[tensor]`` and ``all_reduce`` returns the input
unchanged. The Barlow Twins test pins that the global-batch-normalization fix
does NOT alter the single-process loss (world_size == 1 ⇒ all_reduce is a
no-op), so the multi-GPU correctness fix is provably free for single-GPU users.

Multi-process correctness is covered in
``tests/distributed/test_collectives_dist.py``.
"""

import pytest
import torch

import torch.nn.functional as F

from stable_pretraining.losses import BarlowTwinsLoss, NTXEntLoss
from stable_pretraining.losses.utils import off_diagonal
from stable_pretraining.utils import all_gather, all_reduce


@pytest.mark.unit
class TestCollectivesSingleProcess:
    """Identity-like behavior + Barlow single-process invariance."""

    def test_all_gather_returns_local_only(self):
        x = torch.randn(3, 4)
        out = all_gather(x)
        assert len(out) == 1
        assert torch.equal(torch.cat(out, 0), x)

    def test_all_reduce_returns_input(self):
        x = torch.randn(3, 4)
        out = all_reduce(x)
        assert torch.equal(out, x)

    def test_barlow_single_process_unchanged(self):
        """The global-batch normalization must not change the single-process loss."""
        torch.manual_seed(0)
        z_i = torch.randn(8, 16)
        z_j = torch.randn(8, 16)
        loss = BarlowTwinsLoss(lambd=5e-3)(z_i, z_j)

        def _norm(z):
            return (z - z.mean(0)) / (z.std(0) + 1e-5)

        c = _norm(z_i).T @ _norm(z_j) / z_i.size(0)
        ref = (torch.diagonal(c) - 1).pow(2).sum() + 5e-3 * off_diagonal(c).pow(2).sum()
        assert torch.allclose(loss, ref, atol=1e-6), (loss.item(), ref.item())

    def test_ntxent_single_process_matches_classic_formula(self):
        """Single-process NT-Xent must equal the classic SimCLR formulation.

        Guards that the cross-GPU rank-offset refactor is bit-identical when
        ``world_size == 1`` (rank 0, gather is a no-op).
        """
        torch.manual_seed(1)
        z_i = torch.randn(4, 16)
        z_j = torch.randn(4, 16)
        loss = NTXEntLoss(temperature=0.5)(z_i, z_j)

        a = F.normalize(torch.cat([z_i, z_j], 0), dim=-1)
        logits = a @ a.T / 0.5
        n = 4
        targets = torch.cat([torch.arange(n, 2 * n), torch.arange(n)])
        logits = logits.masked_fill(torch.eye(2 * n, dtype=torch.bool), -torch.inf)
        ref = F.cross_entropy(logits, targets)
        assert torch.allclose(loss, ref, atol=1e-6), (loss.item(), ref.item())
