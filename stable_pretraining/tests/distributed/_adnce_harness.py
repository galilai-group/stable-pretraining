"""Spawn workers for ADNCE checks against an independent global objective."""

import copy
import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def _reference_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float,
    mu: float,
    sigma: float,
) -> torch.Tensor:
    """Evaluate the reference image objective without distributed operations."""
    features = F.normalize(torch.cat((first, second)), dim=1)
    batch_size = first.shape[0]
    losses = []
    for anchor_index in range(2 * batch_size):
        positive_index = (anchor_index + batch_size) % (2 * batch_size)
        negative_indices = [
            index
            for index in range(2 * batch_size)
            if index not in (anchor_index, positive_index)
        ]
        similarity = features[anchor_index] @ features.T
        negatives = similarity[negative_indices]
        weights = torch.exp(-((negatives - mu) ** 2) / (2 * sigma**2)).detach()
        weights = weights / weights.mean()
        positive = torch.exp(similarity[positive_index] / temperature)
        denominator = positive + (weights * torch.exp(negatives / temperature)).sum()
        losses.append(-torch.log(positive / denominator))
    return torch.stack(losses).mean()


def _check_global_equivalence(rank: int, world_size: int) -> None:
    from stable_pretraining.losses import ADNCELoss

    # Batch size one needs negatives supplied by the other process.
    for local_batch in (1, 3):
        for temperature, mu, sigma in ((0.4, 0.7, 0.5), (0.23, -0.1, 0.3)):
            torch.manual_seed(41 + local_batch)
            model = nn.Sequential(nn.Linear(5, 9), nn.Tanh(), nn.Linear(9, 7)).double()
            reference_model = copy.deepcopy(model)
            distributed_model = DistributedDataParallel(model)
            global_batch = world_size * local_batch
            first = torch.randn(global_batch, 5, dtype=torch.float64)
            second = torch.randn(global_batch, 5, dtype=torch.float64)
            start = rank * local_batch
            stop = start + local_batch
            local_input = torch.cat((first[start:stop], second[start:stop]))
            local_features = distributed_model(local_input)
            loss = ADNCELoss(temperature=temperature, mu=mu, sigma=sigma)(
                local_features[:local_batch], local_features[local_batch:]
            )

            reference_features = reference_model(torch.cat((first, second)))
            reference = _reference_loss(
                reference_features[:global_batch],
                reference_features[global_batch:],
                temperature,
                mu,
                sigma,
            )
            average_loss = loss.detach().clone()
            dist.all_reduce(average_loss)
            average_loss /= world_size
            torch.testing.assert_close(average_loss, reference, atol=1e-12, rtol=1e-12)

            loss.backward()
            reference.backward()
            for parameter, reference_parameter in zip(
                model.parameters(), reference_model.parameters(), strict=True
            ):
                assert parameter.grad is not None
                torch.testing.assert_close(
                    parameter.grad, reference_parameter.grad, atol=1e-11, rtol=1e-10
                )

            # Actual DDP parameters must take the same step, not just report
            # the same scalar loss with incorrectly scaled gradients.
            torch.optim.SGD(model.parameters(), lr=0.1).step()
            torch.optim.SGD(reference_model.parameters(), lr=0.1).step()
            for parameter, reference_parameter in zip(
                model.parameters(), reference_model.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    parameter, reference_parameter, atol=1e-12, rtol=1e-11
                )


def _check_input_mismatch(rank: int, world_size: int) -> None:
    from stable_pretraining.losses import ADNCELoss

    cases = (
        ((rank + 1, 7), torch.float32),
        ((2, rank + 5), torch.float32),
        ((2, 7), torch.float64 if rank else torch.float32),
    )
    for shape, dtype in cases:
        first = torch.randn(shape, dtype=dtype, requires_grad=True)
        second = torch.randn(shape, dtype=dtype, requires_grad=True)
        caught = False
        try:
            ADNCELoss()(first, second)
        except ValueError:
            caught = True
        assert caught, f"Rank {rank} did not reject incompatible per-rank inputs"
        # Reaching a collective on every rank proves the error did not leave
        # peers stuck in the subsequent feature gather.
        success = torch.tensor(int(caught))
        dist.all_reduce(success)
        assert success.item() == world_size


def worker(rank: int, world_size: int, rendezvous: str, scenario: str) -> None:
    """Run one two-process scenario with bounded collective waits.

    Args:
        rank: Process index assigned by the parent.
        world_size: Number of participating processes.
        rendezvous: File URL used to initialize the process group.
        scenario: Either ``equivalence`` or ``input_mismatch``.
    """
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        if scenario == "equivalence":
            _check_global_equivalence(rank, world_size)
        elif scenario == "input_mismatch":
            _check_input_mismatch(rank, world_size)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        dist.barrier()
    finally:
        dist.destroy_process_group()
