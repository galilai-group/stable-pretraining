"""Train RC-aux reachability on deterministic one-dimensional trajectories.

The example uses trajectories whose state increases by one at every step. The
ground-truth reachability label is therefore known exactly: a goal at distance
``d`` is reachable if and only if the available budget is at least ``d``.
"""

import argparse
import math

import torch

from stable_pretraining.losses import RCauxLoss


def make_trajectories(starts: torch.Tensor, length: int = 6) -> torch.Tensor:
    """Create deterministic chain trajectories from their initial positions.

    Args:
        starts: Trajectory identity angles with shape ``(batch,)``.
        length: Number of states in each trajectory.

    Returns:
        Trajectory embeddings with shape ``(batch, length, 3)``. The first two
        coordinates identify a trajectory on the unit circle and the third is
        normalized time.
    """
    identity = torch.stack((starts.cos(), starts.sin()), dim=-1)
    identity = identity[:, None].expand(-1, length, -1)
    time = torch.linspace(0, 1, length, device=starts.device, dtype=starts.dtype).view(
        1, length, 1
    )
    return torch.cat((identity, time.expand(starts.shape[0], -1, -1)), dim=-1)


@torch.no_grad()
def evaluate(objective: RCauxLoss, encoded: torch.Tensor) -> dict[str, float]:
    """Evaluate temporal labels and cross-trajectory negatives.

    Args:
        objective: Trained RC-aux objective.
        encoded: Held-out deterministic trajectories.

    Returns:
        Accuracy and mean-probability metrics.
    """
    sources = []
    goals = []
    budgets = []
    labels = []
    max_horizon = objective.max_horizon

    for distance in range(1, max_horizon + 1):
        source = encoded[:, 0]
        goal = encoded[:, distance]
        for budget in range(1, max_horizon + 1):
            sources.append(source)
            goals.append(goal)
            budgets.append(
                torch.full(
                    (encoded.shape[0],),
                    budget,
                    dtype=torch.long,
                    device=encoded.device,
                )
            )
            labels.append(
                torch.full(
                    (encoded.shape[0],),
                    budget >= distance,
                    dtype=torch.bool,
                    device=encoded.device,
                )
            )

    source = torch.cat(sources)
    goal = torch.cat(goals)
    budget = torch.cat(budgets)
    label = torch.cat(labels)
    probability = objective.score_reachability(source, goal, budget)
    temporal_accuracy = ((probability >= 0.5) == label).float().mean()

    negative_goal = encoded[:, -1].roll(shifts=1, dims=0)
    negative_source = encoded[:, 0]
    negative_budget = torch.full(
        (encoded.shape[0],), max_horizon, dtype=torch.long, device=encoded.device
    )
    negative_probability = objective.score_reachability(
        negative_source, negative_goal, negative_budget
    )

    return {
        "temporal_accuracy": temporal_accuracy.item(),
        "positive_probability": probability[label].mean().item(),
        "temporal_negative_probability": probability[~label].mean().item(),
        "batch_negative_accuracy": (negative_probability < 0.5).float().mean().item(),
    }


def main() -> None:
    """Train and evaluate the synthetic RC-aux benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    objective = RCauxLoss(
        embedding_dim=3,
        max_horizon=5,
        hidden_dim=64,
        horizon_dim=16,
        prediction_weight=0.0,
        reachability_weight=1.0,
    )
    optimizer = torch.optim.AdamW(objective.parameters(), lr=3e-3)

    for _ in range(args.steps):
        offset = torch.rand(()) * 2 * math.pi
        starts = (
            torch.arange(args.batch_size).float() * 2 * math.pi / args.batch_size
            + offset
        )
        encoded = make_trajectories(starts)
        predicted = encoded[:, 1:].clone()
        result = objective(encoded, predicted, context_index=1)
        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()

    held_out_starts = (
        torch.arange(args.batch_size).float() * 2 * math.pi / args.batch_size
        + math.pi / args.batch_size
    )
    metrics = evaluate(objective, make_trajectories(held_out_starts))
    metrics["final_loss"] = result["loss"].item()
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
