"""Matched-budget RC-aux validation on real Reacher trajectories.

This experiment trains a small action-conditioned latent world model on an HDF5
Reacher dataset. It compares prediction-only continuation with RC-aux under the
same windows, optimizer, architecture, update count, and evaluation candidates.
The offline planning task asks each model to retrieve the demonstrated action
sequence that reaches a held-out future state among action-sequence distractors.
"""

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from stable_pretraining.losses import RCauxLoss


class ReacherLatentWorldModel(nn.Module):
    """Small residual latent dynamics model for state-based Reacher validation.

    Args:
        observation_dim: Number of state observation features.
        action_dim: Number of action features.
        hidden_dim: Width of the dynamics network.
    """

    def __init__(
        self, observation_dim: int, action_dim: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(observation_dim, observation_dim)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(observation_dim))
            self.encoder.bias.zero_()
        self.dynamics = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, observation_dim),
        )

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode normalized observations into latent states.

        Args:
            observation: Normalized observations with features in the last axis.

        Returns:
            Latent states with the same shape as ``observation``.
        """
        return self.encoder(observation)

    def rollout(
        self, initial_latent: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Predict an open-loop latent rollout.

        Args:
            initial_latent: Initial latent states with shape ``(batch, dim)``.
            actions: Action sequences with shape ``(batch, horizon, action_dim)``.

        Returns:
            Predicted latent states with shape ``(batch, horizon, dim)``.
        """
        latent = initial_latent
        predictions = []
        for step in range(actions.shape[1]):
            delta = self.dynamics(torch.cat((latent, actions[:, step]), dim=-1))
            latent = latent + delta
            predictions.append(latent)
        return torch.stack(predictions, dim=1)

    def trajectory(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a trajectory and predict its future latents.

        Args:
            observations: State sequences with shape ``(batch, horizon + 1, dim)``.
            actions: Aligned actions with shape ``(batch, horizon, action_dim)``.

        Returns:
            Encoded states, open-loop predictions, and an identity-anchor loss.
        """
        encoded = self.encode(observations)
        predicted = self.rollout(encoded[:, 0], actions)
        anchor_loss = F.mse_loss(encoded, observations)
        return encoded, predicted, anchor_loss


def load_reacher_windows(
    dataset_path: Path,
    horizon: int,
    train_episodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load normalized train and test windows without crossing episode borders.

    Args:
        dataset_path: HDF5 file with observation, action, ep_offset, and ep_len.
        horizon: Number of actions and predicted future states per window.
        train_episodes: Number of leading episodes used for training.

    Returns:
        Train observations, train actions, test observations, and test actions.
    """
    with h5py.File(dataset_path, "r") as dataset:
        observations = np.asarray(dataset["observation"], dtype=np.float32)
        actions = np.asarray(dataset["action"], dtype=np.float32)
        episode_offsets = np.asarray(dataset["ep_offset"], dtype=np.int64)
        episode_lengths = np.asarray(dataset["ep_len"], dtype=np.int64)

    if not 0 < train_episodes < len(episode_lengths):
        raise ValueError("train_episodes must leave at least one held-out episode")
    if np.any(episode_lengths <= horizon):
        raise ValueError("every episode must be longer than the rollout horizon")

    train_end = int(
        episode_offsets[train_episodes - 1] + episode_lengths[train_episodes - 1]
    )
    observation_mean = observations[:train_end].mean(axis=0, keepdims=True)
    observation_std = observations[:train_end].std(axis=0, keepdims=True) + 1e-6
    action_mean = np.nanmean(actions[:train_end], axis=0, keepdims=True)
    action_std = np.nanstd(actions[:train_end], axis=0, keepdims=True) + 1e-6
    observations = (observations - observation_mean) / observation_std
    actions = (actions - action_mean) / action_std
    actions = np.nan_to_num(actions)

    train_starts = []
    test_starts = []
    for episode, (offset, length) in enumerate(
        zip(episode_offsets, episode_lengths, strict=True)
    ):
        target = train_starts if episode < train_episodes else test_starts
        target.extend(range(int(offset), int(offset + length - horizon)))

    def build_windows(starts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        start_array = np.asarray(starts, dtype=np.int64)
        observation_indices = start_array[:, None] + np.arange(horizon + 1)[None]
        action_indices = start_array[:, None] + np.arange(horizon)[None]
        return (
            torch.from_numpy(observations[observation_indices]),
            torch.from_numpy(actions[action_indices]),
        )

    train_observations, train_actions = build_windows(train_starts)
    test_observations, test_actions = build_windows(test_starts)
    return train_observations, train_actions, test_observations, test_actions


def train_world_model(
    initial_state: dict[str, torch.Tensor],
    train_observations: torch.Tensor,
    train_actions: torch.Tensor,
    batch_schedule: torch.Tensor,
    device: torch.device,
    rcaux: bool,
    horizon: int,
    learning_rate: float,
) -> tuple[ReacherLatentWorldModel, RCauxLoss | None, float]:
    """Train one matched-budget prediction-only or RC-aux model.

    Args:
        initial_state: Shared initial world-model parameters.
        train_observations: Normalized training observation windows.
        train_actions: Normalized training action windows.
        batch_schedule: Precomputed example indices for every update.
        device: Training device.
        rcaux: Whether to train the reachability auxiliary objective.
        horizon: Open-loop prediction horizon.
        learning_rate: AdamW learning rate.

    Returns:
        Trained world model, optional RC-aux objective, and final total loss.
    """
    model = ReacherLatentWorldModel(
        observation_dim=train_observations.shape[-1],
        action_dim=train_actions.shape[-1],
    ).to(device)
    model.load_state_dict(initial_state)
    objective = None
    parameters = list(model.parameters())
    if rcaux:
        objective = RCauxLoss(
            embedding_dim=train_observations.shape[-1],
            max_horizon=horizon,
            hidden_dim=128,
            horizon_dim=16,
            prediction_weight=1.0,
            reachability_weight=0.35,
            predicted_pair_weight=0.5,
            regularization_weight=0.1,
        ).to(device)
        parameters.extend(objective.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

    final_loss = 0.0
    model.train()
    for indices in batch_schedule:
        observations = train_observations[indices].to(device)
        actions = train_actions[indices].to(device)
        encoded, predicted, anchor_loss = model.trajectory(observations, actions)
        if objective is None:
            prediction_loss = F.mse_loss(predicted, encoded[:, 1:])
            loss = prediction_loss + 0.1 * anchor_loss
        else:
            loss = objective(
                encoded,
                predicted,
                context_index=1,
                regularization_loss=anchor_loss,
            )["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return model, objective, final_loss


@torch.no_grad()
def prediction_rmse(
    model: ReacherLatentWorldModel,
    observations: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> float:
    """Measure held-out multi-horizon latent prediction error.

    Args:
        model: Trained latent world model.
        observations: Held-out observation windows.
        actions: Held-out action windows.
        device: Evaluation device.
        batch_size: Evaluation batch size.

    Returns:
        Root mean squared latent prediction error.
    """
    squared_error = 0.0
    element_count = 0
    model.eval()
    for start in range(0, len(observations), batch_size):
        observation_batch = observations[start : start + batch_size].to(device)
        action_batch = actions[start : start + batch_size].to(device)
        encoded, predicted, _ = model.trajectory(observation_batch, action_batch)
        squared_error += (predicted - encoded[:, 1:]).pow(2).sum().item()
        element_count += predicted.numel()
    return float(np.sqrt(squared_error / element_count))


@torch.no_grad()
def planning_retrieval(
    model: ReacherLatentWorldModel,
    objective: RCauxLoss | None,
    test_observations: torch.Tensor,
    test_actions: torch.Tensor,
    query_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    true_slots: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate demonstrated action-sequence retrieval on held-out windows.

    Args:
        model: Trained latent world model.
        objective: RC-aux objective for reachability-aware costs, when available.
        test_observations: Held-out observation windows.
        test_actions: Held-out action windows.
        query_indices: Window indices defining starts and goals.
        candidate_indices: Distractor window indices per query.
        true_slots: Candidate positions replaced with each demonstrated sequence.
        device: Evaluation device.

    Returns:
        Top-one, top-five, and mean-rank metrics for ordinary and optional
        reachability-aware planning costs.
    """
    model.eval()
    query_count, candidate_count = candidate_indices.shape
    candidate_actions = test_actions[candidate_indices].clone()
    candidate_actions[torch.arange(query_count), true_slots] = test_actions[
        query_indices
    ]
    starts = test_observations[query_indices, 0]
    goals = test_observations[query_indices, -1]

    flat_actions = candidate_actions.reshape(
        query_count * candidate_count,
        candidate_actions.shape[-2],
        candidate_actions.shape[-1],
    ).to(device)
    flat_starts = (
        starts[:, None]
        .expand(-1, candidate_count, -1)
        .reshape(query_count * candidate_count, -1)
        .to(device)
    )
    flat_goals = (
        goals[:, None]
        .expand(-1, candidate_count, -1)
        .reshape(query_count * candidate_count, -1)
        .to(device)
    )
    initial_latent = model.encode(flat_starts)
    goal_latent = model.encode(flat_goals)
    rollout = model.rollout(initial_latent, flat_actions)
    base_cost = (rollout[:, -1] - goal_latent).pow(2).mean(dim=-1)

    def rank_metrics(cost: torch.Tensor) -> dict[str, float]:
        shaped_cost = cost.reshape(query_count, candidate_count)
        order = shaped_cost.argsort(dim=1)
        target = true_slots.to(device)
        ranks = (order == target[:, None]).nonzero()[:, 1] + 1
        return {
            "top1": (ranks == 1).float().mean().item(),
            "top5": (ranks <= min(5, candidate_count)).float().mean().item(),
            "mean_rank": ranks.float().mean().item(),
        }

    metrics = {
        f"ordinary_{name}": value for name, value in rank_metrics(base_cost).items()
    }
    if objective is not None:
        adjusted_cost, _ = objective.planning_cost(
            base_cost,
            rollout,
            goal_latent,
            reachability_weight=0.85,
        )
        metrics.update(
            {
                f"reachability_{name}": value
                for name, value in rank_metrics(adjusted_cost).items()
            }
        )
    return metrics


@torch.no_grad()
def reachability_accuracy(
    model: ReacherLatentWorldModel,
    objective: RCauxLoss,
    observations: torch.Tensor,
    device: torch.device,
    sample_count: int = 256,
) -> dict[str, float]:
    """Evaluate temporal and cross-trajectory reachability labels.

    Args:
        model: RC-aux-trained latent world model.
        objective: Trained RC-aux objective.
        observations: Held-out observation windows.
        device: Evaluation device.
        sample_count: Number of windows used for the diagnostic.

    Returns:
        Temporal accuracy, batch-negative accuracy, and probability separation.
    """
    sample_indices = torch.linspace(
        0, len(observations) - 1, sample_count, dtype=torch.long
    )
    encoded = model.encode(observations[sample_indices].to(device))
    sources = []
    goals = []
    budgets = []
    labels = []
    for distance in range(1, objective.max_horizon + 1):
        for budget in range(1, objective.max_horizon + 1):
            sources.append(encoded[:, 0])
            goals.append(encoded[:, distance])
            budgets.append(
                torch.full((len(encoded),), budget, dtype=torch.long, device=device)
            )
            labels.append(
                torch.full(
                    (len(encoded),), budget >= distance, dtype=torch.bool, device=device
                )
            )

    source = torch.cat(sources)
    goal = torch.cat(goals)
    budget = torch.cat(budgets)
    label = torch.cat(labels)
    probability = objective.score_reachability(source, goal, budget)
    temporal_accuracy = ((probability >= 0.5) == label).float().mean()

    batch_goal = encoded[:, -1].roll(shifts=sample_count // 4, dims=0)
    batch_budget = torch.full(
        (len(encoded),), objective.max_horizon, dtype=torch.long, device=device
    )
    batch_probability = objective.score_reachability(
        encoded[:, 0], batch_goal, batch_budget
    )
    return {
        "temporal_accuracy": temporal_accuracy.item(),
        "positive_probability": probability[label].mean().item(),
        "temporal_negative_probability": probability[~label].mean().item(),
        "batch_negative_accuracy": (batch_probability < 0.5).float().mean().item(),
    }


def run_seed(
    seed: int,
    train_observations: torch.Tensor,
    train_actions: torch.Tensor,
    test_observations: torch.Tensor,
    test_actions: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Run one matched training and evaluation seed.

    Args:
        seed: Random seed.
        train_observations: Training windows.
        train_actions: Training action windows.
        test_observations: Held-out windows.
        test_actions: Held-out action windows.
        args: Parsed command-line options.
        device: Training and evaluation device.

    Returns:
        Per-seed losses, prediction metrics, planning metrics, and diagnostics.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    template = ReacherLatentWorldModel(
        observation_dim=train_observations.shape[-1],
        action_dim=train_actions.shape[-1],
    )
    initial_state = copy.deepcopy(template.state_dict())
    batch_schedule = torch.randint(
        len(train_observations),
        (args.steps, args.batch_size),
        generator=generator,
    )
    baseline, _, baseline_loss = train_world_model(
        initial_state,
        train_observations,
        train_actions,
        batch_schedule,
        device,
        rcaux=False,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
    )
    rcaux_model, objective, rcaux_loss = train_world_model(
        initial_state,
        train_observations,
        train_actions,
        batch_schedule,
        device,
        rcaux=True,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
    )
    if objective is None:
        raise RuntimeError("RC-aux training did not create an objective")

    query_indices = torch.randperm(len(test_observations), generator=generator)[
        : args.queries
    ]
    candidate_indices = torch.randint(
        len(test_observations),
        (len(query_indices), args.candidates),
        generator=generator,
    )
    true_slots = torch.randint(
        args.candidates, (len(query_indices),), generator=generator
    )
    baseline_planning = planning_retrieval(
        baseline,
        None,
        test_observations,
        test_actions,
        query_indices,
        candidate_indices,
        true_slots,
        device,
    )
    rcaux_planning = planning_retrieval(
        rcaux_model,
        objective,
        test_observations,
        test_actions,
        query_indices,
        candidate_indices,
        true_slots,
        device,
    )
    return {
        "seed": seed,
        "baseline_final_loss": baseline_loss,
        "rcaux_final_loss": rcaux_loss,
        "baseline_prediction_rmse": prediction_rmse(
            baseline, test_observations, test_actions, device
        ),
        "rcaux_prediction_rmse": prediction_rmse(
            rcaux_model, test_observations, test_actions, device
        ),
        "lewm_cont": baseline_planning,
        "rcaux_ordinary": {
            name: value
            for name, value in rcaux_planning.items()
            if name.startswith("ordinary_")
        },
        "rcaux_reachability": {
            name: value
            for name, value in rcaux_planning.items()
            if name.startswith("reachability_")
        },
        "reachability_diagnostic": reachability_accuracy(
            rcaux_model, objective, test_observations, device
        ),
    }


def aggregate(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Aggregate the three planning rows over seeds.

    Args:
        results: Per-seed experiment dictionaries.

    Returns:
        Mean and standard deviation for top-one planning accuracy per method.
    """
    rows = {
        "lewm_cont": "ordinary_top1",
        "rcaux_ordinary": "ordinary_top1",
        "rcaux_reachability": "reachability_top1",
    }
    summary = {}
    for row, metric in rows.items():
        values = np.asarray(
            [result[row][metric] for result in results], dtype=np.float64
        )
        summary[row] = {
            "top1_mean": float(values.mean()),
            "top1_std": float(values.std(ddof=0)),
        }
    return summary


def main() -> None:
    """Run the matched-budget Reacher experiment and save structured results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=16)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    windows = load_reacher_windows(args.dataset, args.horizon, args.train_episodes)
    train_observations, train_actions, test_observations, test_actions = windows
    results = []
    for seed in args.seeds:
        result = run_seed(
            seed,
            train_observations,
            train_actions,
            test_observations,
            test_actions,
            args,
            device,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    payload = {
        "protocol": {
            "dataset": str(args.dataset),
            "train_episodes": args.train_episodes,
            "test_episodes": 20 - args.train_episodes,
            "horizon": args.horizon,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "queries": args.queries,
            "candidates": args.candidates,
            "seeds": args.seeds,
            "device": str(device),
        },
        "results": results,
        "summary": aggregate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
