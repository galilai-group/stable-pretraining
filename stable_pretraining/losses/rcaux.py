"""Reachability-calibrated auxiliary objectives for latent world models."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

_ReachabilityPairs = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class BudgetConditionedReachabilityHead(nn.Module):
    """Predict whether one latent state can reach another within a step budget.

    The score is directed because the source and goal embeddings are concatenated
    in order. A learned horizon embedding makes the prediction conditional on the
    number of actions available to the agent.

    Args:
        embedding_dim: Size of each latent state embedding.
        max_horizon: Largest supported planning budget.
        hidden_dim: Width of the multilayer perceptron.
        horizon_dim: Size of the learned horizon embedding.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_horizon: int,
        hidden_dim: int = 512,
        horizon_dim: int = 32,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive")

        self.embedding_dim = embedding_dim
        self.max_horizon = max_horizon
        self.horizon_embedding = nn.Embedding(max_horizon + 1, horizon_dim)
        input_dim = 4 * embedding_dim + horizon_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        """Return reachability logits for source-goal-budget tuples.

        Args:
            source: Source latents with shape ``(..., embedding_dim)``.
            goal: Goal latents with the same shape as ``source``.
            budget: Integer step budgets with shape ``source.shape[:-1]``.

        Returns:
            Reachability logits with shape ``source.shape[:-1]``.

        Raises:
            ValueError: If tensor shapes or budgets are invalid.
        """
        if source.shape != goal.shape:
            raise ValueError("source and goal must have identical shapes")
        if source.ndim < 2 or source.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"source must end in embedding_dim={self.embedding_dim}, "
                f"got shape {tuple(source.shape)}"
            )
        if budget.shape != source.shape[:-1]:
            raise ValueError("budget must match the leading source dimensions")
        if budget.is_floating_point():
            raise ValueError("budget must use an integer dtype")
        if torch.any((budget < 1) | (budget > self.max_horizon)):
            raise ValueError(f"budget values must be between 1 and {self.max_horizon}")

        horizon = self.horizon_embedding(budget.to(dtype=torch.long))
        features = torch.cat(
            (source, goal, torch.abs(source - goal), source * goal, horizon), dim=-1
        )
        return self.network(features).squeeze(-1)


class RCauxLoss(nn.Module):
    """RC-aux objective for learning predictive and plannable latent dynamics.

    The objective combines weighted multi-horizon latent prediction with a
    budget-conditioned reachability classifier. Encoded trajectory pairs provide
    positive examples, too-short budgets provide temporal hard negatives, and
    goals from another trajectory in the batch provide batch negatives. Predicted
    source latents are detached in the reachability branch, so that branch trains
    the geometry readout without creating a shortcut through the rollout model.

    Predictions are aligned with ``encoded[:, context_index:context_index + K]``.
    When ``context_index`` is omitted, the final ``K`` encoded states are used.

    Args:
        embedding_dim: Size of each latent state embedding.
        max_horizon: Largest temporal distance and planning budget to supervise.
        hidden_dim: Width of the reachability multilayer perceptron.
        horizon_dim: Size of the learned horizon embedding.
        prediction_weight: Weight of the multi-horizon prediction objective.
        reachability_weight: Weight of the combined reachability objective.
        predicted_pair_weight: Relative weight of pairs with predicted sources.
        temporal_negative_weight: Weight of too-short-budget negatives.
        batch_negative_weight: Weight of cross-trajectory negatives.
        regularization_weight: Weight of an optional world-model regularizer.
        horizon_weights: Optional positive weights for prediction horizons.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_horizon: int = 5,
        hidden_dim: int = 512,
        horizon_dim: int = 32,
        prediction_weight: float = 1.0,
        reachability_weight: float = 0.35,
        predicted_pair_weight: float = 0.5,
        temporal_negative_weight: float = 1.0,
        batch_negative_weight: float = 1.0,
        regularization_weight: float = 0.0,
        horizon_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive")
        for name, value in (
            ("prediction_weight", prediction_weight),
            ("reachability_weight", reachability_weight),
            ("predicted_pair_weight", predicted_pair_weight),
            ("temporal_negative_weight", temporal_negative_weight),
            ("batch_negative_weight", batch_negative_weight),
            ("regularization_weight", regularization_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        weights = None
        if horizon_weights is not None:
            weights = torch.as_tensor(horizon_weights, dtype=torch.float32)
            if weights.ndim != 1 or len(weights) == 0:
                raise ValueError("horizon_weights must be a non-empty sequence")
            if torch.any(weights <= 0):
                raise ValueError("horizon_weights must be positive")

        self.max_horizon = max_horizon
        self.prediction_weight = prediction_weight
        self.reachability_weight = reachability_weight
        self.predicted_pair_weight = predicted_pair_weight
        self.temporal_negative_weight = temporal_negative_weight
        self.batch_negative_weight = batch_negative_weight
        self.regularization_weight = regularization_weight
        self.register_buffer("horizon_weights", weights)
        self.reachability_head = BudgetConditionedReachabilityHead(
            embedding_dim=embedding_dim,
            max_horizon=max_horizon,
            hidden_dim=hidden_dim,
            horizon_dim=horizon_dim,
        )

    @staticmethod
    def _pair_tensors(
        source: torch.Tensor,
        goal: torch.Tensor,
        distance: int,
        max_horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        budgets = torch.arange(
            1, max_horizon + 1, device=source.device, dtype=torch.long
        )
        shape = (1,) * (source.ndim - 1) + (max_horizon,)
        budgets = budgets.view(shape).expand(source.shape[:-1] + (max_horizon,))
        expanded_source = source.unsqueeze(-2).expand(
            source.shape[:-1] + (max_horizon, source.shape[-1])
        )
        expanded_goal = goal.unsqueeze(-2).expand_as(expanded_source)
        labels = budgets >= distance
        return expanded_source, expanded_goal, budgets, labels

    def _trajectory_pairs(
        self, encoded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sources = []
        goals = []
        budgets = []
        labels = []
        is_batch_negative = []
        batch_size, sequence_length, _ = encoded.shape
        usable_horizon = min(self.max_horizon, sequence_length - 1)

        for distance in range(1, usable_horizon + 1):
            source = encoded[:, :-distance]
            goal = encoded[:, distance:]
            src, dst, budget, label = self._pair_tensors(
                source, goal, distance, self.max_horizon
            )
            sources.append(src.reshape(-1, encoded.shape[-1]))
            goals.append(dst.reshape(-1, encoded.shape[-1]))
            budgets.append(budget.reshape(-1))
            labels.append(label.reshape(-1))
            is_batch_negative.append(
                torch.zeros(label.numel(), dtype=torch.bool, device=encoded.device)
            )

            if batch_size > 1:
                negative_goal = goal.roll(shifts=1, dims=0)
                src, dst, budget, _ = self._pair_tensors(
                    source, negative_goal, distance, self.max_horizon
                )
                positive_budget = budget >= distance
                sources.append(src[positive_budget])
                goals.append(dst[positive_budget])
                selected_budget = budget[positive_budget]
                budgets.append(selected_budget)
                labels.append(torch.zeros_like(selected_budget, dtype=torch.bool))
                is_batch_negative.append(
                    torch.ones_like(selected_budget, dtype=torch.bool)
                )

        return (
            torch.cat(sources),
            torch.cat(goals),
            torch.cat(budgets),
            torch.cat(labels),
            torch.cat(is_batch_negative),
        )

    def _predicted_source_pairs(
        self,
        encoded: torch.Tensor,
        predicted: torch.Tensor,
        context_index: int,
    ) -> _ReachabilityPairs | None:
        sources = []
        goals = []
        budgets = []
        labels = []
        is_batch_negative = []
        batch_size, rollout_length, embedding_dim = predicted.shape

        for source_offset in range(rollout_length - 1):
            source = predicted[:, source_offset].detach()
            remaining = min(self.max_horizon, rollout_length - source_offset - 1)
            for distance in range(1, remaining + 1):
                goal_index = context_index + source_offset + distance
                goal = encoded[:, goal_index]
                src, dst, budget, label = self._pair_tensors(
                    source, goal, distance, self.max_horizon
                )
                sources.append(src.reshape(-1, embedding_dim))
                goals.append(dst.reshape(-1, embedding_dim))
                budgets.append(budget.reshape(-1))
                labels.append(label.reshape(-1))
                is_batch_negative.append(
                    torch.zeros(label.numel(), dtype=torch.bool, device=encoded.device)
                )

                if batch_size > 1:
                    negative_goal = goal.roll(shifts=1, dims=0)
                    src, dst, budget, _ = self._pair_tensors(
                        source, negative_goal, distance, self.max_horizon
                    )
                    positive_budget = budget >= distance
                    sources.append(src[positive_budget])
                    goals.append(dst[positive_budget])
                    selected_budget = budget[positive_budget]
                    budgets.append(selected_budget)
                    labels.append(torch.zeros_like(selected_budget, dtype=torch.bool))
                    is_batch_negative.append(
                        torch.ones_like(selected_budget, dtype=torch.bool)
                    )

        if not sources:
            return None
        return (
            torch.cat(sources),
            torch.cat(goals),
            torch.cat(budgets),
            torch.cat(labels),
            torch.cat(is_batch_negative),
        )

    def _balanced_reachability_loss(
        self,
        pairs: _ReachabilityPairs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source, goal, budget, labels, is_batch_negative = pairs
        logits = self.reachability_head(source, goal, budget)
        labels_float = labels.to(dtype=logits.dtype)
        per_pair = F.binary_cross_entropy_with_logits(
            logits, labels_float, reduction="none"
        )
        positive = labels
        temporal_negative = ~labels & ~is_batch_negative
        batch_negative = is_batch_negative

        positive_loss = per_pair[positive].mean()
        negative_total = logits.new_zeros(())
        negative_weight = 0.0
        if torch.any(temporal_negative):
            negative_total = (
                negative_total
                + self.temporal_negative_weight * per_pair[temporal_negative].mean()
            )
            negative_weight += self.temporal_negative_weight
        if torch.any(batch_negative):
            negative_total = (
                negative_total
                + self.batch_negative_weight * per_pair[batch_negative].mean()
            )
            negative_weight += self.batch_negative_weight
        if negative_weight > 0:
            loss = 0.5 * (positive_loss + negative_total / negative_weight)
        else:
            loss = positive_loss

        probabilities = logits.sigmoid()
        positive_probability = probabilities[positive].mean()
        negative_probability = probabilities[~labels].mean()
        return (
            loss,
            positive_probability,
            negative_probability,
            labels_float.new_tensor(labels.numel()),
        )

    def forward(
        self,
        encoded: torch.Tensor,
        predicted: torch.Tensor,
        context_index: int | None = None,
        regularization_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the RC-aux objective and its diagnostic terms.

        Args:
            encoded: Encoded trajectories with shape ``(batch, time, dim)``.
            predicted: Open-loop predictions with shape ``(batch, horizon, dim)``.
            context_index: Index of the first encoded target aligned to prediction
                horizon one. Defaults to ``time - horizon``.
            regularization_loss: Optional scalar regularizer from the world model.

        Returns:
            Dictionary containing total loss, component losses, reachability
            probabilities, and supervised pair counts.

        Raises:
            ValueError: If sequence shapes or alignment are invalid.
        """
        if encoded.ndim != 3 or predicted.ndim != 3:
            raise ValueError("encoded and predicted must have shape (batch, time, dim)")
        if (
            encoded.shape[0] != predicted.shape[0]
            or encoded.shape[2] != predicted.shape[2]
        ):
            raise ValueError(
                "encoded and predicted batch and embedding dimensions must match"
            )
        if predicted.shape[1] == 0:
            raise ValueError("predicted must contain at least one horizon")
        if encoded.shape[1] < 2:
            raise ValueError("encoded must contain at least two states")

        if context_index is None:
            context_index = encoded.shape[1] - predicted.shape[1]
        if context_index < 1 or context_index + predicted.shape[1] > encoded.shape[1]:
            raise ValueError(
                "context_index does not align predicted with encoded targets"
            )

        target = encoded[:, context_index : context_index + predicted.shape[1]]
        horizon_error = (predicted - target).pow(2).mean(dim=(0, 2))
        if self.horizon_weights is None:
            weights = torch.ones_like(horizon_error)
        else:
            if len(self.horizon_weights) < predicted.shape[1]:
                raise ValueError("horizon_weights is shorter than the rollout horizon")
            weights = self.horizon_weights[: predicted.shape[1]].to(horizon_error)
        prediction_loss = (horizon_error * weights).sum() / weights.sum()

        encoded_result = self._balanced_reachability_loss(
            self._trajectory_pairs(encoded)
        )
        (
            encoded_reachability_loss,
            positive_probability,
            negative_probability,
            encoded_count,
        ) = encoded_result
        predicted_pairs = self._predicted_source_pairs(
            encoded, predicted, context_index
        )
        if predicted_pairs is None:
            predicted_reachability_loss = encoded.new_zeros(())
            predicted_count = encoded.new_zeros(())
        else:
            predicted_result = self._balanced_reachability_loss(predicted_pairs)
            predicted_reachability_loss = predicted_result[0]
            predicted_count = predicted_result[3]

        reachability_loss = (
            encoded_reachability_loss
            + self.predicted_pair_weight * predicted_reachability_loss
        )
        if regularization_loss is None:
            regularization_loss = encoded.new_zeros(())
        if regularization_loss.ndim != 0:
            regularization_loss = regularization_loss.mean()
        loss = (
            self.prediction_weight * prediction_loss
            + self.reachability_weight * reachability_loss
            + self.regularization_weight * regularization_loss
        )

        return {
            "loss": loss,
            "prediction_loss": prediction_loss,
            "reachability_loss": reachability_loss,
            "encoded_reachability_loss": encoded_reachability_loss,
            "predicted_reachability_loss": predicted_reachability_loss,
            "regularization_loss": regularization_loss,
            "positive_reachability": positive_probability,
            "negative_reachability": negative_probability,
            "encoded_pair_count": encoded_count,
            "predicted_pair_count": predicted_count,
        }

    def score_reachability(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        """Return reachability probabilities for planning or evaluation.

        Args:
            source: Source latent embeddings.
            goal: Goal latent embeddings with the same shape as ``source``.
            budget: Integer step budgets for each source-goal pair.

        Returns:
            Reachability probabilities in the closed interval zero to one.
        """
        return self.reachability_head(source, goal, budget).sigmoid()

    def planning_cost(
        self,
        base_cost: torch.Tensor,
        predicted_rollout: torch.Tensor,
        goal: torch.Tensor,
        reachability_weight: float = 0.85,
        floor: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reweight a planner cost using rollout-to-goal reachability.

        Args:
            base_cost: Base planner cost with shape ``predicted_rollout.shape[:-2]``.
            predicted_rollout: Candidate latent rollouts with shape ``(..., K, D)``.
            goal: Goal latents with shape ``(..., D)``.
            reachability_weight: Strength of reachability reweighting.
            floor: Minimum multiplicative cost factor.

        Returns:
            A tuple containing the reweighted cost and maximum reachability score.

        Raises:
            ValueError: If shapes, horizon, or scalar parameters are invalid.
        """
        if predicted_rollout.ndim < 3:
            raise ValueError("predicted_rollout must have shape (..., horizon, dim)")
        if goal.shape != predicted_rollout.shape[:-2] + predicted_rollout.shape[-1:]:
            raise ValueError(
                "goal shape must match rollout leading and embedding dimensions"
            )
        if base_cost.shape != predicted_rollout.shape[:-2]:
            raise ValueError("base_cost shape must match rollout leading dimensions")
        if predicted_rollout.shape[-2] > self.max_horizon:
            raise ValueError("rollout horizon exceeds max_horizon")
        if not 0 <= reachability_weight <= 1:
            raise ValueError("reachability_weight must be between zero and one")
        if not 0 < floor <= 1:
            raise ValueError("floor must be in the interval (0, 1]")

        horizon = predicted_rollout.shape[-2]
        expanded_goal = goal.unsqueeze(-2).expand_as(predicted_rollout)
        budgets = torch.arange(
            horizon, 0, -1, device=predicted_rollout.device, dtype=torch.long
        )
        budgets = budgets.view((1,) * (predicted_rollout.ndim - 2) + (horizon,))
        budgets = budgets.expand(predicted_rollout.shape[:-1])
        probabilities = self.score_reachability(
            predicted_rollout, expanded_goal, budgets
        )
        max_probability = probabilities.amax(dim=-1)
        multiplier = torch.clamp(1 - reachability_weight * max_probability, min=floor)
        return base_cost * multiplier, max_probability
