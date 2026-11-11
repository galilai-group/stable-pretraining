"""Unit tests for the RC-aux objective and forward function."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from stable_pretraining import forward as forward_module
from stable_pretraining.losses import (
    BudgetConditionedReachabilityHead,
    RCauxLoss,
)


@pytest.mark.unit
class TestBudgetConditionedReachabilityHead:
    """Tests for directed, horizon-conditioned reachability scores."""

    def test_output_shape_and_budget_validation(self) -> None:
        head = BudgetConditionedReachabilityHead(
            embedding_dim=4, max_horizon=3, hidden_dim=8, horizon_dim=2
        )
        source = torch.randn(2, 5, 4)
        goal = torch.randn(2, 5, 4)
        budget = torch.ones(2, 5, dtype=torch.long)

        assert head(source, goal, budget).shape == (2, 5)

        with pytest.raises(ValueError, match="between 1 and 3"):
            head(source, goal, torch.zeros_like(budget))

    def test_direction_changes_head_input(self) -> None:
        torch.manual_seed(0)
        head = BudgetConditionedReachabilityHead(
            embedding_dim=3, max_horizon=2, hidden_dim=8, horizon_dim=2
        )
        source = torch.tensor([[1.0, 0.0, -1.0]])
        goal = torch.tensor([[0.0, 2.0, 1.0]])
        budget = torch.ones(1, dtype=torch.long)

        forward_score = head(source, goal, budget)
        reverse_score = head(goal, source, budget)

        assert not torch.allclose(forward_score, reverse_score)


@pytest.mark.unit
class TestRCauxLoss:
    """Tests for loss construction, gradients, and planning scores."""

    def test_weighted_multi_horizon_prediction_loss(self) -> None:
        objective = RCauxLoss(
            embedding_dim=2,
            max_horizon=3,
            hidden_dim=8,
            horizon_dim=2,
            horizon_weights=(1.0, 3.0),
        )
        encoded = torch.zeros(1, 4, 2)
        predicted = torch.stack((torch.ones(1, 2), torch.full((1, 2), 3.0)), dim=1)

        result = objective(encoded, predicted, context_index=2)

        torch.testing.assert_close(result["prediction_loss"], torch.tensor(7.0))

    def test_pair_counts_include_temporal_and_batch_negatives(self) -> None:
        objective = RCauxLoss(
            embedding_dim=2, max_horizon=3, hidden_dim=8, horizon_dim=2
        )
        encoded = torch.randn(2, 4, 2)
        predicted = torch.randn(2, 2, 2)

        result = objective(encoded, predicted, context_index=2)

        assert result["encoded_pair_count"].item() == 64
        assert result["predicted_pair_count"].item() == 12

    def test_reachability_branch_stops_predicted_latent_gradient(self) -> None:
        objective = RCauxLoss(
            embedding_dim=2,
            max_horizon=3,
            hidden_dim=8,
            horizon_dim=2,
            prediction_weight=0.0,
            reachability_weight=1.0,
            predicted_pair_weight=1.0,
        )
        encoded = torch.randn(2, 4, 2, requires_grad=True)
        predicted = torch.randn(2, 2, 2, requires_grad=True)

        result = objective(encoded, predicted, context_index=2)
        result["loss"].backward()

        torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))
        assert encoded.grad is not None
        assert any(
            parameter.grad is not None
            for parameter in objective.reachability_head.parameters()
        )

    def test_planning_cost_uses_best_reachability_score(self) -> None:
        class ConstantHead(nn.Module):
            def forward(
                self,
                source: torch.Tensor,
                goal: torch.Tensor,
                budget: torch.Tensor,
            ) -> torch.Tensor:
                return source.new_zeros(source.shape[:-1])

        objective = RCauxLoss(
            embedding_dim=2, max_horizon=3, hidden_dim=8, horizon_dim=2
        )
        objective.reachability_head = ConstantHead()
        base_cost = torch.tensor([2.0, 4.0])
        rollout = torch.randn(2, 3, 2)
        goal = torch.randn(2, 2)

        cost, score = objective.planning_cost(
            base_cost,
            rollout,
            goal,
            reachability_weight=0.8,
        )

        torch.testing.assert_close(score, torch.full((2,), 0.5))
        torch.testing.assert_close(cost, base_cost * 0.6)


@pytest.mark.unit
class TestRCauxForward:
    """Tests for the composable stable-pretraining forward function."""

    def test_training_output_and_logging(self) -> None:
        encoded = torch.randn(2, 4, 3)
        predicted = torch.randn(2, 2, 3)
        module = Mock()
        module.training = True
        module.world_model = Mock(
            return_value={"embedding": encoded, "prediction": predicted}
        )
        module.rcaux_loss = RCauxLoss(
            embedding_dim=3, max_horizon=3, hidden_dim=8, horizon_dim=2
        )
        module.context_index = 2
        module.log = Mock()
        batch = {"observation": torch.randn(2, 4, 5), "label": torch.arange(2)}

        result = forward_module.rcaux(module, batch, "train")

        assert result["embedding"] is encoded
        assert result["prediction"] is predicted
        assert result["loss"].ndim == 0
        torch.testing.assert_close(result["label"], batch["label"])
        assert module.log.call_count == 10

    def test_evaluation_only_requires_embeddings(self) -> None:
        encoded = torch.randn(2, 4, 3)
        module = Mock()
        module.training = False
        module.world_model = Mock(return_value={"embedding": encoded})

        result = forward_module.rcaux(module, {}, "val")

        assert result == {"embedding": encoded}
