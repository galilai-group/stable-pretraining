"""Unit tests for the RC-aux objective and forward function."""

import math

import lightning as pl
from torch.utils.data import DataLoader

import stable_pretraining as spt

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
def test_planning_cost_uses_remaining_budget_and_excludes_endpoint() -> None:
    class BudgetAndStateHead(nn.Module):
        def forward(self, source, goal, budget):
            return source[..., 0] + budget.to(source) - 4.0

    objective = RCauxLoss(embedding_dim=1, max_horizon=3, hidden_dim=4, horizon_dim=2)
    objective.reachability_head = BudgetAndStateHead()
    base = torch.tensor([[2.0, 4.0]])
    rollout = torch.tensor([[[[1.0], [3.0], [100.0]], [[0.0], [1.0], [100.0]]]])
    goal = torch.zeros(1, 2, 1)
    cost, score = objective.planning_cost(base, rollout, goal, reachability_weight=0.8)
    # The two intermediate logits are (-1, 0) and (-2, -2); the endpoint is ignored.
    expected = torch.tensor([[0.5, 1.0 / (1.0 + math.exp(2.0))]])
    torch.testing.assert_close(score, expected)
    torch.testing.assert_close(cost, base * (1 - 0.8 * expected))


@pytest.mark.unit
def test_single_step_planning_has_no_intermediate_discount() -> None:
    objective = RCauxLoss(embedding_dim=1, max_horizon=1, hidden_dim=4, horizon_dim=2)
    base = torch.tensor([2.0, 4.0])
    cost, score = objective.planning_cost(base, torch.ones(2, 1, 1), torch.zeros(2, 1))
    torch.testing.assert_close(cost, base)
    torch.testing.assert_close(score, torch.zeros_like(base))


@pytest.mark.unit
def test_retrieval_has_one_correct_slot_and_preserves_other_candidates() -> None:
    from examples.rcaux_reacher_offline import sample_retrieval_candidates

    queries = torch.tensor([0, 1, 2])
    generator = torch.Generator().manual_seed(7)
    original = torch.randint(3, (3, 32), generator=generator)
    slots = torch.randint(32, (3,), generator=generator)
    candidates, actual_slots = sample_retrieval_candidates(
        queries, 3, 32, torch.Generator().manual_seed(7)
    )
    torch.testing.assert_close(actual_slots, slots)
    assert ((candidates == queries[:, None]).sum(1) == 1).all()
    torch.testing.assert_close(candidates[torch.arange(3), slots], queries)
    assert ((candidates >= 0) & (candidates < 3)).all()
    unchanged = original != queries[:, None]
    unchanged[torch.arange(3), slots] = False
    torch.testing.assert_close(candidates[unchanged], original[unchanged])
    repeated, repeated_slots = sample_retrieval_candidates(
        queries, 3, 32, torch.Generator().manual_seed(7)
    )
    torch.testing.assert_close(repeated, candidates)
    torch.testing.assert_close(repeated_slots, slots)


class _AnalyticHead(nn.Module):
    """Nonconstant logits with an independently computable scalar expression."""

    def forward(self, source, goal, budget):
        return 0.1 * source[..., 0] + 0.2 * goal[..., 0] + 0.3 * budget.to(source)


@pytest.mark.unit
def test_reachability_matches_hand_enumerated_binary_cross_entropy() -> None:
    """Check labels, budgets, group balancing, and predicted-source alignment."""
    objective = RCauxLoss(
        embedding_dim=1,
        max_horizon=2,
        hidden_dim=4,
        horizon_dim=2,
        temporal_negative_weight=2.0,
        batch_negative_weight=3.0,
        predicted_pair_weight=0.4,
        prediction_weight=0.7,
        reachability_weight=0.6,
        regularization_weight=0.2,
        horizon_weights=(1.0, 3.0),
    ).double()
    objective.reachability_head = _AnalyticHead()
    encoded = torch.tensor(
        [[[0.0], [1.0], [3.0]], [[10.0], [12.0], [15.0]]], dtype=torch.float64
    )
    predicted = torch.tensor([[[0.5], [2.0]], [[11.0], [14.0]]], dtype=torch.float64)

    # Explicit (source, goal, budget) tables for these two length-three paths.
    positives = [
        (0, 1, 1),
        (0, 1, 2),
        (1, 3, 1),
        (1, 3, 2),
        (10, 12, 1),
        (10, 12, 2),
        (12, 15, 1),
        (12, 15, 2),
        (0, 3, 2),
        (10, 15, 2),
    ]
    temporal_negatives = [(0, 3, 1), (10, 15, 1)]
    batch_negatives = [
        (0, 12, 1),
        (0, 12, 2),
        (1, 15, 1),
        (1, 15, 2),
        (10, 1, 1),
        (10, 1, 2),
        (12, 3, 1),
        (12, 3, 2),
        (0, 15, 2),
        (10, 3, 2),
    ]
    predicted_positives = [(0.5, 3, 1), (0.5, 3, 2), (11, 15, 1), (11, 15, 2)]
    predicted_negatives = [(0.5, 15, 1), (0.5, 15, 2), (11, 3, 1), (11, 3, 2)]

    def mean_bce(pairs, label):
        logits = [0.1 * a + 0.2 * b + 0.3 * k for a, b, k in pairs]
        return sum(math.log1p(math.exp(x)) - label * x for x in logits) / len(logits)

    encoded_bce = 0.5 * (
        mean_bce(positives, 1)
        + (2 * mean_bce(temporal_negatives, 0) + 3 * mean_bce(batch_negatives, 0)) / 5
    )
    predicted_bce = 0.5 * (
        mean_bce(predicted_positives, 1) + mean_bce(predicted_negatives, 0)
    )
    expected_reachability = encoded_bce + 0.4 * predicted_bce
    expected_prediction = (0.625 + 3 * 1.0) / 4
    result = objective(
        encoded, predicted, context_index=1, regularization_loss=encoded.new_tensor(2.0)
    )
    assert result["encoded_reachability_loss"].item() == pytest.approx(
        encoded_bce, abs=1e-12
    )
    assert result["predicted_reachability_loss"].item() == pytest.approx(
        predicted_bce, abs=1e-12
    )
    assert result["reachability_loss"].item() == pytest.approx(
        expected_reachability, abs=1e-12
    )
    assert result["loss"].item() == pytest.approx(
        0.7 * expected_prediction + 0.6 * expected_reachability + 0.4, abs=1e-12
    )


class _WorldModel(nn.Module):
    """Trainable encoder and transition for the actual Module interface."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(5, 3)
        self.transition = nn.Linear(3, 3)

    def forward(self, batch):
        encoded = self.encoder(batch["observation"])
        out = {"embedding": encoded}
        if self.training:
            out["prediction"] = self.transition(encoded[:, 1:3])
        return out


@pytest.mark.unit
class TestRCauxForward:
    """Exercise real Modules, submodules, and Lightning metric logging."""

    def test_training_output_and_logging(self, tmp_path) -> None:
        module = spt.Module(
            forward=forward_module.rcaux,
            world_model=_WorldModel(),
            rcaux_loss=RCauxLoss(
                embedding_dim=3, max_horizon=3, hidden_dim=8, horizon_dim=2
            ),
            context_index=2,
            optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
        )
        samples = [
            {"observation": torch.randn(4, 5), "label": torch.tensor(i)}
            for i in range(4)
        ]
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            default_root_dir=str(tmp_path),
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloaders=DataLoader(samples, batch_size=2))
        assert trainer.global_step == 1
        for key in [
            "loss",
            "prediction_loss",
            "reachability_loss",
            "encoded_pair_count",
        ]:
            assert torch.isfinite(trainer.callback_metrics[f"fit/{key}"])

    def test_evaluation_only_requires_embeddings(self) -> None:
        module = spt.Module(forward=forward_module.rcaux, world_model=_WorldModel())
        module.eval()
        batch = {"observation": torch.randn(2, 4, 5), "label": torch.arange(2)}
        result = module(batch, "val")
        torch.testing.assert_close(
            result["embedding"], module.world_model.encoder(batch["observation"])
        )
        torch.testing.assert_close(result["label"], batch["label"])
        assert "loss" not in result and "prediction" not in result

    def test_public_method_uses_existing_forward(self) -> None:
        from stable_pretraining.methods import RCaux

        assert spt.RCaux is RCaux
        module = RCaux(
            _WorldModel(),
            embedding_dim=3,
            max_horizon=3,
            loss_kwargs={"hidden_dim": 8, "horizon_dim": 2},
        )
        assert isinstance(module, spt.Module)
        module.eval()
        batch = {"observation": torch.randn(2, 4, 5)}
        torch.testing.assert_close(
            module(batch, "val")["embedding"],
            forward_module.rcaux(module, batch, "val")["embedding"],
        )
        assert module.rcaux_loss.reachability_weight == 0.35
