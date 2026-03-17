"""Tests for optimizer utility functions including exclude_bias_norm."""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.optim.utils import (
    create_optimizer,
    is_bias_or_norm_param,
    split_params_for_weight_decay,
)


class SimpleModel(nn.Module):
    """Simple model for testing parameter splitting."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        return self.linear2(x)


class TransformerLikeModel(nn.Module):
    """Model with LayerNorm for testing normalization parameter detection."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(10, 32)
        self.layer_norm1 = nn.LayerNorm(32)
        self.attention = nn.Linear(32, 32)
        self.layer_norm2 = nn.LayerNorm(32)
        self.output = nn.Linear(32, 5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.layer_norm2(x)
        return self.output(x)


@pytest.mark.unit
class TestIsBiasOrNormParam:
    """Tests for is_bias_or_norm_param function."""

    def test_bias_parameter_detected(self):
        """Test that bias parameters are correctly identified."""
        param = torch.zeros(10)
        assert is_bias_or_norm_param("layer.bias", param) is True
        assert is_bias_or_norm_param("encoder.fc.bias", param) is True
        assert is_bias_or_norm_param("bias", param) is True

    def test_weight_parameter_not_detected(self):
        """Test that weight parameters are not flagged as bias/norm."""
        param = torch.zeros(10, 10)
        assert is_bias_or_norm_param("layer.weight", param) is False
        assert is_bias_or_norm_param("encoder.fc.weight", param) is False

    def test_norm_layer_parameters_detected(self):
        """Test that normalization layer parameters are detected."""
        param = torch.zeros(10)
        # LayerNorm
        assert is_bias_or_norm_param("layer_norm.weight", param) is True
        assert is_bias_or_norm_param("layer_norm.bias", param) is True
        # BatchNorm
        assert is_bias_or_norm_param("bn.weight", param) is False  # bn without 'norm'
        assert is_bias_or_norm_param("batchnorm.weight", param) is True
        assert is_bias_or_norm_param("batch_norm.weight", param) is True
        # GroupNorm
        assert is_bias_or_norm_param("group_norm.weight", param) is True

    def test_mixed_cases(self):
        """Test various edge cases."""
        param = torch.zeros(10)
        # Not bias or norm
        assert is_bias_or_norm_param("conv.weight", param) is False
        assert is_bias_or_norm_param("linear.weight", param) is False
        # RMSNorm (used in some modern architectures)
        assert is_bias_or_norm_param("rmsnorm.weight", param) is True


@pytest.mark.unit
class TestSplitParamsForWeightDecay:
    """Tests for split_params_for_weight_decay function."""

    def test_basic_splitting(self):
        """Test that parameters are correctly split into groups."""
        model = SimpleModel()
        weight_decay = 0.01

        param_groups = split_params_for_weight_decay(
            model.named_parameters(), weight_decay
        )

        # Should have 2 groups (regular + bias/norm)
        assert len(param_groups) == 2

        # Regular params should have weight_decay
        regular_group = param_groups[0]
        assert regular_group["weight_decay"] == weight_decay

        # Bias/norm params should have weight_decay=0
        bias_norm_group = param_groups[1]
        assert bias_norm_group["weight_decay"] == 0.0

    def test_param_count(self):
        """Test that all parameters are accounted for."""
        model = SimpleModel()
        total_params = sum(1 for p in model.parameters() if p.requires_grad)

        param_groups = split_params_for_weight_decay(model.named_parameters(), 0.01)

        split_params = sum(len(g["params"]) for g in param_groups)
        assert split_params == total_params

    def test_transformer_model_splitting(self):
        """Test splitting on a transformer-like model with LayerNorm."""
        model = TransformerLikeModel()
        param_groups = split_params_for_weight_decay(model.named_parameters(), 0.01)

        # Check that LayerNorm params are in the no-decay group
        bias_norm_params = param_groups[1]["params"]
        # LayerNorm has weight and bias, plus linear biases
        assert len(bias_norm_params) > 0

    def test_frozen_params_excluded(self):
        """Test that frozen parameters are not included in any group."""
        model = SimpleModel()
        # Freeze linear1
        for param in model.linear1.parameters():
            param.requires_grad = False

        param_groups = split_params_for_weight_decay(model.named_parameters(), 0.01)

        split_params = sum(len(g["params"]) for g in param_groups)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert split_params == total_trainable


@pytest.mark.unit
class TestCreateOptimizerWithExcludeBiasNorm:
    """Tests for create_optimizer with exclude_bias_norm flag."""

    def test_exclude_bias_norm_creates_param_groups(self):
        """Test that exclude_bias_norm creates separate parameter groups."""
        model = SimpleModel()
        params = list(model.parameters())
        named_params = list(model.named_parameters())

        opt = create_optimizer(
            params,
            {
                "type": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.01,
                "exclude_bias_norm": True,
            },
            named_params=named_params,
        )

        # Should have 2 param groups
        assert len(opt.param_groups) == 2
        # First group (regular) should have weight_decay
        assert opt.param_groups[0]["weight_decay"] == 0.01
        # Second group (bias/norm) should have weight_decay=0
        assert opt.param_groups[1]["weight_decay"] == 0.0

    def test_exclude_bias_norm_without_named_params_raises(self):
        """Test that exclude_bias_norm requires named_params."""
        model = SimpleModel()
        params = list(model.parameters())

        with pytest.raises(ValueError, match="named_params"):
            create_optimizer(
                params,
                {
                    "type": "AdamW",
                    "lr": 1e-3,
                    "weight_decay": 0.01,
                    "exclude_bias_norm": True,
                },
            )

    def test_without_exclude_bias_norm(self):
        """Test that without exclude_bias_norm, all params have same weight_decay."""
        model = SimpleModel()
        params = list(model.parameters())

        opt = create_optimizer(
            params,
            {
                "type": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.01,
            },
        )

        # Should have 1 param group
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["weight_decay"] == 0.01

    def test_exclude_bias_norm_with_sgd(self):
        """Test exclude_bias_norm with SGD optimizer."""
        model = SimpleModel()
        params = list(model.parameters())
        named_params = list(model.named_parameters())

        opt = create_optimizer(
            params,
            {
                "type": "SGD",
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "exclude_bias_norm": True,
            },
            named_params=named_params,
        )

        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["weight_decay"] == 0.0001
        assert opt.param_groups[1]["weight_decay"] == 0.0

    def test_exclude_bias_norm_preserves_other_hyperparams(self):
        """Test that other hyperparameters are preserved with exclude_bias_norm."""
        model = SimpleModel()
        params = list(model.parameters())
        named_params = list(model.named_parameters())

        opt = create_optimizer(
            params,
            {
                "type": "AdamW",
                "lr": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.05,
                "exclude_bias_norm": True,
            },
            named_params=named_params,
        )

        # Both groups should have the same lr, betas, eps
        for group in opt.param_groups:
            assert group["lr"] == 0.01
            assert group["betas"] == (0.9, 0.999)
            assert group["eps"] == 1e-8

    def test_exclude_bias_norm_with_zero_weight_decay(self):
        """Test exclude_bias_norm when weight_decay is 0 (edge case)."""
        model = SimpleModel()
        params = list(model.parameters())
        named_params = list(model.named_parameters())

        opt = create_optimizer(
            params,
            {
                "type": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.0,  # Already 0
                "exclude_bias_norm": True,
            },
            named_params=named_params,
        )

        # Should still create 2 groups (but both with wd=0)
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["weight_decay"] == 0.0
        assert opt.param_groups[1]["weight_decay"] == 0.0
