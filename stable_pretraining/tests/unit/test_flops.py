import pytest
import torch
import torch.nn as nn

from stable_pretraining.utils.flops import (
    FLOPBudgetExceeded,
    flop_budget,
)


# Apply "unit" marker to all tests in this module
pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_linear():
    """A simple linear layer: 768 -> 768, so 2 * 768 * 768 = 1,179,648 FLOPs per forward."""
    return nn.Linear(768, 768, bias=False)


@pytest.fixture
def mlp():
    """MLP block typical in transformers."""
    return nn.Sequential(
        nn.Linear(768, 3072, bias=False),
        nn.GELU(),
        nn.Linear(3072, 768, bias=False),
    )


@pytest.fixture
def conv_net():
    """Simple conv network."""
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
    )


# ============================================================================
# Tests: Basic FLOP Counting
# ============================================================================


class TestBasicFlopCounting:
    """Tests that FLOP counting works correctly."""

    def test_linear_flops_counted(self, simple_linear):
        """Verify FLOPs are counted for linear layer."""
        x = torch.randn(1, 768)

        with flop_budget(budget=1e12) as counter:
            _ = simple_linear(x)

        expected = 2 * 768 * 768
        assert counter.total_flops == expected

    def test_linear_flops_scale_with_batch(self, simple_linear):
        """Verify FLOPs scale linearly with batch size."""
        with flop_budget(budget=1e12) as counter1:
            _ = simple_linear(torch.randn(1, 768))

        with flop_budget(budget=1e12) as counter2:
            _ = simple_linear(torch.randn(8, 768))

        assert counter2.total_flops == 8 * counter1.total_flops

    def test_mlp_flops_counted(self, mlp):
        """Verify FLOPs are counted for MLP."""
        x = torch.randn(1, 768)

        with flop_budget(budget=1e12) as counter:
            _ = mlp(x)

        assert counter.total_flops >= 2 * 768 * 3072 + 2 * 3072 * 768

    def test_conv_flops_counted(self, conv_net):
        """Verify FLOPs are counted for conv layers."""
        x = torch.randn(1, 3, 32, 32)

        with flop_budget(budget=1e12) as counter:
            _ = conv_net(x)

        assert counter.total_flops > 0

    def test_matmul_flops_counted(self):
        """Verify FLOPs are counted for raw matmul operations."""
        a = torch.randn(64, 128)
        b = torch.randn(128, 256)

        with flop_budget(budget=1e12) as counter:
            _ = torch.matmul(a, b)

        expected = 2 * 64 * 256 * 128
        assert counter.total_flops == expected

    def test_multiple_operations_accumulated(self, simple_linear):
        """Verify FLOPs accumulate across multiple operations."""
        x = torch.randn(1, 768)
        single_forward_flops = 2 * 768 * 768

        with flop_budget(budget=1e12) as counter:
            for _ in range(5):
                _ = simple_linear(x)

        assert counter.total_flops == 5 * single_forward_flops


# ============================================================================
# Tests: Budget Enforcement
# ============================================================================


class TestBudgetEnforcement:
    """Tests that budget is enforced correctly."""

    def test_raises_when_budget_exceeded(self, simple_linear):
        """Verify exception is raised when budget is exceeded."""
        x = torch.randn(1, 768)
        single_forward_flops = 2 * 768 * 768

        budget = int(2.5 * single_forward_flops)

        with pytest.raises(FLOPBudgetExceeded):
            with flop_budget(budget=budget) as _:
                for _ in range(10):
                    _ = simple_linear(x)

    def test_no_raise_when_under_budget(self, simple_linear):
        """Verify no exception when staying under budget."""
        x = torch.randn(1, 768)
        single_forward_flops = 2 * 768 * 768

        budget = int(15 * single_forward_flops)

        with flop_budget(budget=budget) as counter:
            for _ in range(10):
                _ = simple_linear(x)

        assert counter.total_flops == 10 * single_forward_flops

    def test_exact_budget_does_not_raise(self, simple_linear):
        """Verify exact budget usage does not raise."""
        x = torch.randn(1, 768)
        single_forward_flops = 2 * 768 * 768

        budget = single_forward_flops

        with flop_budget(budget=budget) as counter:
            _ = simple_linear(x)

        assert counter.total_flops == budget

    def test_raises_on_first_exceeding_operation(self):
        """Verify exception is raised immediately when budget exceeded."""
        budget = 100

        with pytest.raises(FLOPBudgetExceeded) as exc_info:
            with flop_budget(budget=budget):
                _ = torch.matmul(torch.randn(32, 32), torch.randn(32, 32))

        assert exc_info.value.budget == budget
        assert exc_info.value.current > budget

    def test_exception_contains_correct_info(self, simple_linear):
        """Verify exception contains accurate budget and current values."""
        x = torch.randn(1, 768)
        budget = 1000

        with pytest.raises(FLOPBudgetExceeded) as exc_info:
            with flop_budget(budget=budget):
                _ = simple_linear(x)

        exc = exc_info.value
        assert exc.budget == budget
        assert exc.current == 2 * 768 * 768
        assert len(exc.operation) > 0

    def test_counter_accessible_during_execution(self, simple_linear):
        """Verify counter can be queried during execution."""
        x = torch.randn(1, 768)
        single_forward_flops = 2 * 768 * 768
        budget = int(100 * single_forward_flops)

        flop_history = []

        with flop_budget(budget=budget) as counter:
            for i in range(5):
                _ = simple_linear(x)
                flop_history.append(counter.total_flops)

        assert flop_history == [single_forward_flops * (i + 1) for i in range(5)]


# ============================================================================
# Tests: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_zero_budget_raises_immediately(self):
        """Verify zero budget raises on first operation."""
        with pytest.raises(FLOPBudgetExceeded):
            with flop_budget(budget=0):
                _ = torch.matmul(torch.randn(2, 2), torch.randn(2, 2))

    def test_no_ops_no_flops(self):
        """Verify no FLOPs counted when no operations performed."""
        with flop_budget(budget=1000) as counter:
            x = torch.randn(10, 10)
            _ = x.shape

        assert counter.total_flops == 0

    def test_in_place_operations(self):
        """Verify in-place operations are handled."""
        x = torch.randn(32, 32)
        y = torch.randn(32, 32)

        with flop_budget(budget=1e12) as counter:
            x.add_(y)

        assert counter.total_flops >= 0

    def test_nested_modules(self):
        """Verify nested modules are counted correctly."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(64, 128, bias=False),
                nn.Linear(128, 64, bias=False),
            ),
            nn.Linear(64, 32, bias=False),
        )
        x = torch.randn(1, 64)

        with flop_budget(budget=1e12) as counter:
            _ = model(x)

        expected = (2 * 64 * 128) + (2 * 128 * 64) + (2 * 64 * 32)
        assert counter.total_flops == expected

    def test_large_budget_value(self, simple_linear):
        """Verify large budget values work correctly."""
        x = torch.randn(1, 768)

        with flop_budget(budget=int(1e18)) as counter:
            _ = simple_linear(x)

        assert counter.total_flops == 2 * 768 * 768

    def test_reusable_after_exception(self, simple_linear):
        """Verify we can create new context after exception."""
        x = torch.randn(1, 768)

        with pytest.raises(FLOPBudgetExceeded):
            with flop_budget(budget=100):
                _ = simple_linear(x)

        with flop_budget(budget=1e12) as counter:
            _ = simple_linear(x)

        assert counter.total_flops == 2 * 768 * 768


# ============================================================================
# Tests: Different Layer Types
# ============================================================================


class TestDifferentLayerTypes:
    """Tests for various PyTorch layer types."""

    def test_conv1d(self):
        """Test Conv1d FLOP counting."""
        conv = nn.Conv1d(64, 128, kernel_size=3, bias=False)
        x = torch.randn(1, 64, 100)

        with flop_budget(budget=1e12) as counter:
            _ = conv(x)

        assert counter.total_flops > 0

    def test_conv2d(self):
        """Test Conv2d FLOP counting."""
        conv = nn.Conv2d(3, 64, kernel_size=3, bias=False)
        x = torch.randn(1, 3, 32, 32)

        with flop_budget(budget=1e12) as counter:
            _ = conv(x)

        expected = 2 * 64 * 3 * 3 * 3 * 30 * 30
        assert counter.total_flops == expected

    def test_batch_matmul(self):
        """Test batched matrix multiplication."""
        a = torch.randn(4, 32, 64)
        b = torch.randn(4, 64, 128)

        with flop_budget(budget=1e12) as counter:
            _ = torch.bmm(a, b)

        expected = 2 * 4 * 32 * 128 * 64
        assert counter.total_flops == expected

    def test_attention_mechanism(self):
        """Test self-attention FLOP counting."""
        batch, seq, dim = 2, 16, 64

        q = torch.randn(batch, seq, dim)
        k = torch.randn(batch, seq, dim)
        v = torch.randn(batch, seq, dim)

        with flop_budget(budget=1e12) as counter:
            scores = torch.matmul(q, k.transpose(-2, -1))
            attn = torch.softmax(scores, dim=-1)
            _ = torch.matmul(attn, v)

        flops_qk = 2 * batch * seq * seq * dim
        flops_av = 2 * batch * seq * dim * seq

        assert counter.total_flops >= flops_qk + flops_av


# ============================================================================
# Tests: Exception Message Quality
# ============================================================================


class TestExceptionMessages:
    """Tests for exception message quality."""

    def test_exception_str_contains_budget(self):
        """Verify exception string contains budget info."""
        try:
            with flop_budget(budget=1000):
                _ = torch.matmul(torch.randn(32, 32), torch.randn(32, 32))
        except FLOPBudgetExceeded as e:
            msg = str(e)
            assert "1,000" in msg or "1000" in msg

    def test_exception_str_contains_current(self):
        """Verify exception string contains current FLOP count."""
        try:
            with flop_budget(budget=1000):
                _ = torch.matmul(torch.randn(32, 32), torch.randn(32, 32))
        except FLOPBudgetExceeded as e:
            assert "65" in str(e)

    def test_exception_has_operation_name(self):
        """Verify exception contains operation name."""
        try:
            with flop_budget(budget=1000):
                _ = torch.matmul(torch.randn(32, 32), torch.randn(32, 32))
        except FLOPBudgetExceeded as e:
            assert e.operation != ""
            assert "mm" in e.operation.lower() or "matmul" in e.operation.lower()


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
