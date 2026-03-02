"""Unit tests for the unified queue callback with size management."""

import pytest
import torch
from lightning.pytorch import Trainer

from stable_pretraining.callbacks.queue import (
    OnlineQueue,
    find_or_create_queue_callback,
)


@pytest.mark.unit
class TestUnifiedQueueManagement:
    """Test the unified queue size management functionality."""

    def setup_method(self):
        """Clear class-level shared state between tests."""
        OnlineQueue._shared_queues.clear()
        OnlineQueue._queue_info.clear()

    def test_single_queue_multiple_sizes(self):
        """Test that multiple callbacks with same key but different sizes share a queue."""
        trainer = Trainer()

        # Create first callback requesting 1000 samples
        queue1 = find_or_create_queue_callback(
            trainer=trainer,
            key="features",
            queue_length=1000,
            dim=128,
            dtype=torch.float32,
        )

        assert queue1.requested_length == 1000
        assert queue1.actual_queue_length == 1000

        # Create second callback requesting 5000 samples
        queue2 = find_or_create_queue_callback(
            trainer=trainer,
            key="features",
            queue_length=5000,
            dim=128,
            dtype=torch.float32,
        )

        assert queue2.requested_length == 5000
        assert queue2.actual_queue_length == 5000

        # First queue should now also see the increased actual size
        assert queue1.actual_queue_length == 5000

        # Both should share the same underlying queue
        assert queue1.key == queue2.key
        assert OnlineQueue._queue_info["features"]["max_length"] == 5000

    def test_queue_resizing_preserves_data(self):
        """Test that resizing a queue preserves existing data."""
        trainer = Trainer()

        # Mock a LightningModule for setup
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.callbacks_modules = torch.nn.ModuleDict()
                self._dummy = torch.nn.Linear(1, 1)

        pl_module = MockModule()

        # Create initial queue with size 100
        queue1 = find_or_create_queue_callback(
            trainer=trainer,
            key="test_data",
            queue_length=100,
            dim=10,
        )

        # Manually setup to initialize the queue
        queue1.setup(trainer, pl_module, "fit")

        # Add some data to the queue
        test_data = torch.randn(50, 10)
        OnlineQueue._shared_queues["test_data"].append(test_data)

        initial_data = OnlineQueue._shared_queues["test_data"].get()
        assert len(initial_data) == 50

        # Create second callback requesting larger size
        queue2 = find_or_create_queue_callback(
            trainer=trainer,
            key="test_data",
            queue_length=200,
            dim=10,
        )

        queue2.setup(trainer, pl_module, "fit")

        # Check that data was preserved
        resized_data = OnlineQueue._shared_queues["test_data"].get()
        assert len(resized_data) == 50
        assert torch.allclose(initial_data, resized_data)

    def test_size_based_retrieval(self):
        """Test that each callback gets the correct amount of data."""
        trainer = Trainer()

        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.callbacks_modules = torch.nn.ModuleDict()
                self._dummy = torch.nn.Linear(1, 1)

            def all_gather(self, tensor):
                return tensor.unsqueeze(0)

        pl_module = MockModule()

        # Create callbacks with different sizes
        queue_small = find_or_create_queue_callback(
            trainer=trainer,
            key="shared_features",
            queue_length=100,
            dim=5,
        )

        queue_large = find_or_create_queue_callback(
            trainer=trainer,
            key="shared_features",
            queue_length=500,
            dim=5,
        )

        # Setup queues
        queue_small.setup(trainer, pl_module, "fit")
        queue_large.setup(trainer, pl_module, "fit")

        # Add 300 samples to the shared queue
        test_data = torch.randn(300, 5)
        OnlineQueue._shared_queues["shared_features"].append(test_data)

        # Trigger validation snapshots - pass trainer with world_size=1
        class MockTrainer:
            world_size = 1

        mock_trainer = MockTrainer()
        queue_small.on_validation_epoch_start(mock_trainer, pl_module)
        queue_large.on_validation_epoch_start(mock_trainer, pl_module)

        # Small queue should get last 100 items
        assert queue_small._snapshot.shape[0] == 100
        # Should be the last 100 items from the 300
        expected_small = test_data[-100:]
        assert torch.allclose(queue_small._snapshot, expected_small)

        # Large queue should get all 300 items (less than requested 500)
        assert queue_large._snapshot.shape[0] == 300
        assert torch.allclose(queue_large._snapshot, test_data)

    def test_multiple_keys_independent(self):
        """Test that queues with different keys remain independent."""
        trainer = Trainer()

        queue_a = find_or_create_queue_callback(
            trainer=trainer,
            key="features_a",
            queue_length=1000,
        )

        queue_b = find_or_create_queue_callback(
            trainer=trainer,
            key="features_b",
            queue_length=2000,
        )

        # Different keys should have different actual sizes
        assert queue_a.actual_queue_length == 1000
        assert queue_b.actual_queue_length == 2000

        # Should have separate entries in the registry
        assert "features_a" in OnlineQueue._queue_info
        assert "features_b" in OnlineQueue._queue_info
        assert OnlineQueue._queue_info["features_a"]["max_length"] == 1000
        assert OnlineQueue._queue_info["features_b"]["max_length"] == 2000

    def test_find_existing_with_exact_size(self):
        """Test that finding an existing queue with exact size returns it."""
        trainer = Trainer()

        # Create first queue
        queue1 = find_or_create_queue_callback(
            trainer=trainer,
            key="test",
            queue_length=1234,
        )

        # Find the same queue
        queue2 = find_or_create_queue_callback(
            trainer=trainer,
            key="test",
            queue_length=1234,
        )

        # Should be the same instance
        assert queue1 is queue2

    def test_ordering_preservation(self):
        """Test that the OrderedQueue maintains insertion order."""
        trainer = Trainer()

        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.callbacks_modules = torch.nn.ModuleDict()
                self._dummy = torch.nn.Linear(1, 1)

        pl_module = MockModule()

        queue = find_or_create_queue_callback(
            trainer=trainer,
            key="ordered_test",
            queue_length=10,
            dim=1,
        )

        queue.setup(trainer, pl_module, "fit")

        # Add items that will cause wraparound
        for i in range(15):
            item = torch.tensor([[float(i)]])
            OnlineQueue._shared_queues["ordered_test"].append(item)

        # Get the data - should be last 10 items in order
        result = OnlineQueue._shared_queues["ordered_test"].get()
        expected = torch.tensor(
            [[5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0], [13.0], [14.0]]
        )

        assert torch.allclose(result, expected)

    def test_queue_device_matches_module(self):
        """Test that the queue is created on the same device as pl_module."""
        trainer = Trainer()

        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.callbacks_modules = torch.nn.ModuleDict()
                self._dummy = torch.nn.Linear(1, 1)

        pl_module = MockModule()
        expected_device = next(pl_module.parameters()).device

        queue = find_or_create_queue_callback(
            trainer=trainer,
            key="device_test",
            queue_length=100,
            dim=16,
        )
        queue.setup(trainer, pl_module, "fit")

        shared_queue = OnlineQueue._shared_queues["device_test"]
        assert shared_queue.out.device == expected_device
        assert shared_queue.pointer.device == expected_device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_queue_device_matches_module_cuda(self):
        """Test that the queue lands on GPU when the module is on GPU."""
        trainer = Trainer()

        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.callbacks_modules = torch.nn.ModuleDict()
                self._dummy = torch.nn.Linear(1, 1)

        pl_module = MockModule().cuda()

        queue = find_or_create_queue_callback(
            trainer=trainer,
            key="cuda_device_test",
            queue_length=100,
            dim=16,
        )
        queue.setup(trainer, pl_module, "fit")

        shared_queue = OnlineQueue._shared_queues["cuda_device_test"]
        assert shared_queue.out.is_cuda
        assert shared_queue.pointer.is_cuda

        # Append GPU data — should not raise a device mismatch
        data = torch.randn(10, 16, device="cuda")
        shared_queue.append(data)
        result = shared_queue.get()
        assert result.is_cuda
