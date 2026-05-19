"""Unit tests for the unified queue callback with size management."""

import pytest
import torch
from lightning.pytorch import Trainer

from stable_pretraining.callbacks.queue import (
    OnlineQueue,
    find_or_create_queue_callback,
)


@pytest.fixture(autouse=True)
def _reset_online_queue_state():
    """Ensure each test starts with a clean OnlineQueue class state.

    Tests in this module exercise class-level dicts that previously leaked
    between tests; the cross-trainer wipe (#378) now does this automatically
    on the first ``find_or_create_queue_callback`` / ``setup`` call, but
    resetting up-front keeps tests independent regardless of run order.
    """
    OnlineQueue._shared_queues.clear()
    OnlineQueue._queue_info.clear()
    OnlineQueue._owner_trainer_id = None
    yield
    OnlineQueue._shared_queues.clear()
    OnlineQueue._queue_info.clear()
    OnlineQueue._owner_trainer_id = None


@pytest.mark.unit
class TestUnifiedQueueManagement:
    """Test the unified queue size management functionality."""

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
        class MockModule:
            def __init__(self):
                self.callbacks_modules = {}

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

        class MockModule:
            def __init__(self):
                self.callbacks_modules = {}

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

    def test_state_wiped_between_distinct_trainers(self):
        """Regression test for #378: stale class-level state must not leak.

        A second Trainer.fit() in the same process used to inherit the prior
        run's accumulated queue contents because ``_shared_queues`` and
        ``_queue_info`` are class attributes. The wipe is triggered by the
        first ``setup()`` or ``find_or_create_queue_callback`` call that sees
        a new ``id(trainer)``.
        """

        class MockModule:
            def __init__(self):
                self.callbacks_modules = torch.nn.ModuleDict()
                self.device = torch.device("cpu")

        # --- Run 1: populate queue with trainer A ---
        trainer_a = Trainer()
        pl_module_a = MockModule()
        queue_a = find_or_create_queue_callback(
            trainer=trainer_a, key="features", queue_length=100, dim=8
        )
        queue_a.setup(trainer_a, pl_module_a, "fit")

        # Append enough data that staleness would be visible
        OnlineQueue._shared_queues["features"].append(torch.randn(50, 8))
        assert int(OnlineQueue._shared_queues["features"].pointer.item()) == 50
        assert OnlineQueue._owner_trainer_id == id(trainer_a)

        # --- Run 2: new trainer triggers wipe ---
        trainer_b = Trainer()
        pl_module_b = MockModule()
        queue_b = find_or_create_queue_callback(
            trainer=trainer_b, key="features", queue_length=100, dim=8
        )

        # Wipe should have fired on the find_or_create_queue_callback call
        assert OnlineQueue._owner_trainer_id == id(trainer_b)

        queue_b.setup(trainer_b, pl_module_b, "fit")

        # The fresh queue must be empty — no leaked data from trainer_a
        assert int(OnlineQueue._shared_queues["features"].pointer.item()) == 0
        assert not bool(OnlineQueue._shared_queues["features"].filled.item())

    def test_queue_placed_on_module_device_at_setup(self):
        """Regression test for #379: buffers must follow ``pl_module.device``.

        ``pl_module.to(device)`` runs BEFORE callback ``setup()``, and children
        added afterwards aren't auto-moved. The fix places the OrderedQueue on
        ``pl_module.device`` explicitly. We test with the 'meta' device so the
        check is meaningful even on a CPU-only CI box.
        """

        class MockModuleOnDevice:
            def __init__(self, device):
                self.callbacks_modules = torch.nn.ModuleDict()
                self.device = torch.device(device)

        trainer = Trainer()
        pl_module = MockModuleOnDevice("meta")
        queue = find_or_create_queue_callback(
            trainer=trainer, key="emb", queue_length=64, dim=4
        )
        queue.setup(trainer, pl_module, "fit")

        ordered_q = OnlineQueue._shared_queues["emb"]
        assert ordered_q.pointer.device.type == "meta"
        assert ordered_q.out.device.type == "meta"
        # And the queue is registered into the module's ModuleDict
        assert "ordered_queue_emb" in pl_module.callbacks_modules

    def test_resolve_module_device_handles_various_inputs(self):
        """``_resolve_module_device`` is defensive against missing/odd values."""

        class HasDeviceTensor:
            device = torch.device("cpu")

        class HasDeviceStr:
            device = "cpu"

        class NoDevice:
            pass

        class BadDevice:
            device = 42  # not a device-like value

        assert OnlineQueue._resolve_module_device(HasDeviceTensor()) == torch.device(
            "cpu"
        )
        assert OnlineQueue._resolve_module_device(HasDeviceStr()) == torch.device("cpu")
        assert OnlineQueue._resolve_module_device(NoDevice()) is None
        assert OnlineQueue._resolve_module_device(BadDevice()) is None

    def test_setup_invokes_to_with_module_device(self):
        """Regression test for #379: ``OrderedQueue.to(device)`` is called.

        Verified via a spy on ``OrderedQueue.to`` rather than relying on
        ``meta`` (which fails downstream in resize/append paths). This
        guarantees the call happens even when the device-target equals the
        default CPU placement (where a device-equality assertion would pass
        trivially even without the fix).
        """
        from unittest.mock import patch

        from stable_pretraining.callbacks.queues import OrderedQueue

        class MockModule:
            def __init__(self):
                self.callbacks_modules = torch.nn.ModuleDict()
                self.device = torch.device("cpu")

        trainer = Trainer()
        pl_module = MockModule()

        recorded_calls = []
        real_to = OrderedQueue.to

        def spy_to(self, *args, **kwargs):
            recorded_calls.append((args, kwargs))
            return real_to(self, *args, **kwargs)

        queue = find_or_create_queue_callback(
            trainer=trainer, key="emb", queue_length=64, dim=4
        )
        with patch.object(OrderedQueue, "to", spy_to):
            queue.setup(trainer, pl_module, "fit")

        # `.to(torch.device("cpu"))` must have been invoked with the module's
        # device — proves the fix path executed.
        assert any(
            args and args[0] == torch.device("cpu") for args, _ in recorded_calls
        ), f"OrderedQueue.to was not called with the module device: {recorded_calls}"

    def test_fixed_embeddings_shared_across_multiple_queues_cpu(self):
        """End-to-end CPU test exercising shared storage across queue sizes.

        Three callbacks of different lengths see the same underlying queue and
        each gets exactly its requested tail. Uses deterministic
        ``torch.arange``-based embeddings so the snapshot contents can be
        asserted exactly (not just shape). Exercises:
        - Multi-callback sharing of one OrderedQueue
        - Multiple sequential appends across multiple validation epochs
        - The validation snapshot path on CPU
        - Insertion-order preservation through wraparound
        """

        class MockModule:
            def __init__(self):
                self.callbacks_modules = torch.nn.ModuleDict()
                self.device = torch.device("cpu")

            def all_gather(self, tensor):
                return tensor.unsqueeze(0)

        class MockTrainer:
            world_size = 1

        trainer = Trainer()
        mock_trainer = MockTrainer()
        pl_module = MockModule()

        # Three callbacks for the same key — small/medium/large.
        small = find_or_create_queue_callback(
            trainer=trainer, key="emb", queue_length=10, dim=4
        )
        medium = find_or_create_queue_callback(
            trainer=trainer, key="emb", queue_length=50, dim=4
        )
        large = find_or_create_queue_callback(
            trainer=trainer, key="emb", queue_length=200, dim=4
        )
        for cb in (small, medium, large):
            cb.setup(trainer, pl_module, "fit")

        # Underlying shared queue is sized to the largest request.
        shared = OnlineQueue._shared_queues["emb"]
        assert shared.max_length == 200
        assert shared.pointer.device.type == "cpu"
        assert shared.out.device.type == "cpu"

        # Generate deterministic embeddings: row i is [i, i, i, i] as floats.
        # We'll append 250 of them in batches of 50 to force wraparound on
        # the small (10) and medium (50) queues but only partially fill
        # large (200).
        all_embeddings = (
            torch.arange(250, dtype=torch.float32).unsqueeze(1).expand(-1, 4)
        )
        for start in range(0, 250, 50):
            shared.append(all_embeddings[start : start + 50].clone())

        # After 250 appends, the shared queue (length 200) should contain
        # rows 50..249 in insertion order.
        full_ordered = shared.get()
        assert full_ordered.shape == (200, 4)
        torch.testing.assert_close(full_ordered, all_embeddings[50:250])

        # Run validation: each callback takes its tail.
        for cb in (small, medium, large):
            cb.on_validation_epoch_start(mock_trainer, pl_module)

        # Small (10): last 10 items = rows 240..249
        assert small._snapshot.shape == (10, 4)
        torch.testing.assert_close(small._snapshot, all_embeddings[240:250])
        assert small._snapshot.device.type == "cpu"

        # Medium (50): last 50 items = rows 200..249
        assert medium._snapshot.shape == (50, 4)
        torch.testing.assert_close(medium._snapshot, all_embeddings[200:250])

        # Large (200): last 200 items (queue was sized to 200) = rows 50..249
        assert large._snapshot.shape == (200, 4)
        torch.testing.assert_close(large._snapshot, all_embeddings[50:250])

        # All three snapshots are views/derivatives of the same underlying
        # buffer — verify by checking the tail overlap between them.
        # Last 10 rows of medium == small's full snapshot.
        torch.testing.assert_close(medium._snapshot[-10:], small._snapshot)
        # Last 50 rows of large == medium's full snapshot.
        torch.testing.assert_close(large._snapshot[-50:], medium._snapshot)

        # Cleanup snapshots (end of validation).
        for cb in (small, medium, large):
            cb.on_validation_epoch_end(mock_trainer, pl_module)
            assert cb._snapshot is None

        # Append more data and re-validate — confirms snapshots stay fresh
        # across validation cycles.
        extra = torch.full((20, 4), 999.0)
        shared.append(extra)
        for cb in (small, medium, large):
            cb.on_validation_epoch_start(mock_trainer, pl_module)

        # The 20 new rows are now the most-recent items; small (10) should
        # see only the last 10 of them.
        assert small._snapshot.shape == (10, 4)
        torch.testing.assert_close(small._snapshot, extra[-10:])
        # Medium (50): last 50 = 30 of the original tail (rows 220..249) + 20 extras
        assert medium._snapshot.shape == (50, 4)
        torch.testing.assert_close(medium._snapshot[:30], all_embeddings[220:250])
        torch.testing.assert_close(medium._snapshot[30:], extra)

    def test_ordering_preservation(self):
        """Test that the OrderedQueue maintains insertion order."""
        trainer = Trainer()

        class MockModule:
            def __init__(self):
                self.callbacks_modules = {}

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
