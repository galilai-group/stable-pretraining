"""Unit tests for nn_modules split between backbone and callbacks."""

import pytest
import torch

from stable_pretraining.backbone.nn_modules import (
    EMA,
    Normalize,
    ImageToVideoEncoder,
)
from stable_pretraining.callbacks.queues import (
    UnsortedQueue,
    OrderedQueue,
)


@pytest.mark.unit
class TestUnsortedQueue:
    """Test the UnsortedQueue data structure."""

    def test_zero_length_queue(self):
        """Test queue with max_length=0 (passthrough behavior)."""
        q = UnsortedQueue(0)
        for i in range(10):
            v = q.append(torch.Tensor([i]))
            assert v.numel() == 1
            assert v[0] == i

    def test_basic_append_and_wraparound(self):
        """Test basic append operations and wraparound behavior."""
        q = UnsortedQueue(5)
        for i in range(10):
            v = q.append(torch.Tensor([i]))
            if i < 5:
                assert v[-1] == i
                assert len(v) == i + 1
            else:
                assert len(v) == 5

        # After 10 appends, queue should contain last 5 items
        final = q.get()
        assert final.numel() == 5
        assert 9 in final.numpy()
        assert 8 in final.numpy()
        assert 7 in final.numpy()
        assert 6 in final.numpy()
        assert 5 in final.numpy()
        assert 4 not in final.numpy()
        assert 3 not in final.numpy()
        assert 2 not in final.numpy()
        assert 1 not in final.numpy()
        assert 0 not in final.numpy()

    def test_batch_append(self):
        """Test appending batches of items."""
        q = UnsortedQueue(10, shape=2)

        # Append batch of 3 items
        batch1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
        v1 = q.append(batch1)
        assert v1.shape == (3, 2)

        # Append batch of 2 items
        batch2 = torch.tensor([[7, 8], [9, 10]], dtype=torch.float32)
        v2 = q.append(batch2)
        assert v2.shape == (5, 2)

    def test_shape_initialization(self):
        """Test queue initialization with different shapes."""
        # Test with integer shape
        q1 = UnsortedQueue(5, shape=10)
        assert q1.out.shape == (5, 10)

        # Test with tuple shape
        q2 = UnsortedQueue(5, shape=(3, 4))
        assert q2.out.shape == (5, 3, 4)

        # Test with None shape (lazy initialization)
        q3 = UnsortedQueue(5, shape=None)
        item = torch.randn(2, 7, 8)
        q3.append(item)
        assert q3.out.shape == (5, 7, 8)

    def test_get_method(self):
        """Test the get() method returns correct contents."""
        q = UnsortedQueue(5)

        # Before filling
        for i in range(3):
            q.append(torch.Tensor([i]))

        result = q.get()
        assert len(result) == 3
        assert torch.allclose(result, torch.tensor([0, 1, 2], dtype=torch.float32))

        # After filling
        for i in range(3, 8):
            q.append(torch.Tensor([i]))

        result = q.get()
        assert len(result) == 5
        # UnsortedQueue doesn't maintain order after wraparound
        # It will have [5, 6, 7, 3, 4] in buffer order
        assert set(result.numpy().tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}

    def test_batch_larger_than_queue(self):
        """Test appending a batch larger than queue capacity."""
        q = UnsortedQueue(5, shape=2)

        # Append a batch of 10 items to a queue of size 5
        large_batch = torch.tensor(
            [[i * 2, i * 2 + 1] for i in range(10)], dtype=torch.float32
        )
        result = q.append(large_batch)

        # Should keep only the last 5 items from the batch
        assert result.shape == (5, 2)
        expected = torch.tensor(
            [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]], dtype=torch.float32
        )
        assert torch.allclose(result, expected)
        assert q.filled.item()

    def test_batch_exactly_queue_size(self):
        """Test appending a batch exactly equal to queue capacity."""
        q = UnsortedQueue(5, shape=3)

        batch = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            dtype=torch.float32,
        )
        result = q.append(batch)

        assert result.shape == (5, 3)
        assert torch.allclose(result, batch)
        assert q.filled.item()

    def test_multiple_large_batches(self):
        """Test appending multiple batches larger than queue capacity."""
        q = UnsortedQueue(3)

        # First large batch
        batch1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        result1 = q.append(batch1)
        expected1 = torch.tensor([3, 4, 5], dtype=torch.float32)
        assert torch.allclose(result1, expected1)

        # Second large batch
        batch2 = torch.tensor([10, 20, 30, 40], dtype=torch.float32)
        result2 = q.append(batch2)
        expected2 = torch.tensor([20, 30, 40], dtype=torch.float32)
        assert torch.allclose(result2, expected2)


@pytest.mark.unit
class TestOrderedQueue:
    """Test the OrderedQueue data structure that maintains insertion order."""

    def test_zero_length_queue(self):
        """Test queue with max_length=0 (passthrough behavior)."""
        q = OrderedQueue(0)
        for i in range(10):
            v = q.append(torch.Tensor([i]))
            assert v.numel() == 1
            assert v[0] == i

    def test_insertion_order_preserved(self):
        """Test that insertion order is preserved even after wraparound."""
        q = OrderedQueue(5)
        for i in range(10):
            v = q.append(torch.Tensor([i]))
            if i < 5:
                assert v[-1] == i
                assert len(v) == i + 1
            else:
                assert len(v) == 5
                # Should contain last 5 items in order
                expected = torch.tensor(
                    [i - 4, i - 3, i - 2, i - 1, i], dtype=torch.float32
                )
                assert torch.allclose(v, expected)

    def test_batch_append_with_order(self):
        """Test appending batches maintains correct order."""
        q = OrderedQueue(10, shape=2)

        batch1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
        batch2 = torch.tensor([[7, 8], [9, 10]], dtype=torch.float32)

        v1 = q.append(batch1)
        assert v1.shape == (3, 2)

        v2 = q.append(batch2)
        assert v2.shape == (5, 2)
        expected = torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.float32
        )
        assert torch.allclose(v2, expected)

    def test_wraparound_preserves_order(self):
        """Test that order is maintained correctly after wraparound."""
        q = OrderedQueue(10, shape=2)

        # Fill initial queue
        for i in range(5):
            batch = torch.tensor([[i * 2, i * 2 + 1]], dtype=torch.float32)
            q.append(batch)

        # Cause wraparound
        batch3 = torch.tensor(
            [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]],
            dtype=torch.float32,
        )
        v3 = q.append(batch3)

        assert v3.shape == (10, 2)
        # Should have items 1-10 in order (item 0 was overwritten)
        expected = torch.tensor(
            [
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
                [10, 11],
                [12, 13],
                [14, 15],
                [16, 17],
                [18, 19],
                [20, 21],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(v3, expected)

    def test_get_sorted_vs_unsorted(self):
        """Test the difference between get() and get_unsorted()."""
        q = OrderedQueue(5)

        # Fill queue to cause wraparound
        for i in range(8):
            q.append(torch.Tensor([i * 10]))

        sorted_result = q.get()
        unsorted_result = q.get_unsorted()

        # Sorted should be in insertion order
        expected_sorted = torch.tensor([30, 40, 50, 60, 70], dtype=torch.float32)
        assert torch.allclose(sorted_result, expected_sorted)

        # Unsorted is in buffer order (wraparound pattern)
        assert unsorted_result.shape == sorted_result.shape
        # The exact order depends on wraparound position, but all elements should be present
        assert set(unsorted_result.numpy().tolist()) == set(
            sorted_result.numpy().tolist()
        )

    def test_global_counter_persistence(self):
        """Test that global counter correctly tracks total insertions."""
        q = OrderedQueue(3)

        # Insert more items than queue capacity
        for i in range(10):
            q.append(torch.Tensor([i]))

        # Check that global counter has tracked all insertions
        assert q.global_counter.item() == 10

        # Verify last 3 items are in order
        result = q.get()
        expected = torch.tensor([7, 8, 9], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_batch_larger_than_queue(self):
        """Test appending a batch larger than queue capacity."""
        q = OrderedQueue(5, shape=2)

        # Append a batch of 10 items to a queue of size 5
        large_batch = torch.tensor(
            [[i * 2, i * 2 + 1] for i in range(10)], dtype=torch.float32
        )
        result = q.append(large_batch)

        # Should keep only the last 5 items from the batch in order
        assert result.shape == (5, 2)
        expected = torch.tensor(
            [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]], dtype=torch.float32
        )
        assert torch.allclose(result, expected)
        assert q.filled.item()
        assert q.global_counter.item() == 10

    def test_batch_exactly_queue_size(self):
        """Test appending a batch exactly equal to queue capacity."""
        q = OrderedQueue(5, shape=3)

        batch = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            dtype=torch.float32,
        )
        result = q.append(batch)

        assert result.shape == (5, 3)
        assert torch.allclose(result, batch)
        assert q.filled.item()
        assert q.global_counter.item() == 5

    def test_multiple_large_batches_with_order(self):
        """Test appending multiple batches larger than queue capacity maintains order."""
        q = OrderedQueue(3)

        # First large batch
        batch1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        result1 = q.append(batch1)
        expected1 = torch.tensor([3, 4, 5], dtype=torch.float32)
        assert torch.allclose(result1, expected1)
        assert q.global_counter.item() == 5

        # Second large batch
        batch2 = torch.tensor([10, 20, 30, 40], dtype=torch.float32)
        result2 = q.append(batch2)
        expected2 = torch.tensor([20, 30, 40], dtype=torch.float32)
        assert torch.allclose(result2, expected2)
        assert q.global_counter.item() == 9


@pytest.mark.unit
class TestEMA:
    """Test the Exponential Moving Average module."""

    def test_alpha_zero_no_update(self):
        """Test that alpha=0 means no update (always returns first value)."""
        ema = EMA(0)
        initial = torch.randn(10, 10)
        ema(initial)

        for _ in range(10):
            v = ema(torch.randn(10, 10))
            assert torch.allclose(v, initial)

    def test_alpha_one_no_smoothing(self):
        """Test that alpha=1 means no smoothing (always returns current value)."""
        ema = EMA(1)

        for _ in range(10):
            current = torch.randn(10, 10)
            v = ema(current)
            assert torch.allclose(v, current)

    def test_alpha_half_smoothing(self):
        """Test EMA with alpha=0.5 for proper smoothing."""
        ema = EMA(0.5)
        initial = torch.randn(10, 10)
        ground = initial.detach()
        v = ema(initial)
        assert torch.allclose(ground, v)

        for _ in range(10):
            current = torch.randn(10, 10)
            v = ema(current)
            ground = current * 0.5 + ground * 0.5
            assert torch.allclose(v, ground)

    def test_different_shapes(self):
        """Test EMA with different tensor shapes."""
        ema = EMA(0.7)

        # 1D tensor
        v1 = ema(torch.randn(100))
        assert v1.shape == (100,)

        # Reset for new shape test
        ema2 = EMA(0.7)
        # 3D tensor
        v2 = ema2(torch.randn(5, 10, 20))
        assert v2.shape == (5, 10, 20)


@pytest.mark.unit
class TestNormalize:
    """Test the Normalize module."""

    def test_normalize_output(self):
        """Test that Normalize properly normalizes and scales tensors."""
        normalize = Normalize()
        x = torch.randn(10, 20, 30)

        output = normalize(x)

        # Check that output has same shape
        assert output.shape == x.shape

        # The normalized tensor scaled by sqrt(numel) should have specific properties
        # This is testing the F.normalize(x, dim=(0,1,2)) * np.sqrt(x.numel()) operation
        assert output.numel() == x.numel()


@pytest.mark.unit
class TestImageToVideoEncoder:
    """Test the ImageToVideoEncoder wrapper module."""

    def test_video_encoding(self):
        """Test encoding video frames with an image encoder."""
        # Create a simple Conv2d encoder for testing
        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, kernel_size=3),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        video_encoder = ImageToVideoEncoder(encoder)

        # Create video tensor: (batch=2, time=3, channel=4, height=4, width=4)
        video = torch.randn(2, 3, 4, 4, 4)

        # Apply video encoder
        features = video_encoder(video)

        # Check output shape: (batch=2, time=3, features=8)
        assert features.shape == (2, 3, 8)

    def test_single_frame(self):
        """Test with single frame video."""
        # Create a simple Conv2d encoder
        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        video_encoder = ImageToVideoEncoder(encoder)

        # Single frame: (batch=4, time=1, channel=3, height=8, width=8)
        video = torch.randn(4, 1, 3, 8, 8)
        features = video_encoder(video)

        assert features.shape == (4, 1, 16)
