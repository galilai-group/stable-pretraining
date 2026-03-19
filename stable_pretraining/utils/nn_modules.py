"""Neural network modules and utilities."""

from typing import Iterable, Union

import numpy as np
import torch
import torch.nn.functional as F


class BatchNorm1dNoBias(torch.nn.BatchNorm1d):
    """BatchNorm1d with learnable scale but no learnable bias (center=False).

    This is used in contrastive learning methods like SimCLR where the final
    projection layer uses batch normalization with scale (gamma) but without
    bias (beta). This follows the original SimCLR implementation where the
    bias term is removed from the final BatchNorm layer.

    The bias is frozen at 0 and set to non-trainable, while the weight (scale)
    parameter remains learnable.

    Example:
        ```python
        # SimCLR-style projector
        projector = nn.Sequential(
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128, bias=False),
            spt.utils.nn_modules.BatchNorm1dNoBias(128),  # Final layer: no bias
        )
        ```

    Note:
        This is equivalent to TensorFlow's BatchNorm with center=False, scale=True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False
        with torch.no_grad():
            self.bias.zero_()


class L2Norm(torch.nn.Module):
    """L2 normalization layer that normalizes input to unit length.

    Normalizes the input tensor along the last dimension to have unit L2 norm.
    Commonly used in DINO before the prototypes layer.

    Example:
        ```python
        projector = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
            spt.utils.nn_modules.L2Norm(),  # Normalize to unit length
            nn.Linear(256, 4096, bias=False),  # Prototypes
        )
        ```
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to unit L2 norm.

        Args:
            x: Input tensor [..., D]

        Returns:
            L2-normalized tensor [..., D] where each D-dimensional vector has unit length
        """
        return F.normalize(x, dim=-1, p=2)


class ImageToVideoEncoder(torch.nn.Module):
    """Wrapper to apply an image encoder to video data by processing each frame independently.

    This module takes video data with shape (batch, time, channel, height, width) and applies
    an image encoder to each frame, returning the encoded features.

    Args:
        encoder (torch.nn.Module): The image encoder module to apply to each frame.
    """

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, video):
        # we expect something of the shape
        # (batch, time, channel, height, width)
        batch_size, num_timesteps = video.shape[:2]
        assert video.ndim == 5
        # (BxT)xCxHxW
        video = video.contiguous().flatten(0, 1)
        # (BxT)xF
        features = self.encoder(video)
        # BxTxF
        features = features.contiguous().view(
            batch_size, num_timesteps, features.size(1)
        )
        return features


class Normalize(torch.nn.Module):
    """Normalize tensor and scale by square root of number of elements."""

    def forward(self, x):
        return F.normalize(x, dim=(0, 1, 2)) * np.sqrt(x.numel())


class UnsortedQueue(torch.nn.Module):
    """A queue data structure that stores tensors with a maximum length.

    This module implements a circular buffer that stores tensors up to a
    maximum length. When the queue is full, new items overwrite the oldest ones.

    Args:
        max_length: Maximum number of elements to store in the queue
        shape: Shape of each element (excluding batch dimension). Can be int or tuple
        dtype: Data type of the tensors to store
    """

    def __init__(
        self, max_length: int, shape: Union[int, Iterable[int]] = None, dtype=None
    ):
        super().__init__()
        self.max_length = max_length
        self.register_buffer("pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("filled", torch.zeros((), dtype=torch.bool))
        if shape is None:
            # Initialize with a placeholder shape that will be updated on first append
            self.register_buffer("out", torch.zeros((max_length, 1), dtype=dtype))
            self.register_buffer("initialized", torch.tensor(False))
        else:
            if type(shape) is int:
                shape = (shape,)
            self.register_buffer(
                "out", torch.zeros((max_length,) + tuple(shape), dtype=dtype)
            )
            self.register_buffer("initialized", torch.tensor(True))

    def append(self, item):
        """Append item(s) to the queue.

        Args:
            item: Tensor to append. First dimension is batch size.

        Returns:
            Current contents of the queue
        """
        if self.max_length == 0:
            return item

        # Initialize buffer with correct shape on first use
        if not self.initialized:
            shape = (self.max_length,) + item.shape[1:]
            new_out = torch.zeros(shape, dtype=item.dtype, device=item.device)
            self.out.resize_(shape)
            self.out.copy_(new_out)
            self.initialized.fill_(True)

        batch_size = item.size(0)

        # Handle case where batch is larger than queue capacity
        if batch_size >= self.max_length:
            # Keep only the last max_length items from the batch
            self.out[:] = item[-self.max_length :]
            self.pointer.fill_(0)
            self.filled.fill_(True)
        elif self.pointer + batch_size < self.max_length:
            self.out[self.pointer : self.pointer + batch_size] = item
            self.pointer.add_(batch_size)
        else:
            remaining = self.max_length - self.pointer
            self.out[-remaining:] = item[:remaining]
            self.out[: batch_size - remaining] = item[remaining:]
            self.pointer.copy_(batch_size - remaining)
            self.filled.copy_(True)
        return self.out if self.filled else self.out[: self.pointer]

    def get(self):
        """Get current contents of the queue.

        Returns:
            Tensor containing all items currently in the queue
        """
        return self.out if self.filled else self.out[: self.pointer]

    @staticmethod
    def _test():
        q = UnsortedQueue(0)
        for i in range(10):
            v = q.append(torch.Tensor([i]))
            assert v.numel() == 1
            assert v[0] == i

        q = UnsortedQueue(5)
        for i in range(10):
            v = q.append(torch.Tensor([i]))
            if i < 5:
                assert v[-1] == i
        assert v.numel() == 5
        assert 9 in v.numpy()
        assert 8 in v.numpy()
        assert 7 in v.numpy()
        assert 6 in v.numpy()
        assert 5 in v.numpy()
        assert 4 not in v.numpy()
        assert 3 not in v.numpy()
        assert 2 not in v.numpy()
        assert 1 not in v.numpy()
        assert 0 not in v.numpy()
        return True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        out_key = prefix + "out"
        if out_key in state_dict:
            self.out.resize_(state_dict[out_key].shape)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.out.resize_(state_dict["out"].shape)
        super().load_state_dict(state_dict, strict, assign)


class OrderedQueue(torch.nn.Module):
    """A queue that maintains insertion order of elements.

    Similar to UnsortedQueue but tracks the order in which items were inserted,
    allowing retrieval in the original insertion order even after wraparound.

    Args:
        max_length: Maximum number of elements to store in the queue
        shape: Shape of each element (excluding batch dimension). Can be int or tuple
        dtype: Data type of the tensors to store
    """

    def __init__(
        self, max_length: int, shape: Union[int, Iterable[int]] = None, dtype=None
    ):
        super().__init__()
        self.max_length = max_length
        self.register_buffer("pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("filled", torch.zeros((), dtype=torch.bool))
        self.register_buffer("global_counter", torch.zeros((), dtype=torch.long))

        if shape is None:
            # Initialize with placeholder shapes that will be updated on first append
            self.register_buffer("out", torch.zeros((max_length, 1), dtype=dtype))
            self.register_buffer(
                "order_indices", torch.zeros(max_length, dtype=torch.long)
            )
            self.register_buffer("initialized", torch.tensor(False))
        else:
            if type(shape) is int:
                shape = (shape,)
            self.register_buffer(
                "out", torch.zeros((max_length,) + tuple(shape), dtype=dtype)
            )
            self.register_buffer(
                "order_indices", torch.zeros(max_length, dtype=torch.long)
            )
            self.register_buffer("initialized", torch.tensor(True))

    def append(self, item):
        """Append item(s) to the queue with order tracking.

        Args:
            item: Tensor to append. First dimension is batch size.

        Returns:
            Current contents of the queue in insertion order
        """
        if self.max_length == 0:
            return item

        # Initialize buffer with correct shape on first use
        if not self.initialized:
            shape = (self.max_length,) + item.shape[1:]
            new_out = torch.zeros(shape, dtype=item.dtype, device=item.device)
            self.out.resize_(shape)
            self.out.copy_(new_out)
            self.initialized.fill_(True)

        batch_size = item.size(0)

        # Handle case where batch is larger than queue capacity
        if batch_size >= self.max_length:
            # Keep only the last max_length items from the batch
            self.out[:] = item[-self.max_length :]
            self.order_indices[:] = torch.arange(
                self.global_counter + batch_size - self.max_length,
                self.global_counter + batch_size,
                device=self.order_indices.device,
            )
            self.pointer.fill_(0)
            self.filled.fill_(True)
            self.global_counter.add_(batch_size)
        elif self.pointer + batch_size < self.max_length:
            # Simple case: no wraparound
            self.out[self.pointer : self.pointer + batch_size] = item
            # Assign sequential order indices
            self.order_indices[self.pointer : self.pointer + batch_size] = torch.arange(
                self.global_counter,
                self.global_counter + batch_size,
                device=self.order_indices.device,
            )
            self.pointer.add_(batch_size)
            self.global_counter.add_(batch_size)
        else:
            # Wraparound case
            remaining = self.max_length - self.pointer
            self.out[-remaining:] = item[:remaining]
            self.out[: batch_size - remaining] = item[remaining:]

            # Update order indices for wraparound
            self.order_indices[-remaining:] = torch.arange(
                self.global_counter,
                self.global_counter + remaining,
                device=self.order_indices.device,
            )
            self.order_indices[: batch_size - remaining] = torch.arange(
                self.global_counter + remaining,
                self.global_counter + batch_size,
                device=self.order_indices.device,
            )

            self.pointer.copy_(batch_size - remaining)
            self.filled.copy_(True)
            self.global_counter.add_(batch_size)

        return self.get()

    def get(self):
        """Get current contents sorted by insertion order.

        Returns:
            Tensor containing items sorted by their original insertion order
        """
        if self.filled:
            # Sort by order indices and return corresponding data
            sorted_idx = torch.argsort(self.order_indices)
            return self.out[sorted_idx]
        else:
            # Only sort the filled portion
            sorted_idx = torch.argsort(self.order_indices[: self.pointer])
            return self.out[: self.pointer][sorted_idx]

    def get_unsorted(self):
        """Get current contents without sorting (like UnsortedQueue).

        Returns:
            Tensor containing items in buffer order
        """
        return self.out if self.filled else self.out[: self.pointer]

    @staticmethod
    def _test():
        # Test with max_length = 0
        q = OrderedQueue(0)
        for i in range(10):
            v = q.append(torch.Tensor([i]))
            assert v.numel() == 1
            assert v[0] == i

        # Test with max_length = 5
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

        # Test batch append
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

        # Test wraparound preserves order
        batch3 = torch.tensor(
            [[11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]],
            dtype=torch.float32,
        )
        v3 = q.append(batch3)
        assert v3.shape == (10, 2)
        # Should have items 2-11 in order (0-1 were overwritten)
        expected = torch.tensor(
            [
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
                [13, 14],
                [15, 16],
                [17, 18],
                [19, 20],
                [21, 22],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(v3, expected)

        return True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        out_key = prefix + "out"
        if out_key in state_dict:
            self.out.resize_(state_dict[out_key].shape)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.out.resize_(state_dict["out"].shape)
        super().load_state_dict(state_dict, strict, assign)


class EMA(torch.nn.Module):
    """Exponential Moving Average module.

    Maintains an exponential moving average of input tensors.

    Args:
        alpha: Smoothing factor between 0 and 1.
               0 = no update (always return first value)
               1 = no smoothing (always return current value)
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.item = torch.nn.UninitializedBuffer()

    def forward(self, item):
        """Update EMA and return smoothed value.

        Args:
            item: New tensor to incorporate into the average

        Returns:
            Exponentially smoothed tensor
        """
        if self.alpha < 1 and isinstance(self.item, torch.nn.UninitializedBuffer):
            with torch.no_grad():
                self.item.materialize(
                    shape=item.shape, dtype=item.dtype, device=item.device
                )
                self.item.copy_(item, non_blocking=True)
            return item
        elif self.alpha == 1:
            return item
        with torch.no_grad():
            self.item.mul_(1 - self.alpha)
        output = item.mul(self.alpha).add(self.item)
        with torch.no_grad():
            self.item.copy_(output)
        return output

    @staticmethod
    def _test():
        q = EMA(0)
        R = torch.randn(10, 10)
        q(R)
        for i in range(10):
            v = q(torch.randn(10, 10))
            assert torch.allclose(v, R)
        q = EMA(1)
        R = torch.randn(10, 10)
        q(R)
        for i in range(10):
            R = torch.randn(10, 10)
            v = q(R)
            assert torch.allclose(v, R)

        q = EMA(0.5)
        R = torch.randn(10, 10)
        ground = R.detach()
        v = q(R)
        assert torch.allclose(ground, v)
        for i in range(10):
            R = torch.randn(10, 10)
            v = q(R)
            ground = R * 0.5 + ground * 0.5
            assert torch.allclose(v, ground)
        return True
