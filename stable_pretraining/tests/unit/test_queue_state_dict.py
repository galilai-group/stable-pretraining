"""Unit tests for OrderedQueue / UnsortedQueue checkpoint resume (#400).

These tests reproduce the Lightning-style checkpoint resume path — loading a
saved state via a *parent* module's ``load_state_dict``, which recurses via
``_load_from_state_dict`` (not the public ``load_state_dict``). The previous
``load_state_dict`` override on OrderedQueue was never triggered on that path,
so a fresh deferred-shape queue could not accept a saved buffer with a
different shape (e.g. scalar-label data → ``out`` of shape ``(N,)`` instead of
``(N, 1)``).
"""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.callbacks.queues import OrderedQueue, UnsortedQueue


@pytest.mark.unit
class TestOrderedQueueStateDictRoundTrip:
    """End-to-end checkpoint resume tests for OrderedQueue (#400)."""

    def test_scalar_label_buffer_roundtrip_direct(self):
        """Direct ``load_state_dict`` on the queue with 1D items."""
        src = OrderedQueue(max_length=10)
        # Append 1D scalar-label items — this is what triggers the deferred
        # shape resolution to (max_length,) instead of (max_length, 1).
        src.append(torch.tensor([3, 7, 1, 4, 9], dtype=torch.long))
        assert src.out.shape == (10,)

        dst = OrderedQueue(max_length=10)
        assert dst.out.shape == (10, 1)  # deferred placeholder
        dst.load_state_dict(src.state_dict())

        assert dst.out.shape == src.out.shape
        torch.testing.assert_close(dst.get(), src.get())
        assert int(dst.pointer.item()) == int(src.pointer.item())

    def test_scalar_label_buffer_roundtrip_via_parent_module(self):
        """Regression for #400: the Lightning-style recursive path must work.

        Wraps the queue in a parent ``nn.Module``. Calling
        ``parent.load_state_dict`` exercises ``_load_from_state_dict``, which
        is the path the previous override missed.
        """

        class Parent(nn.Module):
            def __init__(self, max_length):
                super().__init__()
                self.queue = OrderedQueue(max_length=max_length)

        src = Parent(max_length=10)
        src.queue.append(torch.tensor([3, 7, 1, 4, 9], dtype=torch.long))
        assert src.queue.out.shape == (10,)

        dst = Parent(max_length=10)
        assert dst.queue.out.shape == (10, 1)
        # This is the path Lightning takes — it would fail pre-fix.
        dst.load_state_dict(src.state_dict())

        assert dst.queue.out.shape == src.queue.out.shape
        torch.testing.assert_close(dst.queue.get(), src.queue.get())

    def test_2d_feature_buffer_roundtrip_via_parent(self):
        """Non-scalar buffer (2D features) — must also round-trip cleanly."""

        class Parent(nn.Module):
            def __init__(self, max_length):
                super().__init__()
                self.queue = OrderedQueue(max_length=max_length)

        src = Parent(max_length=8)
        src.queue.append(torch.arange(20, dtype=torch.float32).reshape(5, 4))
        assert src.queue.out.shape == (8, 4)

        dst = Parent(max_length=8)
        dst.load_state_dict(src.state_dict())

        assert dst.queue.out.shape == src.queue.out.shape
        torch.testing.assert_close(dst.queue.get(), src.queue.get())

    def test_roundtrip_preserves_pointer_and_filled(self):
        """Scalar bookkeeping tensors round-trip exactly.

        ``pointer``, ``filled``, and ``global_counter`` are all 0-dim
        buffers that must be restored verbatim — otherwise resumed runs
        would lose track of insertion order or wraparound state.
        """

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.queue = OrderedQueue(max_length=5, shape=2, dtype=torch.float32)

        src = Parent()
        # Force wraparound: append 7 items into a queue of length 5.
        items = torch.arange(14, dtype=torch.float32).reshape(7, 2)
        src.queue.append(items)
        assert bool(src.queue.filled.item())

        dst = Parent()
        dst.load_state_dict(src.state_dict())

        assert int(dst.queue.pointer.item()) == int(src.queue.pointer.item())
        assert bool(dst.queue.filled.item()) == bool(src.queue.filled.item())
        assert int(dst.queue.global_counter.item()) == int(
            src.queue.global_counter.item()
        )
        torch.testing.assert_close(dst.queue.get(), src.queue.get())

    def test_loading_into_partially_filled_queue_resizes_correctly(self):
        """Loading resizes ``out`` when destination shape differs from source.

        Exercises the override's resize-then-copy logic when a destination
        queue was already initialized to a different shape than the saved
        source — must still succeed without a shape-mismatch error.
        """

        class Parent(nn.Module):
            def __init__(self, max_length):
                super().__init__()
                self.queue = OrderedQueue(max_length=max_length)

        # src has 1D scalar labels
        src = Parent(max_length=10)
        src.queue.append(torch.tensor([1, 2, 3], dtype=torch.long))

        # dst was initialized with a DIFFERENT shape (2D), then we load src
        dst = Parent(max_length=10)
        dst.queue.append(torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.long))
        assert dst.queue.out.shape == (10, 3)  # got initialized to 2D

        # Loading 1D src into 2D dst — must resize.
        dst.load_state_dict(src.state_dict())
        assert dst.queue.out.shape == src.queue.out.shape == (10,)
        torch.testing.assert_close(dst.queue.get(), src.queue.get())


@pytest.mark.unit
class TestUnsortedQueueStateDictRoundTrip:
    """Same fix applied to UnsortedQueue for parity (#400)."""

    def test_scalar_label_buffer_roundtrip_via_parent_module(self):
        class Parent(nn.Module):
            def __init__(self, max_length):
                super().__init__()
                self.queue = UnsortedQueue(max_length=max_length)

        src = Parent(max_length=10)
        src.queue.append(torch.tensor([3, 7, 1, 4, 9], dtype=torch.long))
        assert src.queue.out.shape == (10,)

        dst = Parent(max_length=10)
        assert dst.queue.out.shape == (10, 1)
        dst.load_state_dict(src.state_dict())

        assert dst.queue.out.shape == src.queue.out.shape
        torch.testing.assert_close(dst.queue.get(), src.queue.get())

    def test_2d_feature_buffer_roundtrip_via_parent(self):
        class Parent(nn.Module):
            def __init__(self, max_length):
                super().__init__()
                self.queue = UnsortedQueue(max_length=max_length)

        src = Parent(max_length=8)
        src.queue.append(torch.arange(20, dtype=torch.float32).reshape(5, 4))

        dst = Parent(max_length=8)
        dst.load_state_dict(src.state_dict())

        torch.testing.assert_close(dst.queue.get(), src.queue.get())
        assert int(dst.queue.pointer.item()) == int(src.queue.pointer.item())
