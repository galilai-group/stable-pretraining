"""Reference parity and state/gradient contracts for All4One components."""

import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from stable_pretraining.utils.all4one import (
    All4OneAttention,
    All4OneQueue,
    build_all4one_predictor,
    build_all4one_projector,
)

pytestmark = pytest.mark.unit


def test_queue_starts_full_with_normalized_random_vectors():
    torch.manual_seed(13)
    expected = F.normalize(torch.randn(11, 8), dim=-1)
    torch.manual_seed(13)
    queue = All4OneQueue(8, queue_size=11)
    torch.testing.assert_close(queue.queue.get(), expected)
    assert queue.queue.pointer.item() == 0
    assert queue.queue.filled.item()
    assert not list(queue.parameters())


def test_queue_returns_sorted_cosine_neighbours_without_mutation():
    queue = All4OneQueue(2, queue_size=5, num_neighbours=3)
    support = torch.tensor([[1.0, 0], [0, 1], [-1, 0], [1, 1], [-1, -1]])
    queue.enqueue(support)
    state = copy.deepcopy(queue.state_dict())
    query = torch.tensor([[8.0, 1.0], [-1.0, 3.0]], requires_grad=True)
    result = queue(query)
    assert result["indices"].tolist() == [[0, 3, 1], [1, 3, 2]]
    expected = F.normalize(support, dim=-1)[result["indices"]]
    torch.testing.assert_close(result["neighbours"], expected)
    assert not result["neighbours"].requires_grad
    for name, value in queue.state_dict().items():
        torch.testing.assert_close(value, state[name])
    queue.enqueue(torch.randn(5, 2))
    torch.testing.assert_close(result["neighbours"], expected)


@pytest.mark.parametrize("sizes", [(2, 2, 3), (5, 1, 4), (7, 2), (0, 3)])
def test_queue_reuses_fifo_wraparound_and_oversized_batch_handling(sizes):
    queue = All4OneQueue(3, queue_size=5, num_neighbours=2)
    history = list(queue.queue.get().clone())
    for size in sizes:
        batch = torch.randn(size, 3, requires_grad=True)
        history.extend(F.normalize(batch.detach(), dim=-1))
        history = history[-5:]
        queue.enqueue(batch)
        pointer = int(queue.queue.pointer)
        chronological = torch.roll(queue.queue.get(), shifts=-pointer, dims=0)
        torch.testing.assert_close(chronological, torch.stack(history))
        assert not queue.queue.out.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_queue_dtype_and_autocast_contract(dtype):
    queue = All4OneQueue(8, 11, 3)
    query = torch.randn(3, 8).to(dtype)
    baseline = queue(query)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = queue(query)
        queue.enqueue(query)
    assert result["neighbours"].dtype == dtype
    torch.testing.assert_close(result["indices"], baseline["indices"])
    torch.testing.assert_close(queue.queue.out[:3], F.normalize(query.float(), dim=-1))
    assert queue.queue.out.dtype == torch.float32


def _reference_position(k, dim):
    denominator = 2 * math.ceil(k / 2)
    return torch.tensor(
        [
            [
                (math.sin if channel % 2 == 0 else math.cos)(
                    position / 10000 ** (2 * (channel // 2) / denominator)
                )
                for position in range(dim)
            ]
            for channel in range(k)
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


@pytest.mark.parametrize("k", [1, 4, 5])
@pytest.mark.parametrize("shifted", [False, True])
def test_attention_matches_reference_encoding_shift_and_first_token(k, shifted):
    attention = All4OneAttention(8, k, num_heads=2, num_layers=2, dropout=0)
    reference = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            8, 2, dim_feedforward=16, dropout=0, batch_first=True
        ),
        num_layers=2,
    )
    reference.load_state_dict(attention.encoder.state_dict())
    neighbours = torch.randn(3, k, 8)
    prediction = torch.randn(3, 8, requires_grad=True)
    reference_prediction = prediction.detach().clone().requires_grad_()
    tokens = neighbours
    if shifted:
        tokens = torch.cat((reference_prediction[:, None], neighbours), dim=1)[:, :k]
    expected = reference(tokens + _reference_position(k, 8))[:, 0]
    actual = attention(neighbours, prediction if shifted else None)
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    for parameter, ref_parameter in zip(
        attention.encoder.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            parameter.grad, ref_parameter.grad, rtol=1e-4, atol=2e-6
        )
    if shifted:
        torch.testing.assert_close(prediction.grad, reference_prediction.grad)
        assert prediction.grad.abs().sum() > 0
    else:
        assert prediction.grad is None


@pytest.mark.parametrize("kind", ["projector", "predictor"])
def test_mlp_builders_match_reference_outputs_gradients_and_batchnorm(kind):
    if kind == "projector":
        actual = build_all4one_projector(6, 12, 8)
        expected = nn.Sequential(
            nn.Linear(6, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
        )
        inputs = torch.randn(4, 6)
    else:
        actual = build_all4one_predictor(8, 12)
        expected = nn.Sequential(
            nn.Linear(8, 12), nn.BatchNorm1d(12), nn.ReLU(), nn.Linear(12, 8)
        )
        inputs = torch.randn(4, 8)
    with torch.no_grad():
        for parameter, ref_parameter in zip(
            actual.parameters(), expected.parameters(), strict=True
        ):
            ref_parameter.copy_(parameter)
    result = actual(inputs)
    reference = expected(inputs)
    torch.testing.assert_close(result, reference)
    weight = torch.randn_like(result)
    (result * weight).sum().backward()
    (reference * weight).sum().backward()
    for parameter, ref_parameter in zip(
        actual.parameters(), expected.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)
    for buffer, ref_buffer in zip(actual.buffers(), expected.buffers(), strict=True):
        torch.testing.assert_close(buffer, ref_buffer)
