import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses.sce import SCELoss


@pytest.fixture
def representations():
    torch.manual_seed(42)

    queries = torch.randn(8, 128, requires_grad=True)
    keys = torch.randn(8, 128)
    queue = torch.randn(100, 128)

    return queries, keys, queue

#make sure loss returns finite scalar
def test_sce_loss_is_finite(representations):
    queries, keys, queue = representations

    loss = SCELoss()(queries, keys, queue)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

#make sure gradients go back
def test_sce_backward(representations):
    queries, keys, queue = representations

    loss = SCELoss()(queries, keys, queue)
    loss.backward()

    assert queries.grad is not None
    assert torch.isfinite(queries.grad).all()
    assert queries.grad.abs().sum() > 0

#reduces back to infonce when disabaled
def test_sce_lambda_one_matches_infonce(representations):
    queries, keys, queue = representations

    sce_loss = SCELoss(lambda_weight=1.0)(
        queries, keys, queue
    )

    q = F.normalize(queries, dim=-1)
    k = F.normalize(keys, dim=-1)
    memory = F.normalize(queue, dim=-1)

    positives = (q * k).sum(dim=-1, keepdim=True)
    negatives = q @ memory.T

    logits = torch.cat([positives, negatives], dim=1) / 0.2

    labels = torch.zeros(
        queries.shape[0],
        dtype=torch.long,
        device=queries.device,
    )

    reference_loss = F.cross_entropy(logits, labels)

    torch.testing.assert_close(sce_loss, reference_loss)

#invalid hyperparameters
@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0},
        {"temperature": -0.1},
        {"target_temperature": 0},
        {"target_temperature": -0.1},
        {"lambda_weight": -0.1},
        {"lambda_weight": 1.1},
    ],
)
def test_sce_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        SCELoss(**kwargs)

#verify changing lambda changes loss
def test_sce_lambda_changes_loss(representations):
    queries, keys, queue = representations

    hard_loss = SCELoss(lambda_weight=1.0)(
        queries, keys, queue
    )

    soft_loss = SCELoss(lambda_weight=0.0)(
        queries, keys, queue
    )

    assert not torch.allclose(hard_loss, soft_loss)

def test_sce_teacher_gradients_detached():
    queries = torch.randn(4, 16, requires_grad=True)
    keys = torch.randn(4, 16, requires_grad=True)
    queue = torch.randn(10, 16, requires_grad=True)

    loss = SCELoss()(
        queries,
        keys.detach(),
        queue.detach(),
    )

    loss.backward()

    assert queries.grad is not None
    assert keys.grad is None
    assert queue.grad is None

#for valid probability distribution
def test_sce_soft_targets_sum_to_one(representations):
    _, keys, queue = representations

    keys = F.normalize(keys, dim=-1)
    queue = F.normalize(queue, dim=-1)

    teacher_logits = torch.cat(
        [
            torch.zeros(keys.shape[0], 1),
            keys @ queue.T,
        ],
        dim=1,
    )

    relational_probs = F.softmax(
        teacher_logits / 0.05,
        dim=1,
    )

    hard_targets = torch.zeros_like(relational_probs)
    hard_targets[:, 0] = 1.0

    lam = 0.5

    targets = (
        lam * hard_targets
        + (1 - lam) * relational_probs
    )

    torch.testing.assert_close(
        targets.sum(dim=1),
        torch.ones(keys.shape[0]),
    )

def test_sce_matches_manual_calculation():
    import torch.nn.functional as F

    queries = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[1.0, 0.0]])
    queue = torch.tensor([
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    temperature = 0.2
    target_temperature = 0.5
    lambda_weight = 0.5

    loss_fn = SCELoss(
        temperature=temperature,
        target_temperature=target_temperature,
        lambda_weight=lambda_weight,
    )

    actual_loss = loss_fn(queries, keys, queue)

    # Student: positive similarity followed by queue similarities
    student_logits = torch.tensor([[1.0, 0.0, -1.0]]) / temperature

    # Teacher: zero positive logit followed by queue similarities
    teacher_logits = torch.tensor([[0.0, 0.0, -1.0]]) / target_temperature
    relational_probs = F.softmax(teacher_logits, dim=1)

    hard_targets = torch.tensor([[1.0, 0.0, 0.0]])

    targets = (
        lambda_weight * hard_targets
        + (1.0 - lambda_weight) * relational_probs
    )

    expected_loss = -(
        targets * F.log_softmax(student_logits, dim=1)
    ).sum(dim=1).mean()

    assert torch.allclose(actual_loss, expected_loss, atol=1e-6)