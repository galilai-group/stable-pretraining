"""Tests for the PCL warm-up model."""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.methods.pcl import PCL


class TinyEncoder(nn.Module):
    """Small encoder for fast CPU tests."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(3, 16)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.linear(x)


def make_model():
    return PCL(
        encoder_name=TinyEncoder(),
        projector_dims=(32, 8),
        queue_length=16,
        temperature=0.2,
    )


@pytest.mark.unit
def test_pcl_warmup_forward_and_backward():
    torch.manual_seed(7)
    model = make_model()
    model.train()

    # Populate the negative queue before testing gradients.
    initial_keys = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
    model.queue.append(initial_keys)

    view1 = torch.randn(4, 3, 8, 8)
    view2 = torch.randn(4, 3, 8, 8)

    output = model(view1, view2)

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.queries.shape == (4, 8)
    assert output.keys.shape == (4, 8)
    assert model.queue.get().shape[0] == 12

    output.loss.backward()

    student_grads = [p.grad for p in model.backbone.student.parameters()]

    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        for g in student_grads
    )

    assert all(p.grad is None for p in model.backbone.teacher.parameters())

    assert all(p.grad is None for p in model.projector.teacher.parameters())


@pytest.mark.unit
def test_pcl_inference_does_not_modify_queue():
    model = make_model()
    model.eval()

    images = torch.randn(2, 3, 8, 8)
    old_queue = model.queue.get().clone()

    output = model(images)

    assert output.embedding.shape == (2, 16)
    assert output.embedding.requires_grad is False
    torch.testing.assert_close(model.queue.get(), old_queue)


@pytest.mark.unit
def test_pcl_empty_queue_warmup():
    model = make_model()
    model.train()

    output = model(
        torch.randn(4, 3, 8, 8),
        torch.randn(4, 3, 8, 8),
    )

    assert torch.isfinite(output.loss)
    torch.testing.assert_close(
        output.loss,
        torch.tensor(0.0),
    )
    assert model.queue.get().shape[0] == 4


@pytest.mark.unit
def test_pcl_combines_instance_and_prototype_losses():
    torch.manual_seed(7)

    model = make_model()
    model.train()

    # Populate the negative queue.
    initial_keys = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
    model.queue.append(initial_keys)

    # Five dataset samples, assigned to five clusters.
    assignments = torch.tensor([0, 1, 2, 3, 4])

    prototypes = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)

    concentrations = torch.full((5,), 0.2)

    model.set_clustering(
        assignments,
        prototypes,
        concentrations,
    )

    # Batch samples 0 and 2.
    indices = torch.tensor([0, 2])

    output = model(
        torch.randn(2, 3, 8, 8),
        torch.randn(2, 3, 8, 8),
        indices=indices,
    )

    assert output.prototype_loss is not None
    assert torch.isfinite(output.prototype_loss)
    assert torch.isfinite(output.instance_loss)

    torch.testing.assert_close(
        output.loss,
        output.instance_loss + output.prototype_loss,
    )

    output.loss.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.student.parameters()
    )


@pytest.mark.unit
def test_pcl_optimizer_and_ema_update():
    """Verify student optimization and teacher EMA updates."""
    torch.manual_seed(7)

    model = make_model()
    model.train()

    # Use a populated queue to obtain nonzero contrastive gradients.
    keys = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
    model.queue.append(keys)

    optimizer = torch.optim.SGD(
        list(model.backbone.student.parameters())
        + list(model.projector.student.parameters()),
        lr=0.1,
    )

    # Save parameters before training.
    student_before = [p.detach().clone() for p in model.backbone.student.parameters()]
    teacher_before = [p.detach().clone() for p in model.backbone.teacher.parameters()]

    view1 = torch.randn(4, 3, 8, 8)
    view2 = torch.randn(4, 3, 8, 8)

    optimizer.zero_grad()
    output = model(view1, view2)
    output.loss.backward()
    optimizer.step()

    # Student should change after gradient descent.
    student_after = list(model.backbone.student.parameters())

    assert any(
        not torch.equal(before, after)
        for before, after in zip(student_before, student_after)
    )

    # Teacher should not be modified by optimizer.step().
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            teacher_before,
            model.backbone.teacher.parameters(),
        )
    )

    # Simulate the EMA update normally triggered by
    # TeacherStudentCallback after an optimizer step.
    model.backbone.update_teacher()
    model.projector.update_teacher()

    # Teacher should now change through EMA.
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            teacher_before,
            model.backbone.teacher.parameters(),
        )
    )

    assert all(p.grad is None for p in model.backbone.teacher.parameters())


@pytest.mark.unit
def test_pcl_dictionary_training_interface():
    torch.manual_seed(7)
    model = make_model()
    model.train()

    model.queue.append(torch.nn.functional.normalize(torch.randn(8, 8), dim=-1))

    batch = {
        "view1": torch.randn(4, 3, 8, 8),
        "view2": torch.randn(4, 3, 8, 8),
        "indices": torch.arange(4),
    }

    state = model(batch, stage="fit")

    assert isinstance(state, dict)
    assert torch.isfinite(state["loss"])
    assert state["loss"].ndim == 0

    state["loss"].backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.student.parameters()
    )


@pytest.mark.unit
def test_pcl_stores_multiple_clusterings():
    model = make_model()

    assignments1 = torch.tensor([0, 1, 2, 3])
    prototypes1 = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    phi1 = torch.full((4,), 0.2)

    assignments2 = torch.tensor([0, 1, 2, 3])
    prototypes2 = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    phi2 = torch.full((6,), 0.3)

    model.set_clusterings(
        [
            (assignments1, prototypes1, phi1),
            (assignments2, prototypes2, phi2),
        ]
    )

    torch.testing.assert_close(model.cluster_assignments, assignments1)
    assert len(model.additional_clusterings) == 1

    second = model.additional_clusterings[0]

    torch.testing.assert_close(second.assignments, assignments2)
    torch.testing.assert_close(second.prototypes, prototypes2)
    torch.testing.assert_close(second.concentrations, phi2)


@pytest.mark.unit
def test_pcl_multiresolution_loss_average():
    torch.manual_seed(7)

    model = make_model()
    model.num_proto_negatives = 1
    model.train()

    model.queue.append(torch.nn.functional.normalize(torch.randn(8, 8), dim=-1))

    assignments1 = torch.tensor([0, 1, 2, 3])
    assignments2 = torch.tensor([0, 2, 4, 1])

    prototypes1 = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    prototypes2 = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)

    phi1 = torch.full((5,), 0.2)
    phi2 = torch.full((6,), 0.3)

    model.set_clusterings(
        [
            (assignments1, prototypes1, phi1),
            (assignments2, prototypes2, phi2),
        ]
    )

    output = model(
        torch.randn(2, 3, 8, 8),
        torch.randn(2, 3, 8, 8),
        indices=torch.tensor([0, 1]),
    )

    assert output.prototype_loss is not None
    assert torch.isfinite(output.prototype_loss)
    torch.testing.assert_close(
        output.loss,
        output.instance_loss + output.prototype_loss,
    )

    output.loss.backward()
    assert any(p.grad is not None for p in model.backbone.student.parameters())


@pytest.mark.unit
def test_pcl_multiresolution_matches_mean_of_individual_losses():
    """Multi-resolution ProtoNCE equals the mean of individual losses."""
    import copy

    torch.manual_seed(7)

    base = make_model()
    base.eval()  # Avoid queue updates during forward passes.
    base.num_proto_negatives = 1

    assignments1 = torch.tensor([0, 1, 2, 3])
    assignments2 = torch.tensor([0, 2, 4, 1])

    prototypes1 = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    prototypes2 = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)

    phi1 = torch.full((5,), 0.2)
    phi2 = torch.full((6,), 0.3)

    model1 = copy.deepcopy(base)
    model2 = copy.deepcopy(base)
    combined = copy.deepcopy(base)

    model1.set_clustering(assignments1, prototypes1, phi1)
    model2.set_clustering(assignments2, prototypes2, phi2)
    combined.set_clusterings(
        [
            (assignments1, prototypes1, phi1),
            (assignments2, prototypes2, phi2),
        ]
    )

    view1 = torch.randn(2, 3, 8, 8)
    view2 = torch.randn(2, 3, 8, 8)
    indices = torch.tensor([0, 1])

    torch.manual_seed(123)
    out1 = model1(view1, view2, indices=indices)

    torch.manual_seed(123)
    out2 = model2(view1, view2, indices=indices)

    torch.manual_seed(123)
    out_combined = combined(view1, view2, indices=indices)

    expected = (out1.prototype_loss + out2.prototype_loss) / 2

    torch.testing.assert_close(
        out_combined.prototype_loss,
        expected,
    )


@pytest.mark.unit
def test_pcl_clustering_state_dict_roundtrip():
    """Check whether clustering state survives model serialization."""
    import io

    torch.manual_seed(7)
    model = make_model()

    assignments1 = torch.tensor([0, 1, 2, 3])
    assignments2 = torch.tensor([0, 2, 4, 1])

    prototypes1 = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    prototypes2 = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)

    phi1 = torch.full((5,), 0.2)
    phi2 = torch.full((6,), 0.3)

    model.set_clusterings(
        [
            (assignments1, prototypes1, phi1),
            (assignments2, prototypes2, phi2),
        ]
    )

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    restored = make_model()
    state = torch.load(buffer, weights_only=True)

    restored.load_state_dict(state, strict=True)

    torch.testing.assert_close(
        restored.cluster_assignments,
        model.cluster_assignments,
    )
    torch.testing.assert_close(
        restored.cluster_prototypes,
        model.cluster_prototypes,
    )
    torch.testing.assert_close(
        restored.cluster_concentrations,
        model.cluster_concentrations,
    )

    assert len(restored.additional_clusterings) == 1
    torch.testing.assert_close(
        restored.additional_clusterings[0].prototypes,
        model.additional_clusterings[0].prototypes,
    )


@pytest.mark.unit
def test_pcl_queue_excludes_current_batch_keys():
    """Current keys must not be used as their own negatives."""
    torch.manual_seed(7)
    model = make_model()
    model.train()

    # Queue starts empty.
    assert model.queue.get().shape[0] == 0

    view1 = torch.randn(4, 3, 8, 8)
    view2 = torch.randn(4, 3, 8, 8)

    first = model(view1, view2)

    # No previous negatives: first loss is zero.
    torch.testing.assert_close(
        first.instance_loss,
        torch.zeros_like(first.instance_loss),
    )

    # Current keys are added only after computing the loss.
    assert model.queue.get().shape[0] == 4
    torch.testing.assert_close(
        model.queue.get(),
        first.keys,
    )

    # Second batch can use the stored keys as negatives.
    second = model(
        torch.randn(4, 3, 8, 8),
        torch.randn(4, 3, 8, 8),
    )

    assert second.instance_loss.item() > 0
    assert model.queue.get().shape[0] == 8
