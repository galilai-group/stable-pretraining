"""DINO asynchronous centers match explicit means and remain constant until applied."""

import pytest
import torch
import torch.distributed as dist

from stable_pretraining.losses.dino import DINOv1Loss, DINOv2Loss

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("distributed", [False, True])
def test_center_updates_are_deferred_idempotent_and_match_ema(tmp_path, distributed):
    if distributed:
        dist.init_process_group(
            "gloo", init_method=(tmp_path / "group").as_uri(), world_size=1, rank=0
        )
    try:
        loss = DINOv1Loss(center_momentum=0.75)
        teacher = torch.arange(24.0).reshape(2, 3, 4).requires_grad_()
        loss.update_center(teacher)
        assert loss.center is None and not loss.updated
        uncentered = loss.softmax_center_teacher(
            teacher, teacher_temp=0.2, update_centers=False
        )
        torch.testing.assert_close(uncentered, torch.softmax(teacher / 0.2, -1))
        centered = loss.softmax_center_teacher(teacher, teacher_temp=0.2)
        expected_center = teacher.detach().mean((0, 1), keepdim=False)[None]
        torch.testing.assert_close(loss.center, expected_center)
        torch.testing.assert_close(
            centered, torch.softmax((teacher - expected_center) / 0.2, -1)
        )
        assert not centered.requires_grad and not loss.center.requires_grad
        loss.apply_center_update()
        torch.testing.assert_close(loss.center, expected_center)
        loss.update_center(teacher + 4)
        loss.apply_center_update()
        torch.testing.assert_close(loss.center, expected_center + 1)
        assert loss.updated
        flat = torch.tensor([[0.2, 0.5], [0.8, 0.4], [0.1, 0.7], [0.6, 0.3]])
        probs = loss.sinkhorn_knopp_teacher(flat, teacher_temp=1.0, n_iterations=20)
        torch.testing.assert_close(probs.sum(1), torch.ones(4))
        torch.testing.assert_close(
            probs.mean(0), torch.full((2,), 0.5), atol=1e-5, rtol=1e-5
        )
    finally:
        if distributed:
            dist.destroy_process_group()


def test_dinov2_without_masked_patches_matches_cls_loss_and_gradients():
    loss = DINOv2Loss(dino_loss_weight=0.3)
    student = torch.randn(3, 2, 4, requires_grad=True)
    teacher = torch.softmax(torch.randn(2, 2, 4), -1)
    actual = loss(student, teacher)
    expected = loss.dino_loss(student, teacher) * 0.3
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0
