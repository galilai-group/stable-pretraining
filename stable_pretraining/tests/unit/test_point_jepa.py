"""Point-JEPA point-cloud primitives and latent-prediction contracts."""

import torch

from stable_pretraining.backbone.point_cloud import (
    PointJEPAMasking,
    PointSequencer,
    PointTokenizer,
)
from stable_pretraining.methods.point_jepa import PointJEPA
from stable_pretraining.methods.point_jepa import (
    _PointJEPAEMA,
    _reference_lr_multiplier,
)
from stable_pretraining.data.point_cloud import PointCloudTransform


def test_tokenizer_and_sequencer():
    """Grouping is local and ordered token gradients reach PointNet."""
    tokenizer = PointTokenizer(num_groups=8, group_size=6, embed_dim=24)
    points = torch.randn(2, 32, 3)
    groups, centers = tokenizer.group(points)
    assert groups.shape == (2, 8, 6, 3)
    assert torch.allclose(groups.norm(dim=-1).min(-1).values, torch.zeros(2, 8))
    tokens, centers = tokenizer(points)
    tokens.retain_grad()
    ordered, ordered_centers = PointSequencer()(tokens, centers)
    assert ordered.shape == (2, 8, 24)
    assert torch.equal(
        ordered_centers[:, 0], centers[torch.arange(2), centers.sum(-1).argmin(-1)]
    )
    assert torch.allclose(
        torch.sort(ordered_centers, dim=1).values, torch.sort(centers, dim=1).values
    )
    ordered.sum().backward()
    assert tokenizer.pointnet_local_mlp[0].weight.grad is not None


def test_pointnet_patch_permutation(monkeypatch):
    """Mini PointNet pooling ignores point order."""
    tokenizer = PointTokenizer(num_groups=2, group_size=6, embed_dim=24).eval()
    groups = torch.randn(2, 2, 6, 3)
    centers = torch.randn(2, 2, 3)
    monkeypatch.setattr(tokenizer, "group", lambda points: (groups, centers))
    tokens, _ = tokenizer(torch.empty(0))
    permutation = torch.randperm(6)
    monkeypatch.setattr(
        tokenizer, "group", lambda points: (groups[:, :, permutation], centers)
    )
    permuted_tokens, _ = tokenizer(torch.empty(0))
    torch.testing.assert_close(tokens, permuted_tokens)


def test_masking_excludes_targets():
    """Four target blocks are contiguous and none is visible in context."""
    torch.manual_seed(7)
    masking = PointJEPAMasking()
    for _ in range(20):
        mask = masking(64, torch.device("cpu"))
        assert mask.target_idx.shape[0] == 4
        assert 9 <= mask.target_idx.shape[1] <= 12
        assert torch.all(mask.target_idx[:, 1:] - mask.target_idx[:, :-1] == 1)
        assert not torch.isin(mask.context_idx, mask.target_idx).any()
        remaining = 64 - mask.target_idx.unique().numel()
        assert int(remaining * 0.40) <= len(mask.context_idx) <= int(remaining * 0.75)


def test_point_jepa_forward_backward():
    """Student, tokenizer, and predictor train while the teacher stays frozen."""
    model = PointJEPA(
        num_groups=8,
        group_size=8,
        embed_dim=48,
        encoder_depth=2,
        encoder_heads=3,
        predictor_dim=24,
        predictor_depth=1,
        predictor_heads=3,
    )
    masks, predictor_inputs, student_inputs, teacher_inputs, sequences = (
        [],
        [],
        [],
        [],
        [],
    )
    handles = [
        model.sequencer.register_forward_hook(
            lambda module, args, output: sequences.append(output)
        ),
        model.masking.register_forward_hook(
            lambda module, args, output: masks.append(output)
        ),
        model.predictor.register_forward_pre_hook(
            lambda module, args, kwargs: predictor_inputs.append(kwargs),
            with_kwargs=True,
        ),
        model.encoder.student.register_forward_pre_hook(
            lambda module, args: student_inputs.append(args)
        ),
        model.encoder.teacher.register_forward_pre_hook(
            lambda module, args: teacher_inputs.append(args)
        ),
    ]
    output = model({"points": torch.randn(2, 64, 3)}, "fit")
    for handle in handles:
        handle.remove()
    mask = masks[0]
    assert len(student_inputs) == len(teacher_inputs) == 1
    assert teacher_inputs[0][0].shape == (2, 8, 48)
    torch.testing.assert_close(
        student_inputs[0][0], teacher_inputs[0][0][:, mask.context_idx]
    )
    torch.testing.assert_close(
        student_inputs[0][1], teacher_inputs[0][1][:, mask.context_idx]
    )
    assert len(predictor_inputs) == 4
    for inputs, target_idx in zip(predictor_inputs, mask.target_idx):
        assert inputs["queries"].shape == (2, target_idx.numel(), 48)
        assert torch.count_nonzero(inputs["queries"]) == 0
        assert inputs["query_mask"].all()
        torch.testing.assert_close(
            inputs["query_pos"],
            model.predictor_position(sequences[0][1][:, target_idx]),
        )
    assert output["predictions"].shape == output["targets"].shape
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.tokenizer.pointnet_local_mlp[0].weight.grad is not None
    assert model.encoder.student.blocks[0].attn.qkv.weight.grad is not None
    assert model.predictor.output_proj.weight.grad is not None
    assert model.encoder_position[0].weight.grad is not None
    assert not hasattr(model.encoder.student, "position")
    assert not hasattr(model.encoder.teacher, "position")
    with torch.no_grad():
        raw_teacher = model.encoder.forward_teacher(*teacher_inputs[0], normalize=False)
        expected_targets = torch.nn.functional.layer_norm(raw_teacher, (48,))[
            :, mask.target_idx
        ]
    torch.testing.assert_close(output["targets"], expected_targets)
    assert all(
        parameter.grad is None for parameter in model.encoder.teacher.parameters()
    )
    model.eval()
    assert model({"points": torch.randn(2, 64, 3)}, "test")["embedding"].shape == (
        2,
        96,
    )


def test_reference_ema_updates_and_resume():
    """Initial copies and subsequent linear EMA survive state restoration."""
    student = torch.nn.Linear(3, 4)
    wrapper = _PointJEPAEMA(student, 0.9995, 0.99999)
    wrapper.ema_total_updates.fill_(10)
    for _ in range(2):
        with torch.no_grad():
            student.weight.add_(1)
        wrapper.update_ema_coefficient(0, 500)
        wrapper.update_teacher()
        torch.testing.assert_close(wrapper.teacher.weight, student.weight)
    previous = wrapper.teacher.weight.detach().clone()
    with torch.no_grad():
        student.weight.add_(1)
    coefficient = 0.9995 + (0.99999 - 0.9995) * 2 / 10
    wrapper.update_ema_coefficient(0, 500)
    wrapper.update_teacher()
    torch.testing.assert_close(
        wrapper.teacher.weight,
        previous * coefficient + student.weight * (1 - coefficient),
    )
    restored = _PointJEPAEMA(torch.nn.Linear(3, 4), 0.9995, 0.99999)
    restored.load_state_dict(wrapper.state_dict())
    assert restored.ema_updates.item() == 3
    assert restored.ema_total_updates.item() == 10
    restored.update_ema_coefficient(400, 500)
    assert (
        abs(restored.ema_coefficient.item() - (0.9995 + (0.99999 - 0.9995) * 3 / 10))
        < 1e-7
    )


def test_reference_lr_epoch_boundaries():
    """Manual optimizer updates preserve the reference's epoch LR values."""
    settings = dict(updates_per_epoch=5, max_epochs=500, warmup_epochs=30, base_lr=1e-3)
    for step in range(5):
        assert abs(1e-3 * _reference_lr_multiplier(step, **settings) - 1e-5) < 1e-12
    for epoch, lr in ((29, 1e-3), (30, 1e-3), (500, 1e-6)):
        assert abs(1e-3 * _reference_lr_multiplier(epoch * 5, **settings) - lr) < 1e-12


def test_batched_point_transform_fps_and_shared_augmentation():
    """FPS selects input points and identical clouds share batch augmentation."""
    cloud = torch.randn(128, 3)
    repeated = torch.stack([cloud, cloud])
    torch.manual_seed(7)
    batch = PointCloudTransform(128).collate([{"points": cloud}, {"points": cloud}])
    torch.testing.assert_close(batch["points"][0], batch["points"][1])
    torch.testing.assert_close(
        batch["points"].mean(1), torch.zeros(2, 3), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        batch["points"].norm(dim=-1).max(1).values, torch.ones(2)
    )
    from stable_pretraining.backbone.point_cloud import farthest_point_indices

    indices = farthest_point_indices(repeated, 32)
    assert indices.shape == (2, 32)
    assert indices[0].unique().numel() == 32
    assert indices[1].unique().numel() == 32
