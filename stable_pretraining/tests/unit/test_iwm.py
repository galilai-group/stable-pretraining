"""Unit tests for IWM-specific behavior."""

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.methods.iwm import IWM

pytestmark = pytest.mark.unit

B, ACTION_DIM = 2, 7


def _tiny_vit(img_size: int = 64):
    import timm

    return timm.create_model(
        "vit_tiny_patch16_224",
        num_classes=0,
        pretrained=False,
        img_size=img_size,
    )


def _model():
    torch.manual_seed(0)
    return IWM(
        model_or_model_name=_tiny_vit(),
        predictor_embed_dim=32,
        predictor_depth=1,
        action_dim=ACTION_DIM,
        num_targets=2,
    ).train()


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, 3, 64, 64, generator=g),
        torch.randn(B, 3, 64, 64, generator=g),
        torch.randn(B, ACTION_DIM, generator=g),
    )


def _forward(model, source, target, action, mask_seed: int = 0, **kwargs):
    """Run a training forward with a fixed mask, so outputs are comparable."""
    torch.manual_seed(mask_seed)
    return model(source, target, action, **kwargs)


def test_iwm_context_is_every_patch_outside_the_masked_blocks():
    model = _model()
    patches = torch.randn(B, 16, 192)

    mask_out = model.masking(patches, grid_h=4, grid_w=4)
    context = set(mask_out.context_idx[0].tolist())
    masked = set(mask_out.target_idx[0].tolist())

    assert masked
    assert context.isdisjoint(masked)
    assert context | masked == set(range(16))


def test_iwm_loss_is_mse_to_layer_normed_targets():
    model = _model()
    output = _forward(model, *_inputs())

    assert output.predictions.shape == output.targets.shape
    assert torch.allclose(output.loss, F.mse_loss(output.predictions, output.targets))
    assert torch.allclose(
        output.targets.mean(-1), torch.zeros_like(output.targets[..., 0]), atol=1e-5
    )


def test_iwm_targets_come_from_the_target_view():
    model = _model()
    source, target, action = _inputs()
    other_source, other_target, _ = _inputs(seed=1)

    base = _forward(model, source, target, action)
    new_target = _forward(model, source, other_target, action)
    new_source = _forward(model, other_source, target, action)

    # The teacher only sees the target; the student and predictor only the source
    assert not torch.allclose(base.targets, new_target.targets)
    assert torch.allclose(base.predictions, new_target.predictions)
    assert torch.allclose(base.targets, new_source.targets)
    assert not torch.allclose(base.predictions, new_source.predictions)


def test_iwm_predictions_are_conditioned_on_the_action():
    model = _model()
    source, target, action = _inputs()

    output = _forward(model, source, target, action)
    zero_action = _forward(model, source, target, torch.zeros_like(action))
    assert not torch.allclose(output.predictions, zero_action.predictions)

    output.loss.backward()
    action_mlp_grads = [p.grad for p in model.predictor.query_cond_net.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in action_mlp_grads)


def test_iwm_targets_do_not_depend_on_the_student():
    model = _model()
    base = _forward(model, *_inputs())
    assert not base.targets.requires_grad

    with torch.no_grad():
        for p in model.encoder.student.parameters():
            p.add_(0.1)
    perturbed = _forward(model, *_inputs())

    assert torch.allclose(base.targets, perturbed.targets)
    assert not torch.allclose(base.predictions, perturbed.predictions)


@pytest.mark.parametrize("embedding_source", ["teacher", "student"])
def test_iwm_embedding_encodes_the_target_view(embedding_source):
    model = _model()
    source, target, action = _inputs()
    other_source, _, _ = _inputs(seed=1)

    base = _forward(model, source, target, action, embedding_source=embedding_source)
    new_source = _forward(
        model, other_source, target, action, embedding_source=embedding_source
    )

    assert base.embedding.shape == (B, 16, 192)
    assert torch.allclose(base.embedding, new_source.embedding)


def test_iwm_rejects_invalid_embedding_source():
    model = _model()
    with pytest.raises(ValueError, match="embedding_source"):
        model(*_inputs(), embedding_source="both")


@pytest.mark.parametrize("missing", ["target", "action"])
def test_iwm_training_requires_target_and_action(missing):
    model = _model()
    source, target, action = _inputs()
    kwargs = {"target": target, "action": action, missing: None}

    with pytest.raises(ValueError, match="requires target and action"):
        model(source, **kwargs)


def test_iwm_eval_returns_student_embedding_without_prediction():
    model = _model().eval()
    images = torch.randn(B, 3, 64, 64)

    with torch.no_grad():
        output = model(images)
        for p in model.encoder.teacher.parameters():
            p.add_(0.1)
        after_teacher_change = model(images)

    assert output.loss.item() == 0.0
    assert output.embedding.shape == (B, 16, 192)
    assert output.predictions is None
    assert output.targets is None
    assert torch.allclose(output.embedding, after_teacher_change.embedding)
