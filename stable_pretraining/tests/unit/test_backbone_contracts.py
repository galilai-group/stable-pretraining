"""Backbone wrappers must preserve weights, gradients, and feature structure."""

from collections import namedtuple
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from stable_pretraining.backbone import utils

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("decay", [0.0, 0.2])
def test_gradient_scaling_matches_explicit_regularization(decay):
    model = nn.Linear(3, 2)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    assert utils.register_lr_scale_hook(model, 0.25, decay) is model
    model(torch.ones(4, 3)).sum().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, (4 + decay * before[name]) * 0.25)
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.parametrize("names", ["a", ["b", "a"], None])
def test_feature_concatenation_preserves_order_and_gradients(names):
    a = torch.arange(12.0).reshape(2, 2, 3).requires_grad_()
    b = torch.arange(24.0).reshape(2, 4, 3).requires_grad_()

    def agg(value):
        return value.mean(-1)

    selected = [a] if names == "a" else [b, a]
    inputs = selected if names is None else {"a": a, "b": b}
    out = utils.FeaturesConcat(agg, names)(inputs)
    torch.testing.assert_close(out, torch.cat([v.mean(-1) for v in selected], 1))
    shapes = {str(i): v.shape for i, v in enumerate(selected)}
    assert utils.FeaturesConcat.get_output_shape(agg, shapes) == out.shape
    out.sum().backward()
    for value in selected:
        torch.testing.assert_close(value.grad, torch.full_like(value, 1 / 3))


def test_feature_shape_requires_nonempty_input():
    with pytest.raises(ValueError, match="empty"):
        utils.FeaturesConcat.get_output_shape(nn.Identity(), [])


def test_hidden_states_are_fresh_differentiable_and_hooks_removable():
    model = nn.Sequential(nn.Sequential(nn.Linear(3, 4), nn.ReLU()), nn.Linear(4, 2))
    wrapper = utils.HiddenStateExtractor(model, ["0.0", "0.1"])
    x = torch.randn(2, 3, requires_grad=True)
    first = wrapper(x)
    torch.testing.assert_close(first.hidden_states["0.0"], model[0][0](x))
    torch.testing.assert_close(first.last_hidden_state, model(x))
    first.last_hidden_state.sum().backward()
    assert x.grad is not None
    second = wrapper(x + 3)
    assert first.hidden_states is not second.hidden_states
    assert first.hidden_states["0.0"] is not second.hidden_states["0.0"]
    wrapper.remove_hooks()
    wrapper.remove_hooks()
    assert wrapper(x).hidden_states == {}
    assert not model[0][0]._forward_hooks


def test_invalid_hidden_layer_does_not_leave_hooks_attached():
    model = nn.Sequential(nn.Linear(3, 2))
    with pytest.raises(ValueError, match="not found"):
        utils.HiddenStateExtractor(model, ["0", "missing.layer"])
    assert not model[0]._forward_hooks


@dataclass
class _Bundle:
    tensor: object
    label: str


_Pair = namedtuple("Pair", "tensor label")


class _NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("offset", torch.ones(2))

    def forward(self, batch, *, extra):
        y = self.linear(batch["x"][0]) + self.offset + extra.tensor
        return {
            "y": y,
            "nested": [_Pair(y, "pair"), _Bundle(y, "bundle"), (y, None)],
            "set": {y},
            "unchanged": extra.label,
        }


def test_shape_inference_preserves_nested_structure_weights_and_buffers():
    model = _NestedModel()
    params = dict(model.named_parameters())
    state = {k: v.clone() for k, v in model.state_dict().items()}
    shapes = utils.get_output_shape(
        model, {"x": [torch.ones(4, 3)]}, extra=_Bundle(torch.ones(4, 2), "label")
    )
    assert shapes == {
        "y": torch.Size([4, 2]),
        "nested": [
            _Pair(torch.Size([4, 2]), "pair"),
            _Bundle(torch.Size([4, 2]), "bundle"),
            (torch.Size([4, 2]), None),
        ],
        "set": {torch.Size([4, 2])},
        "unchanged": "label",
    }
    for name, parameter in model.named_parameters():
        assert parameter is params[name]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name])


def test_shape_inference_restores_parameters_after_forward_failure():
    model = nn.Linear(3, 2)
    weight = model.weight
    before = weight.clone()
    with pytest.raises(RuntimeError):
        utils.get_output_shape(model, torch.zeros(2, 5))
    assert model.weight is weight
    torch.testing.assert_close(model.weight, before)
    assert model(torch.ones(2, 3)).shape == (2, 2)


class _Classifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.kind = kind
        head = nn.Linear(4, 5)
        if kind == "heads":
            self.heads = nn.ModuleDict({"head": head})
        elif kind == "classifier_sequence":
            self.classifier = nn.Sequential(nn.ReLU(), head)
        else:
            setattr(self, kind, head)

    def forward(self, x):
        x = self.encoder(x)
        if self.kind == "heads":
            return self.heads.head(x)
        if self.kind == "classifier_sequence":
            return self.classifier(x)
        return getattr(self, self.kind)(x)


@pytest.mark.parametrize(
    "kind", ["fc", "classifier", "classifier_sequence", "heads", "head", "custom"]
)
@pytest.mark.parametrize("verify", [False, True])
def test_change_embedding_dimension_preserves_pretrained_weights(kind, verify):
    model = _Classifier(kind)
    encoder = model.encoder.weight
    before = encoder.clone()
    kwargs = (
        {"expected_input_shape": (2, 3), "expected_output_shape": (2, 7)}
        if verify or kind == "custom"
        else {}
    )
    updated = utils.set_embedding_dim(model, 7, bias=False, **kwargs)
    assert model.encoder.weight is encoder
    torch.testing.assert_close(model.encoder.weight, before)
    output = updated(torch.ones(2, 3))
    assert output.shape == (2, 7)
    assert output.device.type == "cpu"
    output.sum().backward()
    assert encoder.grad is not None


def test_unknown_head_requires_input_shape():
    with pytest.raises(ValueError, match="expected_input_shape"):
        utils.set_embedding_dim(nn.Identity(), 4)


@pytest.mark.parametrize("structure", ["tuple", "logits"])
def test_embedding_shape_verification_accepts_structured_outputs(structure):
    class StructuredClassifier(_Classifier):
        def forward(self, x):
            out = super().forward(x)
            return (out, "auxiliary") if structure == "tuple" else {"logits": out}

    model = StructuredClassifier("fc")
    before = model.encoder.weight.detach().clone()
    updated = utils.set_embedding_dim(
        model, 7, expected_input_shape=(2, 3), expected_output_shape=(2, 7)
    )
    torch.testing.assert_close(updated.encoder.weight, before)
    out = updated(torch.ones(2, 3))
    assert (out[0] if structure == "tuple" else out["logits"]).shape == (2, 7)


def test_incorrect_embedding_shape_does_not_destroy_encoder():
    model = _Classifier("fc")
    before = model.encoder.weight.detach().clone()
    with pytest.raises(AssertionError):
        utils.set_embedding_dim(
            model, 7, expected_input_shape=(2, 3), expected_output_shape=(2, 9)
        )
    torch.testing.assert_close(model.encoder.weight, before)
    assert model(torch.ones(2, 3)).shape == (2, 7)


def test_teacher_and_student_hidden_state_hooks_are_independent():
    student = utils.HiddenStateExtractor(nn.Sequential(nn.Linear(2, 2)), ["0"])
    wrapper = utils.TeacherStudentWrapper(student, base_ema_coefficient=0.5)
    x = torch.ones(1, 2)
    student_output = wrapper.forward_student(x)
    teacher_output = wrapper.forward_teacher(x * 2)
    assert student_output.hidden_states["0"] is not teacher_output.hidden_states["0"]
    assert student_output.hidden_states["0"].requires_grad
    assert not teacher_output.hidden_states["0"].requires_grad
    torch.testing.assert_close(student._cache["0"], student_output.hidden_states["0"])


@pytest.mark.parametrize(
    "partial,depth,expected",
    [
        (False, 0, ["blocks"]),
        (False, 1, ["blocks.0", "blocks.1"]),
        (True, 1, ["blocks.0", "blocks.1", "extra_blocks.0"]),
        (False, 3, []),
    ],
)
def test_child_module_discovery_respects_depth_and_matching(partial, depth, expected):
    model = nn.ModuleDict(
        {
            "blocks": nn.Sequential(nn.Sequential(nn.Linear(2, 2)), nn.ReLU()),
            "extra_blocks": nn.Sequential(nn.Identity()),
        }
    )
    assert (
        utils.get_children_modules(model, "blocks", L=depth, partial_match=partial)
        == expected
    )


@pytest.mark.parametrize("base,final", [(-0.1, 1.0), (0.5, 1.1)])
def test_teacher_rejects_invalid_ema_coefficients(base, final):
    with pytest.raises(ValueError, match="ema_coefficient"):
        utils.TeacherStudentWrapper(
            nn.Linear(2, 2), base_ema_coefficient=base, final_ema_coefficient=final
        )


@pytest.mark.parametrize("coefficient", [0.0, 0.5, 1.0])
def test_teacher_updates_float_state_and_integer_counters(coefficient):
    wrapper = utils.TeacherStudentWrapper(
        nn.BatchNorm1d(2),
        base_ema_coefficient=coefficient,
        final_ema_coefficient=1.0,
        warm_init=False,
    )
    with torch.no_grad():
        wrapper.student.weight.fill_(3)
        wrapper.student.running_mean.fill_(4)
        wrapper.student.num_batches_tracked.fill_(7)
    wrapper.update_teacher()
    torch.testing.assert_close(
        wrapper.teacher.weight, torch.full((2,), coefficient + 3 * (1 - coefficient))
    )
    torch.testing.assert_close(
        wrapper.teacher.running_mean, torch.full((2,), 4 * (1 - coefficient))
    )
    assert wrapper.teacher.num_batches_tracked.item() == (0 if coefficient == 1 else 7)


def test_teacher_schedule_and_checkpoint_round_trip():
    wrapper = utils.TeacherStudentWrapper(nn.Linear(3, 2), base_ema_coefficient=0.5)
    wrapper.update_ema_coefficient(5, 10)
    assert wrapper.ema_coefficient.item() == pytest.approx(0.75)
    resumed = utils.TeacherStudentWrapper(nn.Linear(3, 2))
    resumed.load_state_dict(wrapper.state_dict())
    x = torch.ones(2, 3, requires_grad=True)
    torch.testing.assert_close(wrapper(x), resumed(x))
    assert not wrapper(x).requires_grad
    assert wrapper.forward_student(x).requires_grad
    wrapper.eval()
    before = wrapper.teacher.weight.clone()
    with torch.no_grad():
        wrapper.student.weight.add_(1)
    wrapper.update_teacher()
    torch.testing.assert_close(wrapper.teacher.weight, before)
    wrapper.update_ema_coefficient(10, 10)
    assert wrapper.ema_coefficient.item() == 1.0


def test_zero_ema_shares_student_without_disabling_student_gradients():
    wrapper = utils.TeacherStudentWrapper(
        nn.Linear(2, 2), base_ema_coefficient=0.0, final_ema_coefficient=0.0
    )
    assert wrapper.teacher is wrapper.student
    wrapper.update_teacher()
    x = torch.ones(1, 2)
    assert not wrapper(x).requires_grad
    wrapper.forward_student(x).sum().backward()
    assert wrapper.student.weight.grad is not None


def _double_hidden_output(module, inputs, output):
    return output * 2


@pytest.mark.parametrize("serialization", ["deepcopy", "pickle"])
@pytest.mark.parametrize("populated", [False, True])
def test_hidden_extractor_owns_copyable_hooks_and_discards_transient_cache(
    serialization, populated
):
    import copy
    import pickle

    backbone = nn.Sequential(nn.Linear(2, 2))
    backbone[0].register_forward_hook(_double_hidden_output)
    original = utils.HiddenStateExtractor(backbone, ["0"])
    original.eval()
    original.extra_configuration = {"purpose": "test"}
    source_output = original(torch.ones(1, 2)) if populated else None
    container = nn.ModuleDict({"features": original, "shared_layer": backbone[0]})
    copied = (
        copy.deepcopy(container)
        if serialization == "deepcopy"
        else pickle.loads(pickle.dumps(container))
    )
    clone = copied["features"]
    assert copied["shared_layer"] is clone.backbone[0]
    assert clone.extra_configuration == original.extra_configuration
    assert not clone.training
    assert clone._cache == {}
    value = torch.full((1, 2), 3.0, requires_grad=True)
    result = clone(value)
    expected = nn.functional.linear(value, backbone[0].weight, backbone[0].bias) * 2
    torch.testing.assert_close(result.hidden_states["0"], expected)
    result.last_hidden_state.sum().backward()
    assert value.grad is not None
    assert clone.backbone[0].weight.grad is not None
    assert backbone[0].weight.grad is None
    if populated:
        assert original._cache["0"] is source_output.hidden_states["0"]
    else:
        assert original._cache == {}
    clone.remove_hooks()
    assert clone(value).hidden_states == {}
    torch.testing.assert_close(clone(value).last_hidden_state, expected)
    assert "0" in original(value).hidden_states
    original.remove_hooks()
    assert len(backbone[0]._forward_hooks) == 1


def test_teacher_handles_nested_hidden_extractors_without_type_specific_logic():
    student = nn.Sequential(
        utils.HiddenStateExtractor(nn.Sequential(nn.Linear(2, 2)), ["0"])
    )
    first = student(torch.ones(1, 2))
    wrapped = utils.TeacherStudentWrapper(student, base_ema_coefficient=0.5)
    result = wrapped.forward_teacher(torch.full((1, 2), 2.0))
    assert not result.hidden_states["0"].requires_grad
    assert student[0]._cache["0"] is first.hidden_states["0"]
    wrapped.teacher[0].remove_hooks()
    assert "0" in wrapped.forward_student(torch.ones(1, 2)).hidden_states
