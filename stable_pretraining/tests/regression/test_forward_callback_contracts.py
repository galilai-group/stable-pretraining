"""Malformed forward outputs fail at the boundary with actionable diagnostics."""

from copy import deepcopy
import types

import pytest
import torch

from ._interaction_helpers import _Trace, _forward, _model, _probe, _run

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]


@pytest.mark.parametrize("stage", ["fit", "validate"])
@pytest.mark.parametrize(
    "problem",
    [
        "missing_embedding",
        "missing_label",
        "feature_width",
        "target_length",
        "target_rank",
    ],
)
def test_probe_reports_bad_input_with_callback_key_and_stage(stage, problem):
    module = _model()

    def malformed(self, batch, current_stage):
        result = _forward(self, batch, current_stage)
        if current_stage == stage:
            if problem == "missing_embedding":
                del result["embedding"]
            elif problem == "missing_label":
                del batch["label"]
            elif problem == "feature_width":
                result["embedding"] = result["embedding"][:, :2]
            elif problem == "target_length":
                batch["label"] = batch["label"][:-1]
            elif problem == "target_rank":
                batch["label"] = batch["label"].unsqueeze(-1)
        return result

    module.forward = types.MethodType(malformed, module)
    probe = _probe(module, name="linear_probe")
    trace = _Trace()
    initial = deepcopy(module.backbone.state_dict())
    with pytest.raises(ValueError) as caught:
        _run(module, [probe, trace], max_epochs=1)
    message = str(caught.value)
    assert "linear_probe" in message and stage in message
    assert (
        "label"
        if problem in ("missing_label", "target_length", "target_rank")
        else "embedding"
    ) in message
    if problem.startswith("missing"):
        assert "Available batch keys" in message and "Available output keys" in message
    else:
        assert "shape" in message
        if problem == "feature_width":
            assert isinstance(caught.value.__cause__, RuntimeError)
        elif problem == "target_rank":
            assert caught.value.__cause__ is not None
    assert len(trace.batches) == (0 if stage == "fit" else 6)
    if stage == "fit":
        for name, value in initial.items():
            torch.testing.assert_close(
                module.backbone.state_dict()[name], value, rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "problem", ["tensor", "none", "missing_loss", "vector_loss", "float_loss"]
)
def test_training_forward_contract_is_checked_before_backward(problem):
    module = _model()

    def malformed(self, batch, stage):
        outputs = _forward(self, batch, stage)
        if problem == "tensor":
            return outputs["embedding"]
        if problem == "none":
            return None
        if problem == "missing_loss":
            del outputs["loss"]
        elif problem == "vector_loss":
            outputs["loss"] = outputs["embedding"].mean(1)
        elif problem == "float_loss":
            outputs["loss"] = 1.0
        return outputs

    module.forward = types.MethodType(malformed, module)
    trace = _Trace()
    with pytest.raises(ValueError) as caught:
        _run(module, [trace], max_epochs=1)
    message = str(caught.value)
    assert "forward" in message and "fit" in message
    assert ("dict" if problem in ("tensor", "none") else "loss") in message
    assert not trace.batches
    assert all(parameter.grad is None for parameter in module.parameters())


def test_validation_forward_may_omit_loss():
    module = _model()

    def forward(self, batch, stage):
        outputs = _forward(self, batch, stage)
        if stage == "validate":
            del outputs["loss"]
        return outputs

    module.forward = types.MethodType(forward, module)
    trainer = _run(module, [_probe(module)], max_epochs=1)
    assert "eval/probe_accuracy" in trainer.callback_metrics


@pytest.mark.parametrize("stage", ["fit", "validate"])
@pytest.mark.parametrize("value", [None, torch.zeros(2, 3)])
def test_probe_rejects_non_dict_forward_output(stage, value):
    module = _model()

    def malformed(self, batch, current_stage):
        return value if current_stage == stage else _forward(self, batch, current_stage)

    module.forward = types.MethodType(malformed, module)
    probe = _probe(module, name="linear_probe")
    with pytest.raises(ValueError, match=f"linear_probe.*{stage}.*forward.*dict"):
        _run(module, [probe], max_epochs=1)
