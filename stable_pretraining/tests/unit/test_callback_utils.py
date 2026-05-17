"""Unit tests for callback helpers."""

import pytest
import torch

from stable_pretraining.callbacks.utils import get_data_from_batch_or_outputs


@pytest.mark.unit
class TestGetDataFromBatchOrOutputs:
    """Tests for :func:`get_data_from_batch_or_outputs` dispatch logic."""

    def test_string_key_returns_scalar(self):
        batch = {"x": torch.tensor([1.0])}
        out = get_data_from_batch_or_outputs("x", batch, outputs=None)
        assert torch.equal(out, batch["x"])

    def test_outputs_take_precedence_over_batch(self):
        batch = {"x": torch.tensor([1.0])}
        outputs = {"x": torch.tensor([2.0])}
        out = get_data_from_batch_or_outputs("x", batch, outputs)
        assert torch.equal(out, outputs["x"])

    def test_falls_back_to_batch_when_missing_in_outputs(self):
        batch = {"x": torch.tensor([1.0])}
        outputs = {"y": torch.tensor([99.0])}
        out = get_data_from_batch_or_outputs("x", batch, outputs)
        assert torch.equal(out, batch["x"])

    def test_outputs_none_falls_back_to_batch(self):
        batch = {"x": torch.tensor([1.0])}
        out = get_data_from_batch_or_outputs("x", batch, outputs=None)
        assert torch.equal(out, batch["x"])

    def test_list_keys_returns_list_in_order(self):
        batch = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        out = get_data_from_batch_or_outputs(["a", "b"], batch, outputs=None)
        assert isinstance(out, list)
        assert torch.equal(out[0], batch["a"])
        assert torch.equal(out[1], batch["b"])

    def test_list_keys_mixes_outputs_and_batch(self):
        batch = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        outputs = {"a": torch.tensor([10.0])}
        out = get_data_from_batch_or_outputs(["a", "b"], batch, outputs)
        assert torch.equal(out[0], outputs["a"])
        assert torch.equal(out[1], batch["b"])

    def test_missing_key_raises_value_error(self):
        batch = {"x": torch.tensor([1.0])}
        with pytest.raises(ValueError, match="not found in batch or outputs"):
            get_data_from_batch_or_outputs("missing", batch, outputs=None)

    def test_missing_key_in_list_raises(self):
        batch = {"a": torch.tensor([1.0])}
        with pytest.raises(ValueError, match="missing"):
            get_data_from_batch_or_outputs(
                ["a", "missing"], batch, outputs={"a": torch.tensor([0.0])}
            )

    def test_caller_name_appears_in_error(self):
        batch = {}
        with pytest.raises(ValueError, match="MyCaller"):
            get_data_from_batch_or_outputs(
                "x", batch, outputs=None, caller_name="MyCaller"
            )
