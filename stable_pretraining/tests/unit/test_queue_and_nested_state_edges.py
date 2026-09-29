"""Shared queue lifecycle and nested state preserve identity and tensor ownership."""

from collections import defaultdict, namedtuple
from dataclasses import dataclass, InitVar
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_pretraining.callbacks.queue import (
    OnlineQueue,
    find_or_create_queue_callback,
)
from stable_pretraining.callbacks.utils import detach_tensors
from stable_pretraining.utils.inspection_utils import (
    broadcast_param_to_list,
    dict_values,
    get_required_fn_parameters,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def isolated_queues(monkeypatch):
    monkeypatch.setattr(OnlineQueue, "_shared_queues", {})
    monkeypatch.setattr(OnlineQueue, "_queue_info", {})
    monkeypatch.setattr(OnlineQueue, "_owner_trainer_id", None)


@pytest.mark.parametrize("prefill", [False, True])
def test_attached_queue_growth_preserves_order_and_only_appends_once(
    isolated_queues, prefill
):
    module = SimpleNamespace(
        callbacks_modules=nn.ModuleDict(), device=torch.device("cpu")
    )
    trainer = SimpleNamespace(callbacks=[], lightning_module=module, world_size=1)
    first = find_or_create_queue_callback(trainer, "feature", 2, dim=1)
    assert first.data is None
    if prefill:
        first.on_train_batch_end(
            trainer, module, {"feature": torch.tensor([1.0, 2.0])}, {}, 0
        )
    second = find_or_create_queue_callback(trainer, "feature", 4, dim=1)
    trainer.callbacks.append(second)
    first.on_train_batch_end(
        trainer, module, {"feature": torch.tensor([3.0, 4.0])}, {}, 1
    )
    second.on_train_batch_end(
        trainer, module, {"feature": torch.tensor([3.0, 4.0])}, {}, 1
    )
    second.on_validation_epoch_start(trainer, module)
    assert second.data.flatten().tolist() == (
        [1.0, 2.0, 3.0, 4.0] if prefill else [3.0, 4.0]
    )
    assert first.actual_queue_length == 4
    second.teardown(trainer, module, "fit")
    assert second.data is None
    first.on_train_batch_end(trainer, module, {"feature": None}, {}, 2)
    assert len(OnlineQueue._shared_queues["feature"].get()) == (4 if prefill else 2)


def test_queue_search_reports_incompatible_candidates_and_reuses_available_size(
    isolated_queues,
):
    first = OnlineQueue("x", 3, dim=2, dtype=torch.float32, verbose=False)
    assert first.actual_queue_length == 3
    trainer = SimpleNamespace(callbacks=[first], lightning_module=None)
    assert (
        find_or_create_queue_callback(trainer, "x", 5, dim=2, create_if_missing=False)
        is first
    )
    for kwargs in ({"dim": 4}, {"dtype": torch.float64}):
        with pytest.raises(ValueError, match="No OnlineQueue.*Available queues"):
            find_or_create_queue_callback(
                trainer, "x", 3, create_if_missing=False, **kwargs
            )


@pytest.mark.parametrize("device", ["invalid-device", None, 42])
def test_invalid_module_device_has_no_accidental_device_transfer(device):
    assert OnlineQueue._resolve_module_device(SimpleNamespace(device=device)) is None


def test_defaultdict_detachment_preserves_factory_cycles_and_shared_tensor():
    tensor = torch.ones(2, requires_grad=True)
    original = defaultdict(list, left=tensor, right=tensor)
    original["self"] = original
    detached = detach_tensors(original)
    assert isinstance(detached, defaultdict)
    assert detached["self"] is detached
    assert detached["left"] is detached["right"]
    assert not detached["left"].requires_grad and tensor.requires_grad
    assert detached["missing"] == []
    assert "missing" not in original


def test_dataclass_required_initvar_can_be_detached_without_reinitialization():
    @dataclass(frozen=True)
    class Record:
        required: InitVar[int]
        value: torch.Tensor

    original = Record(5, torch.ones(2, requires_grad=True))
    result = detach_tensors(original)
    assert type(result) is Record
    assert not result.value.requires_grad
    assert original.value.requires_grad


def test_tensor_free_objects_keep_identity():
    @dataclass
    class Record:
        count: int

    pair = namedtuple("Pair", ["a", "b"])
    for original in (
        Record(1),
        pair(1, 2),
        frozenset({1, 2}),
        SimpleNamespace(count=1),
    ):
        assert detach_tensors(original) is original


def test_tensor_free_attrs_record_keeps_identity():
    attr = pytest.importorskip("attr")

    @attr.define
    class Record:
        value: int

    original = Record(1)
    assert detach_tensors(original) is original


@pytest.mark.parametrize(
    "value,length,expected",
    [
        (None, 3, [None] * 3),
        (5, 2, [5, 5]),
        ([4], 3, [4] * 3),
        ((1, 2), 2, [1, 2]),
        ([], 0, []),
    ],
)
def test_parameter_broadcast_preserves_explicit_values(value, length, expected):
    assert broadcast_param_to_list(value, length, "width") == expected


def test_parameter_broadcast_rejects_ambiguous_lengths():
    with pytest.raises(ValueError, match="width.*target length"):
        broadcast_param_to_list([1, 2], 3, "width")


def test_callable_inspection_respects_required_keyword_only_parameters():
    def callable_(first, optional=3, *, required, last=4):
        return first, optional, required, last

    assert get_required_fn_parameters(callable_) == ["first", "required"]
    assert dict_values(first=1, other=2) == [1, 2]
