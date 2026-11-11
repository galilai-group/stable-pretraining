"""Protocol identity, exact dataset mappings and prediction artifact contracts."""

import copy
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.trainer.states import TrainerFn

import stable_pretraining as spt

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "examples" / "simco_cifar.py")
    )


def _args(example):
    return example["_parser"]().parse_args(
        [
            "--synthetic",
            "--accelerator",
            "cpu",
            "--batch-size",
            "2",
            "--warmup-epochs",
            "0",
        ]
    )


def test_latest_checkpoint_advances_when_validation_accuracy_decreases(
    example, tmp_path
):
    """An earlier best model must not prevent saving the latest epoch state."""
    best, latest = example["_checkpoints"](tmp_path)
    (tmp_path / "checkpoints").mkdir()
    trainer = SimpleNamespace(
        fast_dev_run=False,
        sanity_checking=False,
        state=SimpleNamespace(fn=TrainerFn.FITTING),
        is_global_zero=True,
        loggers=[],
        ckpt_path=None,
        fit_loop=SimpleNamespace(
            epoch_loop=SimpleNamespace(val_loop=SimpleNamespace(_has_run=True))
        ),
        strategy=SimpleNamespace(
            reduce_boolean_decision=lambda value, **kwargs: value,
        ),
    )

    def save_checkpoint(path, weights_only):
        torch.save(
            {"epoch": trainer.current_epoch, "global_step": trainer.global_step},
            path,
        )

    trainer.save_checkpoint = save_checkpoint
    for epoch, accuracy in enumerate((0.2622, 0.2460, 0.2400)):
        trainer.current_epoch = epoch
        trainer.global_step = (epoch + 1) * 175
        trainer.callback_metrics = {
            "eval/linear_probe_top1_accuracy": torch.tensor(accuracy)
        }
        best.on_validation_end(trainer, None)
        latest.on_validation_end(trainer, None)
        best.on_train_epoch_end(trainer, None)
        latest.on_train_epoch_end(trainer, None)
        assert torch.load(latest.best_model_path, weights_only=True) == {
            "epoch": epoch,
            "global_step": (epoch + 1) * 175,
        }

    assert torch.load(best.best_model_path, weights_only=True) == {
        "epoch": 0,
        "global_step": 175,
    }
    assert best.best_model_score.item() == pytest.approx(0.2622)
    assert sorted(path.name for path in (tmp_path / "checkpoints").iterdir()) == [
        "epoch-000.ckpt",
        "latest.ckpt",
    ]


@pytest.mark.parametrize(
    (
        "training_examples",
        "batch_size",
        "epochs",
        "max_steps",
        "batches_per_epoch",
        "planned_optimizer_steps",
    ),
    [
        (45000, 256, 100, -1, 175, 17500),
        (45000, 256, 1, 10, 10, 10),
        (64, 16, 100, 10, 4, 10),
        (64, 16, 1, 10, 4, 4),
        (64, 16, 2, 10, 4, 8),
        (64, 16, 3, 10, 4, 10),
        (64, 16, 100, 1, 1, 1),
        (65, 16, 2, -1, 4, 8),
    ],
)
def test_protocol_step_budget_respects_epochs_steps_and_dropped_last_batch(
    example,
    training_examples,
    batch_size,
    epochs,
    max_steps,
    batches_per_epoch,
    planned_optimizer_steps,
):
    args = _args(example)
    args.batch_size = batch_size
    args.epochs = epochs
    args.max_steps = max_steps
    splits = {
        "training_indices": list(range(training_examples)),
        "validation_indices": [training_examples],
    }
    protocol = example["_protocol"](args, 10, splits)
    assert protocol["training"]["batches_per_epoch"] == batches_per_epoch
    assert protocol["training"]["planned_optimizer_steps"] == planned_optimizer_steps
    assert protocol["optimization"]["interval"] == "step"
    assert protocol["optimization"]["scheduler"]["max_steps"] == planned_optimizer_steps


def test_cosine_schedule_decays_over_the_full_batch_step_budget(example):
    """Manual optimization steps the scheduler after every optimizer update."""
    args = _args(example)
    args.epochs = 4
    splits = {"training_indices": list(range(16)), "validation_indices": [16]}
    protocol = example["_protocol"](args, 10, splits)
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.ones(()))],
        lr=protocol["optimization"]["optimizer"]["lr"],
    )
    scheduler = spt.optim.create_scheduler(
        optimizer, protocol["optimization"]["scheduler"]
    )
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(protocol["training"]["planned_optimizer_steps"]):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    first_epoch_end = protocol["training"]["batches_per_epoch"]
    assert 0 < learning_rates[first_epoch_end] < learning_rates[0]
    assert all(
        later <= earlier + 1e-14
        for earlier, later in zip(learning_rates, learning_rates[1:])
    )
    assert learning_rates[-1] == pytest.approx(0.0, abs=1e-12)


def test_warmup_reaches_peak_then_cosine_decays_over_optimizer_steps(example):
    args = _args(example)
    args.epochs = 4
    args.warmup_epochs = 1
    splits = {"training_indices": list(range(16)), "validation_indices": [16]}
    protocol = example["_protocol"](args, 10, splits)
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.ones(()))],
        lr=protocol["optimization"]["optimizer"]["lr"],
    )
    scheduler = spt.optim.create_scheduler(
        optimizer, protocol["optimization"]["scheduler"]
    )
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(protocol["training"]["planned_optimizer_steps"]):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    warmup_steps = protocol["optimization"]["scheduler"]["warmup_steps"]
    peak = protocol["optimization"]["optimizer"]["lr"]
    assert learning_rates[0] == 0.0
    assert learning_rates[warmup_steps] == pytest.approx(peak)
    assert all(
        a <= b
        for a, b in zip(
            learning_rates[:warmup_steps], learning_rates[1 : warmup_steps + 1]
        )
    )
    assert all(
        a >= b
        for a, b in zip(
            learning_rates[warmup_steps:], learning_rates[warmup_steps + 1 :]
        )
    )
    assert learning_rates[-1] == pytest.approx(0.0, abs=1e-12)


def test_warmup_cannot_consume_the_entire_short_run(example):
    args = _args(example)
    args.epochs = 1
    args.warmup_epochs = 1
    splits = {"training_indices": list(range(16)), "validation_indices": [16]}
    with pytest.raises(ValueError, match="Warmup must be shorter"):
        example["_protocol"](args, 10, splits)


def test_two_device_500_epoch_budget_preserves_global_batch_and_step_count(example):
    args = _args(example)
    args.epochs = 500
    args.warmup_epochs = 10
    args.batch_size = 256
    args.devices = 2
    args.accelerator = "gpu"
    splits = {
        "training_indices": list(range(45000)),
        "validation_indices": list(range(45000, 50000)),
    }
    protocol = example["_protocol"](args, 10, splits)
    training = protocol["training"]
    assert training["global_batch_size"] == 256
    assert training["per_device_batch_size"] == 128
    assert training["world_size"] == training["devices"] == 2
    assert training["strategy"] == "ddp"
    assert training["sync_batchnorm"]
    assert training["distributed_sampler"]
    assert not training["validation_sampler_padding"]
    assert training["batches_per_epoch"] == 175
    assert training["planned_optimizer_steps"] == 87500
    scheduler = protocol["optimization"]["scheduler"]
    assert scheduler["max_steps"] == 87500
    assert scheduler["warmup_steps"] == 1750
    args.devices = 1
    single = example["_protocol"](args, 10, splits)
    assert single["training"]["planned_optimizer_steps"] == 87500
    assert single["training"]["per_device_batch_size"] == 256
    assert not single["training"]["sync_batchnorm"]
    assert example["_fingerprint"](single) != example["_fingerprint"](protocol)


@pytest.mark.parametrize("devices,batch_size", [(0, 256), (-1, 256), (2, 255), (4, 2)])
def test_rejects_invalid_global_batch_or_device_count(example, devices, batch_size):
    args = _args(example)
    args.devices = devices
    args.batch_size = batch_size
    splits = {
        "training_indices": list(range(64)),
        "validation_indices": list(range(32)),
    }
    with pytest.raises(ValueError, match="divisible by devices"):
        example["_protocol"](args, 10, splits)


@pytest.mark.parametrize("split", ["training_indices", "validation_indices"])
def test_rejects_distributed_sampler_padding(example, split):
    args = _args(example)
    args.devices = 2
    splits = {
        "training_indices": list(range(64)),
        "validation_indices": list(range(32)),
    }
    splits[split] = splits[split][:-1]
    with pytest.raises(ValueError, match="avoid sampler padding"):
        example["_protocol"](args, 10, splits)


def test_cpu_ddp_does_not_request_gpu_only_sync_batchnorm(example):
    args = _args(example)
    args.devices = 2
    splits = {
        "training_indices": list(range(64)),
        "validation_indices": list(range(32)),
    }
    training = example["_protocol"](args, 10, splits)["training"]
    assert training["strategy"] == "ddp"
    assert not training["sync_batchnorm"]


def test_protocol_hash_is_canonical_and_sensitive_to_actual_settings(example):
    args = _args(example)
    splits = {"training_indices": [4, 1], "validation_indices": [2, 0]}
    protocol = example["_protocol"](args, 10, splits)
    fingerprint = example["_fingerprint"]
    original = fingerprint(protocol)
    assert original == fingerprint(dict(reversed(list(protocol.items()))))
    assert "training_indices" not in protocol["data"]
    assert protocol["data"]["split_indices_sha256"] == fingerprint(splits)

    changed = copy.deepcopy(protocol)
    changed["optimization"]["optimizer"]["lr"] *= 2
    assert fingerprint(changed) != original
    changed = copy.deepcopy(protocol)
    changed["augmentation"]["ordered_transforms"]["random_grayscale"]["p"] = 0.0
    assert fingerprint(changed) != original
    changed_splits = {**splits, "validation_indices": [0, 2]}
    assert fingerprint(example["_protocol"](args, 10, changed_splits)) != original


def test_real_split_mapping_matches_dataset_and_ignores_model_seed(
    example, monkeypatch
):
    class TinyCIFAR:
        def __init__(self, root, train, download):
            assert train and not download

        def __len__(self):
            return 5004

        def __getitem__(self, index):
            return torch.zeros(3, 32, 32, dtype=torch.uint8), index % 10

    datasets = example["_datasets"]
    monkeypatch.setitem(datasets.__globals__, "CIFAR10", TinyCIFAR)
    args = _args(example)
    args.synthetic = False
    train, validation, classes, splits = datasets(args)
    assert classes == 10
    assert splits["training_indices"] == train.dataset.indices
    assert splits["validation_indices"] == validation.dataset.indices
    assert len(train) == 4 and len(validation) == 5000
    assert set(splits["training_indices"]).isdisjoint(splits["validation_indices"])
    assert sorted(splits["training_indices"] + splits["validation_indices"]) == list(
        range(5004)
    )
    sample = validation[3]
    assert sample["sample_idx"] == 3
    assert sample["label"] == splits["validation_indices"][3] % 10
    args.seed = 123
    assert datasets(args)[3] == splits
    args.split_seed = 123
    assert datasets(args)[3] != splits


def test_synthetic_sources_are_distinguishable_and_labels_match_saved_logits(
    example, tmp_path
):
    args = _args(example)
    _, validation, _, splits = example["_datasets"](args)
    assert splits["training_random_offset"] == 0
    assert splits["validation_random_offset"] == 10000
    assert splits["validation_indices"] == list(range(32))
    protocol = example["_protocol"](args, 10, splits)
    records = protocol["validation_records"]
    writer = spt.OnlineWriter(
        names=records["names"],
        path=tmp_path / records["directory"],
        during="validation",
        all_gather=False,
    )
    sample_indices = torch.tensor([2, 7])
    labels = torch.tensor(
        [validation[index]["label"] for index in sample_indices.tolist()]
    )
    logits = torch.zeros(2, 10)
    logits[torch.arange(2), labels] = 2.0
    outputs = {
        "sample_idx": sample_indices,
        "label": labels,
        "linear_probe_preds": logits,
    }
    module = SimpleNamespace(
        current_epoch=0, local_rank=0, trainer=SimpleNamespace(max_epochs=1)
    )
    writer.on_validation_batch_end(None, module, outputs, None, 0)
    saved = torch.load(next(writer.path.glob("*.pt")), weights_only=True)
    assert set(saved) == set(records["names"])
    torch.testing.assert_close(saved["linear_probe_preds"], logits)
    assert saved["sample_idx"].tolist() == [2, 7]
    torch.testing.assert_close(
        saved["linear_probe_preds"].argmax(dim=1), saved["label"]
    )
    assert (
        example["_METRIC_DEFINITIONS"]["eval/linear_probe_top1_accuracy"]["direction"]
        == "maximize"
    )
