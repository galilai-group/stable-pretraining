"""Exercise the standalone benchmark training and evaluation lifecycle."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks/imagenet100/directclr.py"
spec = importlib.util.spec_from_file_location("directclr_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.fixture(autouse=True)
def restore_benchmark_settings():
    """Keep standalone benchmark settings from leaking into other tests."""
    config = benchmark.spt.get_config()
    previous = copy.deepcopy(vars(config))
    threads = torch.get_num_threads()
    try:
        yield
    finally:
        vars(config).clear()
        vars(config).update(previous)
        benchmark.spt.set(verbose=config.verbose, log_rank=config.log_rank)
        torch.set_num_threads(threads)


def _args(tmp_path, method="directclr"):
    return argparse.Namespace(
        command="train",
        dataset="fake",
        data_dir=tmp_path / "data",
        data_revision="main",
        output=tmp_path / method,
        method=method,
        backbone="resnet18",
        dim=64,
        temperature=0.1,
        epochs=2,
        batch_size=8,
        lr=0.3,
        seed=7,
        workers=0,
        cpu_threads=1,
        accelerator="cpu",
        precision="32-true",
        max_steps=2,
        val_every=1,
        resume=None,
        checkpoint=None,
        probe_epochs=2,
        probe_lr=1.0,
        probe_batch_size=8,
    )


@pytest.mark.parametrize("method", ["directclr", "linear", "full"])
def test_training_checkpoint_resume_and_frozen_evaluation(tmp_path, method):
    args = _args(tmp_path, method)
    trainer = benchmark.train(args)
    assert trainer.global_step == 2
    summary = json.loads((args.output / "training_summary.json").read_text())
    checkpoint_path = Path(summary["checkpoint"])
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    assert len(checkpoint["optimizer_states"]) == 2
    assert checkpoint["optimizer_states"][0]["param_groups"][0]["momentum"] == 0.9
    assert checkpoint["lr_schedulers"][0]["max_steps"] == 8
    assert checkpoint["lr_schedulers"][0]["last_epoch"] == 2
    args.resume = checkpoint_path
    args.max_steps = 4
    resumed = benchmark.train(args)
    assert resumed.global_step == 4
    summary = json.loads((args.output / "training_summary.json").read_text())
    assert summary["updates_this_invocation"] == 2
    assert summary["seconds_per_step_including_setup_and_eval"] == pytest.approx(
        summary["elapsed_seconds"] / 2
    )
    args.command = "evaluate"
    args.checkpoint = Path(summary["checkpoint"])
    args.output = tmp_path / f"{method}-eval"
    benchmark.evaluate(args)
    result = json.loads((args.output / "evaluation.json").read_text())
    assert result["train_samples"] == 32 and result["eval_samples"] == 16
    assert 0 <= result["linear_top1"] <= result["linear_top5"] <= 1
    assert result["checkpoint_sha256"] != "random"
    features = torch.load(args.output / "features.pt", weights_only=True)
    assert features["x_train"].shape == (32, 512)
    assert torch.isfinite(features["x_train"]).all()


def test_refuses_accidental_run_overwrite(tmp_path):
    args = _args(tmp_path)
    benchmark._reserve_output(args)
    with pytest.raises(ValueError, match="already contains"):
        benchmark._reserve_output(args)


def test_full_control_uses_all_features_without_a_projector(tmp_path, monkeypatch):
    widths = []
    loss_class = benchmark.spt.losses.DirectCLRLoss
    original_forward = loss_class.forward

    def record_width(self, first, second):
        widths.append((first.shape[1], second.shape[1]))
        return original_forward(self, first, second)

    monkeypatch.setattr(loss_class, "forward", record_width)
    args = _args(tmp_path, method="full")
    trainer = benchmark.train(args)
    module = trainer.lightning_module
    assert args.dim == 64 and module.subspace_dim == 512
    assert widths and set(widths) == {(512, 512)}
    assert not hasattr(module, "projector")
    assert not any("projector" in name for name, _ in module.named_parameters())
    summary = json.loads((args.output / "training_summary.json").read_text())
    assert summary["benchmark_method"] == "full"
    assert summary["contrastive_loss_dim"] == 512
    assert summary["trainable_projector"] is False


def test_imagenet_transform_and_hf_wrapper_without_download():
    from datasets import ClassLabel, Dataset, Features, Image
    from PIL import Image as PILImage
    from stable_pretraining.data.datasets import HFMapDataset

    image = PILImage.new("RGB", (300, 260), "red")
    raw = HFMapDataset(
        Dataset.from_dict(
            {"image": [image, image], "label": [0, 1]},
            features=Features({"image": Image(), "label": ClassLabel(num_classes=100)}),
        )
    )
    augment, evaluate = benchmark._transforms("imagenet100")
    train = benchmark._wrap(raw, augment, "imagenet100")
    assert len(train[0]["views"]) == 2
    assert train[0]["views"][0]["image"].shape == (3, 224, 224)
    val = benchmark._wrap(raw, evaluate, "imagenet100")
    assert val[0]["image"].shape == (3, 224, 224)


def test_imagenet_training_uses_only_a_local_snapshot(tmp_path, monkeypatch):
    from datasets import ClassLabel, Dataset, Features, Image
    from PIL import Image as PILImage

    snapshot = tmp_path / "snapshot"
    (snapshot / "data").mkdir(parents=True)
    dataset = Dataset.from_dict(
        {"image": [PILImage.new("RGB", (32, 32))], "label": [0]},
        features=Features({"image": Image(), "label": ClassLabel(num_classes=100)}),
    )
    for split in ("train", "validation"):
        dataset.to_parquet(snapshot / "data" / f"{split}-00000-of-00001.parquet")
    requests = []

    def local_snapshot(*args, **kwargs):
        requests.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(benchmark, "snapshot_download", local_snapshot)
    args = _args(tmp_path)
    args.dataset = "imagenet100"
    args.data_revision = "pinned-revision"
    train, validation, classes = benchmark._raw_datasets(args)
    assert classes == 100 and len(train) == len(validation) == 1
    assert train[0]["label"] == 0
    assert requests[0]["local_files_only"] is True
    assert requests[0]["revision"] == "pinned-revision"
    args.command = "prepare"
    benchmark._raw_datasets(args)
    assert requests[-1]["local_files_only"] is False
    (snapshot / "data" / "validation-00000-of-00001.parquet").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        benchmark._raw_datasets(args)


def test_reference_views_keep_labels_and_solarize_only_second_view(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(
        benchmark.spt.data.transforms,
        "ColorJitter",
        lambda **kwargs: lambda batch: batch,
    )
    monkeypatch.setattr(
        torch, "rand", lambda *args, **kwargs: torch.zeros(*args, **kwargs)
    )
    augment, _ = benchmark._transforms("imagenet100", "reference")
    sample = {"image": Image.new("RGB", (300, 300), (200, 200, 200)), "label": 9}
    views = augment(sample)["views"]
    statistics = benchmark.spt.data.static.ImageNet
    mean = torch.tensor(statistics["mean"])[:, None, None]
    std = torch.tensor(statistics["std"])[:, None, None]
    for view, value in zip(views, (200, 55)):
        assert view["label"] == 9 and view["image"].shape == (3, 224, 224)
        expected = (torch.full((3, 224, 224), value / 255) - mean) / std
        torch.testing.assert_close(view["image"], expected)
    assert sample["image"].getpixel((0, 0)) == (200, 200, 200)


def test_distributed_training_drops_incomplete_global_batches(tmp_path, monkeypatch):
    dataset = torch.utils.data.TensorDataset(torch.arange(31))
    args = _args(tmp_path)
    args.devices = 2
    partitions = []
    for rank in (0, 1):
        monkeypatch.setenv("RANK", str(rank))
        batches = list(benchmark._loader(dataset, args, train=True))
        assert len(batches) == len(dataset) // args.batch_size
        assert all(batch[0].shape == (4,) for batch in batches)
        partitions.append(set(torch.cat([batch[0] for batch in batches]).tolist()))
    assert partitions[0].isdisjoint(partitions[1])


@pytest.mark.parametrize("batch_size,devices", [(8, 0), (7, 2), (2, 2)])
def test_invalid_distributed_batch_is_rejected_before_creating_run(
    tmp_path, batch_size, devices
):
    args = _args(tmp_path)
    args.batch_size, args.devices = batch_size, devices
    with pytest.raises(ValueError, match="global batch size"):
        benchmark.train(args)
    assert not args.output.exists()


def test_multi_device_evaluation_is_rejected_before_writing(tmp_path):
    args = _args(tmp_path)
    args.devices = 2
    with pytest.raises(ValueError, match="require devices=1"):
        benchmark.evaluate(args)
    assert not args.output.exists()


@pytest.mark.parametrize(
    "field,value", [("epochs", 200), ("devices", 2), ("augmentation", "reference")]
)
def test_resume_cannot_change_the_training_recipe(tmp_path, field, value):
    args = _args(tmp_path)
    args.devices, args.augmentation = 1, "legacy"
    benchmark._reserve_output(args)
    args.resume = args.output / "checkpoints/last.ckpt"
    setattr(args, field, value)
    with pytest.raises(ValueError, match="same configuration"):
        benchmark._reserve_output(args)


def test_original_single_device_config_remains_resumable(tmp_path):
    args = _args(tmp_path)
    benchmark._reserve_output(args)
    args.devices, args.augmentation = 1, "legacy"
    args.resume = args.output / "checkpoints/last.ckpt"
    benchmark._reserve_output(args)


def test_epoch_end_resume_completes_the_original_optimizer_budget(tmp_path):
    args = _args(tmp_path)
    args.epochs = 4
    for cap in (4, 8, -1):
        args.max_steps = cap
        trainer = benchmark.train(args)
        summary = json.loads((args.output / "training_summary.json").read_text())
        expected = 16 if cap == -1 else cap
        assert trainer.global_step == expected
        assert summary["complete_schedule"] is (cap == -1)
        args.resume = Path(summary["checkpoint"])
        checkpoint = torch.load(args.resume, weights_only=False, map_location="cpu")
        assert checkpoint["lr_schedulers"][0]["last_epoch"] == expected
    assert summary["updates_this_invocation"] == 8


def test_cache_only_prepares_readout_without_scoring_held_out_data(
    tmp_path, monkeypatch
):
    args = _args(tmp_path)
    args.command = "evaluate"
    args.cache_only = True

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("cache preparation must not fit or score a classifier")

    monkeypatch.setattr(benchmark, "_fit_linear_classifier", forbidden_fit)
    benchmark.evaluate(args)
    features = torch.load(args.output / "features.pt", weights_only=True)
    assert features["x_train"].shape == (32, 512)
    assert features["x_val"].shape == (16, 512)
    assert (args.output / "feature_manifest.json").is_file()
    assert not (args.output / "evaluation.json").exists()
    assert not (args.output / "spectrum.pt").exists()
