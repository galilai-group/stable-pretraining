"""Data-isolation and frozen-evaluation checks for the ADNCE example."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader


@pytest.fixture
def example():
    path = Path(__file__).resolve().parents[3] / "examples" / "adnce_cifar10.py"
    spec = importlib.util.spec_from_file_location("adnce_cifar10", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCIFAR10:
    """Small deterministic image source that forbids test-set access."""

    num_classes = 10

    def __init__(self, root, train, download):
        assert train, "The official test split must never be opened by the benchmark."
        self.targets = list(range(self.num_classes)) * 10
        self.data = np.random.default_rng(0).integers(
            0, 256, (len(self.targets), 32, 32, 3), dtype=np.uint8
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return Image.fromarray(self.data[index]), self.targets[index]


class FakeCIFAR100(FakeCIFAR10):
    """Exercise all hundred labels without downloading a dataset."""

    num_classes = 100


@pytest.mark.unit
@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
def test_split_and_two_view_contract(example, monkeypatch, tmp_path, dataset):
    monkeypatch.setattr(example, "CIFAR10", FakeCIFAR10)
    monkeypatch.setattr(example, "CIFAR100", FakeCIFAR100)
    classes = 10 if dataset == "cifar10" else 100
    args = example.parse_args(
        [
            "--dataset",
            dataset,
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--val-size",
            str(classes * 2),
            "--data-dir",
            str(tmp_path),
            "--train-samples",
            str(classes * 4),
            "--probe-samples",
            str(classes * 2),
        ]
    )
    train, probe, val = example.build_loaders(args)
    train_ids = set(train.dataset.dataset.indices)
    probe_ids = set(probe.dataset.dataset.indices)
    val_ids = set(val.dataset.dataset.indices)
    assert len(train_ids) == classes * 4
    assert len(probe_ids) == len(val_ids) == classes * 2
    assert probe_ids <= train_ids
    assert not train_ids.intersection(val_ids)
    source = val.dataset.dataset.dataset
    assert set(source.targets[index] for index in val_ids) == set(range(classes))
    probe_index = probe.dataset.dataset.indices[0]
    pixels = torch.from_numpy(source.data[probe_index]).permute(2, 0, 1).float() / 255
    expected = (pixels - torch.tensor(example.MEAN)[:, None, None]) / torch.tensor(
        example.STD
    )[:, None, None]
    torch.testing.assert_close(probe.dataset[0]["image"], expected)
    batch = next(iter(train))
    assert len(batch["views"]) == 2
    for view in batch["views"]:
        assert view["image"].shape == (8, 3, 32, 32)
        assert view["label"].shape == (8,)
        assert torch.isfinite(view["image"]).all()
    torch.testing.assert_close(batch["views"][0]["label"], batch["views"][1]["label"])
    args.seed = 99
    _, _, other_val = example.build_loaders(args)
    assert val.dataset.dataset.indices == other_val.dataset.dataset.indices


@pytest.mark.unit
@pytest.mark.parametrize("option", ["--val-size", "--train-samples", "--probe-samples"])
def test_cifar100_rejects_too_small_splits(example, option):
    with pytest.raises(SystemExit):
        example.parse_args(["--dataset", "cifar100", option, "99"])


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_features_preserve_normalization_statistics(example, dtype):
    backbone = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 4)).to(dtype=dtype)
    backbone.train()
    before = {key: value.clone() for key, value in backbone.state_dict().items()}
    loader = DataLoader(
        [
            {"image": torch.tensor([1.0, 2.0, 3.0]), "label": 0},
            {"image": torch.tensor([3.0, 4.0, 1.0]), "label": 1},
        ],
        batch_size=2,
    )
    features, labels = example.extract_features(backbone, loader, torch.device("cpu"))
    assert backbone.training
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(labels, [0, 1])
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(parameter.grad is None for parameter in backbone.parameters())


@pytest.mark.unit
def test_reference_architecture_and_matched_initialization(example):
    args = example.parse_args(["--backbone", "resnet18"])
    adnce = example.build_module(args)
    args.method = "simclr"
    simclr = example.build_module(args)
    assert adnce.backbone.conv1.kernel_size == (3, 3)
    assert isinstance(adnce.backbone.maxpool, nn.Identity)
    assert adnce.projector[0].bias is None
    assert adnce.projector[-1].out_features == 128
    for name in ("backbone", "projector"):
        for key, value in getattr(adnce, name).state_dict().items():
            torch.testing.assert_close(value, getattr(simclr, name).state_dict()[key])


@pytest.mark.unit
def test_initial_probe_survives_resume(example):
    args = example.parse_args([])
    original = example.FrozenProbeReport(None, None, args)
    original.initial = {"validation_accuracy": 0.12, "mean_feature_std": 0.03}
    resumed = example.FrozenProbeReport(None, None, args)
    resumed.load_state_dict(original.state_dict())
    assert resumed.initial == original.initial


@pytest.mark.integration
@pytest.mark.parametrize("method", ["adnce", "simclr"])
@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
@pytest.mark.parametrize(
    "accelerator",
    [
        "cpu",
        pytest.param(
            "gpu",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is unavailable"
                ),
            ],
        ),
    ],
)
def test_example_trains_reports_and_resumes(
    example, monkeypatch, tmp_path, method, dataset, accelerator
):
    import json

    from lightning.pytorch.strategies import SingleDeviceStrategy

    import stable_pretraining as spt

    teardown_started = False
    evaluation_devices = []
    original_teardown = SingleDeviceStrategy.teardown
    original_evaluate = example.FrozenProbeReport._evaluate

    def record_teardown(strategy):
        nonlocal teardown_started
        teardown_started = True
        original_teardown(strategy)

    def evaluate_before_teardown(report, module):
        assert not teardown_started, "Probe ran after the model was moved to CPU."
        evaluation_devices.append(module.device.type)
        return original_evaluate(report, module)

    monkeypatch.setattr(SingleDeviceStrategy, "teardown", record_teardown)
    monkeypatch.setattr(
        example.FrozenProbeReport, "_evaluate", evaluate_before_teardown
    )
    monkeypatch.setattr(example, "CIFAR10", FakeCIFAR10)
    monkeypatch.setattr(example, "CIFAR100", FakeCIFAR100)
    classes = 10 if dataset == "cifar10" else 100
    monkeypatch.setattr(
        spt.backbone,
        "from_torchvision",
        lambda *args, **kwargs: nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512)
        ),
    )
    original_threads = torch.get_num_threads()
    original_cache = spt.get_config().cache_dir
    torch.set_num_threads(1)
    common = [
        "--method",
        method,
        "--dataset",
        dataset,
        "--accelerator",
        accelerator,
        "--backbone",
        "resnet18",
        "--batch-size",
        "10",
        "--num-workers",
        "0",
        "--val-size",
        str(classes * 2),
        "--train-samples",
        str(classes * 4),
        "--data-dir",
        str(tmp_path),
    ]
    try:
        first_output = tmp_path / "first"
        example.main(
            common
            + ["--output-dir", str(first_output), "--epochs", "1", "--max-steps", "2"]
        )
        first = json.loads(next(first_output.rglob("evaluation.json")).read_text())
        checkpoint = next(first_output.rglob("last.ckpt"))
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert saved["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(
            1e-3
        )
        assert saved["lr_schedulers"][0]["factor"] == 1.0
        assert first["global_step"] == 2
        assert first["args"]["dataset"] == dataset
        assert first["initial_frozen_ridge"] is not None
        assert first["final_frozen_ridge"] is not None
        assert first["final_frozen_probe_device"].startswith(
            "cuda" if accelerator == "gpu" else "cpu"
        )
        assert teardown_started
        assert evaluation_devices == ["cuda" if accelerator == "gpu" else "cpu"] * 2
        if accelerator == "gpu":
            assert first["rank_zero_peak_cuda_bytes"] > 0
        else:
            assert first["rank_zero_peak_cuda_bytes"] is None
        teardown_started = False
        second_output = tmp_path / "second"
        example.main(
            common
            + [
                "--output-dir",
                str(second_output),
                "--epochs",
                "2",
                "--max-steps",
                "3",
                "--resume",
                str(checkpoint),
            ]
        )
        second = json.loads(next(second_output.rglob("evaluation.json")).read_text())
        assert second["global_step"] == 3
        assert first["initial_frozen_ridge"] == second["initial_frozen_ridge"]
        assert first["train_images"] == second["train_images"] == classes * 4
        assert teardown_started
        assert evaluation_devices == ["cuda" if accelerator == "gpu" else "cpu"] * 3
        if accelerator == "gpu":
            assert second["rank_zero_peak_cuda_bytes"] > 0
    finally:
        torch.set_num_threads(original_threads)
        spt.set(cache_dir=original_cache)


@pytest.mark.integration
@pytest.mark.parametrize("no_training", ["zero_steps", "completed_resume"])
def test_example_reports_once_when_training_is_skipped(
    example, monkeypatch, tmp_path, no_training
):
    import json

    from lightning.pytorch.strategies import SingleDeviceStrategy

    import stable_pretraining as spt

    monkeypatch.setattr(example, "CIFAR10", FakeCIFAR10)
    monkeypatch.setattr(
        spt.backbone,
        "from_torchvision",
        lambda *args, **kwargs: nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512)
        ),
    )
    original_threads = torch.get_num_threads()
    original_cache = spt.get_config().cache_dir
    torch.set_num_threads(1)
    common = [
        "--accelerator",
        "cpu",
        "--backbone",
        "resnet18",
        "--batch-size",
        "10",
        "--num-workers",
        "0",
        "--val-size",
        "20",
        "--train-samples",
        "40",
        "--data-dir",
        str(tmp_path),
        "--epochs",
        "1",
    ]
    try:
        resume = []
        expected_step = 0
        if no_training == "completed_resume":
            first_output = tmp_path / "trained"
            example.main(
                common + ["--output-dir", str(first_output), "--max-steps", "2"]
            )
            resume = ["--resume", str(next(first_output.rglob("last.ckpt")))]
            expected_step = 2

        teardown_started = False
        evaluations = []
        original_teardown = SingleDeviceStrategy.teardown
        original_evaluate = example.FrozenProbeReport._evaluate

        def record_teardown(strategy):
            nonlocal teardown_started
            teardown_started = True
            original_teardown(strategy)

        def record_evaluation(report, module):
            evaluations.append(teardown_started)
            return original_evaluate(report, module)

        monkeypatch.setattr(SingleDeviceStrategy, "teardown", record_teardown)
        monkeypatch.setattr(example.FrozenProbeReport, "_evaluate", record_evaluation)
        output = tmp_path / "skipped"
        example.main(
            common
            + resume
            + ["--output-dir", str(output), "--max-steps", str(expected_step)]
        )
        reports = list(output.rglob("evaluation.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["global_step"] == expected_step
        assert report["initial_frozen_ridge"] is not None
        assert report["final_frozen_ridge"] is not None
        assert report["final_frozen_probe_device"] == "cpu"
        assert report["rank_zero_peak_cuda_bytes"] is None
        assert evaluations == ([False, True] if expected_step == 0 else [True])
    finally:
        torch.set_num_threads(original_threads)
        spt.set(cache_dir=original_cache)


@pytest.mark.unit
def test_resumed_checkpoint_is_not_treated_as_random_initialization(example):
    from types import SimpleNamespace

    report = example.FrozenProbeReport(None, None, example.parse_args([]))

    def unexpected_evaluation(module):
        pytest.fail("A resumed encoder must not be reported as its initial state.")

    report._evaluate = unexpected_evaluation
    trainer = SimpleNamespace(
        is_global_zero=True,
        ckpt_path="existing.ckpt",
        strategy=SimpleNamespace(broadcast=lambda value: value),
    )
    report.on_fit_start(trainer, SimpleNamespace(device=torch.device("cpu")))
    assert report.initial is None


@pytest.mark.unit
def test_frozen_parameterless_backbone_preserves_input_dtype(example):
    backbone = nn.Identity()
    seen_dtypes = []
    backbone.register_forward_pre_hook(
        lambda module, inputs: seen_dtypes.append(inputs[0].dtype)
    )
    loader = DataLoader(
        [
            {"image": torch.tensor([1.0, 2.0], dtype=torch.float64), "label": 0},
        ],
        batch_size=1,
    )
    features, labels = example.extract_features(backbone, loader, torch.device("cpu"))
    assert seen_dtypes == [torch.float64]
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(labels, [0])
