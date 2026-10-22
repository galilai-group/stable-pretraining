"""Paper recipe tests independent of a live CUDA device."""

import importlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def recipe_module():
    path = str(ROOT / "benchmarks/cifar10")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("ssley_recipe")
    finally:
        sys.path.remove(path)


def test_paper_config_and_augmentation(recipe_module):
    config = json.loads((ROOT / "benchmarks/cifar10/ssley-paper.json").read_text())
    assert config["epochs"] == 1000 and config["batch_size"] == 256
    assert config["projector_dims"] == [2048, 2048, 2048]
    assert config["source_commit"] == "11eac929a3cfb8fe28cefa573ef87e08fcf1eb27"
    assert config["crop_scale"] == [0.2, 1.0]
    assert config["solarization_probabilities"] == [0.1, 0.1]
    train, val = recipe_module.augmentations(config)
    sample = {"image": Image.new("RGB", (32, 32)), "label": 2}
    views = train(sample)["views"]
    assert len(views) == 2
    assert all(view["image"].shape == (3, 32, 32) for view in views)
    assert val(sample)["image"].shape == (3, 32, 32)


def test_experiment_configs_differ_only_in_stability_options():
    def load(name):
        return json.loads((ROOT / f"benchmarks/cifar10/ssley-{name}.json").read_text())

    paper, stable, fp32_only = load("paper"), load("stable"), load("fp32-only")
    metadata = {"name", "implementation_revision"}
    stability = {"loss_force_float32", "unadapted_grad_clip_norm"}
    shared = set(paper) - metadata
    assert set(stable) - metadata - stability == shared
    assert set(fp32_only) - metadata - stability == shared
    assert all(paper[key] == stable[key] == fp32_only[key] for key in shared)
    assert stable["loss_force_float32"] and fp32_only["loss_force_float32"]
    assert stable["unadapted_grad_clip_norm"] == 1.0
    assert "unadapted_grad_clip_norm" not in fp32_only


def test_reference_lars_zero_decay_adaptation(recipe_module):
    weight = torch.nn.Parameter(torch.ones(2, 2))
    bias = torch.nn.Parameter(torch.ones(2))
    optimizer = recipe_module.ReferenceLARS(
        [weight, bias],
        lr=0.1,
        weight_decay=0,
        eta=0.02,
        clip_lr=True,
        exclude_bias_n_norm=True,
    )
    weight.grad = torch.ones_like(weight)
    bias.grad = torch.ones_like(bias)
    optimizer.step()
    # The trust ratio makes a 2% update to matrix weights even with zero decay.
    torch.testing.assert_close(weight, torch.full_like(weight, 0.98))
    torch.testing.assert_close(bias, torch.full_like(bias, 0.9))


def test_warmup_endpoints_and_scheduler_resume(recipe_module):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.3)
    scheduler = recipe_module.LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=4, max_epochs=12, warmup_start_lr=3e-5
    )
    rates = []
    for step in range(13):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] == pytest.approx(3e-5)
    assert rates[3] == pytest.approx(0.3)
    assert rates[4] == pytest.approx(0.3)
    assert rates[8] == pytest.approx(0.15)
    assert rates[12] == pytest.approx(0)
    assert all(math.isfinite(rate) for rate in rates)
    saved = scheduler.state_dict()
    other = recipe_module.LinearWarmupCosineAnnealingLR(optimizer, 4, 12)
    other.load_state_dict(saved)
    assert other.get_last_lr() == scheduler.get_last_lr()


def test_classifier_does_not_train_backbone(recipe_module):
    config = json.loads((ROOT / "benchmarks/cifar10/ssley-paper.json").read_text())
    config["projector_dims"] = [16, 16, 8]
    model = recipe_module.PaperSSLEY(config, steps_per_epoch=2)
    assert model.backbone.conv1.padding == (2, 2)
    model.log = Mock()
    batch = {
        "views": [
            {"image": torch.randn(2, 3, 32, 32), "label": torch.tensor([0, 1])}
            for _ in range(2)
        ]
    }
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = model(batch, stage="fit")
        output["classification_loss"].backward()
        assert model.classifier.weight.grad is not None
        assert all(p.grad is None for p in model.backbone.parameters())
        assert all(p.grad is None for p in model.projector.parameters())
        model.zero_grad(set_to_none=True)
        output = model(batch, stage="fit")
        (output["loss"] - output["classification_loss"]).backward()
        assert model.backbone.conv1.weight.grad is not None
    finally:
        torch.set_num_threads(old_threads)


def test_new_allocation_resets_wall_time(recipe_module):
    timer = recipe_module.SessionTimer(duration="00:01:00:00")
    timer.load_state_dict({"time_elapsed": {"train": 3600}})
    assert timer.time_elapsed() == 0
    assert timer.time_remaining() == 3600


def test_stability_clips_only_unadapted_finite_vectors(recipe_module):
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    vector = torch.nn.Parameter(torch.ones(2))
    matrix.grad = torch.full_like(matrix, 100)
    vector.grad = torch.full_like(vector, 100)
    optimizer = recipe_module.ReferenceLARS(
        [matrix, vector], lr=0.3, exclude_bias_n_norm=True
    )
    module = Mock(recipe={"unadapted_grad_clip_norm": 1.0})
    recipe_module.PaperSSLEY.on_before_optimizer_step(module, optimizer)
    torch.testing.assert_close(matrix.grad, torch.full_like(matrix, 100))
    assert vector.grad.norm() <= 1.000001
    vector.grad.fill_(float("inf"))
    recipe_module.PaperSSLEY.on_before_optimizer_step(module, optimizer)
    assert torch.isinf(vector.grad).all()
