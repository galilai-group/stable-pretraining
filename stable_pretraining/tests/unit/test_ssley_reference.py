"""Numerical parity against pinned, unmodified author source fixtures."""

import ast
import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torchvision.models import resnet18

import stable_pretraining as spt

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parents[1] / "fixtures/ssley_reference"


def _source(name):
    tree = ast.parse((FIXTURES / f"{name}.py.txt").read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "solo.utils.misc")
    ]
    scope = {"gather": lambda x: x}
    exec(compile(tree, str(FIXTURES / f"{name}.py.txt"), "exec"), scope)
    return scope


@pytest.fixture(scope="module")
def recipe():
    sys.path.insert(0, str(ROOT / "benchmarks/cifar10"))
    try:
        yield importlib.import_module("ssley_recipe")
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_released_loss_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA AMP parity requires a GPU")
    torch.manual_seed(17)
    dtype = torch.float16 if device == "cuda" else torch.float32
    x, y = [
        torch.randn(16, 64, device=device, dtype=dtype, requires_grad=True) * 0.1
        for _ in range(2)
    ]
    context = (
        torch.autocast("cuda", dtype=torch.float16)
        if device == "cuda"
        else nullcontext()
    )
    with context:
        expected = _source("loss")["ssley_loss_func"](x, y)
        actual = spt.losses.SSLEYLoss(force_float32=False)(x, y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(
        torch.autograd.grad(actual, (x, y)), torch.autograd.grad(expected, (x, y))
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


class _AuthorModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.backbone = resnet18()
        self.backbone.fc = nn.Identity()
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.classifier = nn.Linear(512, 10)
        self.projector = nn.Sequential(
            nn.Linear(512, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, views):
        projections, losses = [], []
        for view in views:
            features = self.backbone(
                view["image"].to(memory_format=torch.channels_last)
            )
            logits = self.classifier(features.detach())
            projections.append(self.projector(features))
            losses.append(
                nn.functional.cross_entropy(logits, view["label"], ignore_index=-1)
            )
        return _source("loss")["ssley_loss_func"](*projections) + sum(losses) / 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initialization_forward_gradients_and_updates(recipe, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA AMP parity requires a GPU")
    config = json.loads((ROOT / "benchmarks/cifar10/ssley-paper.json").read_text())
    width = 2048 if device == "cuda" else 16
    config["projector_dims"] = [width] * 3
    torch.manual_seed(7)
    expected = _AuthorModel(width).to(device)
    torch.manual_seed(7)
    actual = recipe.PaperSSLEY(config, 195).to(device)
    actual.log = Mock()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name in ("backbone", "classifier", "projector"):
            for a, b in zip(
                getattr(actual, name).parameters(), getattr(expected, name).parameters()
            ):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        optimizer = actual.configure_optimizers()[0][0]
        reference_optimizer = _source("lars")["LARS"](
            [
                {"params": expected.backbone.parameters()},
                {
                    "params": expected.classifier.parameters(),
                    "lr": 0.1,
                    "weight_decay": 0,
                },
                {"params": expected.projector.parameters()},
            ],
            lr=0.3,
            weight_decay=1e-4,
            eta=0.02,
            clip_lr=True,
            exclude_bias_n_norm=True,
        )
        _source("scheduler")["LinearWarmupCosineAnnealingLR"](
            reference_optimizer, 1950, 195000, warmup_start_lr=3e-5
        )
        for _ in range(2):
            views = [
                {
                    "image": torch.randn(4, 3, 32, 32, device=device),
                    "label": torch.arange(4, device=device),
                }
                for _ in range(2)
            ]
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                a = actual({"views": views}, stage="fit")["loss"]
                b = expected(views)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            a.backward()
            b.backward()
            for name in ("backbone", "classifier", "projector"):
                for p, q in zip(
                    getattr(actual, name).parameters(),
                    getattr(expected, name).parameters(),
                ):
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            optimizer.step()
            reference_optimizer.step()
            for name in ("backbone", "classifier", "projector"):
                for key, value in getattr(actual, name).state_dict().items():
                    torch.testing.assert_close(
                        value, getattr(expected, name).state_dict()[key], rtol=0, atol=0
                    )
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
    finally:
        torch.set_num_threads(old_threads)


def test_full_schedule_matches_released_source(recipe):
    optimizers = [
        torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.3) for _ in range(2)
    ]
    schedules = [
        constructor(optimizer, 1950, 195000, warmup_start_lr=3e-5)
        for constructor, optimizer in zip(
            [
                recipe.LinearWarmupCosineAnnealingLR,
                _source("scheduler")["LinearWarmupCosineAnnealingLR"],
            ],
            optimizers,
        )
    ]
    for _ in range(195000):
        assert (
            optimizers[0].param_groups[0]["lr"] == optimizers[1].param_groups[0]["lr"]
        )
        for optimizer, scheduler in zip(optimizers, schedules):
            optimizer.step()
            scheduler.step()
