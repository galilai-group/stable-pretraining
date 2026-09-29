"""Execute onboarding recipes, documentation snippets, and resume contracts."""

import argparse
from copy import deepcopy
from pathlib import Path
import re
import subprocess
import sys

from PIL import Image
import pytest
import torch

import stable_pretraining as spt
from stable_pretraining import quickstart

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def isolated_runtime():
    cfg = spt.get_config()
    state = deepcopy(cfg.__dict__)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    cfg.__dict__.clear()
    cfg.__dict__.update(state)
    torch.set_num_threads(threads)


def options(tmp_path, **kw):
    return argparse.Namespace(
        **(
            dict(
                method="simclr",
                data=None,
                epochs=1,
                steps=-1,
                resume=None,
                cache_dir=tmp_path,
            )
            | kw
        )
    )


@pytest.fixture
def image_root(tmp_path):
    root = tmp_path / "images"
    for split in ("train", "val"):
        for label in ("a", "b"):
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (20, 24), color=(30 + i * 10, 80, 120)).save(
                    folder / f"{i}.png"
                )
    return root


@pytest.mark.parametrize("method", ["simclr", "jet-entropy"])
def test_quickstart_trains_validates_saves_and_resumes(tmp_path, method):
    trainer = quickstart._run(options(tmp_path / "first", method=method))
    assert trainer.global_step == 4
    assert 0 <= trainer.callback_metrics["eval/probe_accuracy"] <= 1
    assert torch.isfinite(trainer.callback_metrics["fit/loss"])
    checkpoints = list((tmp_path / "first").rglob("*.ckpt"))
    assert checkpoints
    saved = torch.load(checkpoints[0], weights_only=False, map_location="cpu")
    assert saved["global_step"] == 4
    assert len(saved["optimizer_states"]) == 2
    assert all(opt["state"] for opt in saved["optimizer_states"])
    assert list((tmp_path / "first").rglob("metrics.csv"))
    if method == "jet-entropy":
        assert saved["hyper_parameters"]["scale_eps"] == 1e-4
        assert (
            saved["state_dict"]["backbone._extra_state"]["scale_parameterization"]
            == "exp_floor"
        )
        for metric in (
            "flow/layer_0/log_scale_max",
            "flow/logdet_per_dim_mean",
            "tokens/semantic_energy",
            "tokens/residual_energy",
        ):
            assert torch.isfinite(trainer.callback_metrics[metric])
    continued = quickstart._run(
        options(tmp_path / "second", method=method, epochs=2, resume=checkpoints[0])
    )
    assert continued.global_step == 8
    assert continued.current_epoch == 2


def test_custom_image_training_and_held_out_data(tmp_path, image_root):
    train, val, classes = quickstart._datasets(image_root)
    assert classes == 2
    assert not set(train.dataset.samples) & set(val.dataset.samples)
    trainer = quickstart._run(options(tmp_path / "run", data=image_root))
    assert trainer.global_step == 1
    assert torch.isfinite(trainer.callback_metrics["eval/probe_accuracy"])


def test_reject_mismatched_classes(image_root):
    (image_root / "val" / "b").rename(image_root / "val" / "c")
    with pytest.raises(ValueError, match="same class"):
        quickstart._datasets(image_root)


def test_reject_too_few_images(image_root):
    (image_root / "train" / "a" / "0.png").unlink()
    with pytest.raises(ValueError, match="eight"):
        quickstart._datasets(image_root)


def test_reject_single_class(tmp_path):
    for split in ("train", "val"):
        folder = tmp_path / split / "only"
        folder.mkdir(parents=True)
        for i in range(8):
            Image.new("RGB", (16, 16)).save(folder / f"{i}.png")
    with pytest.raises(ValueError, match="two classes"):
        quickstart._datasets(tmp_path)


@pytest.mark.parametrize("kw", [{"epochs": 0}, {"steps": 0}, {"steps": -2}])
def test_invalid_duration(tmp_path, kw):
    with pytest.raises(ValueError, match="epochs"):
        quickstart._run(options(tmp_path, **kw))


@pytest.mark.parametrize("guide", ["jet", "online_evaluation", "custom_images"])
def test_documented_python_executes(guide, image_root, monkeypatch):
    monkeypatch.chdir(image_root.parent)
    source = (ROOT / "docs/source/guides" / f"{guide}.md").read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", source, flags=re.DOTALL)
    assert blocks
    namespace = {}
    for code in blocks:
        exec(compile(code, f"{guide}.md", "exec"), namespace)


def test_cli_argument_dispatch(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(quickstart, "_run", seen.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quickstart",
            "--method",
            "jet-entropy",
            "--cache-dir",
            str(tmp_path),
            "--steps",
            "2",
        ],
    )
    quickstart.main()
    assert seen[0].method == "jet-entropy" and seen[0].steps == 2


def test_step_limit_counts_backbone_steps(tmp_path):
    trainer = quickstart._run(options(tmp_path, steps=2))
    assert trainer.global_step == 2


def test_new_forward_export_is_lazy_in_fresh_process():
    code = """
import sys
import stable_pretraining as spt
assert 'stable_pretraining.forward' not in sys.modules
assert 'torch' not in sys.modules
assert 'forward' in dir(spt)
assert callable(spt.forward.simclr)
assert spt.forward is sys.modules['stable_pretraining.forward']
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, timeout=120, capture_output=True
    )


def test_jet_objective_rejects_nonfinite_loss():
    from types import SimpleNamespace

    def backbone(images):
        return torch.full((images.shape[0], 2, 2), float("inf")), torch.zeros(
            images.shape[0]
        )

    module = SimpleNamespace(backbone=backbone, entropy_weight=0.01)
    view = {
        "image": torch.ones(2, 3, 16, 16),
        "label": torch.zeros(2, dtype=torch.long),
    }
    with pytest.raises(FloatingPointError):
        quickstart._jet_entropy(module, {"views": [view, view]}, "fit")
