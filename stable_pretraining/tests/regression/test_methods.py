"""Regression tests for all SSL methods.

Runs each method for 1 epoch on tiny fake data (CPU-only, no downloads),
then verifies:
  1. Training completed without error.
  2. The run is indexed in the SQLite registry with status='completed'.
  3. Flattened Hydra config is queryable in the registry.

Designed to run fast in CI (~30s total) and catch breakage in any
method's forward function, loss, or logging pipeline.
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining import forward, losses
from stable_pretraining.manager import Manager
from stable_pretraining.registry import open_registry

pytestmark = pytest.mark.regression


# ---------------------------------------------------------------------------
# Fake dataset (no downloads, CPU-only)
# ---------------------------------------------------------------------------


class FakeDataset(Dataset):
    """Returns dicts with an image and label.

    All data is generated deterministically at construction time
    (including multi-view augmentations) so that seeded runs are
    perfectly reproducible.
    """

    def __init__(self, num_samples=32, image_size=32, num_classes=10, multi_view=False):
        self.num_samples = num_samples
        self.multi_view = multi_view
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(num_samples, 3, image_size, image_size, generator=g)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)
        if multi_view:
            # Pre-generate views so __getitem__ is deterministic
            self.noise1 = torch.randn_like(self.images, generator=g) * 0.1
            self.noise2 = torch.randn_like(self.images, generator=g) * 0.1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.multi_view:
            # Mimics MultiViewTransform output: dict with "views" key
            return {
                "views": [
                    {
                        "image": self.images[idx] + self.noise1[idx],
                        "label": self.labels[idx],
                    },
                    {
                        "image": self.images[idx] + self.noise2[idx],
                        "label": self.labels[idx],
                    },
                ]
            }
        return {"image": self.images[idx], "label": self.labels[idx]}


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

EMBED_DIM = 64
PROJ_DIM = 32
NUM_CLASSES = 10
IMAGE_SIZE = 32


def make_backbone():
    """Tiny CNN backbone for fast testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, EMBED_DIM),
    )


def make_projector(in_dim=EMBED_DIM, out_dim=PROJ_DIM):
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.BatchNorm1d(in_dim),
        nn.ReLU(),
        nn.Linear(in_dim, out_dim),
    )


def make_predictor(in_dim=PROJ_DIM, out_dim=PROJ_DIM):
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.BatchNorm1d(in_dim),
        nn.ReLU(),
        nn.Linear(in_dim, out_dim),
    )


def make_data(multi_view=True):
    ds = FakeDataset(
        num_samples=32,
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        multi_view=multi_view,
    )
    train_dl = DataLoader(ds, batch_size=8, drop_last=True, num_workers=0)
    val_ds = FakeDataset(
        num_samples=16, image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, multi_view=False
    )
    val_dl = DataLoader(val_ds, batch_size=8, num_workers=0)
    return spt.data.DataModule(train=train_dl, val=val_dl)


def make_trainer_cfg():
    return OmegaConf.create(
        {
            "_target_": "lightning.Trainer",
            "max_epochs": 1,
            "accelerator": "cpu",
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        }
    )


# ---------------------------------------------------------------------------
# Method definitions
# ---------------------------------------------------------------------------


def build_simclr():
    return dict(
        module=spt.Module(
            backbone=make_backbone(),
            projector=make_projector(),
            forward=forward.simclr_forward,
            simclr_loss=losses.NTXEntLoss(temperature=0.5),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(multi_view=True),
    )


def build_byol():
    backbone = make_backbone()
    projector = make_projector()
    return dict(
        module=spt.Module(
            backbone=spt.backbone.TeacherStudentWrapper(backbone),
            projector=spt.backbone.TeacherStudentWrapper(projector),
            predictor=make_predictor(),
            forward=forward.byol_forward,
            byol_loss=losses.BYOLLoss(),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(multi_view=True),
    )


def build_vicreg():
    return dict(
        module=spt.Module(
            backbone=make_backbone(),
            projector=make_projector(),
            forward=forward.vicreg_forward,
            vicreg_loss=losses.VICRegLoss(),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(multi_view=True),
    )


def build_barlow_twins():
    return dict(
        module=spt.Module(
            backbone=make_backbone(),
            projector=make_projector(),
            forward=forward.barlow_twins_forward,
            barlow_loss=losses.BarlowTwinsLoss(),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(multi_view=True),
    )


def build_supervised():
    return dict(
        module=spt.Module(
            backbone=make_backbone(),
            classifier=nn.Linear(EMBED_DIM, NUM_CLASSES),
            forward=forward.supervised_forward,
            supervised_loss=nn.CrossEntropyLoss(),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(multi_view=False),
    )


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

METHOD_BUILDERS = {
    "simclr": build_simclr,
    "byol": build_byol,
    "vicreg": build_vicreg,
    "barlow_twins": build_barlow_twins,
    "supervised": build_supervised,
}

# Reference loss values captured on 2026-04-09 with seed=42.
# If any method's loss changes, the forward/loss logic has drifted.
# Re-capture with: python stable_pretraining/tests/regression/_capture_refs.py
REFERENCE_LOSSES = {
    "simclr": ("fit/loss_epoch", 1.645321011543274),
    "byol": ("fit/loss_epoch", 1.822302222251892),
    "vicreg": ("fit/loss_epoch", 18.119741439819336),
    "barlow_twins": ("fit/loss_epoch", 1.7105509042739868),
    "supervised": ("validate/loss_step", 2.3338754177093506),
}


@pytest.mark.parametrize("method_name", list(METHOD_BUILDERS.keys()))
def test_method_trains_and_registers(method_name, tmp_path):
    """Each SSL method completes 1 epoch and is indexed in the registry."""
    import lightning as pl

    pl.seed_everything(42, workers=True)
    builder = METHOD_BUILDERS[method_name]
    components = builder()

    trainer_cfg = make_trainer_cfg()
    manager = Manager(
        trainer=trainer_cfg,
        module=components["module"],
        data=components["data"],
    )
    manager()

    # Verify the run is in the registry
    from stable_pretraining._config import get_config

    reg = open_registry(cache_dir=get_config().cache_dir)

    runs = reg.query(status="completed")
    assert len(runs) >= 1, f"{method_name}: no completed run in registry"

    run = runs[-1]
    assert run.run_dir is not None
    assert run.hparams  # flattened config was stored
    assert "trainer.max_epochs" in run.hparams
    reg.close()


@pytest.mark.parametrize("method_name", list(METHOD_BUILDERS.keys()))
def test_method_matches_reference(method_name, tmp_path):
    """Loss must match the hardcoded reference value (seed=42).

    Catches correctness drift: if the forward function, loss, or
    optimizer logic changes, this test breaks even if the code is
    internally consistent (two identical runs would still agree,
    but both would be wrong).

    To update references after an intentional change, run:
        python stable_pretraining/tests/regression/_capture_refs.py
    """
    import lightning as pl

    pl.seed_everything(42, workers=True)

    builder = METHOD_BUILDERS[method_name]
    components = builder()
    trainer_cfg = make_trainer_cfg()

    manager = Manager(
        trainer=trainer_cfg,
        module=components["module"],
        data=components["data"],
        seed=42,
    )
    manager()

    from stable_pretraining._config import get_config

    reg = open_registry(cache_dir=get_config().cache_dir)
    runs = reg.query(status="completed")
    run = runs[-1]

    ref_key, ref_value = REFERENCE_LOSSES[method_name]
    assert ref_key in run.summary, (
        f"{method_name}: expected summary key '{ref_key}' not found. "
        f"Available: {list(run.summary.keys())}"
    )
    actual = run.summary[ref_key]
    assert actual == pytest.approx(ref_value, abs=1e-5), (
        f"{method_name}: loss drifted! expected={ref_value}, got={actual}. "
        "If this is intentional, re-run _capture_refs.py to update."
    )
    reg.close()
