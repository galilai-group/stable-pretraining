"""Helper to capture reference loss values for regression tests.

Run once, copy the output into REFERENCE_LOSSES in test_methods.py.
"""

import tempfile

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining import forward, losses
from stable_pretraining._config import get_config
from stable_pretraining.manager import Manager
from stable_pretraining.registry import open_registry

EMBED_DIM, PROJ_DIM, NUM_CLASSES, IMAGE_SIZE = 64, 32, 10, 32


def make_backbone():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, EMBED_DIM),
    )


def make_projector():
    return nn.Sequential(
        nn.Linear(EMBED_DIM, EMBED_DIM),
        nn.BatchNorm1d(EMBED_DIM),
        nn.ReLU(),
        nn.Linear(EMBED_DIM, PROJ_DIM),
    )


def make_predictor():
    return nn.Sequential(
        nn.Linear(PROJ_DIM, PROJ_DIM),
        nn.BatchNorm1d(PROJ_DIM),
        nn.ReLU(),
        nn.Linear(PROJ_DIM, PROJ_DIM),
    )


class FakeDataset(Dataset):
    def __init__(self, n=32, multi_view=False, n_views=2):
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        self.labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)
        self.multi_view = multi_view
        if multi_view:
            # Views beyond the first two are drawn after them, so the two-view
            # data (and the references captured on it) do not change.
            self.noises = [
                torch.randn_like(self.images, generator=g) * 0.1 for _ in range(n_views)
            ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.multi_view:
            return {
                "views": [
                    {"image": self.images[i] + noise[i], "label": self.labels[i]}
                    for noise in self.noises
                ]
            }
        return {"image": self.images[i], "label": self.labels[i]}


def make_data(multi_view=True, n_views=2):
    return spt.data.DataModule(
        train=DataLoader(
            FakeDataset(32, multi_view, n_views),
            batch_size=8,
            drop_last=True,
            num_workers=0,
        ),
        val=DataLoader(FakeDataset(16, False), batch_size=8, num_workers=0),
    )


METHODS = {
    "simclr": lambda: dict(
        module=spt.Module(
            backbone=make_backbone(),
            projector=make_projector(),
            forward=forward.simclr,
            simclr_loss=losses.NTXEntLoss(temperature=0.5),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(True),
    ),
    "poly_view": lambda: dict(
        module=spt.Module(
            backbone=make_backbone(),
            projector=make_projector(),
            forward=forward.poly_view,
            poly_view_loss=losses.PVCLoss(temperature=0.5, aggregation="geometric"),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(True, n_views=4),
    ),
    "byol": lambda: dict(
        module=spt.Module(
            backbone=spt.backbone.TeacherStudentWrapper(make_backbone()),
            projector=spt.backbone.TeacherStudentWrapper(make_projector()),
            predictor=make_predictor(),
            forward=forward.byol,
            byol_loss=losses.BYOLLoss(),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(True),
    ),
    "vicreg": lambda: dict(
        module=spt.Module(
            backbone=make_backbone(),
            projector=make_projector(),
            forward=forward.vicreg,
            vicreg_loss=losses.VICRegLoss(),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(True),
    ),
    "barlow_twins": lambda: dict(
        module=spt.Module(
            backbone=make_backbone(),
            projector=make_projector(),
            forward=forward.barlow_twins,
            barlow_loss=losses.BarlowTwinsLoss(),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(True),
    ),
    "supervised": lambda: dict(
        module=spt.Module(
            backbone=make_backbone(),
            classifier=nn.Linear(EMBED_DIM, NUM_CLASSES),
            forward=forward.supervised,
            supervised_loss=nn.CrossEntropyLoss(),
            optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        ),
        data=make_data(False),
    ),
}

trainer_cfg = OmegaConf.create(
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


if __name__ == "__main__":
    import lightning as pl

    print("REFERENCE_LOSSES = {", flush=True)
    for name, builder in METHODS.items():
        tmp = tempfile.mkdtemp()
        get_config()._cache_dir = tmp
        pl.seed_everything(42, workers=True)
        c = builder()
        Manager(trainer=trainer_cfg, module=c["module"], data=c["data"], seed=42)()
        reg = open_registry(cache_dir=tmp)
        run = reg.query(status="completed")[-1]
        loss_keys = [k for k in run.summary if "loss" in k.lower()]
        if loss_keys:
            v = run.summary[loss_keys[0]]
            print(f'    "{name}": ("{loss_keys[0]}", {v!r}),', flush=True)
    print("}", flush=True)
