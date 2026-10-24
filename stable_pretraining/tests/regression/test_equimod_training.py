"""CPU regression tests for EquiMod under the Manager: training and requeue-resume.

Runs the three-view EquiMod pipeline on tiny fake data (no downloads), for
both entry points (``forward.equimod`` on a composable ``Module`` and the
``EquiMod`` class), and checks that a SLURM-style requeue resumes from the
per-epoch checkpoint and reproduces the uninterrupted run.
"""

import json
import signal
import types

import lightning as pl
import pytest
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining import forward, losses
from stable_pretraining.manager import Manager
from stable_pretraining.methods import EquiMod
from stable_pretraining.methods.equimod import EquivarianceModule
from stable_pretraining.registry import open_registry

pytestmark = pytest.mark.regression

EMBED_DIM = 16
IMAGE_SIZE = 32
NUM_CLASSES = 10
SEED = 42


class FakeThreeViewDataset(Dataset):
    """``[original, view1, view2]`` with the parameter records the transforms write.

    Everything is generated once at construction, so seeded runs are exactly
    reproducible and a resumed run sees the same batches as an uninterrupted one.
    """

    def __init__(self, num_samples=32):
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,), generator=g)
        self.views = []
        for _ in range(2):
            noise = torch.randn_like(self.images, generator=g) * 0.1
            crop = torch.stack(
                [
                    torch.randint(0, 8, (num_samples,), generator=g),
                    torch.randint(0, 8, (num_samples,), generator=g),
                    torch.randint(16, 33, (num_samples,), generator=g),
                    torch.randint(16, 33, (num_samples,), generator=g),
                ],
                dim=1,
            ).float()
            factors = 1 + 0.4 * (torch.rand(num_samples, 4, generator=g) - 0.5)
            factors[:, 3] = 0.1 * (torch.rand(num_samples, generator=g) - 0.5)
            order = torch.stack(
                [torch.randperm(4, generator=g) for _ in range(num_samples)]
            )
            jitter = torch.cat([factors, order.float()], dim=1)
            jitter[::5] = 0.0  # every fifth sample skipped the jitter
            self.views.append(
                {
                    "image": self.images + noise,
                    "RandomResizedCrop": crop,
                    "RandomHorizontalFlip": torch.rand(num_samples, generator=g) < 0.5,
                    "ColorJitter": jitter,
                    "RandomGrayscale": torch.rand(num_samples, generator=g) < 0.2,
                }
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        original = {"image": self.images[idx], "label": self.labels[idx]}
        views = [
            {**{k: v[idx] for k, v in view.items()}, "label": self.labels[idx]}
            for view in self.views
        ]
        return {"views": [original, *views]}


class FakeEvalDataset(Dataset):
    """Single clean images with labels for the online probe."""

    def __init__(self, num_samples=16):
        g = torch.Generator().manual_seed(1)
        self.images = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,), generator=g)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


def make_encoder():
    return nn.Sequential(
        nn.Conv2d(3, EMBED_DIM, 3, stride=2, padding=1),
        nn.BatchNorm2d(EMBED_DIM),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


def make_data():
    return spt.data.DataModule(
        train=DataLoader(FakeThreeViewDataset(), batch_size=8, drop_last=True),
        val=DataLoader(FakeEvalDataset(), batch_size=8),
    )


STEPS_PER_EPOCH = 32 // 8
OPTIM = {"optimizer": {"type": "Adam", "lr": 1e-3}}


def build_composable_module():
    """forward.equimod on a plain Module: the YAML-config form."""
    pl.seed_everything(SEED, workers=True)
    return spt.Module(
        backbone=make_encoder(),
        projector=nn.Sequential(
            nn.Linear(EMBED_DIM, 16), nn.BatchNorm1d(16), nn.Linear(16, 8)
        ),
        forward=forward.equimod,
        simclr_loss=losses.NTXEntLoss(temperature=0.5),
        equi=EquivarianceModule(
            EMBED_DIM, projector_dims=(16, 8), param_projector_dims=(8,)
        ),
        equi_loss=losses.NTXEntLoss(temperature=0.2),
        optim=OPTIM,
    )


def build_method_class():
    """The EquiMod class driven by forward.equimod, as the benchmark does."""
    pl.seed_everything(SEED, workers=True)
    model = EquiMod(
        encoder_name=make_encoder(),
        projector_dims=(16, 8),
        equi_projector_dims=(16, 8),
        param_projector_dims=(8,),
    )
    model.forward = types.MethodType(forward.equimod, model)
    model.optim = OPTIM
    return model


BUILDERS = {"module": build_composable_module, "method_class": build_method_class}

# fit/loss_epoch after one seeded epoch of test_equimod_trains_under_manager
# (captured on CPU; re-capture by printing ``summary["fit/loss_epoch"]`` there).
# If this drifts, the forward, loss or optimisation logic changed.
REFERENCE_LOSS = {"module": 6.997955322265625, "method_class": 6.734986305236816}

TRAINER = {
    "accelerator": "cpu",
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


def make_trainer(max_epochs, callbacks=()):
    return pl.Trainer(
        max_epochs=max_epochs, callbacks=list(callbacks), logger=False, **TRAINER
    )


@pytest.fixture
def restore_signals():
    """Snapshot SIGTERM/SIGUSR2 and restore them after the test.

    With ``SLURM_JOB_ID`` set, ``Manager`` installs a process-global SIGTERM
    forwarder; left in place it would outlive the test and poison the runner.
    """
    prior = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR2)}
    yield
    for sig, handler in prior.items():
        signal.signal(sig, handler)


class _PreemptAfterFirstEpoch(pl.Callback):
    """Stop after epoch 1 while keeping the planned horizon (so the LR schedule matches)."""

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def _run_summary(manager) -> dict:
    """Final metrics of the run this Manager trained, read from its sidecar."""
    sidecar = json.loads((manager._run_dir / "sidecar.json").read_text())
    assert sidecar["status"] == "completed", sidecar["status"]
    return sidecar["summary"]


def _registry_has_completed_run() -> bool:
    from stable_pretraining._config import get_config

    reg = open_registry(cache_dir=get_config().cache_dir)
    try:
        return bool(reg.query(status="completed"))
    finally:
        reg.close()


@pytest.mark.parametrize("entry_point", list(BUILDERS))
def test_equimod_trains_under_manager(entry_point):
    """One epoch of real Manager training on both entry points, with a probe."""
    module = BUILDERS[entry_point]()
    probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(EMBED_DIM, NUM_CLASSES),
        loss=nn.CrossEntropyLoss(),
        metrics={"top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES)},
    )
    trainer = make_trainer(max_epochs=1, callbacks=[probe])
    manager = Manager(trainer=trainer, module=module, data=make_data(), seed=SEED)
    manager()

    assert _registry_has_completed_run()
    summary = _run_summary(manager)
    for key in (
        "fit/loss_epoch",
        "fit/inv_epoch",
        "fit/equi_epoch",
        "fit/equi_gain_epoch",
    ):
        assert key in summary, (key, sorted(summary))
        assert torch.isfinite(torch.tensor(summary[key]))
    assert summary["fit/loss_epoch"] > summary["fit/inv_epoch"]
    assert summary["fit/loss_epoch"] == pytest.approx(
        REFERENCE_LOSS[entry_point], abs=1e-3
    ), f"{entry_point}: loss drifted from the reference value"
    assert "eval/linear_probe_top1_epoch" in summary, sorted(summary)


@pytest.mark.parametrize("entry_point", list(BUILDERS))
def test_equimod_requeue_resumes_and_matches_uninterrupted(
    entry_point, monkeypatch, restore_signals
):
    """A SLURM requeue resumes from last.ckpt and ends where a 2-epoch run ends."""
    builder = BUILDERS[entry_point]

    # Reference: two uninterrupted epochs.
    reference = builder()
    reference_manager = Manager(
        trainer=make_trainer(max_epochs=2),
        module=reference,
        data=make_data(),
        seed=SEED,
    )
    reference_manager()
    reference_loss = _run_summary(reference_manager)["fit/loss_epoch"]

    # Leg 1: the same job plans two epochs but is preempted after the first;
    # the Manager's requeue checkpoint is written at the epoch end.
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    first = Manager(
        trainer=make_trainer(max_epochs=2, callbacks=[_PreemptAfterFirstEpoch()]),
        module=builder(),
        data=make_data(),
        seed=SEED,
    )
    first()
    run_dir = first._run_dir
    ckpt_path = run_dir / "checkpoints" / "last.ckpt"
    assert ckpt_path.is_file()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["global_step"] == STEPS_PER_EPOCH, ckpt["global_step"]
    assert "equi.predictor.0.weight" in ckpt["state_dict"]

    # Leg 2: SLURM requeues the job; the Manager must resume into the same run_dir.
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    resumed = builder()
    second = Manager(
        trainer=make_trainer(max_epochs=2),
        module=resumed,
        data=make_data(),
        seed=SEED,
    )
    second()
    assert second._run_dir == run_dir
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["global_step"] == 2 * STEPS_PER_EPOCH, (
        f"resumed run ended at step {ckpt['global_step']}, expected "
        f"{2 * STEPS_PER_EPOCH}: training restarted from scratch or repeated an epoch"
    )

    resumed_loss = _run_summary(second)["fit/loss_epoch"]
    assert resumed_loss == pytest.approx(reference_loss, abs=1e-5), (
        resumed_loss,
        reference_loss,
    )
    resumed_state = resumed.state_dict()
    for key, value in reference.state_dict().items():
        if value.is_floating_point():
            assert torch.allclose(value, resumed_state[key], atol=1e-6), key
        else:
            assert torch.equal(value, resumed_state[key]), key
