"""MMCR training, probing and checkpoint resume through Manager."""

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy

import stable_pretraining as spt


class _Samples(Dataset):
    """Small aligned tensors for an offline training lifecycle test."""

    def __init__(self, multiview):
        self.multiview = multiview
        self.x = torch.randn(12, 5, generator=torch.Generator().manual_seed(43))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        sample = {"image": self.x[index], "label": torch.tensor(index % 2)}
        if self.multiview:
            return {"views": [sample, {**sample, "image": self.x[index] + 0.1}]}
        return sample


@pytest.mark.unit
@pytest.mark.parametrize("api", ["method", "forward"])
def test_manager_probe_and_resume(tmp_path, api, monkeypatch):
    config = spt.get_config()
    # spt.set merges callback flags; isolate them from subsequent tests.
    monkeypatch.setattr(config, "_default_callbacks", config.default_callbacks)
    monkeypatch.setattr(config, "_requeue_checkpoint", config.requeue_checkpoint)
    spt.set(
        cache_dir=str(tmp_path),
        requeue_checkpoint=False,
        default_callbacks={
            "env_dump": False,
            "trainer_info": False,
            "module_summary": False,
            "slurm_info": False,
        },
    )

    def run(epochs, ckpt=None):
        components = dict(
            backbone=nn.Linear(5, 4),
            projector=nn.Linear(4, 3),
            optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
        )
        model = (
            spt.MMCR(**components, lmbda=0.1)
            if api == "method"
            else spt.Module(
                **components,
                forward=spt.forward.mmcr,
                mmcr_loss=spt.losses.MMCRLoss(0.1),
            )
        )
        probe = spt.OnlineProbe(
            model,
            name="probe",
            input="embedding",
            target="label",
            probe=nn.Linear(4, 2),
            loss=nn.CrossEntropyLoss(),
            optimizer={"type": "AdamW", "lr": 1e-3},
            metrics={"accuracy": MulticlassAccuracy(2)},
            verbose=False,
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=epochs,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            default_root_dir=str(tmp_path),
            callbacks=[probe],
        )
        data = spt.data.DataModule(
            train=DataLoader(_Samples(True), batch_size=4),
            val=DataLoader(_Samples(False), batch_size=4),
        )
        before = model.backbone.weight.detach().clone()
        spt.Manager(
            trainer=trainer,
            module=model,
            data=data,
            seed=7,
            ckpt_path=ckpt,
            weights_only=False,
        )()
        assert trainer.global_step == 3 * epochs
        assert not torch.equal(before, model.backbone.weight)
        assert "eval/probe_accuracy" in trainer.callback_metrics
        assert torch.isfinite(trainer.callback_metrics["fit/loss"])
        return trainer.checkpoint_callback.best_model_path

    ckpt = run(1)
    checkpoint = torch.load(ckpt, weights_only=False)
    assert checkpoint["global_step"] == 3 and checkpoint["optimizer_states"]
    run(2, ckpt)
    assert list(tmp_path.rglob("sidecar.json"))
