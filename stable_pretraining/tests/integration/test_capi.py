"""Bounded Manager, resume, mixed-precision and Gloo checks for CAPI."""

from pathlib import Path

import lightning as pl
import pytest
import torch
import torch.multiprocessing as mp

import stable_pretraining as spt
from stable_pretraining.losses.capi import positionwise_sinkhorn

pytestmark = pytest.mark.integration


class _Images(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return {"image": torch.full((3, 32, 32), i / 8), "label": i % 2}


def _model():
    return spt.CAPI(
        embed_dim=32,
        depth=1,
        predictor_depth=1,
        num_heads=4,
        n_registers=4,
        patch_size=8,
        num_clusters=8,
        drop_path_rate=0.2,
        total_steps=12,
        warmup_steps=0,
        ema_momentum=0.5,
        accumulation_steps=2,
    )


@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
@pytest.mark.parametrize(
    "accelerator", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]
)
def test_manager_accumulation_and_resume(tmp_path, precision, accelerator):
    spt.set(cache_dir=str(tmp_path / "runs"), default_callbacks={"env_dump": False})
    model = _model()
    before = model.teacher.patch_embed.weight.detach().clone()
    clusters = model.clustering.weight.detach().clone()
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(_Images(), batch_size=2)
    )
    import torchmetrics

    callbacks = [
        spt.OnlineProbe(
            model,
            name="linear",
            input="embedding",
            target="label",
            probe=torch.nn.Linear(32, 2),
            loss=torch.nn.CrossEntropyLoss(),
            optimizer={"type": "AdamW", "lr": 0.01},
            metrics={"top1": torchmetrics.classification.MulticlassAccuracy(2)},
        )
    ]
    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator=accelerator,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        precision=precision,
    )
    spt.Manager(trainer=trainer, module=model, data=data, seed=0)()
    assert trainer.global_step == 2
    assert int(model.capi_updates) == 2
    assert not torch.equal(before, model.teacher.patch_embed.weight)
    assert not torch.equal(clusters, model.clustering.weight)
    assert all(p.grad is None for p in model.teacher.parameters())
    path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(path)
    resumed = _model()
    resumed_probe = spt.OnlineProbe(
        resumed,
        name="linear",
        input="embedding",
        target="label",
        probe=torch.nn.Linear(32, 2),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer={"type": "AdamW", "lr": 0.01},
        metrics={"top1": torchmetrics.classification.MulticlassAccuracy(2)},
    )
    trainer2 = pl.Trainer(
        callbacks=[resumed_probe],
        accelerator=accelerator,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        precision=precision,
    )
    spt.Manager(
        trainer=trainer2,
        module=resumed,
        data=data,
        seed=0,
        ckpt_path=str(path.resolve()),
        weights_only=False,
    )()
    assert int(resumed.capi_updates) == 4
    assert trainer2.global_step == 4
    assert not resumed.teacher.training
    assert all(torch.isfinite(p).all() for p in resumed.parameters())
    assert trainer2.lr_scheduler_configs[0].scheduler.last_epoch == 4


def _gloo_worker(rank, path, output):
    torch.distributed.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=2
    )
    torch.manual_seed(19)
    all_logits = torch.randn(7, 3, 5) * 0.2
    local = all_logits[:3] if rank == 0 else all_logits[3:]
    assignments = positionwise_sinkhorn(local)
    torch.save(assignments, Path(output) / f"rank{rank}.pt")
    torch.distributed.destroy_process_group()


def test_two_rank_balancing_matches_concatenated_batch(tmp_path):
    mp.spawn(
        _gloo_worker, args=(str(tmp_path / "gloo"), str(tmp_path)), nprocs=2, join=True
    )
    torch.manual_seed(19)
    expected = positionwise_sinkhorn(torch.randn(7, 3, 5) * 0.2)
    actual = torch.cat(
        [torch.load(tmp_path / f"rank{r}.pt", weights_only=True) for r in range(2)]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_manually_composed_module_uses_same_forward_and_ema(tmp_path):
    spt.set(cache_dir=str(tmp_path / "runs"), default_callbacks={"env_dump": False})
    reference = _model()
    names = (
        "backbone",
        "teacher",
        "student_head",
        "clustering",
        "capi_loss",
        "optim",
        "mask_ratio",
        "prediction_subsampling",
        "representation",
        "total_steps",
        "warmup_steps",
        "ema_momentum",
    )
    module = spt.Module(
        forward=spt.forward.capi, **{name: getattr(reference, name) for name in names}
    )
    module.register_buffer("capi_updates", reference.capi_updates.clone())
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[spt.CAPIEMA()],
    )
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(_Images(), batch_size=2)
    )
    spt.Manager(trainer=trainer, module=module, data=data, seed=0)()
    assert int(module.capi_updates) == trainer.global_step == 2
    assert not module.teacher.training
