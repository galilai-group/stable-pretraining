"""LeJEPA ViT-Small on ImageNet-10 (Imagenette) under FSDP2.

Run with: ``torchrun --nproc-per-node=2 lejepa_vit_small_fsdp.py``.
"""

from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.callbacks.earlystop import EpochMilestones
from stable_pretraining.data import transforms

from lejepa_vit_small import (
    LeJEPA,
    _global_transform,
    _local_transform,
    lejepa_forward,
)


def main():
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    def _stop_after_n_epochs(n: int) -> dict[int, float]:
        return {n - 1: float("-inf")}

    num_gpus = 2
    effective_batch_size = 128
    batch_size = effective_batch_size // num_gpus
    num_workers = 16
    max_epochs = 600
    global_views = 2
    all_views = 8

    data_dir = str(get_data_dir("imagenet10"))

    train_transform = transforms.MultiViewTransform(
        {
            **{f"global_{i}": _global_transform() for i in range(global_views)},
            **{
                f"local_{i}": _local_transform()
                for i in range(all_views - global_views)
            },
        }
    )

    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )

    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="train",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=train_transform,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="validation",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=val_transform,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )

    model = LeJEPA(
        encoder_name="vit_small_patch16_224",
        lamb=0.02,
        n_slices=1024,
        n_points=17,
    )

    module = spt.Module(
        model=model,
        forward=lejepa_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": (lr := 4e-4),
                "weight_decay": 0.05,
                "betas": (0.9, 0.999),
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": 10 / max_epochs,
                "start_factor": 0.01,
                "end_lr": lr / 1000,
                "total_steps": (len(data.train) // num_gpus) * max_epochs,
            },
            "interval": "step",
        },
    )

    # The actual sharding happens in ``Module.configure_model`` via ``default_parallelize_fn``,
    # which auto-detects the ``Block`` classes and applies ``fully_shard``
    # per-block.
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(model.embed_dim, 10),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(10),
                    "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
                },
                optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 1e-6},
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=10000,
                metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
                input_dim=model.embed_dim,
                k=20,
            ),
            spt.callbacks.RankMe(
                name="rankme",
                target="embedding",
                queue_length=1000,
                target_shape=model.embed_dim,
            ),
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(Path(__file__).parent / "checkpoints" / "lejepa-vits-fsdp2"),
                filename="lejepa-vits-fsdp2-{epoch:03d}",
                save_top_k=-1,
                every_n_epochs=300,
                save_last=True,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
            EpochMilestones(
                monitor="fit/loss",
                milestones=_stop_after_n_epochs(3),
                direction="min",
                after_validation=True,
            ),
        ],
        logger=pl.pytorch.loggers.WandbLogger(
            entity="stable-ssl",
            project="imagenet10-methods",
            name="lejepa-vits-fsdp2-inet10",
            log_model=False,
        ),
        # FSDP2 / ``ModelParallelStrategy`` does not accept ``"16-mixed"``
        # (fp16 mixed). It supports ``32-true``, ``bf16-mixed``,
        # ``bf16-true``, and ``16-true``. ``bf16-mixed`` is the closest
        # analogue of the DDP variant's ``"16-mixed"``: same mixed-precision
        # forward-in-half-backward-in-full pattern, just with bfloat16 instead
        # of float16 (no loss-scaling needed). A10G/L4/A100 all support bf16.
        precision="bf16-mixed",
        devices=num_gpus,
        accelerator="gpu",
        strategy="fsdp2",
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
