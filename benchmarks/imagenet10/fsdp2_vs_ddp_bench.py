"""Speed / memory micro-benchmark: FSDP2 vs DDP on Imagenette (ImageNet-10).

Trains a SimCLR-style two-view model on real Imagenette for a fixed number of
steps and reports per-GPU peak memory and throughput, so the FSDP2 sharding
gain can be compared against DDP on the same model/batch/hardware.

Run once per strategy on a >=2-GPU node (Lightning spawns one process per GPU)::

    srun --partition=track1 --gres=gpu:2 --ntasks=1 --cpus-per-task=16 \\
        python benchmarks/imagenet10/fsdp2_vs_ddp_bench.py --strategy ddp   --backbone vit_large_patch16_224
    srun --partition=track1 --gres=gpu:2 --ntasks=1 --cpus-per-task=16 \\
        python benchmarks/imagenet10/fsdp2_vs_ddp_bench.py --strategy fsdp2 --backbone vit_large_patch16_224

Each run prints a parseable line on rank 0::

    RESULT strategy=fsdp2 backbone=vit_large_patch16_224 world=2 batch=256 \\
        peak_mem_GiB=12.34 step_s=0.210 imgs_per_s=2438.1

A larger backbone (e.g. ``vit_large``/``vit_huge``) is where FSDP2's
parameter/grad/optimizer-state sharding shows a clear per-GPU memory reduction;
on a tiny model FSDP2's comm overhead dominates and DDP is faster.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.losses import NTXEntLoss


class _BenchCallback(pl.Callback):
    """Records mean step time (post-warmup) and per-GPU peak memory."""

    def __init__(self, warmup: int = 5):
        self.warmup = warmup
        self.times = []
        self._t0 = None
        self._seen = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        self._seen += 1
        if self._seen > self.warmup:
            self.times.append(dt)

    def summarize(self, world_size: int, local_batch: int) -> dict:
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        step_s = statistics.mean(self.times) if self.times else float("nan")
        imgs_per_s = (local_batch * world_size) / step_s if self.times else float("nan")
        return {"peak_mem_GiB": peak, "step_s": step_s, "imgs_per_s": imgs_per_s}


def _build_forward():
    loss_fn = NTXEntLoss(temperature=0.2)

    def forward(self, batch, stage):
        if "image" in batch:  # eval/single-view
            return {"embedding": self.projector(self.backbone(batch["image"]))}
        views = batch["views"] if "views" in batch else list(batch.values())
        v1, v2 = views[0]["image"], views[1]["image"]
        z1 = self.projector(self.backbone(v1))
        z2 = self.projector(self.backbone(v2))
        return {"loss": loss_fn(z1, z2)}

    return forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["ddp", "fsdp2"], required=True)
    parser.add_argument("--backbone", default="vit_large_patch16_224")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    two_view = transforms.MultiViewTransform(
        [
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(**spt.data.static.ImageNet),
            ),
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(**spt.data.static.ImageNet),
            ),
        ]
    )

    data_dir = str(get_data_dir("imagenet10"))
    train_loader = torch.utils.data.DataLoader(
        dataset=spt.data.HFDataset(
            "frgfm/imagenette",
            split="train",
            revision="refs/convert/parquet",
            cache_dir=data_dir,
            transform=two_view,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        shuffle=True,
    )

    backbone = spt.backbone.from_timm(args.backbone, num_classes=0)
    embed_dim = backbone.embed_dim
    module = spt.Module(
        forward=_build_forward(),
        backbone=backbone,
        projector=nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
        ),
        optim={"optimizer": {"type": "AdamW", "lr": 1e-3}, "scheduler": "ConstantLR"},
    )

    bench = _BenchCallback(warmup=5)
    # Under SLURM, Lightning requires devices == ntasks-per-node; off SLURM, use
    # all visible GPUs. (Hardcoding would error the moment the launch width changes.)
    import os

    n_devices = int(
        os.environ.get("SLURM_NTASKS_PER_NODE") or (torch.cuda.device_count() or 1)
    )
    trainer = pl.Trainer(
        strategy=args.strategy,
        accelerator="gpu",
        devices=n_devices,
        precision="bf16-mixed",
        max_steps=args.steps,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[bench],
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(module, train_loader)

    stats = bench.summarize(trainer.world_size, args.batch_size)
    if trainer.is_global_zero:
        print(
            f"RESULT strategy={args.strategy} backbone={args.backbone} "
            f"world={trainer.world_size} batch={args.batch_size} "
            f"peak_mem_GiB={stats['peak_mem_GiB']:.2f} "
            f"step_s={stats['step_s']:.3f} imgs_per_s={stats['imgs_per_s']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
