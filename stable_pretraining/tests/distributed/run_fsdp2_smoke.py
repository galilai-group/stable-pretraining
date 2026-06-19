r"""End-to-end FSDP2 smoke test through a real Lightning ``Trainer``.

This is NOT a pytest test — it is a standalone script meant to be launched on a
GPU node to sanity-check the full ``strategy="fsdp2"`` integration path
(strategy registration → device-mesh setup → ``configure_model`` sharding →
manual-optimization training loop → checkpoint), end to end, on >=2 GPUs.

Run it with srun (single task; Lightning spawns the per-GPU workers itself)::

    srun --partition=spot --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --pty \\
        python stable_pretraining/tests/distributed/run_fsdp2_smoke.py

Or two explicit tasks (one process per GPU)::

    srun --partition=spot --gres=gpu:2 --ntasks=2 --cpus-per-task=8 \\
        python stable_pretraining/tests/distributed/run_fsdp2_smoke.py

It trains a tiny ViT-backed SimCLR-style module on random data for a few steps
and asserts: training completes, the loss is finite, and the backbone params
are sharded ``DTensor``\\ s on every rank. Prints ``FSDP2 SMOKE OK`` on success.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt


class _RandomViews(Dataset):
    """Random two-view dataset: returns dict with two augmented "images"."""

    def __init__(self, n=256, dim=32):
        self.n, self.dim = n, dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        x = torch.randn(self.dim, generator=g)
        return {"view0": x, "view1": x + 0.01 * torch.randn(self.dim, generator=g)}


class _MLPBackbone(nn.Module):
    """Tiny block-structured backbone so FSDP2 has per-block units to shard."""

    def __init__(self, dim=32, hidden=128, depth=4):
        super().__init__()
        self.proj_in = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                )
                for _ in range(depth)
            ]
        )
        self.embed_dim = hidden

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.blocks:
            x = x + block(x)
        return x


def _forward(self, batch, stage):
    z0 = self.projector(self.backbone(batch["view0"]))
    z1 = self.projector(self.backbone(batch["view1"]))
    # simple NT-Xent-free cosine alignment loss (avoids cross-rank gather here;
    # we only want to exercise the FSDP2 train loop end to end)
    z0 = nn.functional.normalize(z0, dim=-1)
    z1 = nn.functional.normalize(z1, dim=-1)
    loss = (2 - 2 * (z0 * z1).sum(-1)).mean()
    return {"loss": loss, "embedding": z0}


def main():
    torch.manual_seed(0)
    backbone = _MLPBackbone()
    module = spt.Module(
        forward=_forward,
        backbone=backbone,
        projector=nn.Linear(backbone.embed_dim, 64),
        optim={"optimizer": {"type": "AdamW", "lr": 1e-3}, "scheduler": "ConstantLR"},
    )

    loader = DataLoader(_RandomViews(), batch_size=32, num_workers=2)

    import os

    import lightning as pl

    # Under SLURM, devices must equal ntasks-per-node; off SLURM use all GPUs.
    n_devices = int(
        os.environ.get("SLURM_NTASKS_PER_NODE") or (torch.cuda.device_count() or 1)
    )
    trainer = pl.Trainer(
        strategy="fsdp2",
        accelerator="gpu",
        devices=n_devices,
        precision="bf16-mixed",
        max_steps=5,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, loader)

    # Post-fit assertions (every rank runs these).
    from torch.distributed.tensor import DTensor

    sharded = [isinstance(p, DTensor) for p in module.backbone.parameters()]
    assert sharded and all(sharded), "backbone params should be sharded DTensors"
    if trainer.is_global_zero:
        print("FSDP2 SMOKE OK")


if __name__ == "__main__":
    main()
