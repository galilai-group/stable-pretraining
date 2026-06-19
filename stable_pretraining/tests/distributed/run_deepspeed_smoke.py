r"""End-to-end DeepSpeed ZeRO-3 smoke test (single-optimizer partial support).

NOT a pytest test — a standalone script to confirm DeepSpeed ZeRO-3 *actually
trains* under this library's manual-optimization loop, on a GPU node. It
overfits a fixed batch and asserts the loss collapses; if the manual-opt loop
silently no-ops under the ZeRO engine (optimizer never steps), the loss stays
flat and the script reports ``DEEPSPEED_RESULT=NOOP``.

**Scope of DeepSpeed support: single optimizer only.** Teacher/student EMA and
the multi-optimizer online-probe callbacks are NOT supported under DeepSpeed
(ZeRO's single-engine model conflicts with them) — use FSDP2 (``strategy="fsdp2"``)
for those. Requires the optional ``deepspeed`` dependency.

Run on a >=2-GPU node (Lightning spawns one process per task)::

    srun --partition=spot --gres=gpu:2 --ntasks=2 --ntasks-per-node=2 \
        python stable_pretraining/tests/distributed/run_deepspeed_smoke.py

Compare against a DDP baseline by passing ``--strategy ddp``.
"""

import argparse
import os

import lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt


class _FixedTargetDataset(Dataset):
    """A handful of distinct inputs mapping to one shared target → overfittable."""

    def __init__(self, n: int = 256, dim: int = 32):
        self.n, self.dim = n, dim
        g = torch.Generator().manual_seed(0)
        self.y = torch.randn(dim, generator=g)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(idx % 8)
        return {"x": torch.randn(self.dim, generator=g), "y": self.y.clone()}


class _LossTrace(pl.Callback):
    def __init__(self):
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and "loss" in outputs:
            self.losses.append(float(outputs["loss"].detach()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="deepspeed_stage_3")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    def forward(self, batch, stage):
        pred = self.backbone(batch["x"])
        return {"loss": nn.functional.mse_loss(pred, batch["y"])}

    module = spt.Module(
        forward=forward,
        backbone=nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        ),
        optim={"optimizer": {"type": "AdamW", "lr": 1e-2}, "scheduler": "ConstantLR"},
    )

    trace = _LossTrace()
    loader = DataLoader(
        _FixedTargetDataset(), batch_size=8, num_workers=2, shuffle=True
    )
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
        callbacks=[trace],
    )
    trainer.fit(module, loader)

    if trainer.is_global_zero:
        ls = trace.losses
        first, last = sum(ls[:3]) / 3, sum(ls[-3:]) / 3
        finite = all(v == v and abs(v) < 1e9 for v in ls)
        trained = finite and last < 0.5 * first
        print(f"loss_first3={first:.4f} loss_last3={last:.4f} steps={len(ls)}")
        print(
            f"DEEPSPEED_RESULT={'OK' if trained else 'NOOP'} "
            f"strategy={args.strategy} devices={n_devices} trained={trained}"
        )


if __name__ == "__main__":
    main()
