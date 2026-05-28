"""Head-to-head benchmark: CPU torchvision aug vs GPU kornia aug on Imagenette.

Measures **end-to-end training throughput** (samples/sec) of the BarlowTwins
ViT-S/16 two-view recipe with two augmentation pipelines:

1. ``cpu`` — the existing :mod:`two_view` pipeline: full torchvision
   augmentation in DataLoader workers.
2. ``gpu`` — the new :mod:`two_view_gpu` pipeline: minimal CPU prep
   (resize+ToImage), kornia augmentation on GPU via
   ``Module.on_after_batch_transfer``, with ``torch.compile`` on the
   deterministic ops.

Run::

    # CPU baseline
    srun --gpus=1 python benchmarks/imagenet10/benchmark_gpu_vs_cpu.py --mode cpu --steps 50
    # GPU augmentation
    srun --gpus=1 python benchmarks/imagenet10/benchmark_gpu_vs_cpu.py --mode gpu --steps 50

Both modes use the same batch size, num_workers, model, and number of steps
so the only thing that varies is where augmentation happens.
"""

import argparse
import sys
import time
from pathlib import Path

import lightning as pl
import torch

sys.path.insert(0, str(Path(__file__).parent))

import stable_pretraining as spt  # noqa: E402
from stable_pretraining.methods.barlow_twins import BarlowTwins  # noqa: E402


def build_cpu(batch_size, num_workers):
    from two_view import attach_forward_and_optim, make_imagenette_data

    data = make_imagenette_data(batch_size=batch_size, num_workers=num_workers)
    module = BarlowTwins(
        encoder_name="vit_small_patch16_224",
        projector_dims=(2048, 2048, 2048),
        lambd=5.1e-3,
    )
    attach_forward_and_optim(
        module,
        BarlowTwins,
        optim={
            "optimizer": {"type": "AdamW", "lr": 5e-4, "weight_decay": 0.05},
            "scheduler": {"type": "ConstantLR"},
            "interval": "step",
        },
    )
    return module, data


def build_gpu(batch_size, num_workers, compile, stacked=True):
    from two_view_gpu import (
        attach_forward_and_optim,
        build_gpu_transform,
        make_imagenette_data_gpu,
    )

    gpu_transform = build_gpu_transform(compile=compile, stacked=stacked)
    data = make_imagenette_data_gpu(
        batch_size=batch_size,
        num_workers=num_workers,
        gpu_transform=gpu_transform,
    )
    module = BarlowTwins(
        encoder_name="vit_small_patch16_224",
        projector_dims=(2048, 2048, 2048),
        lambd=5.1e-3,
    )
    attach_forward_and_optim(
        module,
        BarlowTwins,
        optim={
            "optimizer": {"type": "AdamW", "lr": 5e-4, "weight_decay": 0.05},
            "scheduler": {"type": "ConstantLR"},
            "interval": "step",
        },
    )
    return module, data


def run_benchmark(mode, batch_size, num_workers, steps, warmup, compile, stacked=True):
    if mode == "cpu":
        module, data = build_cpu(batch_size, num_workers)
    else:
        module, data = build_gpu(
            batch_size, num_workers, compile=compile, stacked=stacked
        )

    trainer = pl.Trainer(
        max_steps=steps + warmup,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        limit_val_batches=0,
        callbacks=[_StepTimer(warmup=warmup)],
        precision="16-mixed",
        devices=1,
        accelerator="gpu",
    )
    spt.Manager(trainer=trainer, module=module, data=data)()
    timer = trainer.callbacks[0]
    return timer.report(batch_size)


class _StepTimer(pl.Callback):
    """End-to-end step throughput callback.

    Times from the END of step N to the END of step N+1, which captures
    the *full* step: data load + H2D transfer + on_after_batch_transfer
    (GPU aug) + training_step. Using ``on_train_batch_start`` instead
    would miss everything before the model forward — see the bug fixed
    in benchmark_gpu_vs_cpu.py revision history.
    """

    def __init__(self, warmup: int):
        super().__init__()
        self.warmup = warmup
        self.times = []
        self._t = None
        self._step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self._t is not None and self._step >= self.warmup:
            self.times.append(now - self._t)
        self._t = now
        self._step += 1

    def report(self, batch_size):
        import statistics

        ts = self.times
        return {
            "n": len(ts),
            "mean_step_s": statistics.mean(ts),
            "median_step_s": statistics.median(ts),
            "p10_step_s": sorted(ts)[max(0, len(ts) // 10 - 1)],
            "samples_per_sec": batch_size / statistics.mean(ts),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile in the GPU pipeline (gpu mode only).",
    )
    p.add_argument(
        "--no-stack",
        action="store_true",
        help="Use the two-chain (asymmetric-safe) GPU path instead of stacked single-chain.",
    )
    args = p.parse_args()

    print(
        f"=== Benchmark: mode={args.mode}, bs={args.batch_size}, steps={args.steps} ==="
    )
    out = run_benchmark(
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        warmup=args.warmup,
        compile=not args.no_compile,
        stacked=not args.no_stack,
    )
    print(
        f"  steps timed     : {out['n']}\n"
        f"  mean step (s)   : {out['mean_step_s']:.4f}\n"
        f"  median step (s) : {out['median_step_s']:.4f}\n"
        f"  p10 step (s)    : {out['p10_step_s']:.4f}\n"
        f"  samples / sec   : {out['samples_per_sec']:.1f}"
    )


if __name__ == "__main__":
    main()
