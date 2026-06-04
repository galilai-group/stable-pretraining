"""Per-phase profiler for the two-view SSL training step.

Breaks each step into:

- ``data_load``   : DataLoader.next() — CPU workers produce a batch
- ``h2d``         : explicit batch.to(cuda, non_blocking=True) — first sync
- ``aug``         : GPU augmentation pipeline (gpu mode only; ``aug`` ≈ 0 in cpu mode)
- ``fwd``         : model forward (loss computation)
- ``bwd``         : backward + optimizer step

Timing strategy
---------------
GPU phases (``h2d``, ``aug``, ``fwd``, ``bwd``) are measured with
``torch.cuda.Event`` pairs to capture true device wall time (perf_counter
on the host would only see kernel-launch time). The events are recorded
on the default stream, and we ``cuda.synchronize()`` at end-of-step so
host timing of ``data_load`` is fair.

Caveats
-------
- ``cuda.synchronize()`` between phases serialises the timeline and
  PREVENTS the H2D/aug/fwd overlap that PyTorch gets for free with
  ``non_blocking=True``. The numbers below are PHASE COSTS in isolation,
  not steady-state step time. The ``total`` row is what overlapping
  would converge to in the best case. The real end-to-end step time
  (with implicit overlap) is what ``benchmark_gpu_vs_cpu.py`` reports.

Run::

    srun --gpus=1 python benchmarks/imagenet10/profile_phases.py --mode cpu --steps 30
    srun --gpus=1 python benchmarks/imagenet10/profile_phases.py --mode gpu --steps 30
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def build(mode, batch_size, num_workers, compile_flag):
    """Return (datamodule, model, gpu_transform_or_None)."""
    from stable_pretraining.methods.barlow_twins import BarlowTwins

    model = BarlowTwins(
        encoder_name="vit_small_patch16_224",
        projector_dims=(2048, 2048, 2048),
        lambd=5.1e-3,
    ).cuda()
    if mode == "cpu":
        from two_view import make_imagenette_data

        data = make_imagenette_data(batch_size=batch_size, num_workers=num_workers)
        return data, model, None
    else:
        from two_view_gpu import gpu_two_view_transform, make_imagenette_data_gpu

        data = make_imagenette_data_gpu(batch_size=batch_size, num_workers=num_workers)
        gpu_transform = gpu_two_view_transform(compile=compile_flag).cuda()
        return data, model, gpu_transform


def cuda_event_pair():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def run(mode, batch_size, num_workers, steps, warmup, compile_flag):
    data, model, gpu_transform = build(mode, batch_size, num_workers, compile_flag)
    data.setup("fit")
    loader = data.train_dataloader()

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler("cuda")

    timings = {"data_load": [], "h2d": [], "aug": [], "fwd": [], "bwd": []}

    it = iter(loader)
    for step in range(steps + warmup):
        torch.cuda.synchronize()

        # --- data_load (host) ---
        t0 = time.perf_counter()
        batch = next(it)
        t_data = time.perf_counter() - t0

        # --- h2d ---
        h2d_a, h2d_b = cuda_event_pair()
        h2d_a.record()
        if mode == "cpu":
            # batch is a dict whose "views" is a list of view dicts (CPU)
            for v in batch["views"]:
                v["image"] = v["image"].to("cuda", non_blocking=True)
                if "label" in v and isinstance(v["label"], torch.Tensor):
                    v["label"] = v["label"].to("cuda", non_blocking=True)
            if "label" in batch and isinstance(batch.get("label"), torch.Tensor):
                batch["label"] = batch["label"].to("cuda", non_blocking=True)
        else:
            batch["image"] = batch["image"].to("cuda", non_blocking=True)
            if "label" in batch and isinstance(batch["label"], torch.Tensor):
                batch["label"] = batch["label"].to("cuda", non_blocking=True)
        h2d_b.record()

        # --- aug (gpu mode only) ---
        aug_a, aug_b = cuda_event_pair()
        aug_a.record()
        if mode == "gpu":
            batch = gpu_transform(batch)
        aug_b.record()

        # --- fwd ---
        fwd_a, fwd_b = cuda_event_pair()
        fwd_a.record()
        views = batch["views"]
        v1, v2 = views[0]["image"], views[1]["image"]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model.forward(v1, v2)
            loss = output.loss
        fwd_b.record()

        # --- bwd ---
        bwd_a, bwd_b = cuda_event_pair()
        bwd_a.record()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        bwd_b.record()

        torch.cuda.synchronize()
        if step >= warmup:
            timings["data_load"].append(t_data * 1000)  # ms
            timings["h2d"].append(h2d_a.elapsed_time(h2d_b))
            timings["aug"].append(aug_a.elapsed_time(aug_b))
            timings["fwd"].append(fwd_a.elapsed_time(fwd_b))
            timings["bwd"].append(bwd_a.elapsed_time(bwd_b))

    return timings


def fmt(ts):
    return f"{statistics.mean(ts):7.2f} ms   (median {statistics.median(ts):6.2f}, p10 {sorted(ts)[max(0, len(ts) // 10 - 1)]:6.2f})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    print(
        f"=== phase profile: mode={args.mode}, bs={args.batch_size}, "
        f"workers={args.num_workers}, steps={args.steps}, compile={not args.no_compile} ==="
    )
    timings = run(
        args.mode,
        args.batch_size,
        args.num_workers,
        args.steps,
        args.warmup,
        compile_flag=not args.no_compile,
    )
    total = sum(statistics.mean(timings[k]) for k in timings)
    for k in ("data_load", "h2d", "aug", "fwd", "bwd"):
        mean = statistics.mean(timings[k])
        share = 100 * mean / total
        print(f"  {k:10s}: {fmt(timings[k])}   {share:5.1f}% of total")
    print(f"  {'TOTAL':10s}: {total:7.2f} ms                                  100.0%")
    print(
        f"  samples / sec (single-stream, no overlap): {args.batch_size / (total / 1000):.1f}"
    )


if __name__ == "__main__":
    main()
