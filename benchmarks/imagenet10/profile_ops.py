"""Per-op timing for the GPU augmentation chain.

Runs each kornia-backed transform in isolation on a synthetic
``(B, 3, 224, 224)`` GPU tensor and reports wall time per call. Used to
find which ops dominate the chain (typically GaussianBlur with large
kernel size).

Run::

    srun --gpus=1 python benchmarks/imagenet10/profile_ops.py
"""

import argparse
import statistics

import torch

from stable_pretraining.data import gpu_transforms as gt


def time_op(op, x, iters=30, warmup=5):
    op.cuda()
    # warmup
    for _ in range(warmup):
        _ = op({"image": x})
    torch.cuda.synchronize()
    # measure with cuda events
    times = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        _ = op({"image": x})
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--input-size", type=int, default=256, help="square HxW")
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    B, S, C = args.batch_size, args.input_size, args.crop_size

    print(
        f"=== per-op profile: B={B}, input={S}x{S}, crop={C}x{C}, iters={args.iters} ==="
    )
    x_in = torch.rand(B, 3, S, S, device="cuda")
    x_crop = torch.rand(B, 3, C, C, device="cuda")

    ops = [
        (
            "GPURandomResizedCrop(224)",
            gt.GPURandomResizedCrop(size=C, scale=(0.08, 1.0)),
            x_in,
        ),
        ("GPURandomHorizontalFlip(p=0.5)", gt.GPURandomHorizontalFlip(p=0.5), x_crop),
        (
            "GPUColorJitter(0.4,0.4,0.2,0.1,p=0.8)",
            gt.GPUColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            x_crop,
        ),
        ("GPURandomGrayscale(p=0.2)", gt.GPURandomGrayscale(p=0.2), x_crop),
        (
            "GPUGaussianBlur(k=23,sigma=(0.1,2),p=1.0)",
            gt.GPUGaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=1.0),
            x_crop,
        ),
        (
            "GPUGaussianBlur(k=5,sigma=(0.1,2),p=1.0)",
            gt.GPUGaussianBlur(kernel_size=5, sigma=(0.1, 2.0), p=1.0),
            x_crop,
        ),
        ("GPURandomSolarize(p=0.2)", gt.GPURandomSolarize(p=0.2), x_crop),
        (
            "GPUNormalize(imagenet)",
            gt.GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            x_crop,
        ),
    ]

    print(f"  {'op':40s}  {'mean (ms)':>10s}  {'median':>8s}  {'p10':>8s}")
    print(f"  {'-' * 40}  {'-' * 10}  {'-' * 8}  {'-' * 8}")
    for name, op, x in ops:
        ts = time_op(op, x, iters=args.iters)
        print(
            f"  {name:40s}  {statistics.mean(ts):10.2f}  "
            f"{statistics.median(ts):8.2f}  {sorted(ts)[max(0, len(ts) // 10 - 1)]:8.2f}"
        )


if __name__ == "__main__":
    main()
