"""Apples-to-apples JAX vs PyTorch SimCLR throughput (1 and 2 GPUs).

Measures steady-state images/sec for a backbone + MLP-projector + NT-Xent
training step on *synthetic in-memory* data (so the number reflects compute,
not the input pipeline). Per-GPU batch is fixed; global throughput is reported.
``--model`` selects the encoder (resnet18 / vit_base / vit_large).

JAX (single process, all visible devices via SPMD data-parallel)::

    CUDA_VISIBLE_DEVICES=0   python benchmarks/jax_vs_torch_throughput.py --backend jax --model vit_base
    CUDA_VISIBLE_DEVICES=0,1 python benchmarks/jax_vs_torch_throughput.py --backend jax --model vit_base

PyTorch (1 GPU single process; 2 GPUs via DDP/torchrun)::

    CUDA_VISIBLE_DEVICES=0 python benchmarks/jax_vs_torch_throughput.py --backend torch --model vit_base
    torchrun --nproc_per_node=2 benchmarks/jax_vs_torch_throughput.py --backend torch --model vit_base

Notes:
    JAX always jit-compiles. PyTorch is eager unless ``--compile``. ``--bf16``
    enables bfloat16 mixed precision on both. ``--tune`` (JAX) sets TF32 matmul
    precision + donated jit.
"""

import argparse
import os
import time

WARMUP = 10
STEPS = 40

# model -> (jax_builder(rngs,dtype,img), torch_builder(img), embed_dim)
EMBED = {"resnet18": 512, "vit_base": 768, "vit_large": 1024}


def bench_jax(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    import stable_pretraining.jax as spj

    if args.tune:
        jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    ndev = jax.device_count()
    gb = args.batch_size * ndev
    dtype = jnp.bfloat16 if args.bf16 else None
    rngs = nnx.Rngs(0)
    if args.model == "resnet18":
        backbone = spj.backbone.resnet18(rngs=rngs, dtype=dtype)
    elif args.model == "vit_base":
        backbone = spj.backbone.vit_base(rngs=rngs, img_size=args.img_size, dtype=dtype)
    else:
        backbone = spj.backbone.vit_large(
            rngs=rngs, img_size=args.img_size, dtype=dtype
        )
    model = spj.SimCLR(
        backbone=backbone,
        embed_dim=backbone.embed_dim,
        rngs=rngs,
        projector_dims=(2048, 256),
        optim={"type": "adamw", "learning_rate": 1e-3},
        dtype=dtype,
    )

    from stable_pretraining.jax.optim import create_optimizer

    optimizer = nnx.Optimizer(model, create_optimizer(model.optim), wrt=nnx.Param)
    trainer = spj.Trainer(max_epochs=1, data_parallel=ndev > 1)
    if ndev > 1:
        trainer.optimizer = optimizer
        trainer._setup_data_parallel(model)
        optimizer = trainer.optimizer

    rng = np.random.RandomState(0)

    def mk():
        return jnp.asarray(
            rng.rand(gb, args.img_size, args.img_size, 3).astype("float32")
        )

    batch = {"views": [{"image": mk()}, {"image": mk()}]}
    if ndev > 1:
        batch = trainer._shard(batch)

    from stable_pretraining.jax.trainer import _train_step

    def run():
        state, _ = _train_step(model, optimizer, batch)
        jax.block_until_ready(state["loss"])

    for _ in range(WARMUP):
        run()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        run()
    dt = time.perf_counter() - t0
    prec = ("bf16" if args.bf16 else "f32") + ("+tune" if args.tune else "")
    print(
        f"[JAX {args.model} {prec}] devices={ndev} gb={gb} "
        f"{STEPS * gb / dt:,.0f} img/s  ({1000 * dt / STEPS:.1f} ms/step)",
        flush=True,
    )


def bench_torch(args):
    import numpy as np
    import torch
    import torch.distributed as dist

    from stable_pretraining.losses import NTXEntLoss

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.model == "resnet18":
        import torchvision

        backbone = torchvision.models.resnet18(num_classes=0)
        backbone.fc = torch.nn.Identity()
        embed = 512
    else:
        import timm

        name = {
            "vit_base": "vit_base_patch16_224",
            "vit_large": "vit_large_patch16_224",
        }[args.model]
        backbone = timm.create_model(name, num_classes=0, img_size=args.img_size)
        embed = backbone.num_features
    # Compile the encoder before DDP (keeps the compiled graph off the DDP boundary).
    if args.compile:
        backbone = torch.compile(backbone)
    projector = torch.nn.Sequential(
        torch.nn.Linear(embed, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 256)
    )
    model = torch.nn.Sequential(backbone, projector).to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    loss_fn = NTXEntLoss(temperature=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    rng = np.random.RandomState(rank)
    x = torch.from_numpy(
        rng.rand(2 * args.batch_size, 3, args.img_size, args.img_size).astype("float32")
    ).to(device)

    def run():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            z = model(x)
            loss = loss_fn(z[: args.batch_size], z[args.batch_size :])
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    for _ in range(WARMUP):
        run()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        run()
    dt = time.perf_counter() - t0
    gb = args.batch_size * world
    if rank == 0:
        prec = ("bf16" if args.bf16 else "f32") + ("+compile" if args.compile else "")
        print(
            f"[torch {args.model} {prec}] gpus={world} gb={gb} "
            f"{STEPS * gb / dt:,.0f} img/s  ({1000 * dt / STEPS:.1f} ms/step)",
            flush=True,
        )
    if world > 1:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["jax", "torch"], required=True)
    ap.add_argument("--model", choices=list(EMBED), default="resnet18")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--bf16", action="store_true", help="bfloat16 mixed precision")
    ap.add_argument("--compile", action="store_true", help="torch.compile (torch only)")
    ap.add_argument("--tune", action="store_true", help="JAX: TF32 + donated jit")
    args = ap.parse_args()
    (bench_jax if args.backend == "jax" else bench_torch)(args)


if __name__ == "__main__":
    main()
