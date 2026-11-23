"""Train DAE-Net on full-category geometry through stable-pretraining Manager."""

import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

import stable_pretraining as spt

from protocol import (
    CATEGORIES,
    DEFAULT_CONFIG,
    digest,
    load_config,
    read_records,
    relative_source,
)


class FullCategoryData(spt.data.DataModule):
    """Provide repeated query training and one-pass voxel-only revival.

    Args:
        root: Dataset root containing the author HDF5 files.
        manifest: Prepared full-category training manifest.
        config: Validated training settings.
    """

    def __init__(self, root: Path, manifest: Path, config: dict[str, Any]) -> None:
        dataset = spt.data.DAENetDataset(root, manifest, split="train")
        usage = spt.data.DAENetDataset(
            root, manifest, split="train", load_queries=False
        )
        sampler = spt.data.DAENetExposureSampler(
            dataset, shapes_per_epoch=config["shapes_per_epoch"], seed=config["seed"]
        )
        kwargs = dict(
            batch_size=config["batch_size"],
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )
        if config.get("world_size", 1) > 1:
            train = DataLoader(
                dataset,
                batch_sampler=spt.data.DAENetDistributedBatchSampler(
                    sampler, config["batch_size"]
                ),
                generator=torch.Generator().manual_seed(config["seed"]),
                num_workers=0,
            )
        else:
            train = DataLoader(
                dataset,
                sampler=sampler,
                generator=torch.Generator().manual_seed(config["seed"]),
                **kwargs,
            )
        super().__init__(train=train)
        self.train_sampler = sampler
        self.usage_loader = DataLoader(
            usage, generator=torch.Generator().manual_seed(config["seed"] + 3), **kwargs
        )

    def usage_dataloader(self) -> DataLoader:
        """Return the single traversal used to estimate branch usage.

        Returns:
            Loader visiting every training shape once, without occupancy queries.
        """
        return self.usage_loader


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def verify_inputs(root: Path, directory: Path, config: dict[str, Any]) -> str:
    """Verify prepared input identity and the full-category training contract.

    Args:
        root: Portable data root.
        directory: Directory produced by prepare.py.
        config: Selected training recipe.

    Returns:
        Training-manifest identity for checkpoint continuation.

    Raises:
        ValueError: If a source, manifest or training protocol has changed.
    """
    receipt = json.loads((directory / "prepared.json").read_text())
    for key in ("category", "synset", "protocol", "train_shapes", "query_resolution"):
        if receipt[key] != config[key]:
            raise ValueError(f"Prepared data disagrees with {key}")
    for name, expected in receipt["manifest_sha256"].items():
        relative = relative_source(directory, Path(name))
        if digest(directory / relative) != expected:
            raise ValueError(f"Prepared manifest changed: {name}")
    for name, expected in receipt["source_sha256"].items():
        relative = relative_source(root, Path(name))
        if digest(root / relative) != expected:
            raise ValueError(f"Prepared source changed: {name}")
    rows = read_records(directory / "train.jsonl")
    if len(rows) != config["train_shapes"] or len({r["shape_id"] for r in rows}) != len(
        rows
    ):
        raise ValueError("Training manifest must contain the complete category once")
    for row in rows:
        if (
            row["clean_split"] != "train"
            or row["protocol"] != config["protocol"]
            or row["query_resolution"] != config["query_resolution"]
        ):
            raise ValueError("Training record disagrees with the selected protocol")
    return receipt["manifest_sha256"]["train.jsonl"]


def validate_resume(checkpoint: dict[str, Any], settings: dict[str, Any]) -> int:
    """Reject changed recipes and checkpoints taken inside an epoch.

    Args:
        checkpoint: Trusted checkpoint loaded on CPU.
        settings: Effective recipe and input identity for this invocation.

    Returns:
        Number of completed epochs.

    Raises:
        ValueError: If data, settings, optimization state or epoch boundary differ.
    """
    saved = checkpoint.get("hyper_parameters", {}).get("run_settings")
    if saved != settings:
        raise ValueError(
            "Resume requires unchanged recipe, input identity and device/software environment"
        )
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or not 0 <= epoch < settings["epochs"]:
        raise ValueError("Checkpoint epoch is outside the configured budget")
    if checkpoint.get("global_step") != (epoch + 1) * settings["updates_per_epoch"]:
        raise ValueError("Resume requires a completed-epoch checkpoint")
    spt.callbacks.DAENetRevival.validate_checkpoint(checkpoint)
    if not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers"):
        raise ValueError("Missing optimizer or scheduler state")
    controller = checkpoint.get("callbacks", {}).get("DAENetRevival", {})
    if controller.get("torch_rng_state") is None:
        raise ValueError("Missing controller RNG state")
    if epoch + 1 >= settings["era_length"] and not controller.get("accepted_weights"):
        raise ValueError("Missing accepted era snapshot")
    world_size = settings.get("world_size", 1)
    if world_size > 1:
        if (
            controller.get("schema_version") != 2
            or controller.get("world_size") != world_size
        ):
            raise ValueError("Missing DDP controller schema or changed world size")
        states = controller.get("rank_rng_states")
        if not isinstance(states, list) or len(states) != world_size:
            raise ValueError("Missing per-rank RNG states")
        for state in states:
            if not isinstance(state, dict) or not all(
                name in state
                for name in (
                    "torch_rng_state",
                    "cuda_rng_states",
                    "python_rng_state",
                    "numpy_rng_state",
                    "loader_rng_states",
                )
            ):
                raise ValueError("Missing per-rank RNG fields")
            if (
                settings["accelerator"] == "gpu"
                and len(state["cuda_rng_states"] or []) != 1
            ):
                raise ValueError("Missing per-rank CUDA RNG state")
        return epoch + 1
    if (
        settings["accelerator"] == "gpu"
        and len(controller.get("cuda_rng_states") or []) != 1
    ):
        raise ValueError("Missing single-device CUDA RNG state")
    return epoch + 1


def main(argv: list[str] | None = None) -> None:
    """Train, inspect a recipe, or resume a completed epoch.

    Args:
        argv: CLI arguments, or None for process arguments.

    Note:
        FP32 on one device or multiple single-node DDP ranks is supported.
        Batch size is global. Batch limits mark smoke runs; they
        are saved in checkpoint settings and must remain unchanged on resume.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", type=_positive, default=1)
    parser.add_argument("--strategy", choices=("auto", "ddp"), default="auto")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Completed-epoch checkpoint, or an output directory containing result.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    for flag in ("epochs", "batch-size", "shapes-per-epoch", "era-length"):
        parser.add_argument(f"--{flag}", type=_positive)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--limit-train-batches",
        type=_positive,
        help="Smoke checks only; saved in resume identity",
    )
    parser.add_argument(
        "--stop-after-epochs",
        type=_positive,
        help="Stop at this completed epoch without changing the configured total budget",
    )
    args = parser.parse_args(argv)
    if args.strategy == "ddp" and args.devices < 2:
        parser.error("DDP requires --devices >= 2")
    if args.devices > 1 and platform.system() != "Linux":
        parser.error("DAE-Net DDP currently supports Linux only")
    if args.devices > 1 and (
        int(os.environ.get("LOCAL_WORLD_SIZE", str(args.devices))) != args.devices
        or int(os.environ.get("GROUP_WORLD_SIZE", "1")) != 1
    ):
        parser.error("DAE-Net DDP requires all ranks on a single node")
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) != args.devices:
        parser.error("Launcher WORLD_SIZE must match --devices")
    config = load_config(args.category, args.config)
    for key in ("epochs", "batch_size", "shapes_per_epoch", "era_length", "seed"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if config["era_length"] < 2 or not 0 <= config["seed"] < 2**32:
        parser.error("era-length must be >=2 and seed must be in [0, 2**32)")
    target = args.stop_after_epochs or config["epochs"]
    if target > config["epochs"]:
        parser.error("stop-after-epochs exceeds the total configured epochs")
    manifest_digest = verify_inputs(args.data_root, args.manifest_dir, config)
    exposures = config["train_shapes"] * math.ceil(
        config["shapes_per_epoch"] / config["train_shapes"]
    )
    tail = exposures % config["batch_size"]
    if config["batch_size"] < args.devices or (tail and tail < args.devices):
        parser.error(
            "Every global batch, including the tail, must contain at least devices shapes"
        )
    full_updates = math.ceil(exposures / config["batch_size"])
    updates = min(full_updates, args.limit_train_batches or full_updates)
    settings = {
        **config,
        "exposures_per_epoch": exposures,
        "updates_per_epoch": updates,
        "target_updates": updates * config["epochs"],
        "manifest_sha256": manifest_digest,
        "limit_train_batches": args.limit_train_batches,
        "run_kind": "smoke" if args.limit_train_batches else "full_category",
        "cache_in_memory": False,
        "num_workers": 0,
        "accelerator": args.accelerator,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "lightning": pl.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0)
            if args.accelerator == "gpu" and torch.cuda.is_available()
            else "CPU",
        },
    }
    if args.devices > 1:

        def shards(size: int) -> list[int]:
            quotient, remainder = divmod(size, args.devices)
            return [quotient + int(rank < remainder) for rank in range(args.devices)]

        settings.update(
            world_size=args.devices,
            strategy="ddp_find_unused_parameters_true",
            sampling_protocol="global-exposure-v1",
            rank_batch_sizes=shards(config["batch_size"]),
            rank_tail_batch_sizes=shards(tail) if tail else None,
        )
        if args.accelerator == "gpu" and torch.cuda.is_available():
            settings["environment"]["devices"] = [
                torch.cuda.get_device_name(i) for i in range(args.devices)
            ]
    is_rank_zero = spt.utils.get_rank() == 0
    completed = 0
    if args.resume:
        if args.resume.is_dir():
            args.resume = Path(
                json.loads((args.resume / "result.json").read_text())["checkpoint"]
            )
        completed = validate_resume(
            torch.load(args.resume, map_location="cpu", weights_only=False), settings
        )
        if completed >= target:
            raise ValueError("Requested stop epoch has already completed")
    if is_rank_zero:
        print(
            json.dumps(
                {
                    "settings": settings,
                    "completed_epochs": completed,
                    "stop_epoch": target,
                },
                indent=2,
            ),
            flush=True,
        )
    if args.dry_run:
        return
    if is_rank_zero:
        if (
            not args.resume
            and args.output_dir.exists()
            and any(args.output_dir.iterdir())
        ):
            raise FileExistsError("Use a new output directory for a new run")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run-config.json").write_text(
            json.dumps(settings, indent=2) + "\n"
        )
    pl.seed_everything(config["seed"], workers=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    spt.set(cache_dir=str(args.output_dir.resolve()), requeue_checkpoint=False)
    data = FullCategoryData(args.data_root, args.manifest_dir / "train.jsonl", settings)
    model = spt.DAENet(
        backbone=spt.backbone.DAENetBackbone(branch_num=config["branch_num"]),
        sparse_weight=config["sparse_weight"],
        era_length=config["era_length"],
        lr=config["lr"],
    )
    model.save_hyperparameters({"run_settings": settings})
    saver = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="epoch{epoch:04d}",
        auto_insert_metric_name=False,
        monitor=None,
        save_top_k=1,
        save_last=False,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        # Stable buckets preserve FP32 reduction order across fresh-process resume.
        strategy=settings.get("strategy", "auto"),
        use_distributed_sampler=False,
        precision="32-true",
        deterministic=True,
        benchmark=False,
        max_epochs=target,
        callbacks=[saver],
        logger=CSVLogger(str(args.output_dir), name="logs"),
        num_sanity_val_steps=0,
        limit_val_batches=0,
        limit_train_batches=args.limit_train_batches or 1.0,
        default_root_dir=str(args.output_dir),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    spt.Manager(
        trainer=trainer,
        module=model,
        data=data,
        seed=config["seed"],
        ckpt_path=str(args.resume) if args.resume else None,
        weights_only=False,
    )()
    if trainer.is_global_zero:
        result = {
            "completed_epochs": target,
            "updates": trainer.global_step,
            "checkpoint": str(Path(saver.best_model_path).resolve(strict=True)),
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
