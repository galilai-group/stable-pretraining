"""Paper-aligned SSL-EY CIFAR-10 recipe for a single CUDA GPU."""

import argparse
import json
import math
import time
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, FakeData

import stable_pretraining as spt
from ssley_recipe import PaperSSLEY, SessionTimer, augmentations


class RunLog(pl.Callback):
    """Track training while allowing AMP to handle transient overflow."""

    def __init__(self, output: Path, stop_after_epochs: int | None) -> None:
        self.output = output
        self.stop_after_epochs = stop_after_epochs
        self.started = time.monotonic()
        self.nonfinite_batches = 0
        self.amp_skips = 0
        self.consecutive_nonfinite = 0

    @staticmethod
    def finite_metrics(metrics: dict) -> dict:
        """Represent invalid measurements as null, never as a valid score."""
        return {
            k: float(v) if math.isfinite(float(v)) else None for k, v in metrics.items()
        }

    @staticmethod
    def _check_model(module) -> None:
        bad = [
            name
            for name, value in module.state_dict().items()
            if value.is_floating_point() and not torch.isfinite(value).all()
        ]
        if bad:
            raise FloatingPointError(f"Non-finite model state: {bad[:8]}")

    def _status(self, trainer, state: str, **extra) -> None:
        row = {
            "state": state,
            "epoch": trainer.current_epoch,
            "steps": trainer.global_step,
            "nonfinite_batches": self.nonfinite_batches,
            "amp_skips": self.amp_skips,
            **extra,
        }
        (self.output / "run-status.json").write_text(
            json.dumps(row, indent=2, allow_nan=False)
        )

    def on_train_start(self, trainer, pl_module) -> None:
        """Replace stale paused/failed status when a checkpoint is resumed."""
        self._check_model(pl_module)
        self._status(trainer, "running")

    def on_before_backward(self, trainer, pl_module, loss: torch.Tensor) -> None:
        """Leave scaled-gradient overflow handling to Lightning's GradScaler."""
        if not torch.isfinite(loss).all():
            self._check_model(pl_module)
            if getattr(trainer.precision_plugin, "scaler", None) is None:
                raise FloatingPointError(
                    "Non-finite training loss without AMP GradScaler"
                )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        """Count unscaled invalid gradients without bypassing the scaler's skip."""
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        bad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for group in optimizer.param_groups
            for p in group["params"]
        )
        if bad:
            if scaler is None:
                raise FloatingPointError("Non-finite gradients without AMP GradScaler")
            self.amp_skips += 1
            row = {
                "step": trainer.global_step,
                "event": "amp_skip",
                "scale": scaler.get_scale(),
            }
            with (self.output / "numerics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        """Record failure and retain earlier healthy checkpoints."""
        self._status(trainer, "failed", error=str(exception))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Record overflow separately; stop damaged models or persistent failure."""
        if not torch.isfinite(outputs["loss"]).all():
            self.nonfinite_batches += 1
            self.consecutive_nonfinite += 1
            self._check_model(pl_module)
            if getattr(trainer.precision_plugin, "scaler", None) is None:
                raise FloatingPointError(
                    "Non-finite training loss without AMP GradScaler"
                )
            row = {"step": trainer.global_step, "event": "nonfinite_loss"}
            with (self.output / "numerics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if self.consecutive_nonfinite >= 32:
                raise FloatingPointError(
                    "AMP did not recover after 32 consecutive non-finite batches"
                )
        else:
            self.consecutive_nonfinite = 0
        if trainer.global_step % 50 == 0:
            row = {"epoch": trainer.current_epoch, "step": trainer.global_step}
            (self.output / "progress.json").write_text(json.dumps(row))
            self._status(trainer, "running")
            print("PROGRESS", json.dumps(row), flush=True)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Keep invalid epoch aggregates visible without aborting recovered AMP."""
        self._check_model(pl_module)
        metrics = self.finite_metrics(trainer.callback_metrics)
        if any(
            value is None for key, value in metrics.items() if key.startswith("val/")
        ):
            raise FloatingPointError("Non-finite validation metrics")
        row = {
            "epoch": trainer.current_epoch + 1,
            "step": trainer.global_step,
            "session_seconds": time.monotonic() - self.started,
            "metrics": metrics,
            "nonfinite_metrics": [
                key for key, value in metrics.items() if value is None
            ],
            "nonfinite_batches": self.nonfinite_batches,
            "amp_skips": self.amp_skips,
        }
        with (self.output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print("EPOCH", json.dumps(row, allow_nan=False), flush=True)
        if (
            self.stop_after_epochs
            and trainer.current_epoch + 1 >= self.stop_after_epochs
        ):
            trainer.should_stop = True


def main() -> None:
    """Resolve a recipe, prepare local data, and run or resume with Manager."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "ssley-paper.json"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--precision", choices=["32-true", "16-mixed", "bf16-mixed"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-time", help="Lightning duration DD:HH:MM:SS")
    parser.add_argument("--stop-after-epochs", type=int)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.precision:
        config["precision"] = args.precision
    if args.workers < 0 or args.threads < 1 or args.checkpoint_every_steps < 1:
        parser.error("Invalid worker/thread/checkpoint count")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-steps must be -1 or positive")
    if (
        args.stop_after_epochs is not None
        and not 0 < args.stop_after_epochs <= config["epochs"]
    ):
        parser.error("stop-after-epochs must be within the configured training budget")
    if args.download_only:
        CIFAR10(str(args.data_dir), train=True, download=True)
        CIFAR10(str(args.data_dir), train=False, download=True)
        return
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        parser.error("CUDA GPU unavailable: run this inside a GPU allocation")
    if args.accelerator == "cpu" and config["precision"] != "32-true":
        parser.error("CPU checks require --precision 32-true")
    if config["epochs"] <= config["warmup_epochs"] or config["batch_size"] < 2:
        parser.error("epochs must exceed warmup_epochs; batch size must be >=2")
    resume_steps_per_epoch = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["hyper_parameters"]["recipe"] != config:
            parser.error(
                "Resume recipe differs from the checkpoint; use the original recipe"
            )
        resume_steps_per_epoch = checkpoint["hyper_parameters"]["steps_per_epoch"]
        del checkpoint
    elif (args.output / "resolved.json").exists():
        parser.error(
            "Output already exists: explicitly --resume its checkpoint or choose a new output"
        )
    torch.set_num_threads(args.threads)
    pl.seed_everything(config["seed"], workers=True)
    args.output.mkdir(parents=True, exist_ok=True)
    resolved = {
        "recipe": config,
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "torch": torch.__version__,
        "lightning": pl.__version__,
    }
    (args.output / "resolved.json").write_text(json.dumps(resolved, indent=2))
    spt.set(cache_dir=str(args.output.resolve()), requeue_checkpoint=False)
    if args.smoke:
        raw_train = FakeData(
            size=config["batch_size"] * 2, image_size=(3, 32, 32), num_classes=10
        )
        raw_val = FakeData(
            size=config["batch_size"],
            image_size=(3, 32, 32),
            num_classes=10,
            random_offset=10000,
        )
    else:
        # Downloads are a separate preparation step, avoiding network use on compute nodes.
        raw_train = CIFAR10(str(args.data_dir), train=True, download=False)
        raw_val = CIFAR10(str(args.data_dir), train=False, download=False)
    train_transform, val_transform = augmentations(config)
    train = spt.data.FromTorchDataset(
        raw_train, names=["image", "label"], transform=train_transform
    )
    val = spt.data.FromTorchDataset(
        raw_val, names=["image", "label"], transform=val_transform
    )
    loader_options = dict(
        batch_size=config["batch_size"],
        num_workers=args.workers,
        pin_memory=args.accelerator == "gpu",
        persistent_workers=False,
    )
    train_loader = DataLoader(train, shuffle=True, drop_last=True, **loader_options)
    if resume_steps_per_epoch is not None and resume_steps_per_epoch != len(
        train_loader
    ):
        parser.error(
            "Resume data length differs from checkpoint; do not mix smoke and real data"
        )
    data = spt.data.DataModule(
        train=train_loader, val=DataLoader(val, **loader_options)
    )
    module = PaperSSLEY(config, steps_per_epoch=len(train_loader))
    module.backbone.to(memory_format=torch.channels_last)
    checkpoint_dir = args.output.resolve() / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="latest",
        save_last=True,
        save_top_k=1,
        every_n_train_steps=args.checkpoint_every_steps,
        save_on_train_epoch_end=False,
    )
    log = RunLog(args.output, args.stop_after_epochs)
    callbacks = [log, checkpoint_callback]
    if args.max_time:
        callbacks.append(SessionTimer(duration=args.max_time, interval="step"))
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision=config["precision"],
        max_epochs=config["epochs"],
        max_steps=args.max_steps,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        log_every_n_steps=50,
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        seed=config["seed"],
        ckpt_path=str(args.resume.resolve()) if args.resume else None,
        weights_only=False,
    )()
    trainer.save_checkpoint(checkpoint_dir / "last.ckpt")
    torch.save(
        {k: v.detach().cpu() for k, v in module.backbone.state_dict().items()},
        args.output / "backbone.pt",
    )
    complete = trainer.global_step >= config["epochs"] * len(train_loader)
    report = {
        "state": "complete" if complete else "paused",
        "steps": trainer.global_step,
        "synthetic": args.smoke,
        "checkpoint": str(checkpoint_dir / "last.ckpt"),
        "session_seconds": time.monotonic() - log.started,
        "metrics": log.finite_metrics(trainer.callback_metrics),
        "nonfinite_batches": log.nonfinite_batches,
        "amp_skips": log.amp_skips,
    }
    (args.output / "run-status.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
