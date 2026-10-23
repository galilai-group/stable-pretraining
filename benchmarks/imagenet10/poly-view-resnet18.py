"""Poly-view contrastive learning at a fixed encoder budget, ResNet-18 on ImageNet-10 (Imagenette).

Implements the Fixed Batch setting of :cite:`shidani2024polyview` (Section 3.2):
every step encodes the same number of views ``V``, split as ``K = V // M``
unique images times ``M`` views each, so optimizer steps are proportional to
encoder compute. A run's compute is ``(M / 2) * epochs`` SimCLR-epochs. With
``M = 2`` every poly-view loss is SimCLR's NT-Xent. Backbone, projector,
augmentation, optimizer, learning rate and temperature do not depend on ``M``
or the loss.

Augmentation runs on the GPU (:class:`StackedMultiView`), and each step makes
one backbone call over all ``V`` views, so BatchNorm always normalizes over
``V`` views regardless of ``M``, as in both papers.

Each run logs to ``<log-dir>/<loss>-M<views>-e<epochs>/seed<seed>/``.

Usage::

    # train one run
    python benchmarks/imagenet10/poly-view-resnet18.py train \
        --loss geometric --views 8 --epochs 200 --seed 0

    # kNN + online-linear curves (mean ± std over seeds) for any runs, given
    # as run names under --log-dir or as run/seed directory paths
    python benchmarks/imagenet10/poly-view-resnet18.py plot \
        geometric-M2-e200 \
        geometric-M4-e100 \
        geometric-M8-e50 \
        --x step
"""

import argparse
import dataclasses
import json
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import gpu_transforms as gt, transforms
from stable_pretraining.methods import PolyView
from stable_pretraining.optim import LinearWarmupCosineAnnealing

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402

LOSSES = ("geometric", "arithmetic", "sufficient_statistics")
NUM_CLASSES = 10
PROJECTOR_DIMS = (2048, 2048, 128)
IMAGENET_MEAN = spt.data.static.ImageNet["mean"]
IMAGENET_STD = spt.data.static.ImageNet["std"]
DEFAULT_LOG_DIR = Path(__file__).parent / "logs" / "poly-view-resnet18"


# --- config ---------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One training run. Every field is also a ``train`` command-line flag."""

    views: int = field(default=2, metadata={"help": "views per image, M"})
    epochs: int = field(default=200, metadata={"help": "passes over unique images"})
    seed: int = 0
    loss: str = field(
        default="geometric",
        metadata={
            "choices": LOSSES,
            "help": "poly-view loss (any loss is SimCLR at M=2)",
        },
    )
    views_per_step: int = field(
        default=256, metadata={"help": "encoder budget V = K * M, fixed across arms"}
    )
    temperature: float = 0.1
    base_lr: float = field(
        default=0.3,
        metadata={"help": "LARS lr per 256 SimCLR images; not rescaled with M"},
    )
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    image_size: int = 224
    val_every: int = field(default=5, metadata={"help": "validate every n epochs"})
    num_workers: int = 8
    log_dir: str = str(DEFAULT_LOG_DIR)

    def __post_init__(self):
        if self.views < 2:
            raise ValueError(f"views must be >= 2, got {self.views}")
        if self.views_per_step % self.views:
            raise ValueError(
                f"views_per_step={self.views_per_step} not divisible by views={self.views}"
            )
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs)")
        if self.epochs % self.val_every:
            raise ValueError(
                "epochs must be a multiple of val_every to log the last epoch"
            )

    @property
    def batch_size(self) -> int:
        """Unique images per step, K."""
        return self.views_per_step // self.views

    @property
    def lr(self) -> float:
        # Linear scaling on SimCLR's image count at this budget (V / 2), held
        # fixed for every M as in the paper ("optimizer scaling rule: none").
        return self.base_lr * (self.views_per_step / 2) / 256

    @property
    def run_name(self) -> str:
        return f"{self.loss}-M{self.views}-e{self.epochs}"


# --- data -----------------------------------------------------------------


def _imagenette(split: str, transform, gpu_transform=None):
    # Parquet mirror of frgfm/imagenette (official 9469/3925 split), whose
    # loading script and parquet conversion are no longer served by the Hub.
    return spt.data.HFDataset(
        "ShaomuTan/imagenette",
        "320px",
        split=split,
        cache_dir=str(get_data_dir("imagenet10")),
        transform=transform,
        gpu_transform=gpu_transform,
    )


def _resize_size(cfg: Config) -> int:
    return cfg.image_size * 8 // 7  # 256 for 224, the usual resize/crop ratio


def cpu_train_transform(cfg: Config):
    """Fixed-size uint8->float tensor; all random augmentation happens on GPU."""
    size = _resize_size(cfg)
    return transforms.Compose(
        transforms.Resize((size, size)),
        transforms.ToImage(rgb=True, scale=True),
    )


def eval_transform(cfg: Config):
    size = _resize_size(cfg)
    return transforms.Compose(
        transforms.Resize((size, size)),
        transforms.CenterCrop((cfg.image_size, cfg.image_size)),
        transforms.ToImage(rgb=True, **spt.data.static.ImageNet),
    )


def gpu_train_transform(cfg: Config):
    """SimCLR augmentation, one chain call producing ``M`` views per image."""
    blur_kernel = cfg.image_size // 10 | 1  # ~10% of the image, odd (23 at 224)
    chain = gt.GPUCompose(
        [
            gt.GPURandomResizedCrop(size=cfg.image_size, scale=(0.08, 1.0)),
            gt.GPURandomHorizontalFlip(p=0.5),
            gt.GPUColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            gt.GPURandomGrayscale(p=0.2),
            gt.GPUGaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0), p=0.5),
            gt.GPUNormalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return gt.StackedMultiView(chain, n_views=cfg.views)


def _loader(dataset, cfg: Config, batch_size: int, train: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        shuffle=train,
        drop_last=train,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def build_data(cfg: Config) -> spt.data.DataModule:
    train = _imagenette("train", cpu_train_transform(cfg), gpu_train_transform(cfg))
    val = _imagenette("validation", eval_transform(cfg))
    return spt.data.DataModule(
        train=_loader(train, cfg, cfg.batch_size, train=True),
        val=_loader(val, cfg, cfg.views_per_step, train=False),
    )


# --- model ----------------------------------------------------------------


def poly_view_forward(self, batch, stage):
    """Training: M views from StackedMultiView. Validation: one center crop.

    ``spt.forward.poly_view`` encodes each view in its own backbone call, as
    ``spt.forward.simclr`` does, so BatchNorm normalizes each view separately.
    Here all views go through one call, so BatchNorm normalizes over all of them.
    We concatenate because per-view calls shrink the BatchNorm batch to
    ``K = V / M`` as ``M`` grows, but both papers normalize over the full batch:
    Poly-View with synced BatchNorm (:cite:`shidani2024polyview`, Appendix F.1),
    SimCLR with Global BN (Chen et al., 2020, Section 2.2).
    """
    if "image" in batch:
        return {
            "embedding": self.backbone(batch["image"]),
            "label": batch["label"].long(),
        }

    views = batch["views"]
    # One backbone/projector call over all K * M views, so BatchNorm statistics
    # cover the same V views in every arm rather than K images per view.
    h = self.backbone(torch.cat([v["image"] for v in views]))
    z = self.projector(h)
    loss = self.poly_view_loss(list(z.chunk(len(views))))
    self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
    return {
        "loss": loss,
        "embedding": h.detach(),
        "label": torch.cat([v["label"] for v in views]).long(),
    }


def optim_config(cfg: Config) -> dict:
    warmup_fraction = cfg.warmup_epochs / cfg.epochs

    def scheduler(optimizer, module):
        return LinearWarmupCosineAnnealing(
            optimizer,
            total_steps=module.trainer.estimated_stepping_batches,
            peak_step=warmup_fraction,
        )

    return {
        "optimizer": {
            "type": "LARS",
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "clip_lr": True,
            "eta": 0.02,
            "exclude_bias_n_norm": True,
        },
        "scheduler": scheduler,
        "interval": "step",
    }


def build_module(cfg: Config) -> PolyView:
    module = PolyView(
        encoder_name="resnet18",
        projector_dims=PROJECTOR_DIMS,
        temperature=cfg.temperature,
        loss=cfg.loss,
    )
    module.forward = types.MethodType(poly_view_forward, module)
    module.optim = optim_config(cfg)
    return module


# --- training -------------------------------------------------------------


def _top1() -> torchmetrics.Metric:
    return torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES)


class WallClock(pl.Callback):
    """Log seconds since the start of training at the end of every train epoch."""

    def on_train_start(self, trainer, pl_module):
        self.start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log("time/elapsed_seconds", time.perf_counter() - self.start)


def build_callbacks(module: PolyView) -> list:
    return [
        WallClock(),
        spt.callbacks.OnlineProbe(
            module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=nn.Linear(module.embed_dim, NUM_CLASSES),
            loss=nn.CrossEntropyLoss(),
            metrics={"top1": _top1()},
            optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 0.0},
        ),
        spt.callbacks.OnlineKNN(
            name="knn_probe",
            input="embedding",
            target="label",
            queue_length=10000,
            metrics={"top1": _top1()},
            input_dim=module.embed_dim,
            k=20,
        ),
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]


def build_trainer(cfg: Config, callbacks: list) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=cfg.val_every,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=pl.pytorch.loggers.CSVLogger(
            save_dir=cfg.log_dir, name=cfg.run_name, version=f"seed{cfg.seed}"
        ),
        precision="16-mixed",
        enable_checkpointing=False,
        devices=1,
        accelerator="gpu",
    )


@torch.no_grad()
def extract_features(backbone: nn.Module, loader, device) -> tuple:
    backbone.eval()
    feats, labels = [], []
    for batch in loader:
        feats.append(backbone(batch["image"].to(device)).float().cpu())
        labels.append(batch["label"])
    return torch.cat(feats), torch.cat(labels).long()


def offline_linear_probe(train: tuple, val: tuple, steps: int = 1000) -> float:
    """Full-batch logistic regression on frozen, standardized features.

    Same recipe for every arm, unlike the online probe whose number of updates
    grows with M at a fixed epoch count. Returns validation top-1.
    """
    (x_train, y_train), (x_val, y_val) = train, val
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    x_train, x_val = (x_train - mean) / std, (x_val - mean) / std
    probe = nn.Linear(x_train.size(1), NUM_CLASSES)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(probe(x_train), y_train).backward()
        opt.step()
    with torch.no_grad():
        return (probe(x_val).argmax(1) == y_val).float().mean().item()


def evaluate_offline(cfg: Config, module: PolyView) -> dict:
    def features(split: str) -> tuple:
        dataset = _imagenette(split, eval_transform(cfg))
        loader = _loader(dataset, cfg, cfg.views_per_step, train=False)
        return extract_features(backbone, loader, device)

    # Lightning may move the module back to CPU after fit.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = module.backbone.to(device)
    return {
        "offline_linear_top1": offline_linear_probe(
            features("train"), features("validation")
        )
    }


def _timestamp(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


def train(cfg: Config) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    module = build_module(cfg)
    trainer = build_trainer(cfg, build_callbacks(module))
    run_dir = Path(trainer.logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
    config = dataclasses.asdict(cfg) | {"started_at": _timestamp(started)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    trainer.fit(module, datamodule=build_data(cfg))
    fit_done = datetime.now().astimezone()

    final = evaluate_offline(cfg, module)
    finished = datetime.now().astimezone()
    final |= {
        "started_at": _timestamp(started),
        "finished_at": _timestamp(finished),
        "train_seconds": round((fit_done - started).total_seconds(), 1),
        "total_seconds": round((finished - started).total_seconds(), 1),
    }
    (run_dir / "final.json").write_text(json.dumps(final, indent=2))
    print(f"{cfg.run_name} seed{cfg.seed}: {final}")


# --- plot -----------------------------------------------------------------

KNN = "eval/knn_probe_top1"
ONLINE_LINEAR = "eval/linear_probe_top1_epoch"


def load_run(path: Path, seeds: int | None = None) -> tuple:
    """Label and validation history of one run, pooled over its finished seeds.

    A seed counts as finished once ``train`` has written its ``final.json``.

    Args:
        path (Path): A run directory holding ``seed*/`` subdirectories, or a
            single seed directory.
        seeds (int | None): Use only ``seed0`` .. ``seed{seeds-1}`` of a run
            directory; all seeds if None.

    Returns:
        tuple: ``(label, history)``, where ``history`` has one row per logged
        step with ``epoch``, ``step``, ``seed`` and both probe metrics.
    """
    import pandas as pd

    seed_dirs = sorted(path.glob("seed*"), key=lambda d: int(d.name[4:]))
    if seeds is not None:
        seed_dirs = [d for d in seed_dirs if int(d.name[4:]) < seeds]
    seed_dirs = seed_dirs or [path]
    histories = []
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "metrics.csv"
        if not (seed_dir / "final.json").exists():
            print(f"skipping {seed_dir}: no final.json (unfinished)")
            continue
        history = pd.read_csv(metrics_path).reindex(
            columns=["epoch", "step", KNN, ONLINE_LINEAR]
        )
        histories.append(history.assign(seed=seed_dir.name))
    if not histories:
        raise SystemExit(f"no finished seeds under {path}")

    cfg = json.loads((seed_dirs[0] / "config.json").read_text())
    method = (
        "SimCLR"
        if cfg["views"] == 2
        else f"{cfg['loss'].capitalize()} PVC, M={cfg['views']}"
    )
    seeds = f"{len(histories)} seed" + ("s" if len(histories) > 1 else "")
    label = f"{method} ({cfg['epochs']} ep, {seeds})"
    return label, pd.concat(histories)


def plot_curves(runs: list, x: str, path: Path) -> None:
    """KNN and online-linear top-1 against ``x`` ("epoch" or "step"), mean ± std."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for label, history in runs:
        for ax, metric in zip(axes, (KNN, ONLINE_LINEAR)):
            points = history.dropna(subset=[metric])
            stats = points.groupby(x)[metric].agg(["mean", "std"]).fillna(0) * 100
            xs = stats.index + 1 if x == "epoch" else stats.index
            ax.plot(xs, stats["mean"], marker="o", markersize=3, label=label)
            ax.fill_between(
                xs,
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.2,
            )
    for ax, title in zip(axes, ("kNN top-1 (%)", "Online linear top-1 (%)")):
        ax.set_title(title)
        ax.set_xlabel("epoch" if x == "epoch" else "optimizer step (∝ encoder compute)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# --- command line ---------------------------------------------------------


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    for f in dataclasses.fields(Config):
        parser.add_argument(
            f"--{f.name.replace('_', '-')}",
            type=type(f.default),
            default=f.default,
            choices=f.metadata.get("choices"),
            help=f.metadata.get("help"),
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    _add_config_flags(commands.add_parser("train", help="train one run"))

    plot_parser = commands.add_parser("plot", help="probe curves for chosen runs")
    plot_parser.add_argument(
        "runs",
        nargs="+",
        help="run names under --log-dir (e.g. geometric-M4-e100), or paths to "
        "run or seed directories",
    )
    plot_parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    plot_parser.add_argument("--x", choices=("epoch", "step"), default="epoch")
    plot_parser.add_argument(
        "--seeds", type=int, default=None, help="use only seeds 0..n-1 of each run"
    )
    plot_parser.add_argument(
        "--out", type=Path, default=None, help="default: <log-dir>/curves_<x>.png"
    )

    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "train":
        train(Config(**args))
    else:
        log_dir = args["log_dir"]
        paths = [Path(r) if Path(r).is_dir() else log_dir / r for r in args["runs"]]
        out = args["out"] or log_dir / f"curves_{args['x']}.png"
        plot_curves([load_run(p, args["seeds"]) for p in paths], args["x"], out)


if __name__ == "__main__":
    main()
