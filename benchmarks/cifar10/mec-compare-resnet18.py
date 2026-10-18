"""Controlled MEC vs BYOL vs SimSiam comparison on CIFAR-10 with ResNet-18.

Every run shares the backbone, augmentations, batch size, epoch budget and
online probes; only the method (heads, loss, paper-default optimizer)
changes. MEC ablation flags vary the Taylor order and the lambda warm-up.

Example::

    python benchmarks/cifar10/mec-compare-resnet18.py --method mec --epochs 50
    python benchmarks/cifar10/mec-compare-resnet18.py --method mec --taylor-order 1
    python benchmarks/cifar10/mec-compare-resnet18.py --method byol
"""

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.losses import MECLoss
from stable_pretraining.methods import BYOL, MEC, SimSiam

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402


def two_view_transform():
    """Asymmetric two-view CIFAR augmentation, as in the other CIFAR-10 scripts."""
    views = []
    for min_scale in (0.2, 0.08):
        views.append(
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((32, 32), scale=(min_scale, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToImage(**spt.data.static.CIFAR10),
            )
        )
    return transforms.MultiViewTransform(views)


def make_data(batch_size, num_workers):
    data_dir = str(get_data_dir("cifar10"))
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )
    train = spt.data.FromTorchDataset(
        torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True),
        names=["image", "label"],
        transform=two_view_transform(),
    )
    val = spt.data.FromTorchDataset(
        torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True),
        names=["image", "label"],
        transform=val_transform,
    )
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            train, drop_last=True, shuffle=True, **loader_kwargs
        ),
        val=torch.utils.data.DataLoader(val, **loader_kwargs),
    )


def resnet18():
    backbone = spt.backbone.from_torchvision(
        "resnet18", low_resolution=True, weights=None
    )
    backbone.fc = nn.Identity()
    return backbone


class NormGuardedMECLoss(MECLoss):
    """Experimental: raise ``1/λ`` when needed so that ``||λC||_2 <= max_c_norm``.

    Keeps the Taylor series inside its radius of convergence, as an adaptive
    alternative to the fixed warm-up. Used only for diagnostic runs;
    ``MECLoss(symmetric=True)`` is the recommended fix.
    """

    def __init__(self, max_c_norm=1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_c_norm = max_c_norm

    def forward(self, online_pred, target_proj, lambda_inv_scale=1.0):
        norm = c_spectral_norm(online_pred, target_proj, self.symmetric)
        base = self.lambda_inv(*online_pred.shape)
        scale = max(lambda_inv_scale, norm / (self.max_c_norm * base))
        return super().forward(online_pred, target_proj, scale)


@torch.no_grad()
def c_spectral_norm(online_pred, target_proj, symmetric=False):
    """``||P Zᵀ||_2`` of the L2-normalized inputs, before the ``λ`` scaling."""
    # On CPU so that GPU autocast cannot downcast the matmul to bf16.
    p = F.normalize(online_pred.float().cpu(), dim=-1)
    z = F.normalize(target_proj.float().cpu(), dim=-1)
    c = p @ z.T
    if symmetric:
        c = (c + c.T) / 2
    return torch.linalg.matrix_norm(c, ord=2).item()


def build_method(args):
    """Method module plus its paper-default optimizer, scaled to the batch size."""
    scale = args.batch_size / 256
    if args.method == "mec":
        module = MEC(
            encoder_name=resnet18(),
            taylor_order=args.taylor_order,
            symmetric=args.symmetric_c,
            warmup_epochs=args.warmup_epochs,
        )
        if args.max_c_norm is not None:
            module.mec_loss = NormGuardedMECLoss(
                max_c_norm=args.max_c_norm,
                order=args.taylor_order,
                symmetric=args.symmetric_c,
            )
        optimizer = {
            "type": "SGD",
            "lr": 0.5 * scale,
            "momentum": 0.9,
            "weight_decay": 1e-5,
        }
    elif args.method == "byol":
        module = BYOL(encoder_name=resnet18(), ema_decay_start=0.996)
        optimizer = {
            "type": "LARS",
            "lr": 0.2 * scale,
            "weight_decay": 1.5e-6,
            "exclude_bias_n_norm": True,
        }
    elif args.method == "simsiam":
        module = SimSiam(encoder_name=resnet18())
        optimizer = {
            "type": "SGD",
            "lr": 0.05 * scale,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        }
    else:
        raise ValueError(f"unknown method {args.method}")
    if args.base_lr is not None:
        optimizer["lr"] = args.base_lr * scale
    if args.weight_decay is not None:
        optimizer["weight_decay"] = args.weight_decay
    module.optim = {
        "optimizer": optimizer,
        "scheduler": {"type": "LinearWarmupCosineAnnealing"},
        "interval": "epoch",
    }
    return module


def mec_c_norm(module, output):
    """``||λC||_2`` of the paper's (unsymmetrized) C; the series needs it below 1."""
    batch_size = output.prediction.shape[0] // 2
    p, z = output.prediction[:batch_size], output.target[batch_size:]
    lambda_inv = module.mec_loss.lambda_inv(*p.shape) * module._lambda_inv_scale()
    return c_spectral_norm(p, z) / lambda_inv


def two_view_forward(self, batch, stage):
    """Lightning forward for the two-view method classes; also logs MEC entropy."""
    if "image" in batch:
        output = type(self).forward(self, batch["image"])
        return {"embedding": output.embedding, "label": batch["label"].long()}

    views = batch["views"]
    output = type(self).forward(self, views[0]["image"], views[1]["image"])
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True)
    if getattr(output, "entropy", None) is not None:
        self.log(f"{stage}/mec_entropy", output.entropy, on_step=True, on_epoch=True)
        if self.global_step % 10 == 0:
            self.log(
                f"{stage}/mec_c_norm_max",
                mec_c_norm(self, output),
                on_step=False,
                on_epoch=True,
                reduce_fx="max",
            )
    return {
        "loss": output.loss,
        "embedding": output.embedding.detach(),
        "label": torch.cat([v["label"] for v in views]).long(),
    }


def callbacks(module, embed_dim):
    teacher = (
        [spt.callbacks.TeacherStudentCallback(update_after_backward=True)]
        if isinstance(module.backbone, spt.TeacherStudentWrapper)
        else []
    )
    return [
        *teacher,
        spt.callbacks.OnlineProbe(
            module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=nn.Linear(embed_dim, 10),
            loss=nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(10),
                "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
            },
        ),
        spt.callbacks.OnlineKNN(
            name="knn_probe",
            input="embedding",
            target="label",
            queue_length=20000,
            metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
            input_dim=embed_dim,
            k=10,
        ),
        spt.callbacks.RankMe(
            name="rankme",
            target="embedding",
            queue_length=2048,
            target_shape=embed_dim,
        ),
    ]


def run_name(args):
    name = args.method
    if args.method == "mec":
        name += f"-order{args.taylor_order}-warmup{args.warmup_epochs}"
        if args.symmetric_c:
            name += "-sym"
        if args.max_c_norm is not None:
            name += f"-maxc{args.max_c_norm}"
    if args.batch_size != 256:
        name += f"-bs{args.batch_size}"
    if args.base_lr is not None:
        name += f"-lr{args.base_lr}"
    if args.weight_decay is not None:
        name += f"-wd{args.weight_decay}"
    return f"{name}-ep{args.epochs}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["mec", "byol", "simsiam"], default="mec")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--taylor-order", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--base-lr", type=float, default=None, help="override the paper lr (per 256)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None, help="override the paper value"
    )
    parser.add_argument(
        "--symmetric-c", action="store_true", help="experimental: use (C + Cᵀ) / 2"
    )
    parser.add_argument(
        "--max-c-norm",
        type=float,
        default=None,
        help="experimental: raise lambda_inv so that ||C||_2 stays below this",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit-batches", type=float, default=1.0, help="for quick smoke tests"
    )
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).parent / "logs" / "mec-compare")
    )
    args = parser.parse_args()

    # One run registry per configuration, so each run's metrics.csv is findable.
    spt.set(cache_dir=str(Path(args.out_dir, run_name(args)).resolve()))
    module = build_method(args)
    module.forward = two_view_forward.__get__(module)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
        num_sanity_val_steps=0,
        callbacks=callbacks(module, module.embed_dim),
        precision="bf16-mixed",
        enable_checkpointing=False,
    )
    data = make_data(args.batch_size, args.num_workers)
    spt.Manager(trainer=trainer, module=module, data=data, seed=args.seed)()


if __name__ == "__main__":
    main()
