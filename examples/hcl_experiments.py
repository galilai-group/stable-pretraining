"""
Hard Negative Contrastive Learning (HCL)
========================================

Reproduces the image experiments of HCL (Robinson et al., ICLR 2021) and the
SimCLR baseline it is compared against. ``HCLLoss`` is a drop-in replacement for
``NTXEntLoss`` in the composable :func:`stable_pretraining.forward.simclr`
pipeline, so the only difference between the two runs is the loss.

The recipe follows the official implementation (https://github.com/joshr17/HCL,
``image/main.py`` and ``image/linear.py``):

- **Pretraining**: CIFAR-style ResNet (3x3 stem, no max-pool), projector
  ``Linear(D, 512) -> BN -> ReLU -> Linear(512, 128)``, Adam (lr 1e-3, weight
  decay 1e-6, constant), temperature 0.5, batch size 256, 400 epochs, random
  32x32 crops with SimCLR augmentations. STL-10 pretrains on
  ``train+unlabeled``.
- **Linear evaluation**: frozen encoder, linear classifier trained with Adam
  (lr 1e-3, weight decay 1e-6) for 100 epochs at batch size 512 on the labeled
  training split, evaluated on the test split at native resolution.

Usage::

    # 1) Pretrain: SimCLR baseline, or HCL with the official defaults
    python examples/hcl_experiments.py --phase pretrain --dataset stl10 \\
        --backbone resnet50 --loss simclr
    python examples/hcl_experiments.py --phase pretrain --dataset stl10 \\
        --backbone resnet50 --loss hcl --beta 1.0 --tau-plus 0.1

    # 2) Linear evaluation of a pretraining checkpoint
    python examples/hcl_experiments.py --phase linear --dataset stl10 \\
        --backbone resnet50 --ckpt /abs/path/to/checkpoints/last.ckpt

Pretraining runs are logged by the built-in registry (browse them with
``spt web``); set ``SPT_CACHE_DIR`` to choose where runs and checkpoints go.
"""

import argparse
import json
import os
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
import torchvision
from torch import nn

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms
from stable_pretraining.methods.hcl import HCLLoss

# The official code normalizes every dataset (STL-10 included) with these values.
OFFICIAL_NORM = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "stl10": 10}
FEATURE_DIM = {"resnet18": 512, "resnet50": 2048}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="HCL / SimCLR experiments")
    parser.add_argument("--phase", choices=["pretrain", "linear"], default="pretrain")
    parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "stl10"], default="cifar10"
    )
    parser.add_argument(
        "--backbone", choices=["resnet18", "resnet50"], default="resnet50"
    )
    parser.add_argument("--loss", choices=["simclr", "hcl"], default="hcl")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tau-plus", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=None, help="400 / 100 by phase")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="256 / 512 by phase"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default=None, help="STL-10 root")
    parser.add_argument("--resume", type=str, default=None, help="pretrain checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint to evaluate")
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="cap batches per epoch (smoke tests)",
    )
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 400 if args.phase == "pretrain" else 100
    if args.batch_size is None:
        args.batch_size = 256 if args.phase == "pretrain" else 512
    if args.phase == "linear" and args.ckpt is None:
        parser.error("--phase linear requires --ckpt")
    return args


def train_view() -> transforms.Compose:
    """Official HCL training augmentation: SimCLR augmentations on 32x32 crops."""
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
        transforms.ToImage(**OFFICIAL_NORM),
    )


def test_view() -> transforms.Compose:
    """Official HCL test transform: native resolution, normalization only."""
    return transforms.Compose(transforms.RGB(), transforms.ToImage(**OFFICIAL_NORM))


def get_dataset(name: str, split: str, transform, data_dir: str):
    """Load ``split`` in {"pretrain", "train", "test"} of a dataset.

    CIFAR comes from the Hugging Face Hub; STL-10 is read with torchvision, whose
    ``train+unlabeled`` split is what the official code pretrains on.
    """
    if name == "stl10":
        stl_split = {"pretrain": "train+unlabeled", "train": "train", "test": "test"}
        dataset = torchvision.datasets.STL10(
            root=data_dir, split=stl_split[split], download=True
        )
        return spt.data.FromTorchDataset(
            dataset, names=["image", "label"], transform=transform
        )
    rename = {"img": "image"}
    if name == "cifar100":
        rename["fine_label"] = "label"
    return spt.data.HFDataset(
        path=f"uoft-cs/{name}",
        split="test" if split == "test" else "train",
        transform=transform,
        rename_columns=rename,
    )


def build_backbone(name: str) -> nn.Module:
    """CIFAR-style ResNet, matching the official ``Model`` encoder."""
    backbone = spt.backbone.from_torchvision(name, low_resolution=True)
    backbone.fc = nn.Identity()
    return backbone


def make_loader(dataset, args, shuffle: bool, drop_last: bool = False):
    """DataLoader with persistent workers (worker startup is slow under spawn)."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def default_data_dir() -> str:
    """Same location as ``benchmarks/utils.get_data_dir()``."""
    base = os.environ.get(
        "STABLE_PRETRAINING_DATA_DIR",
        Path.home() / ".cache" / "stable-pretraining" / "data",
    )
    return str(base)


def accelerator_and_precision():
    """Mixed precision on CUDA (bf16 where supported), fp32 elsewhere."""
    if torch.cuda.is_available():
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        return "gpu", precision
    if torch.backends.mps.is_available():
        return "mps", "32-true"
    return "cpu", "32-true"


def pretrain(args):
    """Self-supervised pretraining with the SimCLR or HCL loss."""
    backbone = build_backbone(args.backbone)
    feat_dim = FEATURE_DIM[args.backbone]
    projector = nn.Sequential(
        nn.Linear(feat_dim, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 128),
    )

    if args.loss == "hcl":
        loss = HCLLoss(
            temperature=args.temperature, beta=args.beta, tau_plus=args.tau_plus
        )
    else:
        loss = spt.losses.NTXEntLoss(temperature=args.temperature)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward.simclr,
        simclr_loss=loss,
        optim={
            "optimizer": {"type": "Adam", "lr": args.lr, "weight_decay": 1e-6},
            "scheduler": {"type": "ConstantLR", "factor": 1.0, "total_iters": 0},
            "interval": "epoch",
        },
    )

    data_dir = args.data_dir or default_data_dir()
    view = train_view()
    train_set = get_dataset(
        args.dataset, "pretrain", transforms.MultiViewTransform([view, view]), data_dir
    )
    train_loader = make_loader(train_set, args, shuffle=True, drop_last=True)

    # Online probes need a label for every training image, which STL-10's
    # unlabeled split does not have; its accuracy comes from the linear phase.
    callbacks, val_loader = [], None
    if args.dataset != "stl10":
        num_classes = NUM_CLASSES[args.dataset]
        callbacks = [
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(feat_dim, num_classes),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
                    "top5": torchmetrics.classification.MulticlassAccuracy(
                        num_classes, top_k=5
                    ),
                },
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=20000,
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(num_classes)
                },
                input_dim=feat_dim,
                k=200,
                temperature=args.temperature,
            ),
        ]
        val_set = get_dataset(args.dataset, "test", test_view(), data_dir)
        val_loader = make_loader(val_set, args, shuffle=False)

    accelerator, precision = accelerator_and_precision()
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        seed=args.seed,
        ckpt_path=args.resume,
        weights_only=args.resume is None,
    )
    manager()


@torch.no_grad()
def evaluate(backbone, head, loader, device) -> dict:
    """Top-1 / top-5 accuracy of ``head(backbone(x))`` on ``loader``."""
    backbone.eval()
    head.eval()
    correct1 = correct5 = total = 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        top5 = head(backbone(x)).topk(5, dim=-1).indices
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(dim=-1).sum().item()
        total += y.numel()
    return {"top1": 100 * correct1 / total, "top5": 100 * correct5 / total}


def linear_eval(args):
    """Train a linear classifier on the frozen pretrained encoder (official linear.py)."""
    pl.seed_everything(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    backbone = build_backbone(args.backbone)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    backbone.load_state_dict(
        {
            k.removeprefix("backbone."): v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
    )
    backbone.requires_grad_(False)
    backbone.to(device)

    num_classes = NUM_CLASSES[args.dataset]
    head = nn.Linear(FEATURE_DIM[args.backbone], num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    data_dir = args.data_dir or default_data_dir()
    train_loader = make_loader(
        get_dataset(args.dataset, "train", train_view(), data_dir), args, shuffle=True
    )
    test_loader = make_loader(
        get_dataset(args.dataset, "test", test_view(), data_dir), args, shuffle=False
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        # The official linear.py keeps the frozen encoder in train mode while
        # fitting the classifier, so BatchNorm statistics keep adapting; keep
        # that behavior to stay comparable with the published numbers.
        backbone.train()
        head.train()
        for step, batch in enumerate(train_loader):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            x, y = batch["image"].to(device), batch["label"].to(device)
            loss = criterion(head(backbone(x)), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = evaluate(backbone, head, test_loader, device)
        history.append({"epoch": epoch, **metrics})
        print(
            f"[linear] epoch {epoch}/{args.epochs} test top1 {metrics['top1']:.2f} "
            f"top5 {metrics['top5']:.2f}",
            flush=True,
        )

    result = {
        "ckpt": args.ckpt,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "final": history[-1],
        "best_top1": max(h["top1"] for h in history),
        "history": history,
    }
    out_path = Path(args.ckpt).parent.parent / "linear_eval.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"[linear] final top1 {result['final']['top1']:.2f} -> {out_path}", flush=True
    )


def main():
    """Entry point."""
    args = parse_args()
    if args.phase == "pretrain":
        pretrain(args)
    else:
        linear_eval(args)


if __name__ == "__main__":
    main()
