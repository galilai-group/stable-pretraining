"""Frozen-feature CIFAR-10 evaluation of a RotNet checkpoint (paper protocol).

Trains an object classifier on the frozen feature maps of one NIN block of a
pretrained RotNet, reproducing the official configs
``CIFAR10_MultLayerClassifier_on_RotNet_NIN4blocks_Conv2_feats.py`` (``--head
fc``) and ``CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats.py``
(``--head conv``). The feature extractor stays in eval mode, so its batch-norm
statistics are frozen as in the original code. The RotNet architecture is read
from the checkpoint.

Paper targets (Table 3, 4-block RotNet, block-2 features): 89.06 (``fc``) and
91.16 (``conv``). For Table 1 (other blocks, ``--block``), the official repo has
no configs, so the block-2 schedules are reused.

The official code reports the best test accuracy over epochs; this script
prints both that and the final-epoch accuracy. Accuracy here is weighted by
sample, whereas the official code averages per-batch accuracies (the last
CIFAR-10 test batch holds 16 images), so the two can differ slightly.

Usage::

    python benchmarks/cifar10/rotnet-nin-eval.py --ckpt <run_dir>/checkpoints/last.ckpt
    python benchmarks/cifar10/rotnet-nin-eval.py --ckpt ... --head conv
    python benchmarks/cifar10/rotnet-nin-eval.py --ckpt ... --block 3  # Table 1
"""

import argparse
import math
import sys
from pathlib import Path

import lightning as pl
import torch
import torchvision
from torch import nn

import stable_pretraining as spt
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402

# Normalisation constants of the official CIFAR-10 loader.
ROTNET_CIFAR10 = {
    "mean": [x / 255.0 for x in (125.3, 123.0, 113.9)],
    "std": [x / 255.0 for x in (63.0, 62.1, 66.7)],
}
# Official schedules: (epochs, lr milestones); lr 0.1, x0.2 at each milestone.
SCHEDULES = {"fc": (100, [20, 40, 45]), "conv": (100, [35, 70, 85])}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="checkpoint from rotnet-nin.py")
    parser.add_argument("--block", type=int, default=2, help="NIN block to evaluate")
    parser.add_argument("--head", choices=list(SCHEDULES), default="fc")
    parser.add_argument("--epochs", type=int, help="override; milestones scale")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_data(batch_size: int, num_workers: int):
    """Official CIFAR-10 pipeline: crop + flip for training, none for test."""
    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(**ROTNET_CIFAR10),
    )
    val_transform = transforms.Compose(
        transforms.RGB(), transforms.ToImage(**ROTNET_CIFAR10)
    )
    data_dir = get_data_dir("cifar10")
    cifar_train = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True
    )
    cifar_val = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        spt.data.FromTorchDataset(
            cifar_train, names=["image", "label"], transform=train_transform
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        spt.data.FromTorchDataset(
            cifar_val, names=["image", "label"], transform=val_transform
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def _official_init(module: nn.Module) -> None:
    """Initialisation of the official ``NonLinearClassifier``."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, math.sqrt(2.0 / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()


def build_classifier(head: str, feat_shape: torch.Size, num_classes: int = 10):
    if head == "fc":
        hidden = min(num_classes * 20, 2048)
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(feat_shape), hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )
    else:
        classifier = nn.Sequential(
            spt.backbone.nin_block(feat_shape[0], 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, num_classes),
        )
    _official_init(classifier)
    return classifier


def load_frozen_features(ckpt_path: str, block: int) -> nn.Module:
    """Frozen NIN blocks ``1..block`` of the RotNet stored in ``ckpt_path``."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    num_blocks = hparams.get("num_blocks", 4)
    feature_block = hparams.get("feature_block", 2)
    rotnet = spt.RotNet(
        encoder_name="nin",
        num_blocks=num_blocks,
        feature_block=feature_block,
        rotations=tuple(hparams.get("rotations", (0, 90, 180, 270))),
    )
    weights = {
        k: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(("backbone.", "rotation_head."))
    }
    rotnet.load_state_dict(weights, strict=True)
    nin_blocks = (
        list(rotnet.backbone) + list(rotnet.rotation_head)[: num_blocks - feature_block]
    )
    if not 1 <= block <= len(nin_blocks):
        raise ValueError(f"--block must be in [1, {len(nin_blocks)}], got {block}")
    return spt.backbone.EvalOnly(nn.Sequential(*nin_blocks[:block]))


def classify(self, batch, stage):
    """``forward.supervised`` plus exact top-1 accuracy."""
    out = spt.forward.supervised(self, batch, stage)
    correct = (out["logits"].argmax(dim=1) == batch["label"]).float()
    self.log(
        f"{stage}/top1",
        correct.mean(),
        on_step=False,
        on_epoch=True,
        sync_dist=True,
        batch_size=correct.numel(),
    )
    return out


class BestAccuracy(pl.Callback):
    """Track validation accuracy per epoch, as the official code reports the best.

    The history is part of the checkpoint, so a SLURM requeue keeps the
    epochs that ran before it.
    """

    def __init__(self):
        self.history: list[tuple[int, float]] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        acc = trainer.callback_metrics.get("validate/top1")
        if acc is not None:
            self.history.append((trainer.current_epoch + 1, float(acc)))

    def state_dict(self):
        return {"history": list(self.history)}

    def load_state_dict(self, state_dict):
        self.history = [tuple(h) for h in state_dict.get("history", [])]

    def summary(self) -> str:
        if not self.history:
            return "no validation epochs recorded"
        last_epoch, last = self.history[-1]
        best_epoch, best = max(self.history, key=lambda h: h[1])
        return (
            f"final top-1 {100 * last:.2f} (epoch {last_epoch}), "
            f"best top-1 {100 * best:.2f} (epoch {best_epoch})"
        )


def main():
    args = parse_args()
    # Seed before building the classifier: Manager seeds only when it runs.
    pl.seed_everything(args.seed, workers=True)
    features = load_frozen_features(args.ckpt, args.block)
    with torch.no_grad():
        feat_shape = features(torch.zeros(2, 3, 32, 32)).shape[1:]

    train_loader, val_loader = make_data(args.batch_size, args.num_workers)

    epochs, milestones = SCHEDULES[args.head]
    if args.epochs is not None:
        milestones = [round(m * args.epochs / epochs) for m in milestones]
        epochs = args.epochs
    # Module steps schedulers after every optimizer step, so milestones are in steps.
    milestones = [m * len(train_loader) for m in milestones]
    module = spt.Module(
        forward=classify,
        backbone=features,
        classifier=build_classifier(args.head, feat_shape),
        supervised_loss=nn.CrossEntropyLoss(),
        optim={
            "optimizer": {
                "type": "SGD",
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "nesterov": True,
                "exclude_bias_norm": False,
            },
            "scheduler": {
                "type": "MultiStepLR",
                "milestones": milestones,
                "gamma": 0.2,
            },
        },
    )

    best = BestAccuracy()
    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=1,
        num_sanity_val_steps=0,
        callbacks=[best],
        precision="32-true",
        enable_checkpointing=False,
    )
    data = spt.data.DataModule(train=train_loader, val=val_loader)
    spt.Manager(trainer=trainer, module=module, data=data, seed=args.seed)()
    print(f"block {args.block}, {args.head} head: {best.summary()}")


if __name__ == "__main__":
    main()
