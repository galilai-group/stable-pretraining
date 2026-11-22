"""RotNet pretraining on CIFAR-10 with the paper's Network-in-Network.

Reproduces ``config/CIFAR10_RotNet_NIN4blocks.py`` from the official code
(https://github.com/gidariss/FeatureLearningRotNet): a 4-block NIN trained for
200 epochs with SGD (Nesterov, momentum 0.9, weight decay 5e-4) at lr 0.1,
multiplied by 0.2 at epochs 60, 120 and 160, on batches of 128 images, each
expanded to its four rotations. The paper's text describes a shorter 100-epoch
schedule; the released config, which produced the released model, is used here.

Online kNN and linear probes on block-2 features are for monitoring only. The
paper's numbers come from classifiers trained afterwards on frozen features:
run ``rotnet-nin-eval.py`` on the checkpoint this script prints.

Usage::

    python benchmarks/cifar10/rotnet-nin.py
    python benchmarks/cifar10/rotnet-nin.py --num-blocks 5 --feature-block 2
    python benchmarks/cifar10/rotnet-nin.py --rotations 0 180  # Table 2 ablation
"""

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor

import stable_pretraining as spt
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402

# Normalisation constants of the official CIFAR-10 loader.
ROTNET_CIFAR10 = {
    "mean": [x / 255.0 for x in (125.3, 123.0, 113.9)],
    "std": [x / 255.0 for x in (63.0, 62.1, 66.7)],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--feature-block", type=int, default=2)
    parser.add_argument("--rotations", type=int, nargs="+", default=[0, 90, 180, 270])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_data(batch_size: int, num_workers: int):
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


def main():
    args = parse_args()
    # Seed before building the network: Manager seeds only when it runs.
    pl.seed_everything(args.seed, workers=True)
    train_loader, val_loader = make_data(args.batch_size, args.num_workers)

    module = spt.RotNet(
        encoder_name="nin",
        num_blocks=args.num_blocks,
        feature_block=args.feature_block,
        rotations=tuple(args.rotations),
    )
    # The official LR look-up table (lr x0.2 at epochs 60, 120, 160 of 200).
    # Module steps schedulers after every optimizer step, so milestones are in
    # steps.
    milestones = [round(args.epochs * f) * len(train_loader) for f in (0.3, 0.6, 0.8)]
    module.optim = {
        "optimizer": {
            "type": "SGD",
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "nesterov": True,
            "exclude_bias_norm": False,
        },
        "scheduler": {"type": "MultiStepLR", "milestones": milestones, "gamma": 0.2},
    }

    linear_probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=torch.nn.Linear(module.embed_dim, 10),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        },
    )
    knn_probe = spt.callbacks.OnlineKNN(
        name="knn_probe",
        input="embedding",
        target="label",
        queue_length=20000,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
        input_dim=module.embed_dim,
        k=10,
    )

    # One device, as in the paper: under DDP each rank would see 128 images and
    # keep its own batch-norm statistics instead of one 512-image batch.
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        num_sanity_val_steps=0,
        callbacks=[
            knn_probe,
            linear_probe,
            LearningRateMonitor(logging_interval="step"),
        ],
        precision="32-true",
        enable_checkpointing=False,
    )
    data = spt.data.DataModule(train=train_loader, val=val_loader)
    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=args.seed)
    manager()
    print(f"Pretrained checkpoint: {manager._run_dir / 'checkpoints' / 'last.ckpt'}")


if __name__ == "__main__":
    main()
