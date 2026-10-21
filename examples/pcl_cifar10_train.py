"""Configurable CIFAR-10 training for PCL and its instance-only ablation."""

import argparse
import os

import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.methods.pcl import PCL, PCLClusteringCallback


class IndexedCIFAR10(Dataset):
    """Provide augmented views and stable dataset indices."""

    def __init__(self, root="./data", subset_size=128, seed=42):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=False,
        )

        if not 0 < subset_size <= len(self.dataset):
            raise ValueError("Invalid subset size")

        generator = torch.Generator().manual_seed(seed)
        self.indices = torch.randperm(
            len(self.dataset),
            generator=generator,
        )[:subset_size].tolist()

        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        self.augment = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.cluster_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]

        return {
            "image": self.cluster_transform(image),
            "view1": self.augment(image),
            "view2": self.augment(image),
            "indices": torch.tensor(index, dtype=torch.long),
        }


class PCLLossLogger(Callback):
    """Record total, instance, and prototype losses."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not isinstance(outputs, dict):
            return

        for name in ("loss", "instance_loss", "prototype_loss"):
            value = outputs.get(name)

            if value is not None:
                pl_module.log(
                    f"pcl/{name}",
                    value.detach(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(name == "loss"),
                    logger=True,
                )

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        parts = [f"epoch={trainer.current_epoch}"]

        for name in ("loss", "instance_loss", "prototype_loss"):
            key = f"pcl/{name}"
            if key in metrics:
                parts.append(f"{name}={metrics[key].item():.4f}")

        print("[PCL Loss] " + ", ".join(parts), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PCL or an instance-only ablation on CIFAR-10."
    )

    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cluster-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument(
        "--num-clusters",
        type=int,
        nargs="+",
        default=[32, 64],
    )
    parser.add_argument("--queue-length", type=int, default=256)
    parser.add_argument("--num-proto-negatives", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--accelerator",
        default=os.environ.get("PCL_ACCELERATOR", "cpu"),
    )
    parser.add_argument("--instance-only", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/pcl-cifar10",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a trusted Lightning checkpoint.",
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet50"],
        default="resnet18",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    if not args.instance_only:
        if args.warmup_epochs < 0:
            raise ValueError("--warmup-epochs must be nonnegative")
        if not all(1 <= k <= args.subset_size for k in args.num_clusters):
            raise ValueError("Cluster counts must be between 1 and subset size")
        if any(
            k < args.batch_size + args.num_proto_negatives for k in args.num_clusters
        ):
            raise ValueError(
                "Each cluster count must be at least "
                "batch size + number of prototype negatives"
            )

    dataset = IndexedCIFAR10(
        root=args.data_root,
        subset_size=args.subset_size,
        seed=args.seed,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = PCL(
        encoder_name=args.backbone,
        low_resolution=True,
        projector_dims=(512, 128),
        queue_length=args.queue_length,
        temperature=0.2,
    )

    model.num_proto_negatives = args.num_proto_negatives
    model.optim = {
        "optimizer": {
            "type": "SGD",
            "lr": args.learning_rate,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        }
    }

    callbacks = [
        TeacherStudentCallback(verbose=False),
    ]

    if not args.instance_only:
        cluster_loader = DataLoader(
            dataset,
            batch_size=args.cluster_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        callbacks.append(
            PCLClusteringCallback(
                dataloader=cluster_loader,
                num_clusters=tuple(args.num_clusters),
                warmup_epochs=args.warmup_epochs,
                seed=args.seed,
            )
        )

    callbacks.extend(
        [
            ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename="epoch-{epoch:03d}",
                save_last=True,
                save_top_k=-1,
                every_n_epochs=args.checkpoint_every,
            ),
            PCLLossLogger(),
        ]
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        callbacks=callbacks,
    )

    print("Experiment configuration:", vars(args), flush=True)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        ckpt_path=args.resume,
        weights_only=False if args.resume else None,
    )

    print("PCL CIFAR-10 training completed.")
    print("Optimizer steps:", trainer.global_step)
    print(
        "Clustering levels:",
        0 if args.instance_only else 1 + len(model.additional_clusterings),
    )


if __name__ == "__main__":
    main()
