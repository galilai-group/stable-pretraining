#!/usr/bin/env python

from functools import partial


def main():
    from pathlib import Path

    # m1 = run("gpu", 1)
    m2 = run("gpu", 1)
    # print(m1)
    print(m2)
    Path("./lightning_logs").unlink()


def run(accelerator, devices):
    import os

    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    import lightning as pl
    import torch
    import torchmetrics
    import torchvision
    from lightning.pytorch.loggers import CSVLogger
    from transformers import AutoConfig, AutoModelForImageClassification

    import stable_pretraining as ssl
    from stable_pretraining.data import transforms
    from stable_pretraining.data.datasets import Dataset

    # Create a custom wrapper that adds sample_idx

    class IndexedDataset(Dataset):
        """Custom dataset wrapper that adds sample_idx to each sample."""

        def __init__(self, dataset, transform=None):
            super().__init__(transform)
            self.dataset = dataset

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            sample = {"image": image, "label": label, "sample_idx": idx}
            return self.process_sample(sample)

        def __len__(self):
            return len(self.dataset)

    # pl.seed_everything(0, workers=True)
    # torch.use_deterministic_algorithms(True)

    train_transform = transforms.ToImage()

    # Use torchvision CIFAR-10
    cifar_train = torchvision.datasets.CIFAR10(
        root="~/tmp/cifar10", train=True, download=True
    )
    train_dataset = IndexedDataset(cifar_train, transform=train_transform)

    # Increase batch size for multi-GPU training (64 per GPU * 2 GPUs = 128 total)
    train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=256 // devices,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )

    val_transform = transforms.ToImage()

    # Use torchvision CIFAR-10 for validation
    cifar_val = torchvision.datasets.CIFAR10(
        root="~/tmp/cifar10", train=False, download=True
    )
    val_dataset = IndexedDataset(cifar_val, transform=val_transform)
    val = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=128 // devices, num_workers=8, drop_last=False
    )
    data = ssl.data.DataModule(train=train, val=val)

    def forward(self, batch, stage):
        output = {}
        output["embedding"] = self.backbone(batch["image"])["logits"]
        if stage == "fit":
            assert self.training
            output["loss"] = torch.nn.functional.cross_entropy(
                output["embedding"], batch["label"]
            )
            self.log("loss", output["loss"], sync_dist=True, on_step=True)
        return output

    config = AutoConfig.from_pretrained("microsoft/resnet-18")
    backbone = AutoModelForImageClassification.from_config(config)
    backbone.resnet.embedder.embedder = torch.nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    backbone.resnet.embedder.pooler = torch.nn.Identity()
    backbone.classifier[1] = torch.nn.Linear(512, 10)
    backbone.requires_grad_(True)

    module = ssl.Module(
        backbone=backbone, forward=forward, optim=partial(torch.optim.AdamW, lr=0.001)
    )
    # print("DEVICE", module.device)

    # Configure callbacks
    linear_probe = ssl.callbacks.OnlineProbe(
        module=module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=torch.nn.Linear(10, 10),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        },
    )

    # Configure multi-GPU trainer
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[linear_probe],
        logger=CSVLogger("./"),
        enable_checkpointing=False,
        sync_batchnorm=True,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed",
        strategy="ddp",
    )

    manager = ssl.Manager(trainer=trainer, module=module, data=data)
    manager()
    # manager.evaluate()
    return manager._trainer.logged_metrics


if __name__ == "__main__":
    # run with
    # python train_supervised_multigpu_requeue.py --multirun hydra/launcher=submitit_slurm hydra.launcher.timeout_min=7 hydra.launcher.partition=scavenge hydra.launcher.max_num_timeout=10 hydra.launcher.gpus_per_node=8 hydra.launcher.tasks_per_node=8 ++batch_size=256 ++signal_delay_s=240
    # python train_supervised_multigpu_requeue.py --multirun hydra/launcher=submitit_slurm hydra.launcher.timeout_min=4 hydra.launcher.partition=learnfair hydra.launcher.max_num_timeout=10 hydra.launcher.gpus_per_node=1 hydra.launcher.tasks_per_node=1 ++batch_size=512

    main()
