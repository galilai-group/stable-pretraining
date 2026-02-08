"""
Supervised Learning Example
============================

This example demonstrates how to train models using supervised learning
with stable-pretraining, including support for various datasets like
ImageNet-10, ImageNet-100, and ImageNet-1k.
"""

import hydra


def get_data_loaders(cfg):
    from stable_pretraining.data import transforms
    import stable_pretraining as spt
    import torch
    import numpy as np

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if cfg.get("dataset_name", "inet100") == "inet10":
        path = "frgfm/imagenette"
        name = "full_size"
    elif cfg.get("dataset_name", "inet100") == "inet100":
        path = "ilee0022/ImageNet100"
        name = None
    elif cfg.get("dataset_name", "inet1k") == "inet1k":
        path = "ILSVRC/imagenet-1k"
        name = None

    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=(5, 5), p=1.0),
        transforms.ToImage(mean=mean, std=std),
    )
    train_dataset = spt.data.HFDataset(
        path=path, name=name, split="train", transform=train_transform
    )
    num_classes = int(np.max(train_dataset.dataset["label"]) + 1)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=spt.data.sampler.RepeatedRandomSampler(
            train_dataset, n_views=cfg.get("n_views", 1)
        ),
        batch_size=cfg.get("batch_size", 256),
        num_workers=cfg.get("num_workers", 10),
        drop_last=True,
        pin_memory=True,
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(mean=mean, std=std),
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=spt.data.HFDataset(
            path=path, name=name, split="validation", transform=val_transform
        ),
        batch_size=256,
        num_workers=cfg.get("num_workers", 10),
    )
    return train_loader, val_loader, num_classes


@hydra.main(version_base="1.3")
def main(cfg: dict):
    import stable_pretraining as spt
    import torch
    from transformers import AutoModelForImageClassification, AutoConfig
    import lightning as pl
    import torchmetrics
    from lightning.pytorch.loggers import WandbLogger
    from functools import partial

    import torchvision

    # without transform
    train_loader, val_loader, num_classes = get_data_loaders(cfg)
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    def forward(self, batch, stage):
        batch["embedding"] = self.backbone(batch["image"])["logits"]
        batch["projector"] = self.projector(batch["embedding"])
        if self.training:
            loss = torch.nn.functional.cross_entropy(
                batch["projector"],
                batch["label"],
                label_smoothing=cfg.get("label_smoothing", 0),
            )
            batch["loss"] = loss
        return batch

    config = AutoConfig.from_pretrained(cfg.get("backbone", "microsoft/resnet-18"))
    backbone = AutoModelForImageClassification.from_config(config)
    backbone = spt.backbone.utils.set_embedding_dim(
        backbone, cfg.get("embedding_dim", 2048)
    )
    if cfg.get("projector_arch", "linear") == "linear":
        projector = torch.nn.Linear(
            cfg.get("embedding_dim", 2048), cfg.get("projector_dim", 128)
        )
    elif cfg.get("projector_arch", "linear") == "identity":
        projector = torch.nn.Identity()
        cfg["projector_dim"] = cfg.get("embedding_dim", 2048)
    else:
        projector = torchvision.ops.MLP(
            cfg.get("embedding_dim", 2048),
            hidden_channels=[2048, 2048, cfg.get("projector_dim", 128)],
            norm_layer=torch.nn.BatchNorm1d,
        )
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward,
        hparams=cfg,
        optim={
            "optimizer": partial(
                torch.optim.AdamW,
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-3),
            ),
            "scheduler": "LinearWarmupCosineAnnealing",
        },
    )
    linear_probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=torch.nn.Linear(cfg.get("embedding_dim", 2048), num_classes),
        loss=torch.nn.CrossEntropyLoss(),
        metrics=torchmetrics.classification.MulticlassAccuracy(num_classes),
    )
    linear_probe_proj = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe_proj",
        input="projector",
        target="label",
        probe=torch.nn.Linear(cfg.get("projector_dim", 128), num_classes),
        loss=torch.nn.CrossEntropyLoss(),
        metrics=torchmetrics.classification.MulticlassAccuracy(num_classes),
    )
    lr_monitor = pl.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step", log_momentum=True, log_weight_decay=True
    )
    logger = WandbLogger(project=cfg.get("wandb_project", "supervised_learning"))
    trainer = pl.Trainer(
        max_epochs=cfg.get("max_epochs", 100),
        num_sanity_val_steps=1,
        callbacks=[lr_monitor, linear_probe, linear_probe_proj],
        precision="16-mixed",
        logger=logger,
        sync_batchnorm=True,
        enable_checkpointing=False,
    )
    manager = spt.Manager(trainer=trainer, module=module, data=data_module)
    manager()
    manager.validate()


if __name__ == "__main__":
    """Examples to run:
    HYDRA_FULL_ERROR=1 python supervised_learning.py ++embedding_dim=2048 ++projector_dim=256 ++projector_arch=linear ++dataset_name=inet100 ++max_epochs=50 ++batch_size=256 ++backbone=microsoft/resnet-50
    HYDRA_FULL_ERROR=1 python supervised_learning.py ++embedding_dim=2048 ++projector_dim=256 ++projector_arch=MLP ++dataset_name=inet100 ++max_epochs=50 ++batch_size=256 ++backbone=microsoft/resnet-18
    """
    main()
