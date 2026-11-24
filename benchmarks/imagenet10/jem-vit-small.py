"""JEM ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

import os

import torch
import torchvision
from multicrop import (
    get_data_dir,
    standard_callbacks,
    standard_trainer,
    val_transform,
)

import stable_pretraining as spt
from stable_pretraining.data import transforms


def imagenette(split: str, transform) -> spt.data.FromTorchDataset:
    """Imagenette from its official archive, downloaded on first use."""
    root = get_data_dir("imagenet10")
    images = torchvision.datasets.Imagenette(
        root, split=split, download=not (root / "imagenette2").exists()
    )
    return spt.data.FromTorchDataset(
        images, names=["image", "label"], transform=transform
    )


def main():
    batch_size = 128
    max_epochs = int(os.environ.get("MAX_EPOCHS", 20))

    views = transforms.AlignedMultiCrop(
        {"teacher": 224, "global": 224, **{f"local_{i}": 96 for i in range(4)}},
        transform=transforms.Compose(
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
    )
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=imagenette("train", transforms.Compose(transforms.RGB(), views)),
            batch_size=batch_size,
            num_workers=8,
            drop_last=True,
            persistent_workers=True,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=imagenette("val", val_transform()),
            batch_size=batch_size,
            num_workers=8,
            persistent_workers=True,
        ),
    )

    module = spt.JEM("vit_small_patch16_224")
    trainer = standard_trainer(
        standard_callbacks(module, embed_dim=module.embed_dim),
        max_epochs=max_epochs,
        log_name="jem-vits-inet10",
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
