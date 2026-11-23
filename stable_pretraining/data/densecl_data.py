"""Dataset utilities for DenseCL experiments."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import stable_pretraining as spt

from stable_pretraining.data import transforms as T


# class DenseCLTwoViewTransform:
#     """Generate two independently augmented views of one image."""

#     def __init__(self, transform):
#         self.transform = transform

#     def __call__(self, image):
#         return [
#             self.transform(image),
#             self.transform(image),
#         ]


class DenseCLCIFAR10(Dataset):
    """CIFAR-10 dataset returning stable-pretraining-style dictionaries.

    Training samples contain two augmented image views. Labels are retained
    for optional online evaluation; DenseCL does not use them as SSL targets.
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform=None,
    ):
        self.dataset = datasets.CIFAR10(
            root=str(Path(root).expanduser()), train=train, download=download
        )
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
        }


def make_densecl_cifar10(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
    pin_memory: Optional[bool] = None,
):
    # Download CIFAR-10 + create DenseCL dataloaders.
    # we normalize the

    # Returns:
    # train_loader: Batches contain two augmented views per image.
    # val_loader: Batches contain one deterministic image view and label.

    # 0.4914 - average red channel pixel intensity after converting values form 0, 1 to 0, 255
    # 0.4822 - average green channel pixel intensity after converting values form 0, 1 to 0, 255

    mean = (0.4914, 0.4822, 0.4465)

    # each std for each channel with given means
    std = (0.2470, 0.2435, 0.2616)

    # train_augmentation = transforms.Compose([
    #     transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply([transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1,)], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ToTensor(),                                  # convert pixel values to range 0-255
    #     transforms.Normalize(mean, std),
    # ])

    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),                                  # convert pixel values to range 0-255
    #     transforms.Normalize(mean, std),                        # normalize to standard values for CIFAR-10
    # ])
    augment1 = T.Compose(
        T.RGB(),
        T.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=0.8,
        ),
        T.RandomGrayscale(p=0.2),
        T.ToImage(mean=mean, std=std),
    )

    augment2 = T.Compose(
        T.RGB(),
        T.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=0.8,
        ),
        T.RandomGrayscale(p=0.2),
        T.ToImage(mean=mean, std=std),
    )

    train_transform = T.MultiViewTransform([augment1, augment2])

    # train_dataset = DenseCLCIFAR10(
    #     root=root,
    #     train=True,
    #     download=download,
    #     transform=DenseCLTwoViewTransform(train_augmentation),
    # )

    train_torch_dataset = datasets.CIFAR10(
        root=str(Path(root).expanduser()),
        train=True,
        download=download,
    )
    val_torch_dataset = datasets.CIFAR10(
        root=str(Path(root).expanduser()),
        train=False,
        download=download,
    )

    train_dataset = spt.data.FromTorchDataset(
        train_torch_dataset,
        names=["image", "label"],
        transform=train_transform,
    )

    # val_dataset = DenseCLCIFAR10(
    #     root=root,
    #     train=False,
    #     download=download,
    #     transform=val_transform,
    # )

    val_dataset = spt.data.FromTorchDataset(
        val_torch_dataset,
        names=["image", "label"],
        transform=T.Compose(
            T.RGB(),
            T.ToImage(mean=mean, std=std),
        ),
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
