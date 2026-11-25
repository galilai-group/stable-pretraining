"""CPLearn training on CIFAR-10.

Follows the ResNet-18 recipe of the paper (Appendix L) and the official config
(scripts/pretrain/cifar/cplearn.yaml in github.com/emsansone/CPLearn):
f = 2048, c = 16384, beta = 1, Adam (lr 1e-3, weight decay 1e-4), batch size
64, 1000 epochs, warmup + cosine schedule, asymmetric augmentations (the
second view adds solarization). Reported: 91.3% linear accuracy (Table 2).

Measured with this script (1000 epochs, one RTX 3080 Ti, one seed; offline
linear probe on frozen features): 84.9% linear, NMI 0.471. Setting
``batch_size = 256`` and ``dictionary_size=2048`` gives 89.8% linear, NMI
0.486, and trains about 2.5x faster.
"""

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms
from stable_pretraining.methods.cplearn import CPLearnProjector
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

cplearn_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            # Applied to the PIL image (0-255): 128 matches the official
            # threshold, while 0.5 would invert every pixel.
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**spt.data.static.CIFAR10),
)

data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(
    root=str(data_dir), train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=cplearn_transform,
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

batch_size = 64
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=8,
    drop_last=True,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=8,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone = spt.backbone.from_torchvision(
    "resnet18",
    low_resolution=True,
)
backbone.fc = torch.nn.Identity()

module = spt.Module(
    backbone=backbone,
    projector=CPLearnProjector(512, projector_dim=2048, dictionary_size=16384),
    forward=forward.cplearn,
    cplearn_loss=spt.losses.CPLearnLoss(beta=1.0),
    optim={
        "optimizer": {
            "type": "Adam",
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "step",
    },
)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
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
    input_dim=512,
    k=10,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="cifar10-cplearn",
    name="cplearn-resnet18",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=1000,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
