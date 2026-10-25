"""LFR training on CIFAR-10, following Appendix D.6 of Sui et al. (ICLR 2024).

Pretraining uses unaugmented images only. After training, the frozen encoder
and an untrained copy with the same initialization are each evaluated with
the paper's linear protocol, so the gain over random features is measured
under one budget. The paper reports 64.3% (LFR) and 34.7% (random init) after
400 cycles; pass ``--cycles`` for shorter runs. Training metrics
(``metrics.csv``) and ``summary.json`` are written under ``--output-dir``.

    python benchmarks/cifar10/lfr-resnet18.py --cycles 50 --output-dir runs/lfr
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from torch.utils.data import DataLoader, Subset, TensorDataset

import stable_pretraining as spt
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cycles", type=int, default=400)
parser.add_argument("--predictor-epochs", type=int, default=5)
parser.add_argument("--num-projectors", type=int, default=40)
parser.add_argument("--candidate-ratio", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--probe-epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument(
    "--subset", type=int, default=None, help="use the first N images (smoke runs)"
)
parser.add_argument("--output-dir", type=Path, default=Path("runs/lfr-cifar10"))
args = parser.parse_args()

pl.seed_everything(args.seed, workers=True)
args.output_dir.mkdir(parents=True, exist_ok=True)
# Manager writes metrics under ``<cache_dir>/runs/``. Its per-epoch requeue
# checkpoint (about 1.3 GB with 40 projectors) is only needed to resume
# preempted jobs, so it is off for the 2,400-epoch schedule.
spt.set(cache_dir=str(args.output_dir), requeue_checkpoint=False)

# LFR needs no augmentation: training and evaluation share one transform.
transform = transforms.Compose(
    transforms.RGB(),
    transforms.ToImage(**spt.data.static.CIFAR10),
)

data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(
    root=str(data_dir), train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)
if args.subset is not None:
    cifar_train = Subset(cifar_train, range(args.subset))
    cifar_val = Subset(cifar_val, range(args.subset))

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=transform,
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=transform,
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    drop_last=True,
    shuffle=True,
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)
data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

# CIFAR ResNet-18 as in the authors' code: a 2048-d output layer followed by a
# non-affine BatchNorm. That 2048-d output is the evaluated representation.
dim = 2048
backbone = spt.backbone.from_torchvision(
    "resnet18", low_resolution=True, zero_init_residual=True
)
backbone.fc = nn.Sequential(nn.Linear(512, dim), nn.BatchNorm1d(dim, affine=False))
random_backbone = copy.deepcopy(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
selection_sample = None
if args.candidate_ratio > 1:
    selection_sample = next(iter(train_dataloader))["image"]

module = spt.LFR(
    encoder_name=backbone,
    num_projectors=args.num_projectors,
    candidate_ratio=args.candidate_ratio,
    selection_sample=selection_sample,
    target_dim=dim,
    predictor_epochs=args.predictor_epochs,
)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(dim, 10),
    loss=nn.CrossEntropyLoss(),
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
    input_dim=dim,
    k=10,
)

trainer = pl.Trainer(
    max_epochs=module.max_epochs_for(args.cycles),
    # One device: the paper's batch of 512, and no automatic plain-DDP choice.
    devices=1,
    # Validate once per cycle, after its last predictor epoch.
    check_val_every_n_epoch=1 + args.predictor_epochs,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe],
    precision="16-mixed" if device == "cuda" else "32-true",
    enable_checkpointing=False,
)
spt.Manager(trainer=trainer, module=module, data=data, seed=args.seed)()


@torch.no_grad()
def encode(encoder: nn.Module, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache frozen features and labels."""
    encoder.eval().to(device)
    features, labels = [], []
    for batch in loader:
        features.append(encoder(batch["image"].to(device)).float().cpu())
        labels.append(batch["label"])
    return torch.cat(features), torch.cat(labels)


def linear_eval(encoder: nn.Module) -> dict[str, float]:
    """Paper protocol: SGD (momentum 0.9, no weight decay), batch 512, lr 0.2 cosine."""
    eval_train = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    train_x, train_y = encode(encoder, eval_train)
    test_x, test_y = encode(encoder, val_dataloader)
    generator = torch.Generator().manual_seed(args.seed)
    head = nn.Linear(dim, 10).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.probe_epochs)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=512,
        shuffle=True,
        generator=generator,
    )
    for _ in range(args.probe_epochs):
        for x, y in loader:
            optimizer.zero_grad()
            nn.functional.cross_entropy(head(x.to(device)), y.to(device)).backward()
            optimizer.step()
        scheduler.step()
    with torch.no_grad():
        predictions = head(test_x.to(device)).argmax(1).cpu()
    singular = torch.linalg.svdvals(train_x[:2048])
    p = singular / singular.sum()
    return {
        "test_top1": (predictions == test_y).float().mean().item(),
        # Collapse diagnostics on the training features.
        "mean_feature_std": train_x.std(dim=0).mean().item(),
        "effective_rank": torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item(),
    }


summary = {
    "config": {k: str(v) for k, v in vars(args).items()},
    "hardware": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
    "online_linear_top1": trainer.callback_metrics.get("eval/linear_probe_top1"),
    "online_knn_top1": trainer.callback_metrics.get("eval/knn_probe_accuracy"),
    "lfr": linear_eval(module.backbone),
    "random_init": linear_eval(random_backbone),
    "paper": {"lfr_test_top1": 0.643, "random_init_test_top1": 0.347, "cycles": 400},
}
summary = json.loads(
    json.dumps(summary, default=lambda v: v.item() if torch.is_tensor(v) else str(v))
)
(args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
