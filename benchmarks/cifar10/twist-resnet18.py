"""TWIST training on CIFAR-10 (ResNet-18, two views).

Environment variables select the experiment so one script covers the main
run, the loss/normalisation ablations and unsupervised classification:

- ``MAX_EPOCHS`` (default 400), ``BATCH_SIZE`` (512), ``NUM_WORKERS`` (8)
- ``N_CLASSES`` (2048): output dimension ``C`` of the head. Use 10 to evaluate
  unsupervised classification against the CIFAR-10 labels.
- ``HIDDEN_DIM`` (2048): width of the two hidden layers of the head.
- ``BASE_LR`` (0.5): LARS learning rate for batch size 256, scaled linearly.
- ``SEED`` (0): seed passed to ``spt.Manager``.
- ``VARIANT`` (``full``): one of ``full``, ``no_sharpness``, ``no_diversity``,
  ``no_nbs`` (drops the batch normalisation before the softmax).
- ``LIMIT_TRAIN_BATCHES`` / ``LIMIT_VAL_BATCHES``: for quick smoke runs.

Metrics (loss, its three terms, logit statistics, linear and kNN probes) are
written by the registry logger to the run directory under the spt cache; view
them with ``spt web``. Clustering scores of the argmax predictions on the
validation set are saved there as ``clustering.json``.

Example::

    python benchmarks/cifar10/twist-resnet18.py
    VARIANT=no_nbs MAX_EPOCHS=100 python benchmarks/cifar10/twist-resnet18.py
    N_CLASSES=10 python benchmarks/cifar10/twist-resnet18.py
"""

import json
import os
import sys
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import CSVLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402

VARIANTS = {
    # name: (sharpness_weight, diversity_weight, norm_before_softmax)
    "full": (1.0, 1.0, True),
    "no_sharpness": (0.0, 1.0, True),
    "no_diversity": (1.0, 0.0, True),
    "no_nbs": (1.0, 1.0, False),
}
WARMUP_EPOCHS = 10


class LogitStatistics(pl.Callback):
    """Log the row/column standard deviation of the logits before the final BN.

    These are the diagnostics of the paper's Fig. 3: a small column standard
    deviation means every sample in the batch gets a similar score for a class,
    i.e. low diversity.
    """

    def __init__(self, layer: nn.Module):
        self.stats = {}
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if module.training:
            out = output.detach().float()
            self.stats = {
                "logit_row_std": out.std(dim=1, unbiased=False).mean(),
                "logit_col_std": out.std(dim=0, unbiased=False).mean(),
            }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, value in self.stats.items():
            pl_module.log(f"fit/{name}", value, on_step=True, on_epoch=True)


def make_data(batch_size: int, num_workers: int) -> spt.data.DataModule:
    def view(scale, solarize):
        steps = [
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ]
        if solarize:
            steps.append(transforms.RandomSolarize(threshold=0.5, p=0.2))
        steps.append(transforms.ToImage(**spt.data.static.CIFAR10))
        return transforms.Compose(*steps)

    train_transform = transforms.MultiViewTransform(
        [view((0.2, 1.0), solarize=False), view((0.08, 1.0), solarize=True)]
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )

    data_dir = str(get_data_dir("cifar10"))
    train_dataset = spt.data.FromTorchDataset(
        torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True),
        names=["image", "label"],
        transform=train_transform,
    )
    val_dataset = spt.data.FromTorchDataset(
        torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True),
        names=["image", "label"],
        transform=val_transform,
    )
    return spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            shuffle=True,
            persistent_workers=num_workers > 0,
        ),
        val=torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )


def make_projector(hidden_dim: int, n_classes: int, norm_before_softmax: bool):
    layers = [
        nn.Linear(512, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, n_classes, bias=not norm_before_softmax),
    ]
    if norm_before_softmax:
        layers.append(nn.BatchNorm1d(n_classes, affine=False))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate_clustering(module, dataloader, n_classes: int) -> dict:
    """Score the argmax class predictions on the validation set as a clustering.

    Labels are only used for scoring. Accuracy uses the Kuhn-Munkres one-to-one
    matching between predicted and true classes, so it is reported only when
    ``n_classes`` equals the number of true classes.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    device = next(module.backbone.parameters()).device
    module.eval()
    preds, labels = [], []
    for batch in dataloader:
        logits = module.projector(module.backbone(batch["image"].to(device)))
        preds.append(logits.argmax(dim=-1).cpu())
        labels.append(batch["label"].cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()

    results = {
        "nmi": float(normalized_mutual_info_score(labels, preds)),
        "ari": float(adjusted_rand_score(labels, preds)),
        "used_classes": int(len(set(preds.tolist()))),
    }
    n_labels = int(labels.max()) + 1
    if n_classes == n_labels:
        counts = torch.zeros(n_classes, n_labels, dtype=torch.long)
        for p, t in zip(preds, labels):
            counts[p, t] += 1
        rows, cols = linear_sum_assignment(counts.numpy(), maximize=True)
        results["accuracy"] = float(counts[rows, cols].sum() / len(labels))
    return results


def main():
    max_epochs = int(os.environ.get("MAX_EPOCHS", 400))
    batch_size = int(os.environ.get("BATCH_SIZE", 512))
    num_workers = int(os.environ.get("NUM_WORKERS", 8))
    n_classes = int(os.environ.get("N_CLASSES", 2048))
    hidden_dim = int(os.environ.get("HIDDEN_DIM", 2048))
    base_lr = float(os.environ.get("BASE_LR", 0.5))
    seed = int(os.environ.get("SEED", 0))
    variant = os.environ.get("VARIANT", "full")
    if variant not in VARIANTS:
        raise ValueError(f"VARIANT must be one of {sorted(VARIANTS)}, got {variant!r}")
    sharpness_weight, diversity_weight, norm_before_softmax = VARIANTS[variant]

    data = make_data(batch_size, num_workers)
    steps_per_epoch = len(data.train)
    limit_train = os.environ.get("LIMIT_TRAIN_BATCHES")
    if limit_train is not None:
        steps_per_epoch = min(steps_per_epoch, int(limit_train))
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = max(1, min(WARMUP_EPOCHS, max_epochs // 2) * steps_per_epoch)

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    projector = make_projector(hidden_dim, n_classes, norm_before_softmax)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward.twist,
        hparams={
            "method": "twist",
            "variant": variant,
            "n_classes": n_classes,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "base_lr": base_lr,
            "max_epochs": max_epochs,
            "seed": seed,
        },
        twist_loss=spt.losses.TWISTLoss(
            sharpness_weight=sharpness_weight, diversity_weight=diversity_weight
        ),
        optim={
            # Paper recipe for CNNs: LARS, lr = 0.5 * batch / 256, weight decay
            # 1.5e-6, biases and normalisation parameters excluded from both.
            "optimizer": {
                "type": "LARS",
                "lr": base_lr * batch_size / 256,
                "momentum": 0.9,
                "weight_decay": 1.5e-6,
                "exclude_bias_norm": True,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": warmup_steps,
                "total_steps": total_steps,
            },
            "interval": "step",
        },
    )

    linear_probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(512, 10),
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
        input_dim=512,
        k=10,
    )
    last_linear = [m for m in projector if isinstance(m, nn.Linear)][-1]

    run_name = os.environ.get("RUN_NAME", f"twist-resnet18-{variant}-c{n_classes}")
    logger = CSVLogger(save_dir=str(Path(__file__).parent / "logs"), name=run_name)
    limits = {}
    if limit_train is not None:
        limits["limit_train_batches"] = int(limit_train)
    if os.environ.get("LIMIT_VAL_BATCHES") is not None:
        limits["limit_val_batches"] = int(os.environ["LIMIT_VAL_BATCHES"])
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[knn_probe, linear_probe, LogitStatistics(last_linear)],
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=logger,
        enable_checkpointing=False,
        **limits,
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=seed)
    manager()

    if torch.cuda.is_available():
        module.to("cuda")
    elif torch.backends.mps.is_available():
        module.to("mps")
    results = evaluate_clustering(module, data.val, n_classes)
    results.update(variant=variant, n_classes=n_classes, max_epochs=max_epochs)
    # Manager stores the run (metrics.csv, sidecar.json) in its registry
    # run directory; keep the clustering scores next to them.
    run_dir = Path(getattr(manager, "_run_dir", None) or logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / "clustering.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Unsupervised classification on the validation set: {results}")


if __name__ == "__main__":
    main()
