"""Paper-aligned CIFAR recipe components; kept separate from the generic method."""

from typing import Any

import torch
from lightning.pytorch.callbacks import Timer
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms as T

import stable_pretraining as spt
from ssley_reference_lars import ReferenceLARS
from ssley_reference_scheduler import LinearWarmupCosineAnnealingLR


class SessionTimer(Timer):
    """Give every resumed job its own wall-time budget."""

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Ignore elapsed time from a previous job.

        Args:
            state_dict: Previous callback state, deliberately not restored.
        """
        super().load_state_dict({})


class DictImageTransform:
    """Adapt torchvision's PIL pipeline to stable-pretraining dictionaries."""

    def __init__(self, transform: Any) -> None:
        self.transform = transform

    def __call__(self, sample: dict) -> dict:
        sample = dict(sample)
        sample["image"] = self.transform(sample["image"])
        return sample


def augmentations(recipe: dict) -> tuple[Any, Any]:
    """Build the author-fork CIFAR training and evaluation transforms.

    Args:
        recipe: Configuration with crop scale and two solarization probabilities.

    Returns:
        A MultiViewTransform and a single-image evaluation transform.
    """
    views = []
    for solarization in recipe["solarization_probabilities"]:
        transform = T.Compose(
            [
                T.RandomResizedCrop(
                    32,
                    scale=tuple(recipe["crop_scale"]),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomSolarize(128, p=solarization),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        views.append(DictImageTransform(transform))
    # The pinned fork uses rounded CIFAR statistics in its validation pipeline.
    validation = DictImageTransform(
        T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
    )
    return spt.data.transforms.MultiViewTransform(views), validation


def paper_forward(self: Any, batch: dict, stage: str) -> dict[str, torch.Tensor]:
    """Run SSL-EY plus a detached online classifier in one optimization step.

    Args:
        self: Module with SSL-EY components and an online classifier.
        batch: Two-view training or single-view evaluation dictionary.
        stage: Lightning stage passed through Module.

    Returns:
        SSL outputs and joint loss in training; classification metrics in eval.

    Note:
        The classifier receives detached features, as in solo-learn; its
        supervised loss never updates the encoder or projector.
    """
    views = batch["views"] if self.training else [batch]
    if self.training and len(views) != 2:
        raise ValueError("The CIFAR reference recipe requires two views")
    embeddings, projections, logits, losses, labels = [], [], [], [], []
    for view in views:
        embedding = self.backbone(view["image"].to(memory_format=torch.channels_last))
        logit = self.classifier(embedding.detach())
        if self.training:
            projections.append(self.projector(embedding))
        losses.append(F.cross_entropy(logit, view["label"], ignore_index=-1))
        embeddings.append(embedding)
        logits.append(logit)
        labels.append(view["label"])
    classification_loss = sum(losses) / len(losses)
    out = {
        "embedding": torch.cat(embeddings),
        "label": torch.cat(labels),
        "classification_loss": classification_loss,
    }
    batch_size = len(labels[0])
    if self.training:
        ssl_loss = self.ssley_loss(*projections)
        out["loss"] = ssl_loss + classification_loss
        self.log("fit/ssl_loss", ssl_loss, on_epoch=True, batch_size=batch_size)
        self.log("fit/loss", out["loss"], on_epoch=True, batch_size=batch_size)
        self.log(
            "fit/classification_loss",
            classification_loss,
            on_epoch=True,
            batch_size=batch_size,
        )
    else:
        logits = torch.cat(logits)
        self.top1.update(logits, out["label"])
        self.top5.update(logits, out["label"])
        self.log("val/top1", self.top1, on_epoch=True, batch_size=batch_size)
        self.log("val/top5", self.top5, on_epoch=True, batch_size=batch_size)
    return out


class PaperSSLEY(spt.Module):
    """Wire the published CIFAR protocol into Module and Manager.

    Args:
        recipe: Resolved paper configuration.
        steps_per_epoch: Complete batches per epoch, used for the LR schedule.
    """

    def __init__(self, recipe: dict, steps_per_epoch: int) -> None:
        # Construct in the source order so an identical seed gives identical weights.
        backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
        backbone.conv1.padding = (recipe["stem_padding"],) * 2
        classifier = nn.Linear(512, 10)
        dims = [512, *recipe["projector_dims"]]
        layers = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.extend([nn.BatchNorm1d(dims[index + 1]), nn.ReLU()])
        super().__init__(
            forward=paper_forward,
            backbone=backbone,
            projector=nn.Sequential(*layers),
            ssley_loss=spt.losses.SSLEYLoss(
                force_float32=recipe.get("loss_force_float32", False)
            ),
            classifier=classifier,
            hparams={"recipe": recipe, "steps_per_epoch": steps_per_epoch},
        )
        self.recipe = recipe
        self.steps_per_epoch = steps_per_epoch
        self.top1 = MulticlassAccuracy(10, average="micro")
        self.top5 = MulticlassAccuracy(10, average="micro", top_k=5)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Optionally bound vectors that the reference LARS does not adapt.

        Args:
            optimizer: Optimizer after Lightning has unscaled AMP gradients.

        Note:
            This is enabled only in the separately named stability configuration.
            Invalid gradients must remain untouched for GradScaler to skip them.
        """
        limit = self.recipe.get("unadapted_grad_clip_norm")
        if limit is None:
            return
        parameters = [
            p
            for group in optimizer.param_groups
            if group["exclude_bias_n_norm"]
            for p in group["params"]
            if p.ndim == 1 and p.grad is not None
        ]
        if not parameters or any(not torch.isfinite(p.grad).all() for p in parameters):
            return
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, limit, error_if_nonfinite=True
        )
        self.log("fit/unadapted_grad_norm", norm, on_step=True, on_epoch=True)

    def configure_optimizers(self) -> tuple[list, list]:
        config = self.recipe
        optimizer = ReferenceLARS(
            [
                {"name": "backbone", "params": self.backbone.parameters()},
                {
                    "name": "classifier",
                    "params": self.classifier.parameters(),
                    "lr": config["classifier_lr"],
                    "weight_decay": 0.0,
                },
                {"name": "projector", "params": self.projector.parameters()},
            ],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
            eta=config["eta"],
            clip_lr=config["clip_lr"],
            exclude_bias_n_norm=config["exclude_bias_n_norm"],
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            config["warmup_epochs"] * self.steps_per_epoch,
            config["epochs"] * self.steps_per_epoch,
            config["warmup_start_lr"],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
