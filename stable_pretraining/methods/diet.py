"""DIET: Datum Index as Target."""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.backbone import from_timm
from stable_pretraining.module import Module


class DIET(Module):
    """Learn image representations by predicting each sample's stable index.

    Args:
        encoder_name: A timm model name or a feature-extracting ``nn.Module``.
        num_samples: Number of examples in the training dataset; DIET creates
            one classifier output for each training example.
        label_smoothing: Cross-entropy label smoothing used for the index task.
        low_resolution: Adapt convolutional backbones for small images.
        pretrained: Load pretrained weights when ``encoder_name`` is a string.
        image_size: Square input size used to infer an injected encoder's output
            dimension.

    The training batch must provide stable ``sample_idx`` values. When called
    with a project batch dict, ``image`` and ``sample_idx`` are consumed and
    ``label`` is returned only for downstream evaluation callbacks.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "resnet18",
        num_samples: int = 50_000,
        label_smoothing: float = 0.8,
        low_resolution: bool = False,
        pretrained: bool = False,
        image_size: int = 224,
        optim: dict = None,
    ):
        super().__init__(**({"optim": optim} if optim is not None else {}))
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")

        if isinstance(encoder_name, str):
            self.backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            self.backbone = encoder_name

        self.num_samples = num_samples
        self.label_smoothing = label_smoothing
        self.embed_dim = self._infer_embed_dim(image_size)
        self.index_classifier = nn.Linear(self.embed_dim, num_samples, bias=False)

    def _infer_embed_dim(self, image_size: int) -> int:
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, image_size, image_size)
                embedding = self.backbone(dummy)
        finally:
            self.backbone.train(was_training)
        if embedding.ndim != 2:
            raise ValueError(
                "DIET backbone must return [batch, features], "
                f"got shape {tuple(embedding.shape)}"
            )
        return embedding.shape[-1]

    def forward(self, *args, **kwargs) -> dict:
        """Return the DIET loss during training and backbone features otherwise."""
        stage = kwargs.pop("stage", None)
        sample_idx = kwargs.pop("sample_idx", None)
        keyword_images = kwargs.pop("images", None)
        if kwargs:
            raise TypeError(f"Unexpected forward arguments: {', '.join(kwargs)}")
        if len(args) > 2 or (args and keyword_images is not None):
            raise TypeError("DIET.forward accepts images and optional sample_idx")
        images = args[0] if args else keyword_images
        if images is None:
            raise TypeError("DIET.forward requires images or a batch dict")
        if len(args) == 2:
            if sample_idx is not None:
                raise TypeError("sample_idx was provided more than once")
            sample_idx = args[1]

        labels = None
        if isinstance(images, dict):
            batch = images
            images = batch["image"]
            if sample_idx is None:
                sample_idx = batch.get("sample_idx")
            labels = batch.get("label")

        embedding = self.backbone(images)
        should_train = self.training and stage not in {"validate", "test", "predict"}
        logits = None
        if should_train:
            if sample_idx is None:
                raise ValueError("DIET training requires stable sample_idx targets")
            if sample_idx.ndim != 1 or sample_idx.shape[0] != embedding.shape[0]:
                raise ValueError("sample_idx must have shape [batch_size]")
            targets = sample_idx.to(device=embedding.device, dtype=torch.long)
            if torch.any((targets < 0) | (targets >= self.num_samples)):
                raise ValueError("sample_idx values must be in [0, num_samples)")
            logits = self.index_classifier(embedding)
            loss = F.cross_entropy(
                logits, targets, label_smoothing=self.label_smoothing
            )
            if stage == "fit" and getattr(self, "_trainer", None) is not None:
                self.log(
                    "train/diet_loss",
                    loss.detach(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=True,
                )
        else:
            loss = embedding.new_zeros(())

        return {
            "loss": loss,
            "embedding": embedding,
            "logits": logits,
            "label": labels,
        }
