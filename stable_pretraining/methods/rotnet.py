"""RotNet: unsupervised representation learning by predicting image rotations.

Each image is rotated by 0, 90, 180 and 270 degrees and a network is trained
to classify which rotation it sees. Recognising the rotation requires
recognising the object, so the features transfer to object classification.

References:
    Gidaris, Singh, Komodakis. "Unsupervised Representation Learning by
    Predicting Image Rotations." ICLR 2018. https://arxiv.org/abs/1803.07728

    Official code: https://github.com/gidariss/FeatureLearningRotNet

Example::

    from stable_pretraining.methods import RotNet

    # CIFAR-10 recipe: 4-block NIN, features taken after block 2
    model = RotNet(encoder_name="nin", num_blocks=4, feature_block=2)

    images = torch.randn(8, 3, 32, 32)
    out = model(images)
    out.loss.backward()
    features = out.embedding  # [8, 192]
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining import forward as forward_fns
from stable_pretraining.backbone import NetworkInNetwork, from_timm
from stable_pretraining.backbone.nin import NIN_CHANNELS
from stable_pretraining.forward import ROTNET_ROTATIONS, _rotnet_step


@dataclass
class RotNetOutput(ModelOutput):
    """Output from RotNet forward pass.

    Attributes:
        loss: Rotation-classification cross-entropy.
        embedding: Features of the unrotated images ``[B, D]``.
        rotation_logits: Rotation logits ``[R*B, R]``.
        rotation_accuracy: Fraction of rotations classified correctly.
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    rotation_logits: Optional[torch.Tensor] = None
    rotation_accuracy: Optional[torch.Tensor] = None


class RotNet(Module):
    """RotNet: self-supervised learning by rotation prediction.

    Architecture:
        - **Backbone**: the part of the network whose features are kept. With
          ``encoder_name="nin"`` it is the first ``feature_block`` NIN blocks,
          returning a feature map (``[B, 192, 8, 8]`` on CIFAR-10 for block 2).
        - **Rotation head**: the remaining NIN blocks, global average pooling
          and a linear layer to one logit per rotation. For any other encoder
          it is a linear layer, preceded by global average pooling when the
          encoder returns a feature map.
        - **Loss**: cross-entropy on the rotation labels.

    The paper's best CIFAR-10 features come from block 2 of a 4-block NIN,
    with the later blocks specialising in the pretext task; the
    ``backbone``/``rotation_head`` split encodes that choice.

    Calling the model on a tensor returns a :class:`RotNetOutput`. Calling it
    on a batch dict with a ``stage`` (as ``Module.training_step`` does) runs
    :func:`stable_pretraining.forward.rotnet`, so the class trains directly
    under ``Manager``.

    Args:
        encoder_name: ``"nin"`` for the paper's Network-in-Network, a timm
            model name, or a pre-instantiated ``nn.Module`` returning
            ``[B, D]`` or a feature map ``[B, C, H, W]``.
        num_blocks: NIN depth (``"nin"`` only); the paper uses 3, 4 or 5.
        feature_block: Number of NIN blocks in the backbone (``"nin"`` only);
            must be smaller than ``num_blocks``.
        rotations: Angles in degrees, each a multiple of 90. The label of a
            rotation is its index in this tuple.
        low_resolution: Adapt a timm encoder's first conv for 32x32 inputs.
        pretrained: Load pretrained timm weights.

    Example::

        model = RotNet(encoder_name="nin", num_blocks=4, feature_block=2)
        out = model(torch.randn(8, 3, 32, 32))
        out.loss.backward()
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "nin",
        num_blocks: int = 4,
        feature_block: int = 2,
        rotations: Sequence[int] = ROTNET_ROTATIONS,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()
        self.rotations = tuple(rotations)
        num_rotations = len(self.rotations)
        self.rotnet_loss = nn.CrossEntropyLoss()
        # Recorded so a checkpoint rebuilds the same architecture and rotation
        # labels (``load_from_checkpoint``); a module encoder must be passed again.
        config = {
            "num_blocks": num_blocks,
            "feature_block": feature_block,
            "rotations": list(self.rotations),
            "low_resolution": low_resolution,
            "pretrained": pretrained,
        }
        if isinstance(encoder_name, str):
            config["encoder_name"] = encoder_name
        self.save_hyperparameters({**self.hparams, **config})

        if isinstance(encoder_name, str) and encoder_name == "nin":
            if not 1 <= feature_block < num_blocks:
                raise ValueError(
                    f"feature_block must be in [1, num_blocks), got "
                    f"feature_block={feature_block}, num_blocks={num_blocks}"
                )
            nin = NetworkInNetwork(num_blocks=num_blocks, pool=False)
            self.backbone = nn.Sequential(*nin.blocks[:feature_block])
            self.rotation_head = nn.Sequential(
                *nin.blocks[feature_block:],
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(NIN_CHANNELS, num_rotations),
            )
            # Block 1 ends with 96 channels; every later block has 192.
            self.embed_dim = 96 if feature_block == 1 else NIN_CHANNELS
            return

        if isinstance(encoder_name, str):
            self.backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            self.backbone = encoder_name
        with torch.no_grad():
            was_training = self.backbone.training
            self.backbone.eval()
            out = self.backbone(torch.zeros(1, 3, 224, 224))
            self.backbone.train(was_training)
        if out.ndim == 4:
            self.embed_dim = out.shape[1]
            self.rotation_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.embed_dim, num_rotations),
            )
        else:
            self.embed_dim = out.shape[-1]
            self.rotation_head = nn.Linear(self.embed_dim, num_rotations)

    def forward(
        self,
        images: Union[torch.Tensor, dict[str, Any]],
        stage: Optional[str] = None,
    ) -> Union[RotNetOutput, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            images: Square images ``[B, C, H, W]``, or a batch dict with an
                ``"image"`` key when called by the Lightning training loop.
            stage: Training stage, used only for batch dicts.

        Returns:
            :class:`RotNetOutput` for tensors, or the dict from
            :func:`stable_pretraining.forward.rotnet` for batch dicts.
        """
        if isinstance(images, dict):
            return forward_fns.rotnet(self, images, stage)
        step = _rotnet_step(
            self.backbone, self.rotation_head, images, self.rotations, self.rotnet_loss
        )
        return RotNetOutput(**step)
