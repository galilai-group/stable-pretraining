"""Native Flax-NNX backbones for the JAX backend.

Day-one strategy is native Flax (no torch<->jax bridge). All backbones use the
NHWC (channels-last) layout that XLA prefers and drive BatchNorm/Dropout via the
module's ``train()``/``eval()`` flags.
"""

from .mlp import MLP
from .resnet import (
    BasicBlock,
    Bottleneck,
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from .small import ConvMixer, ResNet9
from .teacher_student import EMAParam, TeacherStudentWrapper
from .vit import ViT, vit_base, vit_large, vit_small, vit_tiny

__all__ = [
    "MLP",
    "ResNet",
    "BasicBlock",
    "Bottleneck",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "ResNet9",
    "ConvMixer",
    "ViT",
    "vit_small",
    "vit_tiny",
    "vit_base",
    "vit_large",
    "TeacherStudentWrapper",
    "EMAParam",
]
