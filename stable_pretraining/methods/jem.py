"""JEM: patch-level self-distillation across views that share one patch grid.

A student ViT without class token sees masked crops of an image; an EMA
teacher sees another crop of the same image, cut on the same patch grid. On
the patches the two crops share, the student's categorical distribution of
each patch is pulled to the teacher's, at several depths of the encoder. Two
regularizers on the student prevent collapse, with no centering, Sinkhorn-Knopp
or KoLeo: an entropy term over the batch, and a term that keeps the
patch-to-patch similarities of earlier teacher layers.

References:
    Seitzer et al. "Scalable Patch-Level Self-Supervised Learning." 2026.
    https://arxiv.org/abs/2610.10013

Example::

    import stable_pretraining as spt
    from stable_pretraining.data import transforms

    views = transforms.AlignedMultiCrop(
        {"teacher": 224, "global": 224, **{f"local_{i}": 96 for i in range(4)}},
        transform=transforms.ToImage(**spt.data.static.ImageNet),
    )
    model = spt.JEM("vit_small_patch16_dinov3")
    # trainer and data (a dataset with ``transform=views``) as for any method
    spt.Manager(trainer=trainer, module=model, data=data)()
"""

from collections.abc import Sequence

import timm
import torch
from torch import nn

from ..backbone import MLP, L2Norm, TeacherStudentWrapper
from ..callbacks import TeacherStudentCallback
from ..losses import JEMLoss
from ..module import Module


class JEM(Module):
    """JEM: patch-wise alignment with global and structural regularization.

    Trains on the views of :class:`~stable_pretraining.data.transforms.AlignedMultiCrop`
    through :func:`~stable_pretraining.forward.jem`. The embedding given to
    callbacks is the mean of the teacher's final patches.

    Args:
        encoder_name: timm model name, built without class token, or an
            initialized timm ``VisionTransformer`` or ``Eva`` made with
            ``dynamic_img_size=True`` and ``class_token=False``. The default
            is the paper's encoder: DINOv3's ViT with four registers and axial
            RoPE.
        num_categories: Size K of the categorical distributions.
        target_layers: Blocks whose patches are aligned, the last block last.
            Default: a third, two thirds and the whole depth.
        structure_layers: Teacher layers whose patch similarities the final
            student patches keep; 0 is the patch embedding. Default: 0 and the
            target layers but the last.
        head_hidden_dim: Hidden width of the heads.
        head_bottleneck_dim: Width before the category logits.
        structure_dim: Output width of the structural heads.
        temperature: Softmax temperature of student and teacher.
        masked_weight: Alignment weight on patches masked for the student.
        visible_weight: Alignment weight on patches the student sees.
        mask_probability: Share of the student views that is masked.
        global_mask_ratio: Share of patches masked, in blocks, in a student
            view of the teacher's size.
        local_mask_ratio: Share of patches masked, at random, in a smaller view.
        global_weight: Weight of the global regularizer after warm-up from 1.
        structure_weight: Weight of the structural regularizer after warm-up
            from 0, and at the end of training.
        warmup_fraction: Share of the training steps the warm-ups take.
        ema_decay_start: Teacher EMA coefficient at the start of training.
        ema_decay_end: Teacher EMA coefficient at the end (cosine schedule).
        drop_path_rate: Stochastic depth of an encoder built from a name.
        pretrained: Load timm's pretrained weights into an encoder built from a
            name; the class token of the checkpoint is left out.
        optim: Optimizer configuration of ``Module``. Default: AdamW with
            linear warm-up and cosine decay.

    Note:
        Defaults follow the paper's ViT-L recipe where they do not depend on
        the model size.
        Under DDP the batch statistics of the global regularizer are gathered
        across GPUs; the other terms are per GPU (see ``JEMLoss``).
    """

    def __init__(
        self,
        encoder_name: str | nn.Module = "vit_small_patch16_dinov3",
        num_categories: int = 4096,
        target_layers: Sequence[int] | None = None,
        structure_layers: Sequence[int] | None = None,
        head_hidden_dim: int = 2048,
        head_bottleneck_dim: int = 256,
        structure_dim: int = 1024,
        temperature: float = 0.1,
        masked_weight: float = 0.8,
        visible_weight: float = 0.2,
        mask_probability: float = 0.8,
        global_mask_ratio: float = 0.65,
        local_mask_ratio: float = 0.5,
        global_weight: float = 0.1,
        structure_weight: tuple[float, float] = (0.05, 1.0),
        warmup_fraction: float = 0.025,
        ema_decay_start: float = 0.992,
        ema_decay_end: float = 1.0,
        drop_path_rate: float = 0.1,
        pretrained: bool = False,
        optim: dict | None = None,
    ):
        from ..forward import jem

        hparams = {
            name: value
            for name, value in locals().items()
            if isinstance(value, (str, int, float, tuple, list))
        }
        encoder = encoder_name
        if isinstance(encoder, str):
            encoder = timm.create_model(
                encoder,
                pretrained=pretrained,
                pretrained_strict=False,
                num_classes=0,
                class_token=False,
                global_pool="",
                dynamic_img_size=True,
                drop_path_rate=drop_path_rate,
            )
        depth = len(encoder.blocks)
        if target_layers is None:
            target_layers = (depth // 3, 2 * depth // 3, depth)
        if structure_layers is None:
            structure_layers = (0, *target_layers[:-1])
        if target_layers[-1] != depth:
            raise ValueError(
                f"target_layers must end with the last block ({depth}), got {target_layers}"
            )
        early = len(target_layers) - 1

        def trunk(out_dim: int) -> MLP:
            return MLP(
                encoder.embed_dim,
                [head_hidden_dim, head_hidden_dim, out_dim],
                activation_layer=nn.GELU,
            )

        def wrap(student: nn.Module) -> TeacherStudentWrapper:
            return TeacherStudentWrapper(
                student,
                base_ema_coefficient=ema_decay_start,
                final_ema_coefficient=ema_decay_end,
            )

        super().__init__(
            forward=jem,
            hparams=hparams,
            backbone=wrap(encoder),
            projector=wrap(
                nn.ModuleList(
                    nn.Sequential(
                        trunk(head_bottleneck_dim),
                        L2Norm(),
                        nn.Linear(head_bottleneck_dim, num_categories, bias=False),
                    )
                    for _ in target_layers
                )
            ),
            structure_projector=nn.ModuleList(
                trunk(structure_dim) for _ in structure_layers
            ),
            jem_loss=JEMLoss(temperature, masked_weight, visible_weight),
            global_mask_ratio=global_mask_ratio,
            local_mask_ratio=local_mask_ratio,
            mask_probability=mask_probability,
            target_layers=tuple(target_layers),
            layer_weights=(*(0.5 / early for _ in range(early)), 1.0),
            structure_layers=tuple(structure_layers),
            global_weight=global_weight,
            structure_weight=tuple(structure_weight),
            warmup_fraction=warmup_fraction,
            optim=optim
            or {
                "optimizer": {
                    "type": "AdamW",
                    "lr": 5e-4,
                    "weight_decay": 0.04,
                    "betas": (0.9, 0.999),
                },
                "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            },
        )
        self.embed_dim = encoder.embed_dim
        self.mask_token = nn.Parameter(torch.zeros(encoder.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def configure_callbacks(self) -> list:
        """The callback that updates the EMA teacher after each optimizer step."""
        return [TeacherStudentCallback()]
