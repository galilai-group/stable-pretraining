"""I-JEPA: Image-based Joint-Embedding Predictive Architecture.

Self-supervised learning via predicting target patch representations
from context patch representations using a lightweight predictor.

References:
    Assran et al. "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture." CVPR 2023.
    https://arxiv.org/abs/2301.08243

Example::

    from stable_pretraining.backbone import IJEPA
    from stable_pretraining.callbacks import TeacherStudentCallback
    import lightning as pl

    # Create model
    model = IJEPA(
        encoder_name="vit_base_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=6,
        num_targets=4,
    )

    # Training with PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=300,
        callbacks=[TeacherStudentCallback()],  # Handles EMA updates
    )
    trainer.fit(model, dataloader)

    # Get encoder for downstream tasks
    encoder = model.encoder.student
"""

from dataclasses import dataclass
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.backbone import (
    FlexibleTransformer,
    IJEPAMasking,
    MaskedEncoder,
    TeacherStudentWrapper,
)
from stable_pretraining import Module


@dataclass
class IJEPAOutput:
    """Output from IJEPA forward pass.

    :ivar loss: Prediction loss (0 in eval mode)
    :ivar embedding: Patch embeddings [B, N, D] for downstream use
    :ivar predictions: Predicted representations [B, N_tgt, D] (or context in eval)
    :ivar targets: Target representations [B, N_tgt, D] (or context in eval)
    :ivar num_targets: Number of target patches (0 in eval)
    :ivar num_context: Number of context patches (all patches in eval)
    """

    loss: torch.Tensor
    embedding: torch.Tensor
    predictions: torch.Tensor
    targets: torch.Tensor
    num_targets: int
    num_context: int


class IJEPA(Module):
    """I-JEPA: Image-based Joint-Embedding Predictive Architecture.

    Architecture:
        - **Context Encoder** (student): Encodes visible/context patches
        - **Target Encoder** (teacher): EMA copy, encodes target patches
        - **Predictor**: Lightweight transformer predicting targets from context

    The context encoder is wrapped with :class:`TeacherStudentWrapper`, enabling
    automatic EMA updates via :class:`TeacherStudentCallback`.

    :param encoder_name: timm model name (e.g., "vit_base_patch16_224")
    :param predictor_embed_dim: Predictor hidden dimension (default: 384)
    :param predictor_depth: Number of predictor blocks (default: 6)
    :param num_targets: Number of target blocks to sample (default: 4)
    :param target_scale: (min, max) fraction of patches per target block
    :param target_aspect_ratio: (min, max) aspect ratio of target blocks
    :param context_scale: (min, max) fraction of non-target patches as context
    :param ema_decay_start: Initial EMA decay (default: 0.996)
    :param ema_decay_end: Final EMA decay (default: 1.0)
    :param pretrained: Load pretrained encoder weights

    Example::

        # Basic usage
        model = IJEPA("vit_base_patch16_224")
        images = torch.randn(4, 3, 224, 224)

        # Training mode: predicts masked targets
        model.train()
        output = model(images)
        output.loss.backward()

        # Eval mode: encodes all patches (no masking)
        model.eval()
        output = model(images)
        features = output.predictions  # [B, N, D]

    Example with Lightning::

        import lightning as pl
        from stable_pretraining.callbacks import TeacherStudentCallback


        class IJEPALightning(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = IJEPA("vit_base_patch16_224")

            def training_step(self, batch, batch_idx):
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                output = self.model(images)
                self.log("loss", output.loss)
                return output.loss

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1.5e-4)


        trainer = pl.Trainer(callbacks=[TeacherStudentCallback()])
        trainer.fit(IJEPALightning(), dataloader)

    Note:
        - Use :class:`TeacherStudentCallback` to handle EMA updates automatically
        - In eval mode, ``num_targets=0`` and all patches are returned as context
        - Access trained encoder via ``model.encoder.student``
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        num_targets: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        context_scale: Tuple[float, float] = (0.85, 1.0),
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        pretrained: bool = False,
    ):
        super().__init__()

        # Encoder with EMA wrapper (enables TeacherStudentCallback)
        base_encoder = MaskedEncoder(
            encoder_name,
            masking=None,
            pretrained=pretrained,
            norm_layer=nn.LayerNorm,  # important that elementwise_affine=True (default param)
        )
        self.encoder = TeacherStudentWrapper(
            base_encoder,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        embed_dim = base_encoder.embed_dim
        num_patches = base_encoder.default_grid_h * base_encoder.default_grid_w

        # Lightweight predictor: cross-attention from target queries to context
        self.predictor = FlexibleTransformer(
            input_dim=embed_dim,
            hidden_dim=predictor_embed_dim,
            output_dim=embed_dim,
            num_patches=num_patches,
            depth=predictor_depth,
            num_heads=max(1, predictor_embed_dim // 64),
            self_attn=True,
            cross_attn=False,
            add_mask_token=True,
            use_adaln=False,
            num_prefix_tokens=0,
            zero_init_output=False,
        )

        # I-JEPA multi-block masking
        self.masking = IJEPAMasking(
            num_targets=num_targets,
            target_scale=target_scale,
            target_aspect_ratio=target_aspect_ratio,
            context_scale=context_scale,
        )

        self.embed_dim = embed_dim
        self._fix_init_weight()

    def _encode(
        self,
        patches: torch.Tensor,
        indices: torch.Tensor,
        grid_h: int,
        grid_w: int,
        encoder: MaskedEncoder,
    ) -> torch.Tensor:
        """Encode patches at specified indices.

        :param patches: All patch embeddings [B, N, D]
        :param indices: Indices to encode [B, K]
        :param grid_h: Patch grid height
        :param grid_w: Patch grid width
        :param encoder: Encoder to use (student or teacher)
        :return: Encoded representations [B, K, D]
        """
        B, _, D = patches.shape
        # patch -> posemb -> mask -> block -> norm
        _, pos = encoder._get_pos_embed(grid_h, grid_w)
        x = patches + pos.expand(B, -1, -1)
        x = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        x = encoder.vit.pos_drop(x)
        x = encoder.vit.blocks(x)
        return encoder.vit.norm(x)

    def forward(
        self, images: torch.Tensor, embedding_source: str = "teacher"
    ) -> IJEPAOutput:
        """Forward pass.

        In training mode:
            - Samples target blocks and context region via :class:`IJEPAMasking`
            - Encodes context through student, targets through teacher (EMA)
            - Predicts target representations from context
            - Returns smooth L1 loss between predictions and targets

        In eval mode:
            - No masking, all patches treated as context
            - Returns encoded features with zero loss
            - Always uses student encoder

        :param images: Input images [B, C, H, W]
        :param embedding_source: Which encoder to use for the embedding output.
            ``"teacher"`` (default) or ``"student"``. Only affects training mode;
            eval mode always uses student.
        :return: :class:`IJEPAOutput` with loss and representations
        """
        if embedding_source not in ("teacher", "student"):
            raise ValueError(
                f"embedding_source must be 'teacher' or 'student', got '{embedding_source}'"
            )

        B = images.shape[0]
        grid_h, grid_w = self.encoder.student._get_grid_size(images)
        student_patches = self.encoder.student.patch_embed(images)
        teacher_patches = self.encoder.teacher.patch_embed(images)

        # Apply masking (returns all patches as context in eval mode)
        mask_out = self.masking(student_patches, grid_h, grid_w)

        if self.training:
            # Context: student sees only context patches
            context = self._encode(
                student_patches,
                mask_out.context_idx,
                grid_h,
                grid_w,
                self.encoder.student,
            )

            with torch.no_grad():
                # Teacher: full forward with ALL patches visible, then select targets
                all_idx = (
                    torch.arange(grid_h * grid_w, device=images.device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )

                teacher_full = self._encode(
                    teacher_patches, all_idx, grid_h, grid_w, self.encoder.teacher
                )  # teacher's vit.norm already applied

                teacher_full_normed = F.layer_norm(
                    teacher_full,
                    [teacher_full.size(-1)],
                    weight=None,
                    bias=None,  # extra norm but affine as per paper
                )
                # Select target patches from the full encoding
                D = teacher_full.size(-1)
                targets = torch.gather(
                    teacher_full_normed,
                    1,
                    mask_out.target_idx.unsqueeze(-1).expand(-1, -1, D),
                )

                # Embedding: reuse teacher_full (unnormed, full sequence) for the probe
                if embedding_source == "teacher":
                    embedding = teacher_full
                else:
                    embedding = self._encode(
                        student_patches, all_idx, grid_h, grid_w, self.encoder.student
                    )

            # Predict target representations via joint self-attention on [context + mask tokens]
            N_tgt = mask_out.target_idx.shape[1]
            # Create dummy queries and just mask them all out
            queries = torch.zeros(
                B, N_tgt, self.embed_dim, device=images.device, dtype=context.dtype
            )
            query_mask = torch.ones(B, N_tgt, device=images.device, dtype=torch.bool)
            predictions = self.predictor(
                context=context,
                queries=queries,
                context_idx=mask_out.context_idx,
                query_idx=mask_out.target_idx,
                query_mask=query_mask,
            )

            loss = F.smooth_l1_loss(predictions, targets, beta=1.0)
        else:
            # Eval: encode all patches through student
            with torch.no_grad():
                context = self._encode(
                    student_patches,
                    mask_out.context_idx,
                    grid_h,
                    grid_w,
                    self.encoder.student,
                )
            predictions = context
            targets = context
            embedding = context
            loss = torch.tensor(0.0, device=images.device)

        return IJEPAOutput(
            loss=loss,
            embedding=embedding,
            predictions=predictions,
            targets=targets,
            num_targets=mask_out.target_idx.shape[1],
            num_context=mask_out.context_idx.shape[1],
        )

    def _fix_init_weight(self):
        """Rescale attention proj and MLP output weights by depth, matching I-JEPA init from the repo."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for encoder in (self.encoder.student, self.encoder.teacher):
            for layer_id, block in enumerate(encoder.vit.blocks):
                rescale(block.attn.proj.weight.data, layer_id + 1)
                rescale(block.mlp.fc2.weight.data, layer_id + 1)

        for layer_id, block in enumerate(self.predictor.blocks):
            rescale(block.attn.proj.weight.data, layer_id + 1)
            rescale(block.mlp.fc2.weight.data, layer_id + 1)
