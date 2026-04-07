"""SALT: Static-teacher Asymmetric Latent Training.

SALT combines ideas from V-JEPA masking with MAE pixel reconstruction
(Stage 1) and latent target prediction with a frozen teacher (Stage 2).

References:
    Li, Xianhang, et al. "Rethinking JEPA: Compute-Efficient Video SSL
    with Frozen Teachers." 2025.
    https://arxiv.org/pdf/2509.24317

Example:
    from stable_pretraining.methods import SALT, MAE
    from stable_pretraining.backbone import MultiBlockMasking

    # Stage 1: MAE with multi-block masking
    stage1 = MAE("vit_tiny_patch16_224", masking=MultiBlockMasking())

    # Stage 2: SALT from Stage 1 checkpoint
    stage2 = SALT.from_checkpoint(
        "stage1.ckpt",
        encoder_name="vit_tiny_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=12,
    )

"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from stable_pretraining.backbone import (
    EvalOnly,
    FlexibleTransformer,
    MaskedEncoder,
)
from stable_pretraining.data.masking import multi_block_mask
from stable_pretraining import Module
from transformers.utils import ModelOutput


@dataclass
class SALTOutput(ModelOutput):
    """Output from SALT forward pass.

    :ivar loss: Prediction loss (L1 between predicted and teacher latents, 0 in eval)
    :ivar embedding: CLS token embedding [B, D]
    :ivar predictions: Predicted representations [B, N_tgt, D] (or None in eval)
    :ivar targets: Teacher target representations [B, N_tgt, D] (or None in eval)
    :ivar num_targets: Number of target patches (0 in eval)
    :ivar num_context: Number of context patches (all patches in eval)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    predictions: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    num_targets: int = None
    num_context: int = None


class SALT(Module):
    """SALT Stage 2: Static-teacher Asymmetric Latent Training.

    Architecture:
        - **Teacher** (frozen): Encodes full unmasked image via EvalOnly(MaskedEncoder)
        - **Student** (trainable): Encodes only context (visible) patches
        - **Predictor**: Lightweight transformer predicting teacher latents at target positions

    :param encoder_name: timm model name (e.g., "vit_tiny_patch16_224")
    :param predictor_embed_dim: Predictor hidden dimension (default: 384)
    :param predictor_depth: Number of predictor blocks (default: 12)
    :param predictor_num_heads: Number of predictor attention heads (default: 16)
    :param num_targets: Number of target blocks for masking (default: 4)
    :param context_scale: (min, max) scale for context block
    :param target_scale: (min, max) scale for each target block
    :param context_aspect_ratio: (min, max) aspect ratio for context block
    :param target_aspect_ratio: (min, max) aspect ratio for target blocks
    :param teacher_state_dict: Optional state dict to load into teacher encoder
    :param pretrained: Load pretrained encoder weights

    Example::

        model = SALT("vit_tiny_patch16_224")
        images = torch.randn(4, 3, 224, 224)

        model.train()
        output = model(images)
        output.loss.backward()

        model.eval()
        output = model(images)
        features = output.embedding  # [B, D]
    """

    def __init__(
        self,
        encoder_name: str = "vit_tiny_patch16_224",
        predictor_embed_dim: int = 384,
        predictor_depth: int = 12,
        predictor_num_heads: int = 16,
        num_targets: int = 4,
        context_scale: Tuple[float, float] = (0.85, 1.0),
        target_scale: Tuple[float, float] = (0.15, 0.2),
        context_aspect_ratio: Tuple[float, float] = (1.0, 1.0),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        teacher_state_dict: dict = None,
        pretrained: bool = False,
    ):
        super().__init__()

        # Frozen teacher
        teacher_encoder = MaskedEncoder(
            encoder_name, masking=None, pretrained=pretrained
        )
        if teacher_state_dict is not None:
            teacher_encoder.load_state_dict(teacher_state_dict)
        self.teacher = EvalOnly(teacher_encoder)

        # Trainable student (no masking — we handle it manually)
        self.student = MaskedEncoder(encoder_name, masking=None, pretrained=pretrained)

        embed_dim = self.student.embed_dim
        num_patches = self.student.default_grid_h * self.student.default_grid_w

        # Predictor with mask token for target queries
        self.predictor = FlexibleTransformer(
            input_dim=embed_dim,
            hidden_dim=predictor_embed_dim,
            output_dim=embed_dim,
            num_patches=num_patches,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            self_attn=True,
            cross_attn=False,
            add_mask_token=True,
            use_adaln=False,
            num_prefix_tokens=0,
            zero_init_output=False,
        )

        # Masking parameters
        self.num_targets = num_targets
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.target_aspect_ratio = target_aspect_ratio

        self.embed_dim = embed_dim

    def _generate_masks(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate multi-block masks for context and targets.

        :param grid_h: Patch grid height
        :param grid_w: Patch grid width
        :param device: Target device
        :return: (context_idx [N_ctx], target_idx [N_tgt]) — 1D index tensors
        """
        block_scales = [self.context_scale] + [self.target_scale] * self.num_targets
        aspect_ratios = [self.context_aspect_ratio] + [
            self.target_aspect_ratio
        ] * self.num_targets

        masks = multi_block_mask(
            grid_h,
            grid_w,
            block_scales=block_scales,
            aspect_ratios=aspect_ratios,
        )

        context_mask = masks[0]  # [H, W], 1=in block
        target_masks = masks[1:]

        # Make context disjoint from targets
        for t in target_masks:
            context_mask = context_mask * (1 - t)

        # Flatten and compute indices
        context_flat = context_mask.flatten().bool()
        target_flat = torch.zeros(grid_h * grid_w, dtype=torch.bool)
        for t in target_masks:
            target_flat = target_flat | t.flatten().bool()

        context_idx = context_flat.nonzero(as_tuple=True)[0].to(device)
        target_idx = target_flat.nonzero(as_tuple=True)[0].to(device)

        return context_idx, target_idx

    def _encode(
        self,
        patches: torch.Tensor,
        indices: torch.Tensor,
        grid_h: int,
        grid_w: int,
        encoder: MaskedEncoder,
    ) -> torch.Tensor:
        """Encode patches at specified indices through an encoder.

        Handles positional embeddings and prefix tokens (CLS).

        :param patches: All patch embeddings [B, N, D]
        :param indices: Indices to encode [B, K] or [K] (will be expanded)
        :param grid_h: Patch grid height
        :param grid_w: Patch grid width
        :param encoder: MaskedEncoder instance
        :return: Encoded representations [B, num_prefix + K, D]
        """
        B, _, D = patches.shape

        # Expand 1D indices to batch dimension
        if indices.dim() == 1:
            indices = indices.unsqueeze(0).expand(B, -1)

        # Add positional embeddings to patches
        prefix_pos, patch_pos = encoder._get_pos_embed(grid_h, grid_w)
        x = patches + patch_pos.expand(B, -1, -1)

        # Gather visible patches
        x = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, D))

        # Prepend prefix tokens (CLS, registers)
        prefix = encoder._get_prefix_tokens(B)
        if prefix is not None:
            if prefix_pos is not None and not encoder.no_embed_class:
                prefix = prefix + prefix_pos
            x = torch.cat([prefix, x], dim=1)

        x = encoder.vit.pos_drop(x)
        x = encoder.vit.blocks(x)
        x = encoder.vit.norm(x)
        return x

    def forward(self, images: torch.Tensor) -> SALTOutput:
        """Forward pass.

        Training: teacher encodes full image, student encodes context only,
        predictor predicts teacher latents at target positions, L1 loss.

        Eval: student encodes full image, returns CLS token embedding, zero loss.

        :param images: Input images [B, C, H, W]
        :return: SALTOutput
        """
        B = images.shape[0]

        if not self.training:
            with torch.no_grad():
                student_out = self.student(images)
            embedding = student_out.encoded[:, 0, :].detach()
            return SALTOutput(
                loss=torch.tensor(0.0, device=images.device),
                embedding=embedding,
                predictions=None,
                targets=None,
                num_targets=0,
                num_context=student_out.encoded.shape[1]
                - self.student.num_prefix_tokens,
            )

        grid_h, grid_w = self.student._get_grid_size(images)
        context_idx, target_idx = self._generate_masks(grid_h, grid_w, images.device)

        N_tgt = target_idx.shape[0]

        # === Teacher forward (frozen, full image) ===
        with torch.no_grad():
            teacher_out = self.teacher(images)
            teacher_patches = teacher_out.encoded[
                :, self.teacher.num_prefix_tokens :, :
            ]
            # Gather target latents
            tgt_expand = (
                target_idx.unsqueeze(0)
                .unsqueeze(-1)
                .expand(B, -1, teacher_patches.shape[-1])
            )
            teacher_targets = torch.gather(teacher_patches, 1, tgt_expand)

        # === Student forward (context patches only) ===
        student_patches = self.student.patch_embed(images)
        encoded = self._encode(
            student_patches, context_idx, grid_h, grid_w, self.student
        )
        student_context = encoded[:, self.student.num_prefix_tokens :, :]

        # === Predictor forward ===
        # Zero queries at target positions, mask_token replaces them
        queries = torch.zeros(
            B, N_tgt, self.embed_dim, device=images.device, dtype=student_context.dtype
        )
        query_mask = torch.ones(B, N_tgt, device=images.device, dtype=torch.bool)

        ctx_idx_batch = context_idx.unsqueeze(0).expand(B, -1)
        tgt_idx_batch = target_idx.unsqueeze(0).expand(B, -1)

        predictions = self.predictor(
            context=student_context,
            queries=queries,
            context_idx=ctx_idx_batch,
            query_idx=tgt_idx_batch,
            query_mask=query_mask,
        )

        # === Loss ===
        loss = F.l1_loss(predictions, teacher_targets)

        embedding = encoded[:, 0, :].detach()

        return SALTOutput(
            loss=loss,
            embedding=embedding,
            predictions=predictions,
            targets=teacher_targets,
            num_targets=N_tgt,
            num_context=context_idx.shape[0],
        )

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        encoder_name: str = "vit_tiny_patch16_224",
        **kwargs,
    ) -> "SALT":
        """Create SALT Stage 2 from a Stage 1 (MAE/VPixel) checkpoint.

        Loads the encoder weights from Stage 1 as the frozen teacher.

        :param ckpt_path: Path to Stage 1 checkpoint
        :param encoder_name: timm model name matching Stage 1
        :param kwargs: Additional arguments for SALT.__init__
        :return: SALT instance with teacher initialized from checkpoint
        """
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Extract encoder weights (handles both "encoder." prefix from MAE
        # and direct state dict)
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state[k.removeprefix("encoder.")] = v

        if not encoder_state:
            # Try using the state dict directly (e.g., if saved without prefix)
            encoder_state = state_dict

        return cls(
            encoder_name=encoder_name,
            teacher_state_dict=encoder_state,
            **kwargs,
        )
