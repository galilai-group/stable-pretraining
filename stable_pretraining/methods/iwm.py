"""IWM: Image World Models.

Extends I-JEPA so the predictor also learns to apply photometric
transformations in latent space: it predicts the representation of a target
view from a masked, corrupted source view, conditioned on the action mapping
one to the other.

References:
    Garrido et al. "Learning and Leveraging World Models in Visual
    Representation Learning." 2024.
    https://arxiv.org/abs/2403.00504

Example::

    from stable_pretraining.methods import IWM

    # Default predictor (18 x 384) is the paper's IWM^Equi_{18,384} configuration
    model = IWM("vit_base_patch16_224")

    # source/target views and the action between them come from the data
    # pipeline: MultiViewTransform({"source": ..., "target": ...}) followed by
    # transforms.IWMAction(). For a full training script (batch adapter,
    # TeacherStudentCallback, Manager), see benchmarks/imagenet10/iwm-vit-small.py.
    output = model(source, target, action)
    output.loss.backward()

    # Get encoder for downstream tasks
    encoder = model.encoder.student
"""

from dataclasses import dataclass
from typing import Tuple, Union

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
from transformers.utils import ModelOutput


@dataclass
class IWMOutput(ModelOutput):
    """Output from IWM forward pass.

    :ivar loss: Prediction loss (0 in eval mode)
    :ivar embedding: Patch embeddings [B, N, D] for downstream use
    :ivar predictions: Predicted target representations [B, N_tgt, D] (None in eval)
    :ivar targets: Target representations [B, N_tgt, D] (None in eval)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    predictions: torch.Tensor = None
    targets: torch.Tensor = None


class IWM(Module):
    """IWM: Image World Models.

    Architecture:
        - **Encoder** (student): encodes the visible patches of the source view
        - **Target encoder** (teacher): EMA copy, encodes the full target view
        - **Predictor** (world model): predicts the target representation at the
          masked positions, with mask tokens conditioned on the action

    The source is masked with the union of ``num_targets`` blocks; the context is
    every other patch. Mask tokens are conditioned on the action by feature
    conditioning: concatenated with it after positional embedding and mixed by a
    3-layer MLP (Appendix A.1). The loss is the mean squared error to the teacher
    representation, layer-normed as in the I-JEPA implementation.

    :param model_or_model_name: timm model name string or pre-instantiated nn.Module
    :param predictor_embed_dim: Predictor hidden dimension (default: 384)
    :param predictor_depth: Number of predictor blocks (default: 18, the paper's
        IWM^Equi_{18,384} configuration; deeper predictors more reliably learn an
        equivariant world model)
    :param action_dim: Size of the action vector (default: 7, as produced by
        :class:`~stable_pretraining.data.transforms.IWMAction`)
    :param num_targets: Number of masked blocks (default: 4)
    :param target_scale: (min, max) fraction of patches per masked block
    :param target_aspect_ratio: (min, max) aspect ratio of masked blocks
    :param ema_decay_start: Initial EMA decay (default: 0.996)
    :param ema_decay_end: Final EMA decay (default: 1.0)
    :param pretrained: Load pretrained encoder weights

    Example::

        model = IWM("vit_base_patch16_224")
        source = torch.randn(4, 3, 224, 224)
        target = torch.randn(4, 3, 224, 224)
        action = torch.randn(4, 7)

        # Training mode: predicts the target at the masked source positions
        model.train()
        output = model(source, target, action)
        output.loss.backward()

        # Eval mode: encodes all patches (no masking)
        model.eval()
        features = model(source).embedding  # [B, N, D]

    Note:
        - Use :class:`TeacherStudentCallback` to handle EMA updates automatically
        - Access trained encoder via ``model.encoder.student``
    """

    def __init__(
        self,
        model_or_model_name: Union[str, nn.Module] = "vit_base_patch16_224",
        predictor_embed_dim: int = 384,
        predictor_depth: int = 18,
        action_dim: int = 7,
        num_targets: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        pretrained: bool = False,
    ):
        super().__init__()

        # Encoder with EMA wrapper (enables TeacherStudentCallback)
        base_encoder = MaskedEncoder(
            model_or_model_name,
            masking=None,
            pretrained=pretrained,
        )
        self.encoder = TeacherStudentWrapper(
            base_encoder,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        embed_dim = base_encoder.embed_dim
        num_patches = base_encoder.default_grid_h * base_encoder.default_grid_w

        # Feature conditioning (Appendix A.1): [mask token + pos, action] -> mask token
        action_mlp = nn.Sequential(
            nn.Linear(predictor_embed_dim + action_dim, predictor_embed_dim),
            nn.ReLU(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.ReLU(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
        )

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
            query_cond_net=action_mlp,
        )

        # Mask = union of target blocks (may overlap); context = all other patches (A.1)
        self.masking = IJEPAMasking(
            num_targets=num_targets,
            target_scale=target_scale,
            target_aspect_ratio=target_aspect_ratio,
            context_scale=(1.0, 1.0),
            allow_target_overlap=True,
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
        self,
        source: torch.Tensor,
        target: torch.Tensor = None,
        action: torch.Tensor = None,
        embedding_source: str = "teacher",
    ) -> IWMOutput:
        """Forward pass.

        In training mode:
            - Masks the source via :class:`IJEPAMasking` and encodes the visible
              patches through the student
            - Encodes the full target through the teacher (EMA)
            - Predicts the target representation at the masked positions,
              conditioned on the action
            - Returns the mean squared error between predictions and targets

        In eval mode:
            - No masking, encodes all source patches through the student
            - Returns encoded features with zero loss

        :param source: Source views [B, C, H, W] (or images to embed in eval)
        :param target: Target views [B, C, H, W] (training only)
        :param action: Action mapping source to target [B, action_dim] (training only)
        :param embedding_source: Which encoder embeds the target view for the
            embedding output: ``"teacher"`` (default) or ``"student"``. Only
            affects training mode; eval mode always uses the student.
        :return: :class:`IWMOutput` with loss and representations
        """
        if embedding_source not in ("teacher", "student"):
            raise ValueError(
                f"embedding_source must be 'teacher' or 'student', got '{embedding_source}'"
            )

        B = source.shape[0]
        grid_h, grid_w = self.encoder.student._get_grid_size(source)
        all_idx = (
            torch.arange(grid_h * grid_w, device=source.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        student_patches = self.encoder.student.patch_embed(source)

        if not self.training:
            with torch.no_grad():
                embedding = self._encode(
                    student_patches, all_idx, grid_h, grid_w, self.encoder.student
                )
            return IWMOutput(
                loss=torch.tensor(0.0, device=source.device), embedding=embedding
            )

        if target is None or action is None:
            raise ValueError("IWM training requires target and action")

        mask_out = self.masking(student_patches, grid_h, grid_w)
        context = self._encode(
            student_patches,
            mask_out.context_idx,
            grid_h,
            grid_w,
            self.encoder.student,
        )

        with torch.no_grad():
            # Teacher: full forward on the target, then select masked positions
            teacher_patches = self.encoder.teacher.patch_embed(target)
            teacher_full = self._encode(
                teacher_patches, all_idx, grid_h, grid_w, self.encoder.teacher
            )
            teacher_full_normed = F.layer_norm(
                teacher_full, [teacher_full.size(-1)]
            )  # extra norm on targets, as in I-JEPA
            D = teacher_full.size(-1)
            targets = torch.gather(
                teacher_full_normed,
                1,
                mask_out.target_idx.unsqueeze(-1).expand(-1, -1, D),
            )

            # Embedding: the target view, which has no destructive augmentations
            if embedding_source == "teacher":
                embedding = teacher_full
            else:
                embedding = self._encode(
                    self.encoder.student.patch_embed(target),
                    all_idx,
                    grid_h,
                    grid_w,
                    self.encoder.student,
                )

        N_tgt = mask_out.target_idx.shape[1]
        queries = torch.zeros(
            B, N_tgt, self.embed_dim, device=source.device, dtype=context.dtype
        )
        query_mask = torch.ones(B, N_tgt, device=source.device, dtype=torch.bool)
        predictions = self.predictor(
            context=context,
            queries=queries,
            context_idx=mask_out.context_idx,
            query_idx=mask_out.target_idx,
            query_mask=query_mask,
            query_cond=action,
        )

        # Squared L2 of the paper, averaged over masked patches as in I-JEPA
        loss = F.mse_loss(predictions, targets)

        return IWMOutput(
            loss=loss,
            embedding=embedding,
            predictions=predictions,
            targets=targets,
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
