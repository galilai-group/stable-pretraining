"""V-JEPA: Video Joint-Embedding Predictive Architecture.

Self-supervised learning on video via predicting target spatio-temporal patch
representations from context patches using a lightweight predictor.  Masking
uses *tube* masking: the same spatial block is masked across **all** temporal
frames, so the predictor must recover coherent motion and appearance across
time from surrounding context.

References:
    Bardes et al. "V-JEPA: Latent Video Prediction for Visual Representation
    Learning." ICLR 2024. https://arxiv.org/abs/2404.08471

Example::

    from stable_pretraining.methods.vjepa import VJEPA
    from stable_pretraining.callbacks import TeacherStudentCallback
    import lightning as pl

    model = VJEPA(
        encoder_name="vit_base_patch16_224",
        num_frames=8,
        predictor_embed_dim=384,
        predictor_depth=6,
        num_targets=8,
    )

    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[TeacherStudentCallback()],
    )
    trainer.fit(model, video_dataloader)

    # Access trained encoder for downstream tasks
    encoder = model.encoder.student
"""

from dataclasses import dataclass
from typing import Tuple

import math
import torch
import torch.nn.functional as F

from stable_pretraining.backbone import (
    FlexibleTransformer,
    MaskedEncoder,
    TeacherStudentWrapper,
    VJEPAMasking,
)
from stable_pretraining.backbone.pos_embed import get_1d_sincos_pos_embed
from stable_pretraining import Module


@dataclass
class VJEPAOutput:
    """Output from VJEPA forward pass.

    :ivar loss: Prediction loss (0 in eval mode)
    :ivar embedding: Mean-pooled spatio-temporal embeddings [B, D] for downstream use
    :ivar predictions: Predicted target representations [B, N_tgt, D]
    :ivar targets: Target representations from teacher [B, N_tgt, D]
    :ivar num_targets: Number of target patches (0 in eval mode)
    :ivar num_context: Number of context patches
    """

    loss: torch.Tensor
    embedding: torch.Tensor
    predictions: torch.Tensor
    targets: torch.Tensor
    num_targets: int
    num_context: int


class VJEPA(Module):
    """V-JEPA: Video Joint-Embedding Predictive Architecture.

    Architecture:
        - **Context Encoder** (student): Encodes visible/context spatio-temporal patches
        - **Target Encoder** (teacher): EMA copy, encodes all patches
        - **Predictor**: Lightweight transformer that predicts target tube representations
          from context

    The encoder processes video clips of shape ``(B, C, T, H, W)``.  Each frame
    is tokenised by the ViT patch embedding into ``N_spatial = (H/p) × (W/p)``
    tokens; the full clip thus yields ``T × N_spatial`` tokens.  Spatial and
    temporal sinusoidal position embeddings are summed and added to every token
    before the transformer blocks run over the full spatiotemporal sequence.

    Masking uses :class:`~stable_pretraining.backbone.VJEPAMasking`: spatial
    blocks are sampled and replicated across all frames to form *tubes*.  The
    context encoder only sees non-tube tokens; the teacher encodes the full
    sequence (all tokens visible) and the target is the tube subset.

    The context encoder is wrapped with :class:`TeacherStudentWrapper`, enabling
    automatic EMA updates via :class:`TeacherStudentCallback`.

    :param encoder_name: timm model name (e.g., ``"vit_base_patch16_224"``)
    :param num_frames: Number of video frames per clip (default: 8)
    :param predictor_embed_dim: Predictor hidden dimension (default: 384)
    :param predictor_depth: Number of predictor transformer blocks (default: 6)
    :param num_targets: Number of target tubes to sample (default: 8)
    :param target_scale: (min, max) fraction of *spatial* patches per tube block
    :param target_aspect_ratio: (min, max) aspect ratio of spatial blocks
    :param context_scale: (min, max) fraction of non-target tokens kept as context
    :param ema_decay_start: Initial EMA decay (default: 0.996)
    :param ema_decay_end: Final EMA decay (default: 1.0)
    :param pretrained: Load pretrained encoder weights

    Example::

        model = VJEPA("vit_small_patch16_224", num_frames=8)

        # Training mode: tube-masked prediction
        model.train()
        videos = torch.randn(4, 3, 8, 224, 224)
        output = model(videos)
        output.loss.backward()

        # Eval mode: encode all patches, zero loss
        model.eval()
        output = model(videos)
        features = output.embedding  # [B, D]

    Example with Lightning::

        import types
        import lightning as pl
        from stable_pretraining.callbacks import TeacherStudentCallback


        def vjepa_forward(self, batch, stage):
            output = VJEPA.forward(self, batch["video"])
            self.log(
                f"{stage}/loss",
                output.loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            return {
                "loss": output.loss,
                "embedding": output.embedding.detach()
                if self.training
                else output.embedding,
                **({"label": batch["label"].long()} if "label" in batch else {}),
            }


        module = VJEPA("vit_base_patch16_224", num_frames=8)
        module.forward = types.MethodType(vjepa_forward, module)

        trainer = pl.Trainer(
            max_epochs=200,
            callbacks=[
                TeacherStudentCallback(update_frequency=1, update_after_backward=True)
            ],
        )

    Note:
        - Use :class:`TeacherStudentCallback` for EMA updates (same as I-JEPA)
        - In eval mode, ``num_targets=0`` and all tokens are returned as context
        - Access trained encoder via ``model.encoder.student``
        - ``embedding`` is the mean-pooled teacher (or student) representation
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        num_frames: int = 8,
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        num_targets: int = 8,
        target_scale: Tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        context_scale: Tuple[float, float] = (1.0, 1.0),
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        pretrained: bool = False,
    ):
        super().__init__()

        self.num_frames = num_frames

        # 2D ViT encoder shared across frames (student, wrapped with EMA teacher)
        base_encoder = MaskedEncoder(
            encoder_name,
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
        default_grid_h = base_encoder.default_grid_h
        default_grid_w = base_encoder.default_grid_w
        num_spatio_temporal = num_frames * default_grid_h * default_grid_w

        # Lightweight predictor over the full T*N_spatial sequence
        self.predictor = FlexibleTransformer(
            input_dim=embed_dim,
            hidden_dim=predictor_embed_dim,
            output_dim=embed_dim,
            num_patches=num_spatio_temporal,
            depth=predictor_depth,
            num_heads=max(1, predictor_embed_dim // 64),
            self_attn=True,
            cross_attn=False,
            add_mask_token=True,
            use_adaln=False,
            num_prefix_tokens=0,
            pos_embed_type="sincos_1d",
            zero_init_output=False,
        )

        # V-JEPA tube masking (spatial blocks replicated across all frames)
        self.masking = VJEPAMasking(
            num_targets=num_targets,
            target_scale=target_scale,
            target_aspect_ratio=target_aspect_ratio,
            context_scale=context_scale,
        )

        self.embed_dim = embed_dim
        self._fix_init_weight()

    def _encode_video(
        self,
        videos: torch.Tensor,
        indices: torch.Tensor,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        encoder: MaskedEncoder,
    ) -> torch.Tensor:
        """Encode selected spatio-temporal patches from a video clip.

        Applies the 2D ViT patch embedding frame-by-frame, adds spatial and
        temporal sinusoidal position embeddings, then selects the desired
        patches by flat index and runs them through the transformer blocks.

        :param videos: Video tensor [B, C, T, H, W]
        :param indices: Flat spatio-temporal patch indices [B, K],
            where index ``t * grid_h * grid_w + h * grid_w + w`` refers to
            frame ``t``, row ``h``, column ``w``.
        :param grid_t: Number of frames (T)
        :param grid_h: Spatial grid height
        :param grid_w: Spatial grid width
        :param encoder: MaskedEncoder (student or teacher)
        :return: Encoded representations [B, K, D]
        """
        B, C, T, H, W = videos.shape
        N_spatial = grid_h * grid_w
        D = encoder.embed_dim
        device = videos.device
        dtype = next(encoder.parameters()).dtype

        # 1. Patch-embed all frames: (B*T, C, H, W) -> (B, T, N_spatial, D)
        frames = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        patches = encoder.patch_embed(frames)  # (B*T, N_spatial, D)
        patches = patches.reshape(B, T, N_spatial, D)  # (B, T, N_spatial, D)

        # 2. Spatial positional embedding (position within a frame, same for all t)
        #    _get_pos_embed returns (prefix_pos, patch_pos); we only need patch_pos.
        _, spatial_pos = encoder._get_pos_embed(grid_h, grid_w)  # (1, N_spatial, D)

        # 3. Temporal positional embedding (which frame, broadcast over spatial)
        temp_pos_np = get_1d_sincos_pos_embed(D, grid_t)  # (T, D) tensor
        temp_pos = temp_pos_np.to(device=device, dtype=dtype)  # (T, D)

        # 4. Add both PEs to patch tokens
        #    spatial_pos: (1, N_spatial, D) -> unsqueeze(1) -> (1, 1, N_spatial, D)
        #    temp_pos:    (T, D)            -> view         -> (1, T, 1, D)
        patches = (
            patches
            + spatial_pos.unsqueeze(1)  # broadcast (B, T, N_spatial, D)
            + temp_pos.view(1, T, 1, D)  # broadcast (B, T, N_spatial, D)
        )

        # 5. Flatten to (B, T*N_spatial, D) and select patches by index
        x = patches.reshape(B, T * N_spatial, D)
        x = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, D))

        # 6. Run through ViT trunk (dropout -> blocks -> norm)
        x = encoder.vit.pos_drop(x)
        x = encoder.vit.blocks(x)
        return encoder.vit.norm(x)

    def forward(
        self, videos: torch.Tensor, embedding_source: str = "teacher"
    ) -> VJEPAOutput:
        """Forward pass.

        In **training** mode:
            - Samples target tubes and context via :class:`VJEPAMasking`
            - Student encoder sees only context tokens
            - Teacher encoder (EMA, no grad) encodes the full sequence; tube
              tokens are selected as targets
            - Predictor attends over ``[context + masked query tokens]`` and
              outputs predictions at target positions
            - Smooth L1 loss between predictions and (layer-normalised) targets

        In **eval** mode:
            - No masking; all tokens treated as context
            - Returns zero loss and mean-pooled student embeddings

        :param videos: Input video clips [B, C, T, H, W].
            T must equal ``self.num_frames`` (or the model's default grid).
        :param embedding_source: Which encoder to use for the ``embedding``
            field: ``"teacher"`` (default) or ``"student"``.  Eval mode always
            uses student.
        :return: :class:`VJEPAOutput`
        """
        if embedding_source not in ("teacher", "student"):
            raise ValueError(
                f"embedding_source must be 'teacher' or 'student', "
                f"got '{embedding_source}'"
            )

        B, C, T, H, W = videos.shape
        grid_h, grid_w = self.encoder.student._get_grid_size(
            videos[:, :, 0]  # single-frame spatial grid
        )
        grid_t = T
        N_spatial = grid_h * grid_w
        N_total = grid_t * N_spatial

        # Compute patch embeddings via the student's patch_embed for masking
        # (Masking only needs the shape; we pass a dummy tensor of correct size)
        device = videos.device
        dtype = next(self.encoder.student.parameters()).dtype
        dummy = torch.empty(B, N_total, self.embed_dim, device=device, dtype=dtype)
        mask_out = self.masking(dummy, grid_t, grid_h, grid_w)

        if self.training:
            # --- Context: student encodes only non-target tokens ---
            context = self._encode_video(
                videos,
                mask_out.context_idx,
                grid_t,
                grid_h,
                grid_w,
                self.encoder.student,
            )

            with torch.no_grad():
                # --- Teacher: encode full sequence, then select target tokens ---
                all_idx = (
                    torch.arange(N_total, device=device).unsqueeze(0).expand(B, -1)
                )
                teacher_full = self._encode_video(
                    videos,
                    all_idx,
                    grid_t,
                    grid_h,
                    grid_w,
                    self.encoder.teacher,
                )  # [B, T*N, D]

                # Extra LayerNorm on targets (affine-free, as in I-JEPA)
                teacher_normed = F.layer_norm(teacher_full, [teacher_full.size(-1)])

                # Gather tube target tokens
                D = teacher_full.size(-1)
                targets = torch.gather(
                    teacher_normed,
                    1,
                    mask_out.target_idx.unsqueeze(-1).expand(-1, -1, D),
                )  # [B, N_tgt, D]

                # Embedding for downstream probes
                if embedding_source == "teacher":
                    embedding = teacher_full.mean(dim=1)  # [B, D]
                else:
                    embedding = self._encode_video(
                        videos,
                        all_idx,
                        grid_t,
                        grid_h,
                        grid_w,
                        self.encoder.student,
                    ).mean(dim=1)

            # --- Predictor: joint attention over [context + masked queries] ---
            N_tgt = mask_out.target_idx.shape[1]
            queries = torch.zeros(
                B, N_tgt, self.embed_dim, device=device, dtype=context.dtype
            )
            query_mask = torch.ones(B, N_tgt, device=device, dtype=torch.bool)
            predictions = self.predictor(
                context=context,
                queries=queries,
                context_idx=mask_out.context_idx,
                query_idx=mask_out.target_idx,
                query_mask=query_mask,
            )  # [B, N_tgt, D]

            loss = F.smooth_l1_loss(predictions, targets, beta=1.0)

        else:
            # Eval: encode all tokens through student (no masking)
            with torch.no_grad():
                all_idx = (
                    torch.arange(N_total, device=device).unsqueeze(0).expand(B, -1)
                )
                context = self._encode_video(
                    videos,
                    all_idx,
                    grid_t,
                    grid_h,
                    grid_w,
                    self.encoder.student,
                )
            predictions = context
            targets = context
            embedding = context.mean(dim=1)
            loss = torch.tensor(0.0, device=device)

        return VJEPAOutput(
            loss=loss,
            embedding=embedding,
            predictions=predictions,
            targets=targets,
            num_targets=mask_out.target_idx.shape[1],
            num_context=mask_out.context_idx.shape[1],
        )

    def _fix_init_weight(self):
        """Rescale attention-proj and MLP output weights by depth (I-JEPA init)."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for enc in (self.encoder.student, self.encoder.teacher):
            for layer_id, block in enumerate(enc.vit.blocks):
                rescale(block.attn.proj.weight.data, layer_id + 1)
                rescale(block.mlp.fc2.weight.data, layer_id + 1)

        for layer_id, block in enumerate(self.predictor.blocks):
            rescale(block.attn.proj.weight.data, layer_id + 1)
            rescale(block.mlp.fc2.weight.data, layer_id + 1)
