"""SimDINOv2 masked-patch machinery.

SimDINOv2 pairs DINOv2's CLS-token self-distillation with iBOT-style masked-patch
prediction. The patch term hides a set of patches from the student, and this
module decides *which* patches those are.

**Polarity.** ``True`` means the patch is **masked out** (hidden from the
student). This is the opposite of the ``1 = in block = visible`` convention used
by :func:`stable_pretraining.data.masking.multi_block_mask`, which serves
SALT, I-JEPA, and the MAE-style transforms. The two must never be interchanged;
the polarity here is fixed by :class:`~stable_pretraining.losses.SimDINOv2Loss`,
which expects a mask of hidden positions.

These helpers port the official ``simdinov2`` masking, whose reference
implementation lives in ``simdinov2/data/masking.py`` (``MaskingGenerator`` for
one mask, plus the collate function that assigns masks across a batch). Only the
batched form is kept here, because only the batched form has the seams the
method needs: one shared RNG pool, per-row ratios fixed up front by the
stratification ladder, and integer bitmasks rather than ``[h, w]`` tensors.

The helpers are private to the method. Other methods in this codebase keep their
own masking inline -- see the ``_random_mask`` helpers in ``ibot``, ``dinov2``,
and ``msn`` -- and none of them share an implementation, so this stays local
until a second method actually needs it.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import L2Norm, TeacherStudentWrapper
from stable_pretraining.losses import SimDINOv2Loss


class _UniformPool:
    """Replenishable stream of floats drawn from one ``torch.Generator``.

    Drawing the whole batch's randomness up front is what keeps mask generation
    off the critical path: the block-placement loop is inherently sequential per
    row, but every random decision it makes can be vectorized in a single call.
    Measured on a 14x14 grid, drawing per-decision costs ~55 ms per step for 256
    rows versus ~25 ms for this pooled form.
    """

    def __init__(self, generator: torch.Generator | None, size: int):
        self.generator = generator
        self.values = self._draw(size)

    def _draw(self, size: int) -> list[float]:
        return torch.rand(size, generator=self.generator).tolist()

    def next(self) -> float:
        if not self.values:
            # Refill unconditionally rather than imposing a draw budget. The
            # placement loop is bounded by construction -- it stops at the
            # first zero-delta attempt and otherwise raises ``count`` by at
            # least one per outer pass -- so refilling cannot spin forever, and
            # capping it here would only introduce failures upstream never has.
            self.values = self._draw(1024)
        return self.values.pop()


def _place_blocks(
    height: int,
    width: int,
    target: int,
    min_num_patches: int,
    max_num_patches: int,
    log_aspect_ratio: tuple[float, float],
    pool: _UniformPool,
) -> int:
    """Accumulate rectangular blocks into a Python-int bitmask.

    Faithful to ``simdinov2/data/masking.py``: each attempt draws a target area
    and an aspect ratio, converts them to a ``(h, w)`` block that must fit with a
    strict inequality on both axes, and is accepted only when it covers at least
    one previously-unmasked cell without exceeding the remaining budget. Ten
    attempts are made per block, and the first accepted block ends them.

    Args:
        height: Grid height in patches.
        width: Grid width in patches.
        target: Desired number of masked cells; placement stops early on ``0``.
        min_num_patches: Lower bound of a block's target area.
        max_num_patches: Upper bound of a block's target area.
        log_aspect_ratio: Natural-log aspect ratio bounds, sampled uniformly.
        pool: Source of uniform ``[0, 1)`` floats.

    Returns:
        int: Bitmask over ``height * width`` cells, bit ``i * width + j`` set when
        cell ``(i, j)`` is masked.
    """
    lo, hi = log_aspect_ratio
    bits = 0
    count = 0

    while count < target:
        budget = min(target - count, max_num_patches)
        delta = 0
        for _ in range(10):
            area = pool.next() * (budget - min_num_patches) + min_num_patches
            aspect_ratio = math.exp(pool.next() * (hi - lo) + lo)
            h = int(round(math.sqrt(area * aspect_ratio)))
            w = int(round(math.sqrt(area / aspect_ratio)))

            if w < width and h < height:
                top = int(pool.next() * (height - h + 1))
                left = int(pool.next() * (width - w + 1))
                row_bits = ((1 << w) - 1) << left
                block = 0
                for row in range(h):
                    block |= row_bits << ((top + row) * width)

                fresh = (~bits) & block
                n_fresh = fresh.bit_count()
                if 0 < n_fresh <= budget:
                    bits |= fresh
                    delta = n_fresh
            if delta > 0:
                break

        if delta == 0:
            break
        count += delta

    assert bits < 1 << (height * width), "bitmask overflowed the grid"
    return bits


def _unpack_masks(rows: list[int], n_patches: int) -> torch.Tensor:
    """Unpack row bitmasks into a ``[len(rows), n_patches]`` bool tensor.

    Single source of truth for bit layout: cell ``c`` lives at bit ``c``, where
    ``c = row * width + col``. Anything that unpacks a bitmask must go through
    here -- re-deriving the index arithmetic is how a driver ends up reading
    every ``width``-th bit. Each int is written little-endian and unpacked
    LSB-first within every byte, matching ``(bits >> c) & 1``.
    """
    n_bytes = (n_patches + 7) // 8
    buf = bytearray(b"".join(row.to_bytes(n_bytes, "little") for row in rows))
    packed = torch.frombuffer(buf, dtype=torch.uint8).view(len(rows), n_bytes)
    shifts = torch.arange(8, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(len(rows), n_bytes * 8)[:, :n_patches].bool()


def _log_aspect_bounds(
    min_aspect: float, max_aspect: float | None
) -> Tuple[float, float]:
    if min_aspect <= 0:
        raise ValueError(f"min_aspect must be positive, got {min_aspect}")
    if max_aspect is None:
        max_aspect = 1 / min_aspect
    if max_aspect <= 0:
        raise ValueError(f"max_aspect must be positive, got {max_aspect}")
    if max_aspect < min_aspect:
        raise ValueError(
            f"max_aspect ({max_aspect}) must be >= min_aspect ({min_aspect})"
        )
    return math.log(min_aspect), math.log(max_aspect)


def _simdinov2_patch_masks(
    n_rows: int,
    n_patches: int,
    grid_size: tuple[int, int],
    mask_ratio_min_max: tuple[float, float] = (0.1, 0.5),
    mask_sample_probability: float = 0.5,
    max_num_patches: int | None = None,
    min_num_patches: int = 4,
    min_aspect: float = 0.3,
    max_aspect: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Assign per-row block masks the way SimDINOv2's collate function does.

    Each row is one ``(view, sample)`` pair. Only ``int(n_rows *
    mask_sample_probability)`` rows receive a mask, their target ratios are
    **stratified** across ``mask_ratio_min_max`` so no two rows draw from the
    same band, the remaining rows get empty masks, and finally all rows are
    permuted -- so masked rows are scattered across views rather than
    front-loaded.

    Args:
        n_rows: Number of ``(view, sample)`` rows. Upstream uses
            ``n_global_crops * batch_size``, pooling every view together.
        n_patches: Patches per row, i.e. ``grid_h * grid_w``.
        grid_size: Patch grid as ``(height, width)``.
        mask_ratio_min_max: ``(min, max)`` masked-patch ratio band. Default is
            ``(0.1, 0.5)``, the official value.
        mask_sample_probability: Fraction of rows that receive a mask. Default is
            0.5, the official value.
        max_num_patches: Cap on a single block's area. Defaults to
            ``n_patches // 2``.
        min_num_patches: Lower bound of a block's target area. Default is 4.
        min_aspect: Minimum block aspect ratio. Default is 0.3.
        max_aspect: Maximum block aspect ratio. Defaults to ``1 / min_aspect``.
        generator: Optional ``torch.Generator`` for reproducible masks.

    Returns:
        torch.Tensor: ``[n_rows, n_patches]`` bool tensor, ``True`` where masked.
        Rows are ordered row-major over ``(view, sample)`` to match the
        ``flatten(0, 1)`` feature layout
        :class:`~stable_pretraining.losses.SimDINOv2Loss` expects.

    Raises:
        ValueError: If the row count, grid or ratio/probability arguments are
            inconsistent.

    Example:
        >>> masks = _simdinov2_patch_masks(n_rows=16, n_patches=196, grid_size=(14, 14))
        >>> masks.shape
        torch.Size([16, 196])
        >>> masks.dtype == torch.bool
        True
        >>> int(masks.any(dim=1).sum()) <= 8
        True
    """
    if n_rows < 1:
        raise ValueError(f"n_rows must be >= 1, got {n_rows}")
    if grid_size[0] * grid_size[1] != n_patches:
        raise ValueError(
            f"grid_size {grid_size} has {grid_size[0] * grid_size[1]} cells but "
            f"n_patches={n_patches}"
        )
    ratio_min, ratio_max = mask_ratio_min_max
    if not 0 < ratio_min <= ratio_max <= 1:
        raise ValueError(
            "mask_ratio_min_max must satisfy 0 < min <= max <= 1, "
            f"got {mask_ratio_min_max}"
        )
    if not 0.0 <= mask_sample_probability <= 1.0:
        raise ValueError(
            f"mask_sample_probability must be in [0, 1], got {mask_sample_probability}"
        )
    if min_num_patches < 1:
        raise ValueError(f"min_num_patches must be >= 1, got {min_num_patches}")
    if max_num_patches is None:
        max_num_patches = n_patches // 2
    if max_num_patches < 1:
        raise ValueError(f"max_num_patches must be >= 1, got {max_num_patches}")
    aspect_bounds = _log_aspect_bounds(min_aspect, max_aspect)

    n_masked_rows = int(n_rows * mask_sample_probability)
    # One shared pool for the whole batch; placement stays sequential per row
    # because each block depends on what previous blocks already covered.
    pool = _UniformPool(generator, max(64, n_rows * 16))

    # Stratify so every masked row draws its ratio from its own sub-band.
    bands = torch.linspace(ratio_min, ratio_max, n_masked_rows + 1).tolist()
    rows = []
    for index in range(n_rows):
        if index < n_masked_rows:
            low, high = bands[index], bands[index + 1]
            target = int(n_patches * (pool.next() * (high - low) + low))
        else:
            target = 0
        rows.append(
            _place_blocks(
                grid_size[0],
                grid_size[1],
                target,
                min_num_patches,
                max_num_patches,
                aspect_bounds,
                pool,
            )
        )

    # Upstream shuffles the mask list before stacking, which is the same as
    # stacking then permuting: it decides *which* rows get masked, not what
    # shape those masks have.
    order = torch.randperm(n_rows, generator=generator).tolist()
    return _unpack_masks([rows[i] for i in order], n_patches)


@dataclass
class SimDINOv2Output(ModelOutput):
    """Structured output of the :class:`SimDINOv2` SSL method."""

    loss: torch.Tensor = None
    loss_cls: torch.Tensor = None
    loss_patch: torch.Tensor = None
    comp_loss: torch.Tensor = None
    expa_loss: torch.Tensor = None
    global_comp_loss: torch.Tensor = None
    embedding: torch.Tensor = None


def _split_cls_patches(
    features: torch.Tensor, has_cls: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split a token sequence into ``(cls, patches)``.

    Args:
        features: Backbone tokens ``[B, T, D]``, or ``[B, D]`` when the backbone
            already pools.
        has_cls: Whether the sequence carries a CLS token at position 0.

    Returns:
        Tuple of ``(cls, patches)``; ``patches`` is ``None`` for pooled outputs.
    """
    if features.ndim == 2:
        return features, None
    if has_cls:
        return features[:, 0], features[:, 1:]
    return features.mean(dim=1), features


def _cosine_projector(in_dim: int, hidden_dim: int, bottleneck_dim: int) -> nn.Module:
    """Standard 3-layer MLP ending in L2-norm.

    SimDINOv2 drops the prototype layer that DINOv2 needs for Sinkhorn-Knopp: every
    term in the loss is a cosine similarity, so normalizing here makes the raw dot
    product upstream scores numerically identical.

    Args:
        in_dim: Backbone embedding dimension.
        hidden_dim: Hidden width of the two inner layers.
        bottleneck_dim: Output width before the final norm.

    Returns:
        nn.Module: The projection head.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, bottleneck_dim),
        L2Norm(),
    )


class SimDINOv2(Module):
    """SimDINOv2: cosine self-distillation with a coding-rate and masked patches.

    SimDINOv2 removes DINOv2's prototype heads, softmax, centering, and
    Sinkhorn-Knopp normalization. What replaces them:

    - **Compression**: mean cosine similarity between every student CLS view and
      every teacher CLS view, excluding same-index pairs.
    - **Expansion**: Gaussian coding rate of the student CLS features, from the
      eigenvalues of their uncentered Gram matrix. This is what keeps the
      features from collapsing.
    - **Masked patches**: cosine similarity between student and teacher patch
      features at the positions the student had hidden behind
      :data:`mask_token`.

    The teacher is an EMA copy of the student that only sees unmasked global
    crops. The student sees the global crops with a mask applied, plus every
    local crop unmasked.

    Masked positions are drawn per ``(view, sample)`` row by
    :func:`_simdinov2_patch_masks`, whose ``True`` means masked.

    :param encoder_name: timm ViT name (default ``"vit_small_patch16_224"``) or a
        pre-built ``nn.Module``. Multi-crop requires the backbone to accept
        variable input sizes; pass ``encoder_kwargs={"dynamic_img_size": True}``
        for timm ViTs.
    :param projector_hidden_dim: Hidden width of the 3-layer MLP (default 2048).
    :param projector_bottleneck_dim: Output width before the final norm
        (default 256).
    :param cls_loss_weight: Weight on the CLS term (default 1.0).
    :param patch_loss_weight: Weight on the masked-patch term (default 1.0).
    :param mask_ratio_min_max: ``(min, max)`` masked-patch ratio band (default
        ``(0.1, 0.5)``, the official value).
    :param mask_sample_probability: Fraction of ``(view, sample)`` rows that
        receive a mask at all (default 0.5, the official value). Because the loss
        normalizes by row count, this also acts as a scale factor on the patch
        term.
    :param max_num_patches: Cap on one block's area; defaults to ``N // 2``.
    :param min_num_patches: Lower bound on a block's area (default 4).
    :param min_aspect: Minimum block aspect ratio (default 0.3).
    :param max_aspect: Maximum block aspect ratio; defaults to ``1 / min_aspect``.
    :param loss: Optional :class:`~stable_pretraining.losses.SimDINOv2Loss`;
        defaults to the paper's settings.
    :param grid_size: Patch grid as ``(height, width)``. Inferred from a dummy
        ``patch_embed`` forward when ``None``.
    :param image_size: Input size used for the grid inference (default 224).
    :param ema_decay_start: Initial backbone/head EMA (default 0.996).
    :param ema_decay_end: Final EMA (default 0.9995).
    :param encoder_kwargs: Extra kwargs forwarded to ``timm.create_model``.
    :param pretrained: Load pretrained timm weights.

    Note:
        EMA updates need :class:`~stable_pretraining.callbacks.TeacherStudentCallback`
        registered on the trainer, as with every other teacher/student method.

    Example::

        model = SimDINOv2(
            "vit_small_patch16_224", encoder_kwargs={"dynamic_img_size": True}
        )

        out = model(
            global_views=[torch.randn(8, 3, 224, 224) for _ in range(2)],
            local_views=[torch.randn(8, 3, 96, 96) for _ in range(6)],
        )
        out.loss.backward()
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_hidden_dim: int = 2048,
        projector_bottleneck_dim: int = 256,
        cls_loss_weight: float = 1.0,
        patch_loss_weight: float = 1.0,
        mask_ratio_min_max: Tuple[float, float] = (0.1, 0.5),
        mask_sample_probability: float = 0.5,
        max_num_patches: Optional[int] = None,
        min_num_patches: int = 4,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
        loss: Optional[nn.Module] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        image_size: int = 224,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 0.9995,
        encoder_kwargs: Optional[dict] = None,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            import timm

            kwargs = dict(encoder_kwargs or {})
            kwargs.setdefault("dynamic_img_size", True)  # multi-crop: 224 + 96 etc.
            base = timm.create_model(
                encoder_name,
                num_classes=0,
                pretrained=pretrained,
                **kwargs,
            )
        else:
            base = encoder_name

        with torch.no_grad():
            seq = base.forward_features(torch.zeros(1, 3, image_size, image_size))
        self._has_cls = (
            hasattr(base, "cls_token")
            and base.cls_token is not None
            and seq.shape[1] > 1
        )
        self.embed_dim = seq.shape[-1]

        self.backbone = TeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.cls_head = TeacherStudentWrapper(
            _cosine_projector(
                self.embed_dim, projector_hidden_dim, projector_bottleneck_dim
            ),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.patch_head = TeacherStudentWrapper(
            _cosine_projector(
                self.embed_dim, projector_hidden_dim, projector_bottleneck_dim
            ),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        if loss is None:
            loss = SimDINOv2Loss(
                cls_loss_weight=cls_loss_weight,
                patch_loss_weight=patch_loss_weight,
            )
        self.loss_fn = loss

        self.mask_ratio_min_max = mask_ratio_min_max
        self.mask_sample_probability = mask_sample_probability
        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

        # One grid for the whole run: the backbone resizes positions by input
        # resolution, so the local crops would otherwise need their own masks.
        self.grid_size = grid_size or self._infer_grid(base, image_size)
        self.n_patches = self.grid_size[0] * self.grid_size[1]

    @staticmethod
    def _infer_grid(base: nn.Module, image_size: int) -> Tuple[int, int]:
        """Read the patch grid off a dummy ``patch_embed`` forward.

        Args:
            base: The timm backbone, built with ``dynamic_img_size=True``.
            image_size: Spatial size to probe at.

        Returns:
            Tuple of ``(grid_height, grid_width)``.
        """
        with torch.no_grad():
            pe = base.patch_embed(torch.zeros(1, 3, image_size, image_size))
        # ``dynamic_img_size=True`` returns [B, H', W', D];
        # flat models return [B, N, D].
        if pe.ndim == 4:
            return pe.shape[1], pe.shape[2]
        n = pe.shape[1]
        side = math.isqrt(n)
        if side * side != n:
            raise ValueError(
                f"cannot infer a square patch grid from {n} patches; "
                f"pass grid_size=(height, width) explicitly"
            )
        return side, side

    def _encode(self, vit, images, mask=None) -> torch.Tensor:
        x = vit.patch_embed(images)
        # With ``dynamic_img_size=True`` patch_embed returns 4D [B, H', W', D]
        # and ``_pos_embed`` *requires* 4D. For the mask substitution we
        # reshape to 3D, apply the mask, then reshape back.
        is_4d = x.ndim == 4
        if mask is not None:
            if is_4d:
                B_, H_, W_, D_ = x.shape
                x = x.reshape(B_, H_ * W_, D_)
            # ``True`` (masked) positions get the mask token; ``.to(x.dtype)``
            # keeps the substitution in the autocast dtype.
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
            if is_4d:
                x = x.reshape(B_, H_, W_, D_)
        x = vit._pos_embed(x)
        if hasattr(vit, "patch_drop"):
            x = vit.patch_drop(x)
        if hasattr(vit, "norm_pre"):
            x = vit.norm_pre(x)
        x = vit.blocks(x)
        return vit.norm(x)

    def _global_mask(self, n_rows: int, device) -> torch.Tensor:
        """Draw the SimDINOv2 mask for the global crops.

        Args:
            n_rows: Number of ``(view, sample)`` rows, i.e.
                ``n_global_views * batch_size``.
            device: Device the mask must live on.

        Returns:
            torch.Tensor: ``[n_rows, n_patches]`` bool tensor, ``True`` where masked.
        """
        masks = _simdinov2_patch_masks(
            n_rows=n_rows,
            n_patches=self.n_patches,
            grid_size=self.grid_size,
            mask_ratio_min_max=self.mask_ratio_min_max,
            mask_sample_probability=self.mask_sample_probability,
            max_num_patches=self.max_num_patches,
            min_num_patches=self.min_num_patches,
            min_aspect=self.min_aspect,
            max_aspect=self.max_aspect,
        )
        return masks.to(device)

    def forward(
        self,
        global_views: Optional[Sequence[torch.Tensor]] = None,
        local_views: Optional[Sequence[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> SimDINOv2Output:
        """Run one SimDINOv2 step.

        Args:
            global_views: Sequence of ``[B, 3, H, W]`` global crops. The teacher
                sees these unmasked and the student sees them masked.
            local_views: Sequence of ``[B, 3, h, w]`` local crops. The student
                sees these unmasked, and they lead the student view ordering that
                the loss relies on.
            images: Evaluation-only batch ``[B, 3, H, W]``. Returns the teacher
                embedding with a zero loss.

        Returns:
            SimDINOv2Output: Loss and detached diagnostics.

        Raises:
            ValueError: If neither ``global_views`` nor ``images`` is given.
        """
        if images is not None:
            with torch.no_grad():
                feats = self.backbone.forward_teacher(images)
                cls, _ = _split_cls_patches(feats, self._has_cls)
            return SimDINOv2Output(
                loss=torch.zeros((), device=cls.device, dtype=cls.dtype),
                embedding=cls.detach(),
            )

        if not global_views:
            raise ValueError("SimDINOv2.forward needs global_views or images")

        global_views = list(global_views)
        local_views = list(local_views or [])
        n_global = len(global_views)
        n_local = len(local_views)
        global_imgs = torch.cat(global_views, dim=0)
        B = global_views[0].shape[0]
        n_rows = n_global * B

        mask = self._global_mask(n_rows, global_imgs.device)

        with torch.no_grad():
            t_feats = self._encode(self.backbone.teacher, global_imgs, mask=None)
            t_cls, t_patches = _split_cls_patches(t_feats, self._has_cls)
            t_cls_feats = self.cls_head.forward_teacher(t_cls).view(n_global, B, -1)
            t_patch_feats = self.patch_head.forward_teacher(t_patches.flatten(0, 1))

        s_feats_g = self._encode(self.backbone.student, global_imgs, mask=mask)
        s_cls_g, s_patches_g = _split_cls_patches(s_feats_g, self._has_cls)
        s_cls_feats_g = self.cls_head.forward_student(s_cls_g).view(n_global, B, -1)
        s_patch_feats = self.patch_head.forward_student(s_patches_g.flatten(0, 1))

        if n_local > 0:
            local_imgs = torch.cat(local_views, dim=0)
            s_feats_l = self._encode(self.backbone.student, local_imgs, mask=None)
            s_cls_l, _ = _split_cls_patches(s_feats_l, self._has_cls)
            s_cls_feats_l = self.cls_head.forward_student(s_cls_l).view(n_local, B, -1)
            # Locals first: ``SimDINOv2Loss`` reads ``student[:2]`` for the
            # expansion term and the no_diag exclusion, and upstream feeds it
            # local crops there.
            s_cls_feats = torch.cat([s_cls_feats_l, s_cls_feats_g], dim=0)
        else:
            s_cls_feats = s_cls_feats_g

        # ``flatten(0, 1)`` already lays patches out row-major over
        # ``(rows, patches)``, which is the order ``mask.flatten()`` indexes in.
        mask_flat = mask.flatten()
        student_masked = s_patch_feats[mask_flat]
        teacher_masked = t_patch_feats[mask_flat]

        loss, metrics = self.loss_fn(
            student_cls_feats=s_cls_feats,
            teacher_cls_feats=t_cls_feats,
            student_patch_feats=student_masked,
            teacher_patch_feats=teacher_masked,
            patch_mask=mask,
        )

        return SimDINOv2Output(
            loss=loss,
            loss_cls=metrics["loss_cls"],
            loss_patch=metrics.get("loss_patch"),
            comp_loss=metrics["comp_loss"],
            expa_loss=metrics["expa_loss"],
            global_comp_loss=metrics["global_comp_loss"],
            # Teacher backbone CLS, matching DINO/DINOv2/iBOT: ``[n_global * B,
            # embed_dim]`` with no prototype layer and no per-sample slicing.
            embedding=t_cls.detach(),
        )
