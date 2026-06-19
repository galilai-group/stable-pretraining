"""GPU-side batched transforms for high-throughput training.

These transforms operate on **already-collated** batches (after the
DataLoader has produced a batch tensor). They run inside
``Module.on_after_batch_transfer``, which Lightning calls right after
the batch has been moved to the target device.

How users wire them in
----------------------
**Preferred:** attach the transform to the **dataset** via the
``gpu_transform=`` constructor argument::

    train_ds = spt.data.HFDataset(
        "frgfm/imagenette",
        split="train",
        transform=cpu_transform,  # CPU side: decode + resize only
        gpu_transform=train_aug,  # GPU side: kornia / GPUCompose chain
    )

The module finds it through the active DataLoader; train / val / test
each carry their own spec naturally.

**Fallback** (when wrapping a third-party dataset you can't modify):
attach to the :class:`DataModule` as ``gpu_transform=...`` (callable) or
``gpu_transform={"train": ..., "val": ...}`` (per-stage dict).

Setting ``gpu_transform`` on the :class:`Module` is **not** supported
and is rejected at ``on_train_start`` — the dataset/DataModule paths
are the only sanctioned entry points.

Why GPU augmentation
--------------------
The CPU transforms in :mod:`stable_pretraining.data.transforms` are
applied per-sample inside the DataLoader workers. For SSL workloads
with heavy multi-view augmentation, those workers often become the
bottleneck. Moving augmentation to the GPU:

1. Frees CPU workers to do more I/O / decode in parallel,
2. Vectorises augmentation across the batch (one kernel launch, not B),
3. Composes with non-blocking H2D transfers (``pin_memory=True`` on the
   DataLoader) so the H2D copy overlaps with the previous step's
   compute for free.

This is **not** a full FFCV port — FFCV's largest wins come from its
custom binary file format + mmap + Numba-JIT'd threaded workers, which
replace the ``DataLoader`` substrate entirely. A GPU-augmentation layer
recovers the augmentation/transfer slice of that win (measured ~1.77×
on BarlowTwins ViT-S/16 / Imagenette / H200, see
``benchmarks/imagenet10/RESULTS.md``), and stacks cleanly with any
DataLoader-based backend (HF datasets, torchvision, Lance, webdataset).

API
---
Top-level pieces:

- :class:`ToDevice` walks the batch dict and moves all tensor leaves to
  a target device. Idempotent — tensors already on-device are left
  alone, so backends like DALI/FFCV that already emit GPU tensors
  aren't taxed.
- :class:`GPUCompose` chains a list of GPU transforms and (by default)
  wraps each tensor op in ``torch.compile`` for kernel fusion.
- :class:`StackedMultiView` / :class:`MultiView` produce N augmented
  views from one source tensor: the stacked variant runs a single chain
  on a ``(N*B, ...)`` tensor (symmetric SSL — Barlow Twins, SimCLR,
  VICReg, NNCLR) and is ~1.2× faster than the per-view variant; the
  per-view variant supports asymmetric recipes (BYOL, DINO
  student/teacher).

The concrete kornia-backed wrappers (:class:`GPUNormalize`,
:class:`GPURandomResizedCrop`, etc.) each take ``source``/``target``
dict keys, mirroring the CPU transform API in :mod:`transforms`. They
record kornia's sampled per-sample parameters into the batch dict
(``batch["GPUColorJitter"]``, etc.) matching the existing CPU
``Transform.get_name`` convention.
"""

from typing import Optional, Sequence, Tuple, Union

import kornia.augmentation as K
import torch
from torch import nn


def _walk_tensors(obj, fn):
    """Recursively map ``fn`` over every torch.Tensor leaf, preserving structure."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _walk_tensors(v, fn)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _walk_tensors(v, fn)
        return obj
    if isinstance(obj, tuple):
        return tuple(_walk_tensors(v, fn) for v in obj)
    return obj


class ToDevice(nn.Module):
    """Move all tensor leaves of a batch dict to a target device (idempotent).

    Tensors already on the resolved device are returned unchanged, so this
    is a safe no-op when Lightning has already transferred the batch (which
    it does by default in ``transfer_batch_to_device`` with ``non_blocking=True``
    when ``DataLoader(pin_memory=True)``). The H2D overlap with compute is
    provided by PyTorch's hidden memcpy stream — you don't need to manage
    your own ``cuda.Stream`` to get it.

    DDP: Lightning calls ``torch.cuda.set_device(local_rank)`` per process,
    so ``device=None`` (auto) resolves to ``cuda:local_rank`` independently
    in each rank without further wiring.

    Args:
        device: Target device. Accepts:

            - ``None`` (default): auto — uses ``torch.cuda.current_device()``
              if CUDA is available, else ``cpu``. Recommended for DDP.
            - ``"cuda"`` / ``"cpu"``: string spec; ``"cuda"`` is normalized
              to ``cuda:current_device()`` for reliable idempotence checks.
            - ``torch.device(...)`` or explicit ``"cuda:0"``: used as-is.

        non_blocking: Pass ``non_blocking=True`` to ``.to()``. Only useful if
            the source tensor is in pinned memory
            (``DataLoader(pin_memory=True)``). Defaults to ``True``.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = True,
    ):
        super().__init__()
        self._device_spec = device  # resolved lazily so DDP rank is honored
        self.non_blocking = non_blocking

    @property
    def device(self) -> torch.device:
        spec = self._device_spec
        if spec is None:
            if torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device())
            return torch.device("cpu")
        d = torch.device(spec)
        # Normalize bare "cuda" to a specific index so `tensor.device == d`
        # works (a bare `cuda` device with index=None doesn't compare equal
        # to a tensor's `cuda:0` device).
        if d.type == "cuda" and d.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return d

    def _move(self, t: torch.Tensor) -> torch.Tensor:
        target = self.device
        if t.device == target:
            return t
        return t.to(target, non_blocking=self.non_blocking)

    def forward(self, batch):
        return _walk_tensors(batch, self._move)


class _GPUTransformBase(nn.Module):
    """Common base: read ``batch[source]``, apply :meth:`_apply_op`, write ``batch[target]``.

    Handles dict and list/tuple-of-views fan-out so transforms compose with
    :class:`MultiViewTransform`. Subclasses implement :meth:`_apply_op`
    (``Tensor -> Tensor``) and may override :meth:`_record_params` to write
    augmentation metadata into the batch dict (see :class:`_KorniaWrap`).

    The ``source``/``target`` keys mirror the CPU transform API in
    :mod:`stable_pretraining.data.transforms`.
    """

    def __init__(self, source: str = "image", target: str = "image"):
        super().__init__()
        self.source = source
        self.target = target

    def _apply_op(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _record_params(self, batch, n_calls: int) -> None:
        """Optionally write per-forward augmentation params into the batch.

        ``n_calls`` is the number of times :meth:`_apply_op` was invoked for
        this forward (1 for a single tensor, N for multi-view inputs).
        Override in subclasses that wrap stochastic ops.
        """
        return None

    def _get_record_name(self, batch) -> str:
        """Pick a unique key in ``batch`` for recording params, like the CPU API."""
        base = type(self).__name__
        if base not in batch:
            return base
        i = 0
        while f"{base}_{i}" in batch:
            i += 1
        return f"{base}_{i}"

    def forward(self, batch):
        src = batch[self.source]
        if isinstance(src, dict):
            out = {k: self._apply_op(v) for k, v in src.items()}
            n_calls = len(src)
        elif isinstance(src, (list, tuple)):
            mapped = [self._apply_op(v) for v in src]
            out = type(src)(mapped) if isinstance(src, tuple) else mapped
            n_calls = len(src)
        else:
            out = self._apply_op(src)
            n_calls = 1
        batch[self.target] = out
        self._record_params(batch, n_calls)
        return batch


class _GPUTensorOp(_GPUTransformBase):
    """Deterministic tensor op (compile-friendly).

    Subclasses implement :meth:`_op` (``Tensor -> Tensor``). When wrapped by
    a :class:`GPUCompose` with ``compile=True``, ``_op`` is replaced with a
    ``torch.compile``'d version for kernel fusion.
    """

    def __init__(self, source: str = "image", target: str = "image"):
        super().__init__(source=source, target=target)
        self._compiled_op = None

    def _op(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _enable_compile(self, **compile_kwargs):
        """Called by :class:`GPUCompose` to wrap :meth:`_op` in ``torch.compile``."""
        self._compiled_op = torch.compile(self._op, **compile_kwargs)

    def _apply_op(self, x: torch.Tensor) -> torch.Tensor:
        return self._compiled_op(x) if self._compiled_op is not None else self._op(x)


class GPUCompose(nn.Module):
    """Sequentially apply a list of GPU transforms.

    Each transform must accept and return the batch dict. Tensor-op
    transforms (subclasses of :class:`_GPUTensorOp`) optionally get their
    inner op wrapped in ``torch.compile`` when ``compile=True``.

    Args:
        transforms: Sequence of GPU transforms (e.g. :class:`ToDevice`,
            :class:`GPUNormalize`, ...). Applied in order.
        compile: If ``True`` (default), wrap each :class:`_GPUTensorOp`'s
            tensor op with ``torch.compile``. The first batch will pay a
            one-time compile cost; subsequent batches benefit from kernel
            fusion. Compile is skipped on CPU.
        compile_mode: Mode passed to ``torch.compile`` (e.g.
            ``"reduce-overhead"`` for low launch overhead at the cost of
            extra memory). Defaults to ``None`` (PyTorch's default mode).
        dynamic: ``dynamic`` flag forwarded to ``torch.compile``. Defaults
            to ``False`` (static shapes — best for fixed-size batches).
    """

    def __init__(
        self,
        transforms: Sequence[nn.Module],
        compile: bool = True,
        compile_mode: Optional[str] = None,
        dynamic: bool = False,
    ):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.compile = compile and torch.cuda.is_available()
        self.compile_mode = compile_mode
        self.dynamic = dynamic
        if self.compile:
            kwargs = {"dynamic": dynamic}
            if compile_mode is not None:
                kwargs["mode"] = compile_mode
            for t in self.transforms:
                if isinstance(t, _GPUTensorOp):
                    t._enable_compile(**kwargs)

    def forward(self, batch):
        for t in self.transforms:
            batch = t(batch)
        return batch


class GPUNormalize(_GPUTensorOp):
    """Normalize a batched image tensor with per-channel mean/std on GPU.

    Expects input of shape ``(B, C, H, W)`` with float dtype. Pure pointwise
    op — fuses well with adjacent transforms under ``torch.compile``.

    Args:
        mean: Per-channel mean.
        std: Per-channel std.
        source: Dict key to read from. Defaults to ``"image"``.
        target: Dict key to write to. Defaults to ``"image"``.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(source=source, target=target)
        self.register_buffer(
            "mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        )

    def _op(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class _KorniaWrap(_GPUTransformBase):
    """Thin wrapper that runs a kornia augmentation and records its sampled params.

    Stochastic kornia augmentations sample independent random parameters per
    sample by default (``same_on_batch=False``). After each forward, kornia
    stores the sampled batch of params on the op as ``op._params`` — we
    surface those into the batch dict under the wrapper class name
    (matching the CPU :class:`Transform.get_name` convention), so downstream
    code (consistency losses, EMA teachers, etc.) can reproduce or
    correlate against the augmentation.

    Compile note: kornia stochastic ops do not currently compile cleanly
    (graph breaks on per-sample random sampling), so :class:`GPUCompose`
    skips ``torch.compile`` for kornia-backed transforms.

    Multi-view note: when the input ``source`` is a list/dict of views, the
    recorded params are a list/dict of the per-view sampled params, in the
    same structure as the input.

    Args:
        kornia_op: A kornia augmentation module (e.g.
            ``kornia.augmentation.RandomHorizontalFlip``).
        record_params: If ``True`` (default), write sampled params into the
            batch dict after each forward. Set ``False`` to skip recording.
    """

    def __init__(
        self,
        kornia_op: nn.Module,
        source: str = "image",
        target: str = "image",
        record_params: bool = True,
    ):
        super().__init__(source=source, target=target)
        self.kornia_op = kornia_op
        self.record_params = record_params
        # Track the most recent params for each forward call (1 per view in
        # the multi-view case). Kornia overwrites op._params on every call,
        # so we have to snapshot it inside _apply_op.
        self._last_call_params: list = []

    def _apply_op(self, x: torch.Tensor) -> torch.Tensor:
        out = self.kornia_op(x)
        if self.record_params:
            # ``op._params`` is the dict of per-sample sampled params for
            # this last call. Snapshot it (kornia will overwrite next call).
            params = getattr(self.kornia_op, "_params", None)
            self._last_call_params.append(params)
        return out

    def _record_params(self, batch, n_calls: int) -> None:
        if not self.record_params:
            return
        snapshot = self._last_call_params
        self._last_call_params = []  # reset for next forward
        if not snapshot:
            return
        key = self._get_record_name(batch)
        # For single-view input we expose the params dict directly; for
        # multi-view we expose a list of dicts (one per view) so the
        # structure mirrors batch[self.target].
        batch[key] = snapshot[0] if n_calls == 1 else snapshot


class GPURandomResizedCrop(_KorniaWrap):
    """Batched random resized crop on GPU (kornia-backed).

    Each sample gets independent crop params. Matches the SSL convention
    of crops sampled uniformly in area within ``scale`` and aspect ratio
    in ``ratio``.

    Args:
        size: Output spatial size ``(H, W)`` or a single int.
        scale: Range of area to sample (default ``(0.08, 1.0)``).
        ratio: Range of aspect ratio (default ``(0.75, 1.333)``).
        p: Probability of applying (default ``1.0``).
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        p: float = 1.0,
        source: str = "image",
        target: str = "image",
    ):
        if isinstance(size, int):
            size = (size, size)
        super().__init__(
            kornia_op=K.RandomResizedCrop(size=size, scale=scale, ratio=ratio, p=p),
            source=source,
            target=target,
        )


class GPURandomHorizontalFlip(_KorniaWrap):
    """Batched per-sample horizontal flip on GPU (kornia-backed)."""

    def __init__(
        self,
        p: float = 0.5,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(
            kornia_op=K.RandomHorizontalFlip(p=p),
            source=source,
            target=target,
        )


class GPUColorJitter(_KorniaWrap):
    """Batched per-sample color jitter on GPU (kornia-backed).

    Args mirror torchvision's ColorJitter. Each sample gets independent
    params when ``same_on_batch=False`` (kornia default).
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        p: float = 1.0,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(
            kornia_op=K.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=p,
            ),
            source=source,
            target=target,
        )


class GPUGaussianBlur(_KorniaWrap):
    """Batched per-sample Gaussian blur on GPU (kornia-backed).

    Args:
        kernel_size: Single int (square kernel) or ``(kH, kW)`` tuple.
        sigma: ``(low, high)`` range sampled per sample.
        p: Probability of applying.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 23,
        sigma: Tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
        source: str = "image",
        target: str = "image",
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        super().__init__(
            kornia_op=K.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigma, p=p),
            source=source,
            target=target,
        )


class GPURandomGrayscale(_KorniaWrap):
    """Batched per-sample random grayscale on GPU (kornia-backed)."""

    def __init__(
        self,
        p: float = 0.2,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(
            kornia_op=K.RandomGrayscale(p=p),
            source=source,
            target=target,
        )


class GPURandomSolarize(_KorniaWrap):
    """Batched per-sample random solarize on GPU (kornia-backed).

    Args:
        thresholds: Threshold range for solarize (default ``0.1`` matches kornia).
        additions: Per-pixel intensity offset range (default ``0.1``).
        p: Probability of applying.
    """

    def __init__(
        self,
        thresholds: float = 0.1,
        additions: float = 0.1,
        p: float = 0.2,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(
            kornia_op=K.RandomSolarize(thresholds=thresholds, additions=additions, p=p),
            source=source,
            target=target,
        )


class GPURandomErasing(_KorniaWrap):
    """Batched per-sample random erasing on GPU (kornia-backed)."""

    def __init__(
        self,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
        p: float = 0.5,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(
            kornia_op=K.RandomErasing(scale=scale, ratio=ratio, value=value, p=p),
            source=source,
            target=target,
        )


class GPURandAugment(_KorniaWrap):
    """Batched RandAugment on GPU (kornia ``auto.RandAugment``).

    The automatic-augmentation policy used by the SOTA supervised ViT recipes
    (DeiT/AugReg). Operates on float images in ``[0, 1]`` so place it **before**
    :class:`GPUNormalize` in a :class:`GPUCompose` chain. Per-sample params are
    not recorded (the auto-augment sub-policy structure isn't a flat dict).

    Args:
        n: Number of sequential sub-policies applied per image (default 2).
        m: Magnitude in ``[0, 30]`` (default 10).
    """

    def __init__(
        self,
        n: int = 2,
        m: int = 10,
        source: str = "image",
        target: str = "image",
    ):
        from kornia.augmentation.auto import RandAugment

        super().__init__(
            kornia_op=RandAugment(n=n, m=m),
            source=source,
            target=target,
            record_params=False,
        )


class RandomMixupCutmix(nn.Module):
    """GPU-native Mixup / CutMix with soft (smoothed) targets.

    Applied at the **batch** level (it mixes samples *and* their labels), so it
    is meant to be called inside the training ``forward`` on the already-on-GPU
    batch — not inside a per-image :class:`GPUCompose` chain. Each call picks
    Mixup or CutMix per batch (by ``switch_prob``), or passes through with
    probability ``1 - prob``. Returns the mixed images and a
    ``(B, num_classes)`` soft-label matrix with label smoothing folded in, ready
    for ``F.cross_entropy(logits, soft_labels)``.

    This reproduces the mixing used by DeiT/AugReg-style supervised training.

    Args:
        num_classes: Number of classes (for one-hot soft targets).
        mixup_alpha: Beta parameter for Mixup (``0`` disables Mixup).
        cutmix_alpha: Beta parameter for CutMix (``0`` disables CutMix).
        prob: Probability of applying any mixing to a given batch.
        switch_prob: Given mixing is applied, probability of CutMix vs Mixup.
        label_smoothing: Smoothing applied to the (mixed) one-hot targets.
    """

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.prob = float(prob)
        self.switch_prob = float(switch_prob)
        self.label_smoothing = float(label_smoothing)

    def _smooth_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        off = self.label_smoothing / self.num_classes
        on = 1.0 - self.label_smoothing + off
        y = torch.full((labels.shape[0], self.num_classes), off, device=labels.device)
        return y.scatter_(1, labels.long().view(-1, 1), on)

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        """Mix a batch. Returns ``(mixed_images, soft_labels)``."""
        target = self._smooth_one_hot(labels)
        if self.prob == 0.0 or float(torch.rand(())) >= self.prob:
            return images, target

        perm = torch.randperm(images.shape[0], device=images.device)
        use_cutmix = self.cutmix_alpha > 0.0 and (
            self.mixup_alpha <= 0.0 or float(torch.rand(())) < self.switch_prob
        )
        if use_cutmix:
            lam = float(
                torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
            )
            _, _, h, w = images.shape
            r = (1.0 - lam) ** 0.5
            cut_h, cut_w = int(h * r), int(w * r)
            cy, cx = int(torch.randint(h, ())), int(torch.randint(w, ()))
            y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
            x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
            images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))  # true area ratio
        else:
            lam = float(
                torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
            )
            images = lam * images + (1.0 - lam) * images[perm]

        target = lam * target + (1.0 - lam) * target[perm]
        return images, target


class StackedMultiView(nn.Module):
    """Run a single GPU augmentation chain on a stacked ``(n_views * B, ...)`` batch.

    The source tensor is repeated along the batch dim, run through one
    chain (kornia samples independent random params per sample, so each
    repeat gets a different augmentation), then split back into N views.

    Valid for **symmetric** SSL where every view uses the same recipe
    (Barlow Twins, SimCLR, VICReg, NNCLR). For asymmetric methods (BYOL,
    DINO student/teacher) use :class:`MultiView` instead.

    Output schema matches the existing CPU :class:`MultiViewTransform`:
    ``batch[views_key]`` becomes a list of N dicts, each with ``{"image":
    view_tensor, "label": label}``. The original ``batch[source]`` key
    is removed to free GPU memory.

    Args:
        chain: The GPU augmentation pipeline (e.g. :class:`GPUCompose`).
        n_views: How many views to produce per sample (default 2).
        source: Input key (default ``"image"``).
        views_key: Output key for the list of views (default ``"views"``).
        label_key: Label key to copy into each view dict (default ``"label"``).
    """

    def __init__(
        self,
        chain: nn.Module,
        n_views: int = 2,
        source: str = "image",
        views_key: str = "views",
        label_key: str = "label",
    ):
        super().__init__()
        self.chain = chain
        self.n_views = n_views
        self.source = source
        self.views_key = views_key
        self.label_key = label_key

    def forward(self, batch):
        src = batch[self.source]
        B = src.shape[0]
        stacked = src.repeat(self.n_views, *([1] * (src.ndim - 1)))
        out = self.chain({self.source: stacked})[self.source]
        views_list = list(torch.split(out, B, dim=0))
        label = batch.get(self.label_key)
        batch[self.views_key] = [{"image": v, "label": label} for v in views_list]
        del batch[self.source]
        return batch


class MultiView(nn.Module):
    """Run a list of per-view chains on the same source batch (asymmetric SSL).

    Use this when each view needs a different augmentation recipe
    (e.g. BYOL's online vs target views, DINO's student vs teacher,
    SimCLR-with-Solarize on view 2 only). For symmetric recipes prefer
    :class:`StackedMultiView` — it's faster (one chain call instead of N).

    Output schema matches :class:`StackedMultiView`.

    Args:
        chains: Per-view augmentation pipelines, one per output view.
        source / views_key / label_key: see :class:`StackedMultiView`.
    """

    def __init__(
        self,
        chains,
        source: str = "image",
        views_key: str = "views",
        label_key: str = "label",
    ):
        super().__init__()
        self.chains = nn.ModuleList(chains)
        self.source = source
        self.views_key = views_key
        self.label_key = label_key

    def forward(self, batch):
        src = batch[self.source]
        label = batch.get(self.label_key)
        views = []
        for chain in self.chains:
            out = chain({self.source: src})[self.source]
            views.append({"image": out, "label": label})
        batch[self.views_key] = views
        del batch[self.source]
        return batch


__all__ = [
    "ToDevice",
    "GPUCompose",
    "GPUNormalize",
    "GPURandomResizedCrop",
    "GPURandomHorizontalFlip",
    "GPUColorJitter",
    "GPUGaussianBlur",
    "GPURandomGrayscale",
    "GPURandomSolarize",
    "GPURandomErasing",
    "StackedMultiView",
    "MultiView",
]
