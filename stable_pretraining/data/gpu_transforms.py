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

import math
import queue
import threading
import warnings
from collections import deque
from typing import Iterable, Optional, Sequence, Tuple, Union

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
        cropping_mode: kornia's crop implementation. ``"slice"`` (kornia's
            default, kept for reproducibility) crops and resizes each sample
            separately; ``"resample"`` warps the whole batch at once and is
            ~3x faster on large batches (e.g. 1536 frames on an H100:
            21 ms vs 64 ms per call). Same parameter distribution.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        p: float = 1.0,
        source: str = "image",
        target: str = "image",
        cropping_mode: str = "slice",
    ):
        if isinstance(size, int):
            size = (size, size)
        super().__init__(
            kornia_op=K.RandomResizedCrop(
                size=size, scale=scale, ratio=ratio, p=p, cropping_mode=cropping_mode
            ),
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


class GPUVideoSequential(_KorniaWrap):
    """Apply kornia augmentations to clips with one parameter set per clip.

    The image wrappers above sample independent parameters per image, so on
    clips flattened to frames every frame would get its own crop / flip /
    jitter. Video SSL recipes (V-JEPA, LeVJEPA, ...) need the same
    augmentation on all frames of a clip; this wraps the ops in
    ``kornia.augmentation.VideoSequential(same_on_frame=True)``.

    Args:
        *ops: kornia augmentations (e.g. ``K.RandomResizedCrop``) or the
            kornia-backed wrappers of this module (their inner op is used).
        data_format: Clip layout, ``"BTCHW"`` (default) or ``"BCTHW"``.
        source: Dict key to read from. Defaults to ``"image"``.
        target: Dict key to write to. Defaults to ``"image"``.
        record_params: Record kornia's sampled params into the batch.

    Example:
        >>> import kornia.augmentation as K
        >>> aug = GPUVideoSequential(
        ...     K.RandomResizedCrop(
        ...         (96, 96), scale=(0.02, 0.4), cropping_mode="resample"
        ...     ),
        ...     K.RandomHorizontalFlip(),
        ... )
        >>> batch = aug({"image": torch.rand(2, 16, 3, 128, 128)})
        >>> batch["image"].shape
        torch.Size([2, 16, 3, 96, 96])
    """

    def __init__(
        self,
        *ops: nn.Module,
        data_format: str = "BTCHW",
        source: str = "image",
        target: str = "image",
        record_params: bool = True,
    ):
        inner = [op.kornia_op if isinstance(op, _KorniaWrap) else op for op in ops]
        super().__init__(
            kornia_op=K.VideoSequential(
                *inner, data_format=data_format, same_on_frame=True
            ),
            source=source,
            target=target,
            record_params=record_params,
        )


# ---------------------------------------------------------------------------
# Batched multi-crop (one call for all views of a multi-crop recipe)
# ---------------------------------------------------------------------------

_GRAY = (0.2989, 0.587, 0.114)  # torchvision's rgb_to_grayscale weights


def _gray(x: torch.Tensor) -> torch.Tensor:
    """Luma over the channel dim (-3), keepdim."""
    w = torch.tensor(_GRAY, device=x.device, dtype=x.dtype).view(3, 1, 1)
    return (x * w).sum(-3, keepdim=True)


def _rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
    """RGB in [0, 1] (channel dim -3) -> HSV with hue in [0, 1)."""
    r, g, b = x.unbind(-3)
    maxc, argmax = x.max(-3)
    delta = maxc - x.min(-3).values
    safe = torch.where(delta > 0, delta, torch.ones_like(delta))
    h = torch.where(
        argmax == 0,
        ((g - b) / safe) % 6,
        torch.where(argmax == 1, (b - r) / safe + 2, (r - g) / safe + 4),
    )
    h = torch.where(delta > 0, h / 6, torch.zeros_like(h))
    s = torch.where(maxc > 0, delta / torch.where(maxc > 0, maxc, 1), 0)
    return torch.stack((h, s, maxc), -3)


def _hsv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_rgb_to_hsv`."""
    h, s, v = (t.unsqueeze(-3) for t in x.unbind(-3))
    n = torch.tensor([5.0, 3.0, 1.0], device=x.device, dtype=x.dtype).view(3, 1, 1)
    k = (n + h * 6) % 6
    return v - v * s * torch.clamp(torch.minimum(k, 4 - k), 0, 1)


def _sample_rrc_boxes(
    h: torch.Tensor,
    w: torch.Tensor,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    attempts: int = 10,
) -> torch.Tensor:
    """Vectorized ``torchvision.transforms.RandomResizedCrop.get_params``.

    Args:
        h: Valid image heights, shape ``(N,)``.
        w: Valid image widths, shape ``(N,)``.
        scale: Range of crop area as a fraction of the image area.
        ratio: Range of crop aspect ratio (w / h), sampled log-uniformly.
        attempts: Rejection-sampling attempts before the center-crop fallback.

    Returns:
        Boxes ``(N, 4)`` as ``(x1, y1, x2, y2)`` in pixels (``x2 - x1`` wide).
    """
    n, dev = h.shape[0], h.device
    h, w = h.float(), w.float()
    area = (h * w)[:, None]
    target = area * torch.empty(n, attempts, device=dev).uniform_(*scale)
    log_r = torch.empty(n, attempts, device=dev).uniform_(
        math.log(ratio[0]), math.log(ratio[1])
    )
    ar = torch.exp(log_r)
    cw = torch.round(torch.sqrt(target * ar))
    ch = torch.round(torch.sqrt(target / ar))
    ok = (cw > 0) & (cw <= w[:, None]) & (ch > 0) & (ch <= h[:, None])
    first = ok.int().argmax(1, keepdim=True)
    found = ok.any(1)
    cw = cw.gather(1, first).squeeze(1)
    ch = ch.gather(1, first).squeeze(1)
    in_ratio = w / h
    fw = torch.where(
        in_ratio < ratio[0],
        w,
        torch.where(in_ratio > ratio[1], torch.round(h * ratio[1]), w),
    )
    fh = torch.where(
        in_ratio < ratio[0],
        torch.round(w / ratio[0]),
        torch.where(in_ratio > ratio[1], h, h),
    )
    cw = torch.where(found, cw, fw)
    ch = torch.where(found, ch, fh)
    y0 = torch.where(
        found,
        torch.floor(torch.rand(n, device=dev) * (h - ch + 1)),
        torch.div(h - ch, 2, rounding_mode="floor"),
    )
    x0 = torch.where(
        found,
        torch.floor(torch.rand(n, device=dev) * (w - cw + 1)),
        torch.div(w - cw, 2, rounding_mode="floor"),
    )
    return torch.stack((x0, y0, x0 + cw, y0 + ch), 1)


_VIEW_GROUP_KEYS = {
    "n_views",
    "size",
    "scale",
    "ratio",
    "hflip_p",
    "color_jitter",
    "color_jitter_p",
    "grayscale_p",
}


class GPUMultiCrop(nn.Module):
    """All views of a multi-crop recipe in a few batched kernels, for images or clips.

    Replaces a :class:`MultiView` of per-view kornia chains (one random-resized
    crop + photometric chain per view) with one fused crop + colour kernel per
    view group (Triton, uint8 CUDA input) or, elsewhere, one batched
    ``roi_align`` plus batched colour ops. For LeVJEPA's recipe (1 global + 10
    local views of 96 x 16-frame 256 px clips) the kornia chains cost
    ~380-890 ms per batch on an H100; the fused kernel takes ~5-8 ms on
    L40S / RTX PRO 6000 class GPUs (the unfused path 115-250 ms).

    Crops follow ``torchvision.transforms.RandomResizedCrop`` (area in
    ``scale``, log-uniform aspect ratio in ``ratio``, 10 attempts, then a
    center crop) and are resampled with ``torchvision.ops.roi_align``, whose
    adaptive sampling averages ``ceil(crop / output)`` points per output pixel
    (an antialiased downscale). Colour jitter matches torchvision's
    ``ColorJitter`` (brightness, contrast against each frame's mean luma,
    saturation, hue in HSV, clamping after each op, one random op order per
    call), followed by random grayscale and horizontal flip. Every parameter
    is drawn per (sample, view) and shared by all frames of a clip.

    Args:
        groups: View groups, each a dict with ``n_views`` (int), ``size``
            (int or ``(H, W)``), ``scale`` (area range), and optional
            ``ratio`` (default ``(3/4, 4/3)``), ``hflip_p`` (default 0),
            ``color_jitter`` (``(brightness, contrast, saturation, hue)``,
            default none), ``color_jitter_p`` (default 1), ``grayscale_p``
            (default 0).
        source: Key of the input: images ``(B, C, H, W)`` or clips
            ``(B, T, C, H, W)``, uint8 (0-255) or float in ``[0, 1]``.
        views_key: Output key.
        label_key: Label key copied into every view dict (if present).
        size_key: Optional key of per-sample valid ``(H, W)`` sizes, shape
            ``(B, 2)``, for padded batches (see :class:`GPUDecodeJPEG`);
            crops are then sampled inside the valid region only. Padding must
            replicate the edge pixels (as :class:`GPUDecodeJPEG` does): bilinear
            sampling at a crop's border reads up to one pixel beyond it.
        output: ``"views"`` writes a list of view dicts ``{"image": (B, [T,]
            C, h, w)}`` (the :class:`MultiView` schema); ``"groups"`` writes
            one tensor per group, ``(B, n_views, [T,] C, h, w)``, for
            consumers that want the views stacked.
        record_params: Write the sampled boxes and photometric params into
            ``batch["GPUMultiCrop"]``.
        backend: ``"triton"`` runs crop, resize and all colour ops in one
            fused kernel per view group (uint8 RGB CUDA input; reads the
            source once and writes uint8 once, several times faster and far
            less memory traffic than the unfused path); ``"torch"`` uses
            ``roi_align`` plus elementwise torch ops (any device and dtype);
            ``"auto"`` (default) picks Triton when it applies. Both draw
            identical parameters; outputs agree to rounding.

    Output dtype matches the input (uint8 in, uint8 out). The source key is
    removed from the batch to free memory.

    Example:
        >>> mc = GPUMultiCrop(
        ...     [
        ...         dict(n_views=1, size=224, scale=(0.8, 1.0)),
        ...         dict(
        ...             n_views=10,
        ...             size=96,
        ...             scale=(0.02, 0.4),
        ...             hflip_p=0.5,
        ...             color_jitter=(0.4, 0.4, 0.2, 0.1),
        ...             color_jitter_p=0.8,
        ...             grayscale_p=0.2,
        ...         ),
        ...     ]
        ... )
        >>> out = mc(
        ...     {
        ...         "image": torch.randint(
        ...             0, 256, (2, 16, 3, 256, 256), dtype=torch.uint8
        ...         )
        ...     }
        ... )
        >>> len(out["views"]), out["views"][1]["image"].shape
        (11, torch.Size([2, 16, 3, 96, 96]))
    """

    def __init__(
        self,
        groups: Sequence[dict],
        source: str = "image",
        views_key: str = "views",
        label_key: str = "label",
        size_key: Optional[str] = None,
        output: str = "views",
        record_params: bool = True,
        backend: str = "auto",
    ):
        super().__init__()
        if backend not in ("auto", "triton", "torch"):
            raise ValueError(
                f"backend must be 'auto', 'triton' or 'torch', got {backend!r}"
            )
        self.backend = backend
        if output not in ("views", "groups"):
            raise ValueError(f"output must be 'views' or 'groups', got {output!r}")
        self.groups = []
        for g in groups:
            unknown = set(g) - _VIEW_GROUP_KEYS
            if unknown:
                raise ValueError(f"unknown view-group keys {sorted(unknown)}")
            size = g["size"]
            self.groups.append(
                dict(
                    n_views=int(g["n_views"]),
                    size=(size, size) if isinstance(size, int) else tuple(size),
                    scale=tuple(g["scale"]),
                    ratio=tuple(g.get("ratio", (3.0 / 4.0, 4.0 / 3.0))),
                    hflip_p=float(g.get("hflip_p", 0.0)),
                    color_jitter=g.get("color_jitter"),
                    color_jitter_p=float(g.get("color_jitter_p", 1.0)),
                    grayscale_p=float(g.get("grayscale_p", 0.0)),
                )
            )
        self.source, self.views_key, self.label_key = source, views_key, label_key
        self.size_key, self.output, self.record_params = size_key, output, record_params

    @staticmethod
    def _sample_params(g: dict, B: int, n: int, dev: torch.device) -> dict:
        """Photometric params per (sample, view) and the colour-op order."""
        params: dict = {"order": []}
        if g["color_jitter"] is not None:
            bri, con, sat, hue = g["color_jitter"]
            on = torch.rand(B, n, device=dev) < g["color_jitter_p"]

            def factor(amount, center):
                f = torch.empty(B, n, device=dev).uniform_(
                    center - amount, center + amount
                )
                return torch.where(on, f, torch.full_like(f, center))

            params.update(
                applied=on,
                brightness=factor(bri, 1.0).clamp_min(0),
                contrast=factor(con, 1.0).clamp_min(0),
                saturation=factor(sat, 1.0).clamp_min(0),
                hue=factor(hue, 0.0),
            )
            amounts = (bri, con, sat, hue)
            params["order"] = [
                op for op in torch.randperm(4).tolist() if amounts[op] > 0
            ]
        if g["grayscale_p"] > 0:
            params["grayscale"] = torch.rand(B, n, device=dev) < g["grayscale_p"]
        if g["hflip_p"] > 0:
            params["hflip"] = torch.rand(B, n, device=dev) < g["hflip_p"]
        return params

    @staticmethod
    def _apply_color(y: torch.Tensor, params: dict) -> torch.Tensor:
        """Colour jitter + grayscale on ``y`` (B, n, T, C, h, w) in [0, 1]."""
        B, n = y.shape[:2]
        bshape = (B, n, 1, 1, 1, 1)
        for op in params["order"]:
            if op == 0:
                y = (y * params["brightness"].view(bshape)).clamp_(0, 1)
            elif op == 1:
                mean = _gray(y).mean((-3, -2, -1), keepdim=True)
                f = params["contrast"].view(bshape)
                y = (f * y + (1 - f) * mean).clamp_(0, 1)
            elif op == 2:
                f = params["saturation"].view(bshape)
                y = (f * y + (1 - f) * _gray(y)).clamp_(0, 1)
            else:
                hsv = _rgb_to_hsv(y)
                hsv[..., 0, :, :] = (
                    hsv[..., 0, :, :] + params["hue"].view(B, n, 1, 1, 1)
                ) % 1.0
                y = _hsv_to_rgb(hsv).clamp_(0, 1)
        if "grayscale" in params:
            y = torch.where(params["grayscale"].view(bshape), _gray(y).expand_as(y), y)
        return y

    def _use_fused(self, x: torch.Tensor) -> bool:
        if self.backend == "torch":
            return False
        ok = x.is_cuda and x.dtype == torch.uint8 and x.shape[-3] == 3
        if ok:
            try:
                from . import _multicrop_triton  # noqa: F401
            except ImportError:
                ok = False
        if self.backend == "triton" and not ok:
            raise RuntimeError(
                "backend='triton' needs Triton and a uint8 RGB CUDA input"
            )
        return ok

    def forward(self, batch: dict) -> dict:
        """Replace ``batch[source]`` by the augmented views.

        Args:
            batch: Batch dict with ``batch[source]`` on the target device.

        Returns:
            The batch with ``batch[views_key]`` set (see ``output``).
        """
        from torchvision.ops import roi_align

        x = batch[self.source]
        is_clip = x.dim() == 5
        if not is_clip and x.dim() != 4:
            raise ValueError(
                f"expected (B, C, H, W) or (B, T, C, H, W), got {tuple(x.shape)}"
            )
        if not is_clip:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        is_uint8 = x.dtype == torch.uint8
        fused = self._use_fused(x)
        frames = None
        if not fused:
            frames = x.reshape(B * T, C, H, W)
            frames = frames.float().div_(255) if is_uint8 else frames.float()
        if self.size_key is not None and self.size_key in batch:
            hw = batch[self.size_key].to(x.device)
            vh, vw = hw[:, 0], hw[:, 1]
        else:
            vh = torch.full((B,), H, device=x.device)
            vw = torch.full((B,), W, device=x.device)

        label = batch.get(self.label_key)
        views, group_out, record = [], [], []
        frame_idx = torch.arange(B * T, device=x.device, dtype=torch.float32).view(
            B, 1, T
        )
        for g in self.groups:
            n, (oh, ow) = g["n_views"], g["size"]
            boxes = _sample_rrc_boxes(
                vh.repeat_interleave(n), vw.repeat_interleave(n), g["scale"], g["ratio"]
            ).view(B, n, 4)
            params = self._sample_params(g, B, n, x.device) if C == 3 else {"order": []}
            if fused:
                from ._multicrop_triton import crop_color

                y = crop_color(
                    x,
                    boxes,
                    (oh, ow),
                    jitter=params if params["order"] else None,
                    order=params["order"],
                    gray=params.get("grayscale"),
                    flip=params.get("hflip"),
                )
            else:
                rois = torch.cat(
                    (
                        frame_idx.expand(B, n, T).unsqueeze(-1),
                        boxes.view(B, n, 1, 4).expand(B, n, T, 4),
                    ),
                    -1,
                ).reshape(-1, 5)
                y = roi_align(frames, rois, (oh, ow), 1.0, -1, True).view(
                    B, n, T, C, oh, ow
                )
                if C == 3:
                    y = self._apply_color(y, params)
                if "hflip" in params:
                    y = torch.where(
                        params["hflip"].view(B, n, 1, 1, 1, 1), y.flip(-1), y
                    )
                if is_uint8:
                    y = y.mul_(255).round_().to(torch.uint8)
            if self.record_params:
                rec = {k: v for k, v in params.items() if k != "order"}
                rec["boxes"] = boxes
                record.append(rec)
            if not is_clip:
                y = y.squeeze(2)
            group_out.append(y)
            views.extend({"image": y[:, v], self.label_key: label} for v in range(n))

        del batch[self.source]
        batch[self.views_key] = group_out if self.output == "groups" else views
        if self.record_params:
            batch[type(self).__name__] = record
        return batch


# ---------------------------------------------------------------------------
# GPU JPEG decode
# ---------------------------------------------------------------------------


class GPUDecodeJPEG(nn.Module):
    """Decode JPEG bytes into a padded uint8 batch with nvJPEG (batched).

    Lets DataLoader workers ship encoded bytes only (no decode, no
    augmentation on the CPU); decoding then runs on the GPU inside
    ``Module.on_after_batch_transfer`` (or :class:`GPUPrefetcher`). Uses
    ``torchvision.io.decode_jpeg(list, device=...)``, whose batched nvJPEG
    path runs on the GPU's hardware JPEG engines where present (A100, H100,
    B200, ...) and on CUDA cores otherwise; on a CPU device it falls back to
    libjpeg.

    Most of the cost is host-side (bitstream parsing and per-image
    submission, about 25 us per image); nvJPEG releases the GIL meanwhile, so
    :class:`GPUPrefetcher` hides it behind training by running it in a
    background thread.

    Input ``batch[source]``: a sequence of ``B`` encoded images (``bytes`` or
    1-D uint8 CPU tensors), or for clips the layout ``default_collate``
    produces from per-sample lists of ``T`` frames: ``T`` sequences of ``B``.
    Images of different sizes are padded to the batch maximum by replicating
    their last row and column (so bilinear crops at the border match an
    unpadded image); the valid ``(H, W)`` per sample is written to
    ``batch[size_key]`` (all frames of a clip are assumed to share one size).

    Args:
        source: Key holding the encoded bytes.
        target: Output key: ``(B, 3, H, W)`` or ``(B, T, 3, H, W)`` uint8.
        size_key: Output key for valid sizes ``(B, 2)``; default
            ``f"{target}_size"``.
        device: Decode device; default CUDA if available, else CPU.
    """

    def __init__(
        self,
        source: str = "image_bytes",
        target: str = "image",
        size_key: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.source, self.target = source, target
        self.size_key = size_key or f"{target}_size"
        self._device_spec = device

    @property
    def device(self) -> torch.device:
        """Resolved decode device."""
        if self._device_spec is not None:
            return torch.device(self._device_spec)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _as_cpu_uint8(x) -> torch.Tensor:
        if torch.is_tensor(x):
            # nvJPEG reads the bitstream from host memory regardless of the
            # decode device (torchvision requirement).
            return x.cpu() if x.device.type != "cpu" else x
        # Zero-copy view of the (read-only) bytes: the decoder never writes
        # to its input, and copying 1-2k bitstreams per batch costs ~4 ms.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given buffer is not writable"
            )
            return torch.frombuffer(x, dtype=torch.uint8)

    @staticmethod
    def _pad_stack(imgs: list[torch.Tensor]) -> torch.Tensor:
        shapes = [tuple(i.shape[1:]) for i in imgs]
        groups: dict[tuple[int, int], list[int]] = {}
        for k, hw in enumerate(shapes):
            groups.setdefault(hw, []).append(k)
        if len(groups) == 1:
            return torch.stack(imgs)
        Hm = max(h for h, _ in groups)
        Wm = max(w for _, w in groups)
        out = imgs[0].new_empty(len(imgs), 3, Hm, Wm)
        # One stack + edge-replicate + scatter per distinct size: a per-image
        # loop costs ~3 launches per image (~40 ms for 1.5k frames).
        for (h, w), idx in groups.items():
            g = torch.stack([imgs[k] for k in idx])
            if w < Wm:
                g = torch.cat([g, g[..., w - 1 :].expand(-1, -1, -1, Wm - w)], dim=3)
            if h < Hm:
                g = torch.cat([g, g[:, :, h - 1 :].expand(-1, -1, Hm - h, -1)], dim=2)
            out.index_copy_(0, torch.tensor(idx, device=out.device), g)
        return out

    def forward(self, batch: dict) -> dict:
        """Decode ``batch[source]`` into ``batch[target]`` and ``batch[size_key]``.

        Args:
            batch: Batch dict holding encoded JPEG bytes.

        Returns:
            The batch with decoded uint8 images or clips.
        """
        from torchvision.io import ImageReadMode, decode_jpeg

        enc = batch[self.source]
        is_clip = len(enc) > 0 and isinstance(enc[0], (list, tuple))
        if is_clip:
            T, B = len(enc), len(enc[0])
            flat = [enc[t][b] for b in range(B) for t in range(T)]
        else:
            B, T = len(enc), 1
            flat = list(enc)
        imgs = decode_jpeg(
            [self._as_cpu_uint8(e) for e in flat],
            mode=ImageReadMode.RGB,
            device=self.device,
        )
        out = self._pad_stack(imgs)
        sizes = torch.tensor(
            [list(imgs[b * T].shape[1:]) for b in range(B)], device=out.device
        )
        out = out.view(B, T, *out.shape[1:]) if is_clip else out
        del batch[self.source]
        batch[self.target] = out
        batch[self.size_key] = sizes
        return batch


# ---------------------------------------------------------------------------
# Overlapped GPU preprocessing
# ---------------------------------------------------------------------------


class _PrefetchedBatch(dict):
    """A batch whose GPU transform already ran in :class:`GPUPrefetcher`.

    ``Module.on_after_batch_transfer`` skips its GPU transform for these.
    Lightning's device transfer rebuilds mappings with ``type(batch)(...)``,
    so the marker survives it.
    """


class GPUPrefetcher:
    """Iterate a DataLoader with H2D copy + GPU transform overlapped with training.

    While the training step for batch ``k`` runs, batch ``k + 1`` (up to
    ``depth`` batches ahead) is copied to the device and passed through
    ``transform`` on a separate CUDA stream. Without it, the GPU transform
    runs inside ``on_after_batch_transfer`` on the training stream, i.e. in
    series with every step. Enable for an spt :class:`DataModule` with
    ``DataModule(..., gpu_prefetch=True)``.

    With ``background=True`` (default) the copy and transform are issued from
    a background thread, so their host-side work (e.g. nvJPEG bitstream
    parsing in :class:`GPUDecodeJPEG`, Python launch overhead) also overlaps
    with the training loop instead of running between steps. GPU-side
    overlap needs free GPU resources: kernels that occupy every SM for a
    whole step (e.g. persistent megakernels) leave the side stream only the
    gaps, but the hardware JPEG engines are separate from the SMs.

    Args:
        loader: The (CPU) DataLoader.
        transform: Batch-level GPU transform (e.g. :class:`GPUCompose`,
            :class:`GPUMultiCrop`); ``None`` only prefetches the copy.
        device: Target device; default the current CUDA device (CPU if none,
            in which case the transform runs synchronously).
        depth: Batches prepared ahead of the one being consumed.
        background: Prepare batches in a background thread.

    Note:
        Attribute access falls through to ``loader`` (``dataset``,
        ``sampler``, ``batch_size``, ...). Lightning does not inject a
        distributed sampler into non-DataLoader iterables, so under DDP give
        the wrapped DataLoader a ``DistributedSampler`` yourself. With
        ``background=True`` random transforms draw from the shared torch RNG
        in another thread, so their draws interleave nondeterministically
        with the training step's.
    """

    def __init__(
        self,
        loader: Iterable,
        transform: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        depth: int = 1,
        background: bool = True,
    ):
        self.loader = loader
        if device is None:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.transform = transform
        if isinstance(transform, nn.Module):
            transform.to(self.device)
        self.depth = max(int(depth), 1)
        self.background = background

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name):
        if name == "loader":  # not yet set (e.g. during unpickling)
            raise AttributeError(name)
        return getattr(self.loader, name)

    def _prepare(self, batch):
        batch = _walk_tensors(batch, lambda t: t.to(self.device, non_blocking=True))
        if self.transform is not None:
            batch = self.transform(batch)
        return batch

    def _submit(self, it, stream):
        batch = next(it)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            batch = self._prepare(batch)
            event = torch.cuda.Event()
            event.record(stream)
        return batch, event

    def _consume(self, batch, event):
        current = torch.cuda.current_stream(self.device)
        current.wait_event(event)

        def keep_alive(t):
            # Allocated on the side stream, used on the training stream:
            # stop the caching allocator from reusing the memory early.
            if t.is_cuda:
                t.record_stream(current)
            return t

        _walk_tensors(batch, keep_alive)
        return _PrefetchedBatch(batch)

    def __iter__(self):
        if self.device.type != "cuda":
            for batch in self.loader:
                yield _PrefetchedBatch(self._prepare(batch))
            return
        stream = torch.cuda.Stream(self.device)
        it = iter(self.loader)
        if self.background:
            yield from self._iter_threaded(it, stream)
            return
        pending = deque()

        def submit() -> None:
            try:
                pending.append(self._submit(it, stream))
            except StopIteration:
                pass

        for _ in range(self.depth):
            submit()
        while pending:
            batch, event = pending.popleft()
            out = self._consume(batch, event)
            submit()
            yield out

    def _iter_threaded(self, it, stream):
        done = object()
        q: queue.Queue = queue.Queue(maxsize=self.depth)
        stop = threading.Event()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def produce() -> None:
            try:
                while not stop.is_set():
                    try:
                        item = self._submit(it, stream)
                    except StopIteration:
                        break
                    if not put(item):
                        return
                put(done)
            except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
                put(e)

        worker = threading.Thread(target=produce, name="GPUPrefetcher", daemon=True)
        worker.start()
        try:
            while True:
                item = q.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield self._consume(*item)
        finally:
            stop.set()
            worker.join()


__all__ = [
    "ToDevice",
    "GPUCompose",
    "GPUVideoSequential",
    "GPUMultiCrop",
    "GPUDecodeJPEG",
    "GPUPrefetcher",
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
