"""Inspection helpers for video encoder factories.

Provides:

- :func:`count_parameters` — total trainable parameter count.
- :func:`summarize` — param count + (optional) output shapes from a real
  forward pass.
- :func:`print_video_zoo` — print a reference table covering every video
  factory in this subpackage (params + output shapes).

The zoo print uses ``torch.device("meta")`` to count parameters of large
models without allocating real memory, so calling ``print_video_zoo`` on
a small dev box doesn't OOM on ``gigantic`` (~2B) presets.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .cosmos import (
    cosmos_base,
    cosmos_giant,
    cosmos_gigantic,
    cosmos_huge,
    cosmos_large,
    cosmos_small,
    cosmos_tiny,
)
from .magvit2 import (
    magvit2_base,
    magvit2_giant,
    magvit2_gigantic,
    magvit2_huge,
    magvit2_large,
    magvit2_small,
    magvit2_tiny,
)
from .predrnn import (
    predrnn_v2_base,
    predrnn_v2_huge,
    predrnn_v2_large,
    predrnn_v2_small,
    predrnn_v2_tiny,
)
from .videomamba import (
    videomamba_base,
    videomamba_giant,
    videomamba_gigantic,
    videomamba_huge,
    videomamba_large,
    videomamba_small,
    videomamba_tiny,
)


# Threshold below which we instantiate on the real device and run a forward
# pass to capture output shapes. Models above this size are param-counted on
# meta device only.
_REAL_INSTANCE_MAX_PARAMS = 500_000_000


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count.

    :param model: Any ``nn.Module``.
    :param trainable_only: Count only parameters with ``requires_grad=True``.
    """
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


def _meta_param_count(factory: Callable[..., nn.Module], **kwargs) -> int:
    """Instantiate on ``meta`` device and return the parameter count.

    No real memory is allocated, so this works on every preset up to
    ``gigantic``.
    """
    with torch.device("meta"):
        m = factory(**kwargs)
    return sum(p.numel() for p in m.parameters())


def summarize(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Inspect a built model.

    :param model: An already-instantiated encoder.
    :param input_shape: Optional input shape for a dry forward pass. If
        provided, the returned dict includes ``feature_shape`` and (when
        applicable) ``pooled_shape``.
    :return: ``{"params": int, "feature_shape": tuple?, "pooled_shape": tuple?}``.
    """
    info: Dict[str, Any] = {"params": count_parameters(model, trainable_only=False)}

    if input_shape is not None:
        model_was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                x = torch.zeros(*input_shape)
                out = model(x)
            if hasattr(out, "feature_map") and out.feature_map is not None:
                info["feature_shape"] = tuple(out.feature_map.shape)
            if hasattr(out, "pooled") and out.pooled is not None:
                info["pooled_shape"] = tuple(out.pooled.shape)
            if "feature_shape" not in info and torch.is_tensor(out):
                info["feature_shape"] = tuple(out.shape)
        finally:
            if model_was_training:
                model.train()

    return info


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    return f"{n / 1e3:.0f}K"


def _row(name: str, n_params: int, input_shape: Any, feat: Any, pooled: Any) -> str:
    in_s = str(input_shape) if input_shape is not None else "—"
    feat_s = str(feat) if feat is not None else "(skipped)"
    pool_s = str(pooled) if pooled is not None else "—"
    return (
        f"| {name:<22s} | {_format_params(n_params):>7s} | "
        f"{in_s:<20s} | {feat_s:<26s} | {pool_s:<12s} |"
    )


def print_video_zoo(
    input_hw: int = 64,
    input_t: int = 8,
    *,
    skip_forward_above: int = _REAL_INSTANCE_MAX_PARAMS,
) -> None:
    """Print a reference table for every video encoder factory.

    :param input_hw: Spatial side length of the dummy input ``(H = W)``. Kept
        small (default 64) so the forward pass is cheap.
    :param input_t: Number of frames in the dummy input. Default 8.
    :param skip_forward_above: Parameter-count threshold above which the
        model is param-counted only (no forward pass).

    Output is a Markdown-style table. Three columns:

    - **Model**: factory name.
    - **Params**: total parameter count.
    - **Feature shape**: shape of the encoder's main feature tensor for a
      ``(1, 3, T, H, H)`` input.
    - **Pooled**: shape of the pooled feature, if any.
    """
    # Per-family factory list + extra constructor kwargs needed (notably
    # VideoMamba which needs ``img_size``, ``num_frames``, ``patch_size``).
    zoo: List[Tuple[str, Callable[..., nn.Module], Dict[str, Any]]] = []

    for f in (
        magvit2_tiny,
        magvit2_small,
        magvit2_base,
        magvit2_large,
        magvit2_huge,
        magvit2_giant,
        magvit2_gigantic,
    ):
        zoo.append(("MAGVIT-v2", f, {}))
    for f in (
        predrnn_v2_tiny,
        predrnn_v2_small,
        predrnn_v2_base,
        predrnn_v2_large,
        predrnn_v2_huge,
    ):
        zoo.append(("PredRNN-v2", f, {"num_frames": input_t}))
    for f in (
        cosmos_tiny,
        cosmos_small,
        cosmos_base,
        cosmos_large,
        cosmos_huge,
        cosmos_giant,
        cosmos_gigantic,
    ):
        zoo.append(("Cosmos", f, {}))
    for f in (
        videomamba_tiny,
        videomamba_small,
        videomamba_base,
        videomamba_large,
        videomamba_huge,
        videomamba_giant,
        videomamba_gigantic,
    ):
        zoo.append(
            (
                "VideoMamba",
                f,
                {
                    "img_size": input_hw,
                    "num_frames": input_t,
                    "patch_size": (1, 16, 16),
                },
            )
        )

    input_shape = (1, 3, input_t, input_hw, input_hw)
    print(
        "| {:<22s} | {:>7s} | {:<20s} | {:<26s} | {:<12s} |".format(
            "Model", "Params", "Input", "Feature shape", "Pooled"
        )
    )
    print(
        "|"
        + "-" * 24
        + "|"
        + "-" * 9
        + "|"
        + "-" * 22
        + "|"
        + "-" * 28
        + "|"
        + "-" * 14
        + "|"
    )

    current_family = None
    for family, factory, kwargs in zoo:
        if family != current_family:
            current_family = family
            total_width = 24 + 9 + 22 + 28 + 14 + 5  # 5 column separators
            label = f"  *{family}*"
            print(f"|{label}" + " " * (total_width - len(label) - 1) + "|")

        n_params = _meta_param_count(factory, **kwargs)
        feat_shape: Any = None
        pooled_shape: Any = None
        if n_params <= skip_forward_above:
            model = factory(**kwargs)
            info = summarize(model, input_shape=input_shape)
            feat_shape = info.get("feature_shape")
            pooled_shape = info.get("pooled_shape")

        print(_row(factory.__name__, n_params, input_shape, feat_shape, pooled_shape))
