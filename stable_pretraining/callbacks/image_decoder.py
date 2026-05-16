"""Online image-reconstruction probe.

:class:`OnlineImageDecoder` trains an image decoder on frozen encoder
features alongside the main self-supervised objective. It is a thin
specialisation of :class:`~stable_pretraining.callbacks.OnlineProbe`: the
probe machinery (gradient detach, independent optimizer/scheduler, metric
logging, prediction-key routing) is inherited unchanged; this class only
adds the decoder construction, sensible defaults for image-regression
training, and shape-mismatch diagnostics.

Use it to visualise what an encoder captures by reconstructing the input
image (RGB, depth, segmentation logits, anything with a fixed channel
count) from its embedding.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch import LightningModule
from loguru import logger as logging

from ..backbone.decoders import build_image_decoder
from .probe import OnlineProbe
from .utils import log_header


class _ShapeCheckedMSELoss(nn.Module):
    """MSELoss with an explicit shape-mismatch guard.

    ``nn.MSELoss`` that raises a clear error when ``pred`` and ``target``
    disagree in shape. Torch's broadcasting will sometimes silently align
    ``(B, C, H, W)`` with ``(B, C, H, W-1)`` along a singleton, or it will
    raise a generic "size mismatch" without telling the user which key in
    the batch was the problem. We catch the mismatch up-front with an
    actionable message.
    """

    def __init__(self, name: str, input_key: str, target_key: str):
        super().__init__()
        self.mse = nn.MSELoss()
        self._name = name
        self._input_key = input_key
        self._target_key = target_key

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"[{self._name}] decoder output and target shape disagree: "
                f"pred={tuple(pred.shape)} (from input '{self._input_key}'), "
                f"target={tuple(target.shape)} (from key '{self._target_key}'). "
                f"Check that image_shape=(C,H,W) matches your target tensor."
            )
        return self.mse(pred, target)


class OnlineImageDecoder(OnlineProbe):
    """Online probe that decodes embeddings back to images.

    The decoder is constructed once at init time from ``image_shape``,
    ``embed_dim``, and (for ViT inputs) ``patch_size``. From there the
    callback behaves exactly like :class:`OnlineProbe`: features under
    ``input`` are detached and passed through the decoder, MSE is computed
    against ``target``, and the optimizer/scheduler are managed
    independently of the main model.

    The reconstruction is written to ``outputs[f"{name}_preds"]`` on every
    forward pass, so other callbacks (visualisers, video writers, custom
    metrics) can read it directly via
    :func:`~stable_pretraining.utils.get_data_from_batch_or_outputs`.
    For example, ``OnlineImageDecoder(name="recon", ...)`` exposes the
    reconstructed image at the key ``"recon_preds"``.

    Parameters
    ----------
    module
        The :class:`~stable_pretraining.LightningModule` being trained.
    name
        Unique identifier (used for logging, prediction key, metric
        namespace, and stored under ``pl_module.callbacks_modules[name]``).
    input
        Key in ``batch`` or ``outputs`` holding the embedding.
        ``(B, D)`` for CNN decoding, ``(B, P, D)`` for ViT decoding.
    target
        Key in ``batch`` or ``outputs`` holding the target image,
        shape ``(B, C, H, W)`` matching ``image_shape``.
    image_shape
        ``(channels, height, width)`` of the reconstruction target.
        Currently ``height == width`` (square images only).
    embed_dim
        Feature dimension ``D`` of the input embedding.
    kind
        ``"auto"`` (default), ``"cnn"``, or ``"vit"``. ``"auto"`` resolves
        to ``"vit"`` if ``patch_size`` is set, otherwise ``"cnn"``.
    patch_size
        Patch side length for the ViT decoder. Must divide ``image_shape[1]``.
    decoder_kwargs
        Extra kwargs forwarded to the underlying decoder
        (e.g. ``base_channels``, ``decoder_dim``, ``depth``).
    loss
        Reconstruction loss. Defaults to MSE with a shape-mismatch guard.
    metrics
        Defaults to :class:`torchmetrics.MeanSquaredError`.
    optimizer, scheduler, accumulate_grad_batches, gradient_clip_val,
    gradient_clip_algorithm, verbose
        Forwarded to :class:`OnlineProbe`. Optimizer defaults to
        ``Adam(lr=1e-3, weight_decay=0)`` (a more sensible choice for a
        conv/transformer decoder than the probe's default LARS).

    Example:
    -------
    Decode a pooled vector to a 64x64 RGB image::

        spt.callbacks.OnlineImageDecoder(
            module=module,
            name="recon_rgb",
            input="embedding",
            target="image",
            image_shape=(3, 64, 64),
            embed_dim=768,
        )

    Decode a token grid to a 96x96 depth map::

        spt.callbacks.OnlineImageDecoder(
            module=module,
            name="recon_depth",
            input="patch_tokens",
            target="depth",
            image_shape=(1, 96, 96),
            embed_dim=384,
            patch_size=8,  # triggers the ViT decoder
        )
    """

    def __init__(
        self,
        module: LightningModule,
        name: str,
        input: str,
        target: str,
        image_shape: Tuple[int, int, int],
        embed_dim: int,
        kind: str = "auto",
        patch_size: Optional[int] = None,
        decoder_kwargs: Optional[dict] = None,
        loss: Optional[callable] = None,
        metrics: Optional[Union[dict, tuple, list, torchmetrics.Metric]] = None,
        optimizer: Optional[Union[str, dict, partial, torch.optim.Optimizer]] = None,
        scheduler: Optional[
            Union[str, dict, partial, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = None,
        gradient_clip_algorithm: str = "norm",
        verbose: bool = None,
    ) -> None:
        if len(image_shape) != 3:
            raise ValueError(
                f"OnlineImageDecoder[{name}]: image_shape must be (C, H, W); "
                f"got {image_shape}."
            )
        C, H, W = image_shape
        self.image_shape = (int(C), int(H), int(W))
        self.embed_dim = int(embed_dim)
        self.kind = kind
        self.patch_size = patch_size

        log_header("OnlineImageDecoder")
        logging.info(f"  name: {name}")
        logging.info(f"  input key: {input!r} (embedding, D={embed_dim})")
        logging.info(f"  target key: {target!r} (image, shape=(C,H,W)={image_shape})")
        if patch_size is not None:
            P = (H // patch_size) ** 2
            logging.info(
                "  decoder kind: ViT (auto)"
                if kind == "auto"
                else f"  decoder kind: {kind}"
            )
            logging.info(
                f"  expected input shape: (B, P={P}, D={embed_dim}) "
                f"with patch_size={patch_size}, grid={H // patch_size}x{W // patch_size}"
            )
        else:
            logging.info(
                "  decoder kind: CNN (auto)"
                if kind == "auto"
                else f"  decoder kind: {kind}"
            )
            logging.info(f"  expected input shape: (B, D={embed_dim})")
        logging.info(f"  expected target shape: (B, {C}, {H}, {W})")

        decoder = build_image_decoder(
            embed_dim=self.embed_dim,
            image_shape=self.image_shape,
            kind=kind,
            patch_size=patch_size,
            decoder_kwargs=decoder_kwargs,
        )

        if loss is None:
            loss = _ShapeCheckedMSELoss(name=name, input_key=input, target_key=target)
        if metrics is None:
            metrics = torchmetrics.MeanSquaredError()
        if optimizer is None:
            optimizer = partial(torch.optim.Adam, lr=1e-3, weight_decay=0.0)

        super().__init__(
            module=module,
            name=name,
            input=input,
            target=target,
            probe=decoder,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            metrics=metrics,
            verbose=verbose,
        )
