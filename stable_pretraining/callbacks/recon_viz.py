"""Reconstruction visualisation callback.

:class:`ReconViz` renders the predictions produced by
:class:`~stable_pretraining.callbacks.OnlineImageDecoder` (or any callback that
writes an image tensor into ``outputs``/``batch``) as a side-by-side
``target | reconstruction`` view and logs it through the trainer's media
logger. It is the rendering counterpart the ``OnlineImageDecoder`` docstring
defers to: the decoder writes reconstructions to ``outputs[f"{name}_preds"]``
and leaves rendering to "other callbacks (visualisers, video writers)".

It is deliberately decoupled from any particular model:

* Denormalisation stats are *injected* via ``pixel_mean`` / ``pixel_std`` rather
  than reached for on the module. When both are ``None`` denorm is a no-op and
  the tensors are only clamped to ``[0, 1]``.
* Two output modes: ``"grid"`` (a plain ``(B, C, H, W)`` batch → a
  ``make_grid``-style image, logged with ``logger.log_image``) and ``"video"``
  (a trajectory-contiguous flat layout → an MP4, logged with
  ``logger.log_video``). Each is guarded on logger capability so headless runs
  never crash.

One callback can render several decoders at once via the ``specs`` list.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Sequence, Tuple, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from .utils import get_data_from_batch_or_outputs, log_header, resolve_verbose

# A single spec: (preds_key_or_name, target_key, caption)
Spec = Tuple[str, str, Optional[str]]


class ReconViz(Callback):
    """Render decoder reconstructions as ``target | recon`` media each val epoch.

    Pairs one-to-one with :class:`OnlineImageDecoder`: point a spec at the
    decoder's ``name`` and ReconViz reads its ``f"{name}_preds"`` output, stitches
    each reconstruction next to its target, and logs the result.

    On ``on_validation_batch_end`` (first batch of the first loader) the first ``N`` rows
    of predictions and their matching targets are cached (detached, float, CPU).
    On ``on_validation_epoch_end`` the cache is denormalised, clamped to
    ``[0, 1]``, stitched (``target | separator | recon``) and emitted, then
    cleared.

    Register this callback after ``OnlineImageDecoder`` (or the callback that
    writes predictions) because both use ``on_validation_batch_end``.

    Args:
        specs: List of ``(preds_key_or_name, target_key, caption)`` tuples. The
            first element is resolved against ``outputs``/``batch`` by trying
            ``f"{key}_preds"`` first (the ``OnlineImageDecoder`` convention) then
            ``key`` verbatim, so both ``"recon"`` and ``"recon_preds"`` work.
            ``caption`` may be ``None``.
        mode: ``"grid"`` (default) treats the cached rows as a plain
            ``(B, C, H, W)`` batch and logs a grid image via ``log_image``.
            ``"video"`` treats them as a trajectory-contiguous flat layout
            (``n_items`` trajectories × ``seq_len`` transitions) and logs an MP4
            via ``log_video``.
        pixel_mean: Per-channel mean used at normalisation time, broadcastable to
            ``(C, 1, 1)`` (a length-``C`` sequence is accepted and reshaped).
            Reconstruction is ``x * pixel_std + pixel_mean``. Must be given
            together with ``pixel_std``; when both are ``None`` denorm is a no-op.
        pixel_std: Per-channel std used at normalisation time (see ``pixel_mean``).
        n_items: ``"grid"`` — number of images shown. ``"video"`` — number of
            trajectories (``n_traj``) shown side by side per frame.
        seq_len: ``"video"`` only — transitions per trajectory. Required when
            ``mode="video"``; the flat layout is assumed to be ``n_items *
            seq_len`` contiguous rows.
        fps: ``"video"`` only — frames per second of the emitted MP4.
        verbose: Enable per-emit info logging. ``None`` defers to the global
            verbosity setting.

    Example:
        Pair with :class:`OnlineImageDecoder` to log a ``target | recon`` grid::

            dec = spt.callbacks.OnlineImageDecoder(
                module=module,
                name="recon",
                input="embedding",
                target="image",
                image_shape=(3, 64, 64),
                embed_dim=768,
            )
            viz = spt.callbacks.ReconViz(
                [("recon", "image", "recon")],
                mode="grid",
            )
    """

    _SEPARATOR_WIDTH = 2
    _SEPARATOR_VALUE = 1.0  # white column between target and reconstruction
    _PADDING = 2
    _PADDING_VALUE = 1.0

    def __init__(
        self,
        specs: Sequence[Spec],
        mode: str = "grid",
        pixel_mean: Optional[Union[Sequence[float], torch.Tensor]] = None,
        pixel_std: Optional[Union[Sequence[float], torch.Tensor]] = None,
        n_items: int = 8,
        seq_len: Optional[int] = None,
        fps: int = 2,
        verbose: bool = None,
    ) -> None:
        super().__init__()
        if mode not in ("grid", "video"):
            raise ValueError(f"ReconViz: mode must be 'grid' or 'video'; got {mode!r}.")
        if (pixel_mean is None) != (pixel_std is None):
            raise ValueError(
                "ReconViz: pixel_mean and pixel_std must both be given or both "
                "be None (got "
                f"pixel_mean={'set' if pixel_mean is not None else 'None'}, "
                f"pixel_std={'set' if pixel_std is not None else 'None'})."
            )
        if mode == "video" and seq_len is None:
            raise ValueError("ReconViz: mode='video' requires seq_len to be set.")

        self.specs: List[Spec] = [self._normalize_spec(s) for s in specs]
        if not self.specs:
            raise ValueError("ReconViz: specs must not be empty.")
        self.mode = mode
        self.pixel_mean = self._as_broadcastable(pixel_mean)
        self.pixel_std = self._as_broadcastable(pixel_std)
        self.n_items = int(n_items)
        self.seq_len = None if seq_len is None else int(seq_len)
        self.fps = int(fps)
        for name in ("n_items", "fps", "seq_len"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"ReconViz: {name} must be positive; got {value}.")
        self.verbose = resolve_verbose(verbose)

        # {preds_name: {"preds": Tensor, "target": Tensor, "caption": str}}
        self._cache: dict = {}
        self._warned_missing: set = set()

        log_header("ReconViz")
        logging.info(f"  mode: {mode}")
        logging.info(f"  specs: {[(p, t) for p, t, _ in self.specs]}")
        logging.info(
            "  denorm: "
            + (
                "identity (no pixel_mean/pixel_std)"
                if self.pixel_mean is None
                else "x * pixel_std + pixel_mean"
            )
        )
        if mode == "grid":
            logging.info(f"  n_items: {self.n_items} images")
        else:
            logging.info(
                f"  n_items: {self.n_items} trajectories, "
                f"seq_len: {self.seq_len}, fps: {self.fps}"
            )

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _normalize_spec(spec: Spec) -> Spec:
        if len(spec) == 2:
            preds, target = spec
            caption = None
        elif len(spec) == 3:
            preds, target, caption = spec
        else:
            raise ValueError(
                "ReconViz: each spec must be (preds_key_or_name, target_key"
                f"[, caption]); got {spec!r}."
            )
        return (str(preds), str(target), caption)

    @staticmethod
    def _as_broadcastable(v) -> Optional[torch.Tensor]:
        if v is None:
            return None
        t = torch.as_tensor(v, dtype=torch.float32).detach().cpu().clone()
        if t.dim() == 1:  # (C,) -> (C, 1, 1)
            t = t.view(-1, 1, 1)
        return t

    def _read_preds(self, name: str, batch, outputs) -> Optional[torch.Tensor]:
        """Resolve a preds spec, trying ``{name}_preds`` before ``name``."""
        for key in (f"{name}_preds", name):
            if (outputs is not None and key in outputs) or (
                isinstance(batch, dict) and key in batch
            ):
                return get_data_from_batch_or_outputs(
                    key, batch, outputs, caller_name="ReconViz"
                )
        return None

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Invert normalisation (if stats given) and clamp to ``[0, 1]``."""
        if self.pixel_mean is not None and self.pixel_std is not None:
            x = x * self.pixel_std.to(x.dtype) + self.pixel_mean.to(x.dtype)
        return x.clamp(0.0, 1.0)

    def _stitch(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """``target | separator-column | pred`` along width. Inputs ``(C, H, W)``."""
        C, H, _ = target.shape
        sep = target.new_full((C, H, self._SEPARATOR_WIDTH), self._SEPARATOR_VALUE)
        return torch.cat([target, sep, pred], dim=2)

    def _make_grid(self, tiles: List[torch.Tensor], ncol: int) -> torch.Tensor:
        """Arrange equal-shaped ``(C, H, W)`` tiles into a padded grid canvas."""
        n = len(tiles)
        C, H, W = tiles[0].shape
        nrow = (n + ncol - 1) // ncol
        pad = self._PADDING
        canvas = tiles[0].new_full(
            (C, nrow * (H + pad) - pad, ncol * (W + pad) - pad),
            self._PADDING_VALUE,
        )
        for i, tile in enumerate(tiles):
            r, c = divmod(i, ncol)
            y, x = r * (H + pad), c * (W + pad)
            canvas[:, y : y + H, x : x + W] = tile
        return canvas

    @staticmethod
    def _to_hwc_uint8(img_chw: torch.Tensor):
        """``(C, H, W)`` float ``[0, 1]`` -> ``(H, W, 3)`` uint8 numpy array."""
        arr = (img_chw.clamp(0.0, 1.0) * 255).round().to(torch.uint8)
        arr = arr.permute(1, 2, 0).contiguous().cpu().numpy()
        if arr.shape[2] == 1:  # grayscale -> RGB for viewers / codecs
            arr = arr.repeat(3, axis=2)
        return arr

    @staticmethod
    def _loggers_with(trainer: Trainer, method: str) -> list:
        return [
            lgr
            for lgr in list(getattr(trainer, "loggers", []) or [])
            if callable(getattr(lgr, method, None))
        ]

    # -- hooks ---------------------------------------------------------------
    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if (
            batch_idx != 0
            or dataloader_idx != 0
            or trainer.global_rank != 0
            or getattr(trainer, "sanity_checking", False)
        ):
            return

        n_rows = self._rows_to_cache()
        for preds_name, target_key, caption in self.specs:
            preds = self._read_preds(preds_name, batch, outputs)
            if preds is None:
                self._warn_missing(preds_name)
                continue
            try:
                target = get_data_from_batch_or_outputs(
                    target_key, batch, outputs, caller_name="ReconViz"
                )
            except ValueError:
                self._warn_missing(target_key)
                continue

            if (
                not isinstance(preds, torch.Tensor)
                or not isinstance(target, torch.Tensor)
                or preds.ndim != 4
                or preds.shape != target.shape
                or any(size == 0 for size in preds.shape)
            ):
                logging.warning(
                    f"ReconViz[{preds_name}]: expected matching non-empty BCHW "
                    "prediction and target tensors; skipping."
                )
                continue

            self._cache[preds_name] = {
                "preds": preds[:n_rows].detach().float().cpu().clone(),
                "target": target[:n_rows].detach().float().cpu().clone(),
                "caption": caption if caption is not None else preds_name,
            }

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            self._cache.clear()
            return
        try:
            if trainer.global_rank != 0:
                return
            step = getattr(trainer, "global_step", None)
            for preds_name, entry in self._cache.items():
                preds = self._denormalize(entry["preds"])
                target = self._denormalize(entry["target"])
                tag = f"val/{preds_name}"
                caption = f"{entry['caption']} (epoch {trainer.current_epoch})"
                if self.mode == "grid":
                    self._emit_grid(trainer, tag, target, preds, step, caption)
                else:
                    self._emit_video(trainer, tag, target, preds, step, caption)
        finally:
            self._cache.clear()

    # -- frame construction (pure, no I/O) -----------------------------------
    def _build_grid_image(self, target: torch.Tensor, preds: torch.Tensor):
        """Stitch each ``(C, H, W)`` row as ``target | recon`` into one grid image.

        ``target`` and ``preds`` are aligned ``(B, C, H, W)`` batches: row ``i``
        of one is paired with row ``i`` of the other. Returns an ``(H, W, 3)``
        uint8 array.
        """
        tiles = [self._stitch(target[i], preds[i]) for i in range(preds.shape[0])]
        return self._to_hwc_uint8(self._make_grid(tiles, ncol=1))

    def _build_video_frames(
        self, target: torch.Tensor, preds: torch.Tensor, n_traj: int
    ) -> list:
        """Build one frame per transition from a trajectory-contiguous layout.

        ``target``/``preds`` are flat ``(n_traj * seq_len, C, H, W)`` batches in
        trajectory-contiguous order (row ``j * seq_len + t`` is trajectory ``j``
        at transition ``t``). Frame ``t`` stacks the ``n_traj`` ``target | recon``
        pairs at that transition. Returns a list of ``(H, W, 3)`` uint8 arrays.
        """
        rows = n_traj * self.seq_len
        C, H, W = preds.shape[1:]
        preds = preds[:rows].reshape(n_traj, self.seq_len, C, H, W)
        target = target[:rows].reshape(n_traj, self.seq_len, C, H, W)
        frames = []
        for t in range(self.seq_len):
            tiles = [self._stitch(target[j, t], preds[j, t]) for j in range(n_traj)]
            frames.append(self._to_hwc_uint8(self._make_grid(tiles, ncol=1)))
        return frames

    # -- emission ------------------------------------------------------------
    def _emit_grid(self, trainer, tag, target, preds, step, caption) -> None:
        loggers = self._loggers_with(trainer, "log_image")
        if not loggers:
            logging.warning(
                f"ReconViz[{tag}]: no logger exposes log_image; skipping grid."
            )
            return
        image = self._build_grid_image(target, preds)
        for lgr in loggers:
            try:
                lgr.log_image(key=tag, images=[image], step=step, caption=[caption])
            except Exception as e:  # never take a run down over a figure
                logging.warning(
                    f"ReconViz[{tag}]: {type(lgr).__name__}.log_image failed: {e}"
                )
        if self.verbose:
            logging.info(f"  {tag}: logged recon grid of {preds.shape[0]} images")

    def _emit_video(self, trainer, tag, target, preds, step, caption) -> None:
        loggers = self._loggers_with(trainer, "log_video")
        if not loggers:
            logging.warning(
                f"ReconViz[{tag}]: no logger exposes log_video; skipping video."
            )
            return

        n_available = preds.shape[0]
        n_traj = min(self.n_items, n_available // self.seq_len)
        if n_traj < 1:
            logging.warning(
                f"ReconViz[{tag}]: need at least seq_len={self.seq_len} rows for a "
                f"video but only cached {n_available}; skipping."
            )
            return
        frames = self._build_video_frames(target, preds, n_traj)

        try:
            import imageio
        except ImportError:
            logging.warning(f"ReconViz[{tag}]: imageio is unavailable; skipping video.")
            return

        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            imageio.mimsave(path, frames, fps=self.fps)
            for lgr in loggers:
                try:
                    lgr.log_video(
                        key=tag,
                        videos=[path],
                        step=step,
                        caption=[caption],
                        fps=self.fps,
                    )
                except Exception as e:  # never take a run down over a video
                    logging.warning(
                        f"ReconViz[{tag}]: {type(lgr).__name__}.log_video failed: {e}"
                    )
        except Exception as e:  # encoding failed (e.g. no ffmpeg)
            logging.warning(f"ReconViz[{tag}]: video encoding failed: {e}")
            return
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if self.verbose:
            logging.info(
                f"  {tag}: logged recon video ({n_traj} trajectories, "
                f"{self.seq_len} frames)"
            )

    # -- misc ----------------------------------------------------------------
    def _rows_to_cache(self) -> int:
        if self.mode == "video":
            return self.n_items * self.seq_len
        return self.n_items

    def _warn_missing(self, key: str) -> None:
        if key not in self._warned_missing:
            logging.warning(
                f"ReconViz: key {key!r} not found in batch or outputs; "
                "skipping this spec."
            )
            self._warned_missing.add(key)
