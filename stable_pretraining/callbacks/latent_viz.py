"""Online latent space visualization callback with dimensionality reduction.

This callback learns a 2D projection of high-dimensional features while preserving
neighborhood structure using a contrastive loss between high-D and low-D similarities.
"""

from functools import partial
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from torch import Tensor

from ..utils.distance_metrics import compute_pairwise_distances_chunked

from .queue import OnlineQueue, find_or_create_queue_callback
from .utils import log_header
from ..optim.utils import create_optimizer
from ..optim.lr_scheduler import create_scheduler


class LatentViz(Callback):
    """Online latent visualization callback with neighborhood-preserving dimensionality reduction.

    This callback learns a 2D projection that preserves neighborhood structure from
    high-dimensional features. It uses a contrastive loss that attracts neighbors
    and repels non-neighbors in the 2D space.

    The loss function is:
        L = -∑_{ij} P_{ij} log Q_{ij} - ∑_{i,j ∈ Neg(i)} log(1 - Q_{ij})

    where:
        - P_{ij} is the high-D neighborhood graph (based on k-NN)
        - Q_{ij} is the similarity in the learned 2D space
        - Neg(i) is the set of negative samples for point i

    Args:
        module: The spt.Module being trained.
        name: Unique identifier for this callback instance.
        input: Key in batch dict containing input features to visualize.
        target: Optional key in batch dict containing labels for coloring plots.
            If None, points will be plotted without color coding.
        projection: The projection module to train (maps high-D to 2D). Can be:
            - nn.Module instance
            - callable that returns a module
            - Hydra config to instantiate
        queue_length: Size of the circular buffer for features.
        k_neighbors: Number of nearest neighbors for building P matrix.
        n_negatives: Number of negative samples per positive pair.
        optimizer: Optimizer configuration. If None, uses AdamW.
        scheduler: Learning rate scheduler configuration. If None, uses ConstantLR.
        accumulate_grad_batches: Number of batches to accumulate gradients.
        update_interval: Compute projection loss every N training batches (default: 10).
            Uses detached features queued by previous training batches.
        warmup_epochs: Number of epochs to wait before starting projection training (default: 0).
            Allows main model to stabilize before learning 2D projections.
        distance_metric: Metric for computing distances in high-D space.
        plot_interval: Interval (in epochs) for plotting 2D visualization.
        save_dir: Optional directory to save plots. If None, saves to 'latent_viz_{name}'.
        input_dim: Expected dimensionality of input features (for queue).
        verbose: Log attraction and repulsion separately; None uses global verbosity.

    Note:
        The callback owns its projection, optimizer, and scheduler; their state
        (including accumulated gradients) is saved in Lightning's callback state.
        Projection training runs in float32 on rank zero using that rank's queued,
        detached features. Validation visualizes features gathered from all ranks.
        It does not change the encoder loss, optimizer list, or global step.
        With no model optimizer, use a finite max_epochs rather than max_steps.
        Validation writes an NPZ file with ``coordinates`` and optional ``labels``;
        it needs at least one validation batch. Queues fill during training, so
        projection training starts only once at least two cached features exist.

    Example:
        >>> viz = spt.callbacks.LatentViz(
        ...     module,
        ...     name="latent",
        ...     input="embedding",
        ...     target="label",
        ...     projection=torch.nn.Linear(512, 2),
        ...     input_dim=512,
        ... )
        >>> trainer = pl.Trainer(callbacks=[viz])
    """

    def __init__(
        self,
        module: LightningModule,
        name: str,
        input: str,
        target: Optional[str],
        projection: torch.nn.Module,
        queue_length: int = 2048,
        k_neighbors: int = 15,
        n_negatives: int = 5,
        optimizer: Optional[Union[str, dict, partial, torch.optim.Optimizer]] = None,
        scheduler: Optional[
            Union[str, dict, partial, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        accumulate_grad_batches: int = 1,
        update_interval: int = 10,
        warmup_epochs: int = 0,
        distance_metric: Literal["euclidean", "cosine"] = "euclidean",
        plot_interval: int = 10,
        save_dir: Optional[str] = None,
        input_dim: Optional[Union[int, tuple, list]] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        for key, value in {
            "queue_length": queue_length,
            "k_neighbors": k_neighbors,
            "n_negatives": n_negatives,
            "update_interval": update_interval,
            "plot_interval": plot_interval,
            "accumulate_grad_batches": accumulate_grad_batches,
        }.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if queue_length < 2:
            raise ValueError("queue_length must be at least 2")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be nonnegative")
        if distance_metric not in ("euclidean", "cosine"):
            raise ValueError("distance_metric must be euclidean or cosine")
        super().__init__()
        self.name = name
        self.accumulate_grad_batches = accumulate_grad_batches
        self._optimizer_config = optimizer
        self._scheduler_config = scheduler
        self._optimizer = None
        self._scheduler = None
        self._restore_state = None
        self._accumulated_batches = 0

        self.input = input
        self.target = target
        self.queue_length = queue_length

        self.k_neighbors = k_neighbors
        self.n_negatives = n_negatives
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs
        self.distance_metric = distance_metric
        self.plot_interval = plot_interval
        self.save_dir = save_dir

        self._queue_dim = tuple(input_dim) if isinstance(input_dim, list) else input_dim
        if isinstance(input_dim, (list, tuple)):
            input_dim = int(np.prod(input_dim))
        self.input_dim = input_dim

        from .utils import resolve_verbose

        self._projection_config = projection
        self.verbose = resolve_verbose(verbose)

        self._input_queue = None
        self._target_queue = None
        # Construct factories without consuming the encoder's random stream.
        with torch.random.fork_rng(devices=[]):
            self.module = self.configure_model(module)
        if {id(p) for p in self.module.parameters()} & {
            id(p) for p in module.parameters()
        }:
            raise ValueError(
                "LatentViz projection must not share parameters with the model"
            )

        log_header("LatentViz")
        logging.info(f"  name: {name}")
        logging.info(f"  input: {input}")
        logging.info(
            f"  target: {target if target else 'None (no labels for coloring)'}"
        )
        logging.info(f"  queue_length: {queue_length}")
        logging.info(f"  k_neighbors: {k_neighbors}")
        logging.info(f"  negative_samples: {n_negatives}")
        logging.info(f"  update_interval: {update_interval} batches")
        logging.info(f"  warmup_epochs: {warmup_epochs}")
        logging.info(f"  accumulate_grad_batches: {accumulate_grad_batches}")

    def configure_model(self, pl_module: LightningModule) -> torch.nn.Module:
        """Build the independently owned projection.

        Args:
            pl_module: The training module (kept for API compatibility).

        Returns:
            The configured projection network.
        """
        if isinstance(self._projection_config, torch.nn.Module):
            projection_module = self._projection_config
        elif callable(self._projection_config):
            projection_module = self._projection_config()
        else:
            projection_module = instantiate(self._projection_config, _convert_="object")

        return projection_module

    def setup_optimizer(self, pl_module: LightningModule) -> torch.optim.Optimizer:
        """Create the projection optimizer.

        Args:
            pl_module: The training module (kept for API compatibility).

        Returns:
            AdamW by default, or the explicitly configured optimizer.
        """
        if self._optimizer_config is None:
            return torch.optim.AdamW(
                self.module.parameters(), lr=1e-3, weight_decay=1e-2
            )
        return create_optimizer(
            self.module.parameters(),
            self._optimizer_config,
            named_params=self.module.named_parameters(),
        )

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Acquire shared queues for detached features and optional labels.

        Args:
            trainer: The active trainer.
            pl_module: The training module.
            stage: The Lightning lifecycle stage.
        """
        if stage != "fit":
            return

        self._input_queue = find_or_create_queue_callback(
            trainer,
            self.input,
            self.queue_length,
            self._queue_dim,
            torch.float32 if self.input_dim is not None else None,
            gather_distributed=True,
            create_if_missing=True,
        )
        logging.info(f"  input queue: {self.input}")

        if self.target is not None:
            self._target_queue = find_or_create_queue_callback(
                trainer,
                self.target,
                self.queue_length,
                None,
                torch.long,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"  target queue: {self.target}")

    @property
    def state_key(self) -> str:
        """Identify independent visualization instances in checkpoints."""
        return f"{type(self).__name__}[name={self.name}]"

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize independent float32 training and restore callback state.

        Args:
            trainer: The active trainer.
            pl_module: The training module, whose device is now available.
        """
        if not trainer.is_global_zero:
            return
        self.module.to(device=pl_module.device, dtype=torch.float32)
        self._optimizer = self.setup_optimizer(pl_module)
        self._scheduler = (
            torch.optim.lr_scheduler.ConstantLR(self._optimizer, factor=1.0)
            if self._scheduler_config is None
            else create_scheduler(
                self._optimizer, self._scheduler_config, module=pl_module
            )
        )
        if self._restore_state is not None:
            state = self._restore_state
            self._accumulated_batches = state["accumulated_batches"]
            if state["optimizer"] is not None:
                self._optimizer.load_state_dict(state["optimizer"])
                self._scheduler.load_state_dict(state["scheduler"])
            for name, parameter in self.module.named_parameters():
                grad = state["gradients"].get(name)
                parameter.grad = None if grad is None else grad.to(parameter).clone()
            self._restore_state = None

    def state_dict(self) -> dict:
        """Return projection training state, including unfinished accumulation.

        Returns:
            State stored under this callback's unique checkpoint key.
        """
        if self._restore_state is not None:
            return self._restore_state
        return {
            "projection": self.module.state_dict(),
            "accumulated_batches": self._accumulated_batches,
            "optimizer": self._optimizer.state_dict()
            if self._optimizer is not None
            else None,
            "scheduler": self._scheduler.state_dict()
            if self._scheduler is not None
            else None,
            "gradients": {
                name: p.grad.detach().clone()
                for name, p in self.module.named_parameters()
                if p.grad is not None
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore weights now and defer optimizer restoration until device setup.

        Args:
            state_dict: State previously returned by this callback.
        """
        self.module.load_state_dict(state_dict["projection"])
        self._restore_state = state_dict

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Dict, batch_idx: int
    ) -> None:
        """Train on previous batches independently of queue callback ordering.

        Args:
            trainer: The active trainer.
            pl_module: The encoder module, used only for logging.
            batch: The current training batch.
            batch_idx: Index used for update and accumulation intervals.
        """
        if not trainer.is_global_zero or trainer.current_epoch < self.warmup_epochs:
            return
        self._accumulated_batches += 1
        queue = (
            OnlineQueue._shared_queues.get(self.input)
            if batch_idx % self.update_interval == 0
            else None
        )
        features = queue.get() if queue is not None else None
        if features is not None and len(features) >= 2:
            features = features[-self.queue_length :].detach()
            features = features.reshape(len(features), -1)
            features = features.to(next(self.module.parameters()))
            devices = [features.device.index] if features.is_cuda else []
            # Neither negative sampling nor projection dropout may change the
            # encoder's RNG stream. This optimizer never uses Lightning's scaler.
            with (
                torch.random.fork_rng(devices=devices),
                torch.enable_grad(),
                torch.autocast(device_type=features.device.type, enabled=False),
            ):
                self.module.train()
                # Custom projections may modify their inputs in place.
                loss = self._compute_loss(features, self.module(features.clone()))
                (loss / self.accumulate_grad_batches).backward()
            pl_module.log(
                f"train/{self.name}_loss",
                loss.detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
                batch_size=len(features),
            )
            if self.verbose:
                for component in ("attraction", "repulsion"):
                    pl_module.log(
                        f"train/{self.name}_{component}_loss",
                        getattr(self, f"_last_{component}_loss"),
                        on_step=True,
                        on_epoch=True,
                        sync_dist=False,
                        rank_zero_only=True,
                        batch_size=len(features),
                    )
        # Epoch boundaries and checkpoint resumes must not reset a partial window.
        if self._accumulated_batches >= self.accumulate_grad_batches:
            self._accumulated_batches = 0
            if any(p.grad is not None for p in self.module.parameters()):
                self._optimizer.step()
                self._scheduler.step()
                self._optimizer.zero_grad(set_to_none=True)

    def _compute_loss(
        self,
        x_high: Tensor,
        z_2d: Tensor,
    ) -> Tensor:
        """Compute the neighborhood-preserving loss.

        Loss = -∑_{ij} P_{ij} log Q_{ij} - ∑_{i,j ∈ Neg(i)} log(1 - Q_{ij})

        Args:
            x_high: High-dimensional features [N, D]
            z_2d: 2D projections [N, 2]
        """
        n_samples = x_high.size(0)
        if n_samples < 2:
            self._last_attraction_loss = self._last_repulsion_loss = 0.0
            return z_2d.sum() * 0
        # cdist does not support half/bfloat16 on CPU; fp32 also avoids
        # underflow in log probabilities under mixed precision.
        if x_high.dtype in (torch.float16, torch.bfloat16):
            x_high = x_high.float()
        if z_2d.dtype in (torch.float16, torch.bfloat16):
            z_2d = z_2d.float()
        chunk_size = 256 if n_samples > 1000 else -1
        distances = compute_pairwise_distances_chunked(
            x_high, x_high, metric=self.distance_metric, chunk_size=chunk_size
        )
        distances.fill_diagonal_(float("inf"))
        k = min(self.k_neighbors, n_samples - 1)
        neighbors = distances.topk(k=k, dim=1, largest=False).indices
        squared_distances = compute_pairwise_distances_chunked(
            z_2d, z_2d, metric="squared_euclidean", chunk_size=chunk_size
        )
        q = 1.0 / (2.0 + squared_distances)
        attraction = -q.gather(1, neighbors).clamp_min(1e-10).log().mean()
        repulsion = z_2d.sum() * 0
        if k < n_samples - 1:
            # Positive pairs and self-pairs must never also be repelled.
            candidates = torch.ones_like(distances)
            candidates.fill_diagonal_(0)
            candidates.scatter_(1, neighbors, 0)
            negatives = torch.multinomial(
                candidates, self.n_negatives * k, replacement=True
            )
            repulsion = -torch.log1p(-q.gather(1, negatives)).mean()
        self._last_attraction_loss = attraction.item()
        self._last_repulsion_loss = repulsion.item()
        return attraction + repulsion

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save the queued projection once per eligible validation epoch.

        Args:
            trainer: The active trainer.
            pl_module: The training module.
            outputs: Validation outputs.
            batch: The validation batch.
            batch_idx: Index within the validation loader.
            dataloader_idx: Index of the validation loader.
        """
        # Queue snapshots exist between epoch-start and epoch-end hooks,
        # regardless of whether their callbacks precede or follow this one.
        if (
            trainer.global_rank != 0
            or trainer.sanity_checking
            or batch_idx != 0
            or dataloader_idx != 0
            or trainer.current_epoch < self.warmup_epochs
            or trainer.current_epoch % self.plot_interval != 0
            or self._input_queue is None
        ):
            return
        features = self._input_queue.data
        if features is None or features.numel() == 0:
            return
        labels = self._target_queue.data if self._target_queue is not None else None
        if labels is not None and labels.numel() == 0:
            labels = None
        was_training = self.module.training
        self.module.eval()
        try:
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=next(self.module.parameters()).device.type,
                    enabled=False,
                ),
            ):
                features = features.reshape(len(features), -1)
                coordinates = self.module(
                    features.to(next(self.module.parameters()), copy=True)
                )
            self._plot_2d_embeddings(
                coordinates, labels, trainer.current_epoch, trainer
            )
        finally:
            self.module.train(was_training)

    def _plot_2d_embeddings(
        self, z_2d: Tensor, labels: Optional[Tensor], epoch: int, trainer: Trainer
    ) -> None:
        """Save 2D embeddings to file and log to experiment tracker."""
        import os

        # Save coordinates to NPZ file
        z_2d = z_2d.detach().cpu()
        # NumPy has no bfloat16 dtype, including for true-bfloat16 training.
        if z_2d.dtype == torch.bfloat16:
            z_2d = z_2d.float()
        z_2d_np = z_2d.numpy()
        labels_np = labels.cpu().numpy() if labels is not None else None
        if self.save_dir is not None:
            save_dir = self.save_dir
        else:
            save_dir = f"latent_viz_{self.name}"
        # Resolve relative paths against trainer.default_root_dir.
        # Prefer cache_dir when default_root_dir is CWD (outside Manager).
        if not os.path.isabs(save_dir):
            from pathlib import Path as _Path

            from stable_pretraining._config import get_config

            root = trainer.default_root_dir
            cfg = get_config()
            if cfg.cache_dir is not None and root == str(_Path().resolve()):
                root = cfg.cache_dir
            save_dir = os.path.join(root, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"epoch_{epoch:04d}.npz")
        save_data = {"coordinates": z_2d_np}
        if labels_np is not None:
            save_data["labels"] = labels_np
        np.savez_compressed(save_path, **save_data)

        logging.info(f"  saved 2D coordinates to {save_path}")

        logging.info(
            f"  2D coordinates saved to disk at epoch {epoch} "
            f"(visualization data in {save_path})"
        )

    @property
    def projection_module(self) -> torch.nn.Module:
        """Alias for self.module for backward compatibility."""
        return self.module
