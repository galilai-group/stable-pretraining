"""Era decisions, branch revival, and resumable state for DAE-Net training."""

import copy
import math
import random
from collections.abc import Iterable
from typing import Any

import lightning as pl
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.trainer.states import TrainerFn
from torch import Tensor

from ..backbone import DAENetBackbone
from ..utils.distributed import get_world_size


class DAENetRevival(pl.Callback):
    """Manage DAE-Net era scheduling.

    Args:
        era_length: Number of training epochs per era. Must be at least two.
            Defaults to 10.

    Note:
        Targets single-device FP32 or Linux single-node DDP training.
        For exact restart checks, use strategy="ddp_find_unused_parameters_true"
        so gradient bucket rebuilding does not change FP32 reduction order.
        DDP requires global exposure batch sharding, no gradient accumulation,
        and zero DataLoader workers. Usage and revival run on rank zero before
        broadcasting weights; each rank clears Adam history. Exact resumption
        targets completed epochs with the same world size, data and environment.
    """

    def __init__(self, era_length: int = 10) -> None:
        super().__init__()
        if type(era_length) is not int or era_length < 2:
            raise ValueError("era_length must be an integer and >= 2")
        self.era_length = era_length

        self.branch_ages: Tensor | None = None
        self._era_iou_sum: float = 0.0
        self._era_iou_batches: int = 0

        self.accepted_iou: float = 0.0
        self.accepted_epoch: int | None = None
        self._accepted_weights: dict[str, Tensor] | None = None
        self._pending_rng_state: dict[str, Any] | None = None
        self._world_size = 1
        self._rank_rng_states: list[dict[str, Any]] | None = None
        self._completed_epoch: int | None = None

    @staticmethod
    def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
        """Require an epoch checkpoint taken after controller finalization.

        Args:
            checkpoint: Full Lightning checkpoint, loaded on CPU.

        Raises:
            ValueError: If epoch-end controller state is not available.

        Note:
            Older checkpoints have no controller marker. Their Lightning loop
            progress must establish that the batch loop reached epoch end.
        """
        epoch = checkpoint.get("epoch")
        state = checkpoint.get("callbacks", {}).get("DAENetRevival", {})
        if "completed_epoch" in state:
            valid = type(epoch) is int and state["completed_epoch"] == epoch
        else:
            processed = (
                checkpoint.get("loops", {})
                .get("fit_loop", {})
                .get("epoch_progress", {})
                .get("total", {})
                .get("processed")
            )
            valid = type(epoch) is int and processed == epoch + 1
        if not valid:
            raise ValueError(
                "DAE-Net resume requires a completed-epoch checkpoint after controller finalization"
            )

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Validate the epoch boundary when restoring a training run.

        Args:
            trainer: Trainer restoring the checkpoint.
            pl_module: Module being restored.
            checkpoint: Full Lightning checkpoint.

        Raises:
            ValueError: If training resumes before controller finalization.
        """
        # Evaluation loads weights without continuing the training state machine.
        if trainer.state.fn == TrainerFn.FITTING:
            self.validate_checkpoint(checkpoint)

    def _broadcast_weights(self, backbone: DAENetBackbone) -> None:
        with torch.no_grad():
            for value in backbone.state_dict().values():
                dist.broadcast(value, src=0)

    def _distributed_revival(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        payload = [None]
        if trainer.is_global_zero:
            try:
                usage, count = self.collect_usage(
                    pl_module.backbone,
                    trainer.datamodule.usage_dataloader(),
                    pl_module.device,
                )
                self.revive_branches(pl_module.backbone, usage, count)
                payload[0] = {"error": None, "branch_ages": self.branch_ages}
            except Exception as error:
                payload[0] = {"error": f"{type(error).__name__}: {error}"}
        # Every peer sees the failure before waiting for network broadcasts.
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if result["error"] is not None:
            raise RuntimeError(
                f"DAE-Net usage/revival failed on rank zero: {result['error']}"
            )
        self.branch_ages = result["branch_ages"].clone()
        self._broadcast_weights(pl_module.backbone)

    def _distributed_finish_era(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, era_iou: float
    ) -> bool:
        """Share rank zero's decision or error before broadcasting model weights."""
        payload = [None]
        if trainer.is_global_zero:
            try:
                accepted = self.finish_era(
                    pl_module.backbone, era_iou, trainer.current_epoch
                )
                payload[0] = {
                    "error": None,
                    "accepted": accepted,
                    "accepted_iou": self.accepted_iou,
                    "accepted_epoch": self.accepted_epoch,
                }
            except Exception as error:
                payload[0] = {"error": f"{type(error).__name__}: {error}"}
        # A rank-zero exception must not leave peers waiting for tensor broadcasts.
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if result["error"] is not None:
            raise RuntimeError(
                f"DAE-Net era decision failed on rank zero: {result['error']}"
            )
        self.accepted_iou = result["accepted_iou"]
        self.accepted_epoch = result["accepted_epoch"]
        self._broadcast_weights(pl_module.backbone)
        if result["accepted"]:
            self._accepted_weights = {
                name: value.detach().cpu().clone()
                for name, value in pl_module.backbone.state_dict().items()
            }
        return result["accepted"]

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Restore checkpoint RNG after Lightning finishes training setup.

        Args:
            trainer: Trainer starting or resuming the training loop.
            pl_module: DAE-Net module on its training device.

        Raises:
            RuntimeError: If the saved CUDA RNG cannot be restored in the
                current device environment.

        Note:
            Restoration happens once. DataLoader generators must be independent
            of the global Torch RNG so iterator setup cannot shift revival draws.
        """
        world_size = getattr(trainer, "world_size", 1)
        if self._pending_rng_state is not None and world_size != self._world_size:
            raise RuntimeError("DAE-Net resume requires the same world size")
        self._world_size = world_size
        if self._rank_rng_states is not None and self._pending_rng_state is not None:
            state = self._rank_rng_states[trainer.global_rank]
            cuda_states = state["cuda_rng_states"]
            if (pl_module.device.type == "cuda") != (cuda_states is not None):
                raise RuntimeError("DAE-Net resume requires the same device type")
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if cuda_states is not None:
                torch.cuda.set_rng_state(cuda_states[0].cpu(), pl_module.device)
            random.setstate(state["python_rng_state"])
            np.random.set_state(state["numpy_rng_state"])
            for name, value in state["loader_rng_states"].items():
                loader = getattr(trainer.datamodule, name)
                loader.generator.set_state(value.cpu())
            self._pending_rng_state = None
            return
        if self._pending_rng_state is None:
            return

        cuda_states = self._pending_rng_state["cuda_rng_states"]
        if cuda_states is not None:
            if (
                not torch.cuda.is_available()
                or len(cuda_states) != torch.cuda.device_count()
            ):
                raise RuntimeError(
                    "Checkpoint RNG requires the same visible CUDA device count"
                )
        elif pl_module.device.type == "cuda":
            raise RuntimeError(
                "Cannot exactly resume CUDA training without CUDA RNG state"
            )

        # Setup may consume random numbers after load_state_dict has run.
        torch.set_rng_state(self._pending_rng_state["torch_rng_state"])
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if "python_rng_state" in self._pending_rng_state:
            random.setstate(self._pending_rng_state["python_rng_state"])
            np.random.set_state(self._pending_rng_state["numpy_rng_state"])
        self._pending_rng_state = None

    def weight_at(self, epoch: int) -> float:
        """Compute the regularization weight for a training epoch.

        Args:
            epoch: Non-negative, zero-based training epoch.

        Returns:
            Exponential warm up weights within the current era.

        Raises:
            ValueError: If epoch is not a non-negative integer.
        """
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")

        epoch_in_era = epoch % self.era_length
        progress = epoch_in_era / (self.era_length - 1)
        return math.exp(math.log(0.001) * (1.0 - progress))

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Apply warmup and perform revival at later era boundaries.

        Args:
            trainer: Trainer providing epoch, optimizer, and data module.
            pl_module: DAE-Net module whose backbone will be updated.

        Raises:
            RuntimeError: If the device count, optimizer count, or usage
                loader interface is unsupported.
            TypeError: If the optimizer is not a native Adam optimizer.

        Note:
            RNG restoration happens once in on_train_start, before any resumed
            branch revival. DDP loader seeds are derived from sampler seed/epoch.
        """
        self._completed_epoch = None
        if trainer.world_size > 1:
            from lightning.pytorch.strategies import DDPStrategy
            from ..data.dae_net import DAENetDistributedBatchSampler

            if (
                trainer.num_nodes != 1
                or not isinstance(trainer.strategy, DDPStrategy)
                or trainer.precision != "32-true"
                or trainer.accumulate_grad_batches != 1
            ):
                raise RuntimeError(
                    "DAE-Net DDP requires one node, FP32 and no gradient accumulation"
                )
            if not isinstance(
                trainer.train_dataloader.batch_sampler, DAENetDistributedBatchSampler
            ):
                raise RuntimeError(
                    "DAE-Net DDP requires DAENetDistributedBatchSampler and use_distributed_sampler=False"
                )
            loader = trainer.train_dataloader
            if loader.num_workers != 0 or loader.generator is None:
                raise RuntimeError(
                    "DAE-Net DDP requires num_workers=0 and an independent DataLoader generator"
                )
            # Lightning creates an iterator before on_train_start on resume.
            # Epoch-derived state removes that extra base-seed draw; with zero
            # workers it cannot affect dataset reads or query-offset sampling.
            loader.generator.manual_seed(
                loader.batch_sampler.sampler.seed + trainer.current_epoch
            )

        epoch = trainer.current_epoch
        self._era_iou_sum = 0.0
        self._era_iou_batches = 0
        pl_module.regularization_weight = self.weight_at(epoch)

        if epoch == 0 or epoch % self.era_length != 0:
            return

        if len(trainer.optimizers) != 1:
            raise RuntimeError("DAE-Net revival requires a single optimizer")

        optimizer = trainer.optimizers[0]
        if type(optimizer) is not torch.optim.Adam:
            raise TypeError("optimizer must be torch.optim.Adam")

        usage_loader_fn = getattr(trainer.datamodule, "usage_dataloader", None)

        if not callable(usage_loader_fn):
            raise RuntimeError(
                "DAE-Net revival requires data_module.usage_dataloader()"
            )

        if trainer.world_size > 1:
            self._distributed_revival(trainer, pl_module)
        else:
            usage, num_shapes = self.collect_usage(
                backbone=pl_module.backbone,
                loader=usage_loader_fn(),
                device=pl_module.device,
            )
            self.revive_branches(
                backbone=pl_module.backbone, usage=usage, num_shapes=num_shapes
            )
        self.reset_adam_state(optimizer)

    @torch.no_grad()
    def collect_usage(
        self,
        backbone: DAENetBackbone,
        loader: Iterable[dict[str, Any]],
        device: torch.device,
    ) -> tuple[Tensor, int]:
        """Sum branch existence over a single training-set traversal.

        Args:
            backbone: DAE-Net backbone already placed on device.
            loader: Voxel batches visiting each training record exactly once.
            device: Device on which to encode voxels.

        Returns:
            A CPU tensor of existence sums with shape [K], and the number
            of shapes processed.

        Raises:
            ValueError: If the loader yields no shapes or existence is nonfinite.

        Note:
            Run on the unwrapped backbone on rank zero in DDP. This method has
            no collectives and restores the previous train/eval mode.
        """
        usage = torch.zeros(backbone.branch_num, dtype=torch.float32, device=device)

        num_shapes = 0
        was_training = backbone.training

        try:
            backbone.eval()

            for batch in loader:
                voxels = batch["voxels"].to(device)
                existence = backbone.encode(voxels)["existence"]

                usage += existence.squeeze(-1).sum(dim=0)
                num_shapes += voxels.shape[0]
        finally:
            backbone.train(was_training)

        if num_shapes == 0:
            raise ValueError("usage loader must contain at least one shape")

        if not torch.isfinite(usage).all().item():
            raise ValueError("branch existence sums must be finite")

        return usage.cpu(), num_shapes

    @torch.no_grad()
    def revive_branches(
        self,
        backbone: DAENetBackbone,
        usage: Tensor,
        num_shapes: int,
    ) -> list[int]:
        """Revive dead branches and then the oldest remaining branch.

        Args:
            backbone: DAE-Net backbone whose branches will be reset.
            usage: Summed existence per branch, with shape [K].
            num_shapes: Number of shapes included in usage.

        Returns:
            Branch indices reset, in execution order.

        Raises:
            ValueError: If the shape count, usage, or age dimensions are invalid.

        Note:
            Call once at each era boundary after era zero. Adam state reset
            and checkpoint persistence are handled separately.
        """
        if type(num_shapes) is not int or num_shapes <= 0:
            raise ValueError("num_shapes must be a positive integer")
        if usage.shape != (backbone.branch_num,):
            raise ValueError("usage must have shape [branch_num]")
        if not torch.isfinite(usage).all().item():
            raise ValueError("usage must be finite")

        if self.branch_ages is None:
            self.branch_ages = torch.zeros(
                backbone.branch_num,
                dtype=torch.long,
            )

        elif self.branch_ages.shape != (backbone.branch_num,):
            raise ValueError("branch_ages must have shape [branch_num]")

        self.branch_ages += 1
        revived: list[int] = []

        dead_indices = (usage < 0.1 * num_shapes).nonzero(as_tuple=True)[0].tolist()

        for branch_id in dead_indices:
            backbone.reset_branch(branch_id)
            self.branch_ages[branch_id] = 0
            revived.append(branch_id)

        oldest = int(self.branch_ages.argmax().item())
        if self.branch_ages[oldest].item() > 0:
            backbone.reset_branch(oldest)
            self.branch_ages[oldest] = 0
            revived.append(oldest)

        return revived

    def reset_adam_state(self, optimizer: torch.optim.Optimizer) -> None:
        """Clear Adam history for every parameter in the optimizer.

        Args:
            optimizer: The existing Adam optimizer used for DAE-Net training.

        Raises:
            TypeError: If optimizer is not a torch.optim.Adam instance.

        Note:
            Parameter values, parameter groups, and optimizer identity are
            preserved. Pending gradients are cleared at the era boundary.
        """
        if type(optimizer) is not torch.optim.Adam:
            raise TypeError("optimizer must be torch.optim.Adam")

        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()

    @torch.no_grad()
    def batch_reconstruction_iou(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        distributed: bool = False,
    ) -> float:
        """Compute occupancy IoU pooled across a training batch.

        Args:
            prediction: Maximum branch occupancy with shape [B, Q, 1].
            target: Binary occupancy targets with the same shape.
            distributed: Pool counts across all training ranks before division.

        Returns:
            Intersection divided by union over all batch entries.

        Raises:
            ValueError: If shapes differ, values are nonfinite, or union is zero.

        Note:
            This is a pooled batch metric, not a mean of per-shape IoUs.
        """
        valid = prediction.shape == target.shape and (
            torch.isfinite(prediction).all().item()
            and torch.isfinite(target).all().item()
        )
        if distributed and get_world_size() > 1:
            valid_tensor = prediction.new_tensor(int(valid))
            dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
            valid = bool(valid_tensor.item())
        if not valid:
            raise ValueError(
                "prediction and target shapes must match and values must be finite"
            )

        predicted_inside = prediction > 0.5
        target_inside = target > 0.5

        intersection = (predicted_inside & target_inside).sum()
        union = (predicted_inside | target_inside).sum()
        if distributed and get_world_size() > 1:
            counts = torch.stack((intersection, union))
            dist.all_reduce(counts)
            intersection, union = counts.unbind()

        if union.item() == 0:
            raise ValueError("occupancy IoU is undefined for an empty union")

        return (intersection.float() / union.float()).item()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, Tensor],
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Accumulate batch IoU during the final epoch of each era.

        Args:
            trainer: Trainer providing the current epoch.
            pl_module: Module being trained.
            outputs: Training outputs containing maximum occupancy predictions.
            batch: Input batch containing occupancy targets.
            batch_idx: Index of the completed training batch.
        """
        if trainer.current_epoch % self.era_length != self.era_length - 1:
            return

        batch_iou = self.batch_reconstruction_iou(
            prediction=outputs["occupancy_max"],
            target=batch["occupancy"],
            distributed=trainer.world_size > 1,
        )
        self._era_iou_sum += batch_iou
        self._era_iou_batches += 1

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log the reconstruction score at the end of each era.

        Args:
            trainer: Trainer providing the current epoch.
            pl_module: Module used to log the era score.

        Raises:
            RuntimeError: If the final epoch supplied no batch IoUs or rank zero
                could not complete the distributed era decision.
        """
        if trainer.current_epoch % self.era_length != self.era_length - 1:
            self._finalize_epoch_state(trainer, pl_module)
            return

        if self._era_iou_batches == 0:
            raise RuntimeError("Cannot score an era without training batches")

        era_iou = self._era_iou_sum / self._era_iou_batches

        if trainer.world_size > 1:
            accepted = self._distributed_finish_era(trainer, pl_module, era_iou)
        else:
            accepted = self.finish_era(
                backbone=pl_module.backbone,
                era_iou=era_iou,
                epoch=trainer.current_epoch,
            )

        pl_module.log(
            "fit/era_reconstruction_iou",
            era_iou,
            on_step=False,
            on_epoch=True,
        )

        pl_module.log(
            "fit/era_accepted",
            float(accepted),
            on_step=False,
            on_epoch=True,
        )
        self._finalize_epoch_state(trainer, pl_module)

    def _finalize_epoch_state(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Capture distributed RNG before marking this epoch resumable."""
        if trainer.world_size == 1:
            self._completed_epoch = trainer.current_epoch
            return
        numpy_state = np.random.get_state()
        state = {
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": [torch.cuda.get_rng_state(pl_module.device)]
            if pl_module.device.type == "cuda"
            else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": (
                numpy_state[0],
                numpy_state[1].tolist(),
                *numpy_state[2:],
            ),
            "loader_rng_states": {
                name: loader.generator.get_state()
                for name in ("train", "val", "test", "usage_loader")
                if (loader := getattr(trainer.datamodule, name, None)) is not None
                and getattr(loader, "generator", None) is not None
            },
        }
        self._world_size = trainer.world_size
        self._rank_rng_states = [None] * trainer.world_size
        dist.all_gather_object(self._rank_rng_states, state)
        self._completed_epoch = trainer.current_epoch

    @torch.no_grad()
    def finish_era(
        self,
        backbone: DAENetBackbone,
        era_iou: float,
        epoch: int,
    ) -> bool:
        """Accept the current network or restore the accepted network.

        Args:
            backbone: Network whose parameters and buffers are managed.
            era_iou: Mean batch IoU from the final epoch of the era.
            epoch: Zero-based epoch that produced the score.

        Returns:
            True if the current network was accepted, otherwise False.

        Raises:
            ValueError: If the score or epoch is invalid.
            RuntimeError: If rejection occurs without an accepted snapshot.

        Note:
            Rejection restores network state only. Branch ages, optimizer
            state, RNG state, and training progress are not restored.
        """
        if not math.isfinite(era_iou) or not 0.0 <= era_iou <= 1.0:
            raise ValueError("era_iou must be finite and within [0, 1]")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")

        # Match upstream's strict 5% reconstruction-error threshold.
        accepted = (1.0 - era_iou) < 1.05 * (1.0 - self.accepted_iou)

        if accepted:
            snapshot = {
                name: value.detach().cpu().clone()
                for name, value in backbone.state_dict().items()
            }
            self._accepted_weights = snapshot
            self.accepted_iou = era_iou
            self.accepted_epoch = epoch
        else:
            if self._accepted_weights is None:
                raise RuntimeError("Cannot roll back without accepted weights")

            backbone.load_state_dict(self._accepted_weights, strict=True)

        return accepted

    def state_dict(self) -> dict[str, Any]:
        """Return an independent snapshot of controller state.

        Returns:
            Controller state with tensor copies stored on CPU.

        Note:
            No collectives execute here. In DDP, schema version 2 includes
            per-rank Torch CPU/current-device CUDA, Python, NumPy and loader RNG
            snapshots gathered after epoch-end era decisions. Lightning saves
            current model, optimizer, and training progress separately.
        """
        rng_state = self._pending_rng_state
        extra = {}
        if self._rank_rng_states is not None:
            rng_state = self._rank_rng_states[0]
            extra = {
                "schema_version": 2,
                "world_size": self._world_size,
                "rank_rng_states": copy.deepcopy(self._rank_rng_states),
            }
        if rng_state is None:
            numpy_state = np.random.get_state()
            rng_state = {
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": (
                    numpy_state[0],
                    numpy_state[1].tolist(),
                    *numpy_state[2:],
                ),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_initialized()
                    else None
                ),
            }
        return {
            **extra,
            "completed_epoch": self._completed_epoch,
            **{
                name: copy.deepcopy(rng_state[name])
                for name in ("python_rng_state", "numpy_rng_state")
                if name in rng_state
            },
            "torch_rng_state": rng_state["torch_rng_state"].clone(),
            "cuda_rng_states": (
                None
                if rng_state["cuda_rng_states"] is None
                else [state.clone() for state in rng_state["cuda_rng_states"]]
            ),
            "era_length": self.era_length,
            "branch_ages": (
                None
                if self.branch_ages is None
                else self.branch_ages.detach().cpu().clone()
            ),
            "era_iou_sum": self._era_iou_sum,
            "era_iou_batches": self._era_iou_batches,
            "accepted_iou": self.accepted_iou,
            "accepted_epoch": self.accepted_epoch,
            "accepted_weights": (
                None
                if self._accepted_weights is None
                else {
                    name: value.detach().cpu().clone()
                    for name, value in self._accepted_weights.items()
                }
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore controller state produced by state_dict.

        Args:
            state_dict: Saved controller state produced by state_dict().

        Raises:
            ValueError: If the saved era length differs from the controller.
            KeyError: If a required field is missing.

        Note:
            Stages RNG restoration for on_train_start. Current network and
            optimizer state are restored separately by Lightning. Checkpoints
            must include the Torch RNG fields for exact resumption.
        """
        saved_era_length = state_dict["era_length"]
        if type(saved_era_length) is not int or saved_era_length != self.era_length:
            raise ValueError("Checkpoint era_length does not match the controller")

        world_size = state_dict.get("world_size", 1)
        rank_states = state_dict.get("rank_rng_states")
        if state_dict.get("schema_version", 1) not in (1, 2):
            raise ValueError("Unsupported DAE-Net checkpoint schema")
        if world_size > 1:
            if (
                state_dict.get("schema_version") != 2
                or not isinstance(rank_states, list)
                or len(rank_states) != world_size
            ):
                raise ValueError("Missing per-rank RNG states")
            for state in rank_states:
                for name in (
                    "torch_rng_state",
                    "cuda_rng_states",
                    "python_rng_state",
                    "numpy_rng_state",
                    "loader_rng_states",
                ):
                    if name not in state:
                        raise ValueError(f"Missing per-rank RNG field: {name}")

        restored = {
            name: state_dict[name]
            for name in (
                "branch_ages",
                "era_iou_sum",
                "era_iou_batches",
                "accepted_iou",
                "accepted_epoch",
                "accepted_weights",
                "torch_rng_state",
                "cuda_rng_states",
            )
        }

        ages = restored["branch_ages"]
        restored["branch_ages"] = None if ages is None else ages.detach().cpu().clone()

        weights = restored["accepted_weights"]
        restored["accepted_weights"] = (
            None
            if weights is None
            else {name: value.detach().cpu().clone() for name, value in weights.items()}
        )

        pending_rng_state = {
            **{
                name: copy.deepcopy(state_dict[name])
                for name in ("python_rng_state", "numpy_rng_state")
                if name in state_dict
            },
            "torch_rng_state": restored["torch_rng_state"].detach().cpu().clone(),
            "cuda_rng_states": (
                None
                if restored["cuda_rng_states"] is None
                else [
                    state.detach().cpu().clone()
                    for state in restored["cuda_rng_states"]
                ]
            ),
        }

        self.branch_ages = restored["branch_ages"]
        self._era_iou_sum = restored["era_iou_sum"]
        self._era_iou_batches = restored["era_iou_batches"]
        self.accepted_iou = restored["accepted_iou"]
        self.accepted_epoch = restored["accepted_epoch"]
        self._accepted_weights = restored["accepted_weights"]
        self._pending_rng_state = pending_rng_state
        self._completed_epoch = state_dict.get("completed_epoch")
        self._world_size = world_size
        self._rank_rng_states = copy.deepcopy(rank_states)
