"""Learning From Randomness (LFR): predict frozen random projections of the data.

References:
    Sui et al. "Self-supervised Representation Learning from Random Data
    Projectors." ICLR 2024. https://arxiv.org/abs/2310.07756

Example::

    import lightning as pl
    import stable_pretraining as spt

    model = spt.LFR(encoder_name="resnet18", low_resolution=True, num_projectors=40)
    trainer = pl.Trainer(max_epochs=model.max_epochs_for(cycles=400))

See ``benchmarks/cifar10/lfr-resnet18.py`` for the paper's CIFAR-10 setup.
"""

import math
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.forward import lfr
from stable_pretraining.losses import BatchwiseBarlowTwinsLoss


def beta_init_(module: nn.Module) -> nn.Module:
    """Draw every ``Conv2d`` weight from a Beta(0.5, 0.5) rescaled to ``[-1, 1]``.

    The U-shaped distribution pushes weights toward ``-1`` and ``1``, which the
    paper uses to obtain diverse edge-like random image features.

    Args:
        module: Network whose convolutions are re-initialized in place.

    Returns:
        The same module, for chaining.
    """
    beta = torch.distributions.Beta(torch.tensor(0.5), torch.tensor(0.5))
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d):
                layer.weight.copy_(2 * beta.sample(layer.weight.shape) - 1)
    return module


def weight_dropout_(module: nn.Module, rate: float = 0.4) -> nn.Module:
    """Permanently zero a random ``rate`` fraction of each linear/conv weight.

    This is DropConnect applied once at initialization; the projector is frozen
    afterwards, so the same weights stay zero for the whole run.

    Args:
        module: Network whose ``Linear``/``Conv1d``/``Conv2d`` weights are sparsified.
        rate: Fraction of entries set to zero in each weight tensor.

    Returns:
        The same module, for chaining.
    """
    if not 0 <= rate < 1:
        raise ValueError("rate must be in [0, 1)")
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weight = layer.weight.view(-1)
                weight[
                    torch.randperm(weight.numel())[: round(rate * weight.numel())]
                ] = 0
    return module


def random_image_projector(dim: int, in_channels: int = 3) -> nn.Module:
    """Build the paper's frozen four-layer CNN image target.

    Channels ``in -> 8 -> 16 -> 32 -> 32`` with ReLU, max-pooling after every
    two convolutions and a linear map to ``dim``. As in the authors' code,
    :func:`weight_dropout_` runs first and :func:`beta_init_` then redraws the
    convolutions, so the convolutions are dense and the Linear map keeps 40%
    zeros. The final
    adaptive pooling is the identity for ``32 x 32`` inputs and lets the same
    projector accept larger images.

    Args:
        dim: Output (target) dimension.
        in_channels: Number of input image channels.

    Returns:
        A randomly initialized projector mapping ``[B, C, H, W]`` to ``[B, dim]``.
    """
    projector = nn.Sequential(
        nn.Conv2d(in_channels, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(8),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, dim),
    )
    # Same order as the authors' build_target: dropout, then Beta init, which
    # overwrites the convolutions, so only the Linear layer stays sparse.
    return beta_init_(weight_dropout_(projector))


def predictor_mlp(in_dim: int, out_dim: int, hidden_dim: int = 256) -> nn.Module:
    """Build the paper's four-layer ReLU prediction head.

    Args:
        in_dim: Encoder embedding width.
        out_dim: Width of the projector output to predict.
        hidden_dim: Width of the three hidden layers.

    Returns:
        A trainable MLP mapping ``[B, in_dim]`` to ``[B, out_dim]``.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _subsample(sample: torch.Tensor, max_samples: int) -> torch.Tensor:
    """Deterministically keep at most ``max_samples`` rows of ``sample``."""
    if sample.shape[0] <= max_samples:
        return sample
    keep = torch.randperm(sample.shape[0], generator=torch.Generator().manual_seed(0))
    return sample[keep[:max_samples].to(sample.device)]


def _signature(projector: nn.Module, sample: torch.Tensor) -> torch.Tensor:
    """Unit-norm, flattened ``B x B`` cosine-similarity matrix of a projector."""
    projector = projector.to(sample.device)
    was_training = projector.training
    projector.eval()
    with torch.no_grad():
        output = projector(sample)
    projector.train(was_training)
    if output.ndim != 2 or output.shape[0] != sample.shape[0]:
        raise ValueError("each projector must return [B, D]")
    normalized = F.normalize(output.float(), dim=1)
    return F.normalize((normalized @ normalized.T).flatten(), dim=0)


def _greedy_dpp(signatures: torch.Tensor, count: int) -> list[int]:
    """Greedy MAP selection of a DPP whose kernel is the signatures' Gram matrix."""
    kernel = (signatures @ signatures.T).double()
    # Signatures have unit norm, so every candidate ties at the first step;
    # start from the first candidate instead of breaking the tie by rounding.
    selected: list[int] = [0]
    remaining = set(range(1, signatures.shape[0]))
    for _ in range(count - 1):
        chosen = kernel[selected][:, selected]
        cross = kernel[:, selected]
        conditional = cross @ torch.linalg.pinv(chosen) * cross
        scores = kernel.diagonal() - conditional.sum(dim=1)
        index = max(remaining, key=lambda i: (scores[i].item(), -i))
        selected.append(index)
        remaining.remove(index)
    return selected


def select_diverse_projectors(
    projectors: Sequence[nn.Module],
    sample: torch.Tensor,
    count: int,
    max_samples: int = 1000,
) -> list[int]:
    """Greedily select the most diverse projectors, as the paper's DPP step.

    Each candidate is summarized by its normalized ``B x B`` cosine-similarity
    matrix on ``sample``; the kernel between candidates is the cosine
    similarity of those summaries. Greedy conditional-variance maximization is
    the standard MAP approximation of a determinantal point process and matches
    the authors' ``fast-map-dpp`` routine.

    Args:
        projectors: Randomly initialized candidate projectors.
        sample: One unlabeled batch.
        count: Number of projectors to retain.
        max_samples: Rows of ``sample`` to use at most. Signatures grow with
            the square of this; the authors cap it at 1000.

    Returns:
        Candidate indices in selection order.
    """
    if not 1 <= count <= len(projectors):
        raise ValueError("count must be between 1 and len(projectors)")
    if sample.shape[0] < 2:
        raise ValueError("selection sample needs at least two examples")
    sample = _subsample(sample, max_samples)
    signatures = torch.stack([_signature(p, sample) for p in projectors])
    return _greedy_dpp(signatures, count)


def _output_dim(module: nn.Module, sample: torch.Tensor) -> int:
    """Width of ``module``'s output on ``sample``, run on the module's device."""
    parameter = next(module.parameters(), None)
    if parameter is not None:
        sample = sample.to(parameter.device)
    return module(sample).shape[-1]


@contextmanager
def _frozen(module: nn.Module) -> Iterator[None]:
    """Temporarily disable gradients for ``module``, restoring each flag after."""
    flags = [p.requires_grad for p in module.parameters()]
    module.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)


class LFR(Module):
    """Learning From Randomness :cite:`sui2024lfr`.

    An encoder learns from **one unaugmented view** by predicting the outputs
    of several frozen, randomly initialized projectors, each through its own
    predictor head, under the batch-wise Barlow Twins loss.

    Training alternates as in Algorithm 1 of the paper: each *cycle* is one
    encoder epoch (predictors frozen) followed by ``predictor_epochs``
    predictor epochs (encoder frozen, its BatchNorm statistics included). Use
    :meth:`max_epochs_for` to size the Trainer. Both phases share one
    optimizer with separate encoder and predictor parameter groups; the frozen
    half has no gradients, so its weights, momentum and weight decay are
    untouched, while online probes and other callbacks keep training every
    step.

    Args:
        encoder_name: timm model name or a prebuilt encoder returning
            ``[B, E]`` embeddings.
        projectors: Custom frozen target networks returning ``[B, D_k]``. By
            default, ``num_projectors * candidate_ratio`` instances of
            :func:`random_image_projector` are created.
        predictors: Custom heads, one per retained projector, mapping
            ``[B, E]`` to ``[B, D_k]``. Defaults to :func:`predictor_mlp`.
        num_projectors: Number of projectors (and predictors) kept. The paper
            uses 40 on CIFAR-10 and 6 on most other datasets.
        target_dim: Output width of the default projectors.
        predictor_hidden_dim: Hidden width of the default predictors.
        predictor_epochs: Predictor epochs after each encoder epoch.
        lambd: Off-diagonal weight of the batch-wise Barlow Twins loss.
        candidate_ratio: When ``> 1``, draw ``candidate_ratio`` times more
            default projectors and keep the most diverse ones (the paper uses
            10). Requires ``selection_sample``.
        selection_sample: Unlabeled batch for diversity selection, needed
            whenever more projectors than predictors are available. Also used
            to infer default predictor widths; without it, a dummy
            ``3``-channel image is used, so pass ``predictors`` or a sample
            for non-image data.
        low_resolution: Adapt a timm encoder's stem to ``32 x 32`` inputs.
        pretrained: Load pretrained timm weights.
        input_key: Batch key holding the inputs.
        **kwargs: Passed to :class:`Module`, including ``optim``. Without
            ``optim``, LFR follows the authors' recipe: SGD (lr 0.03,
            momentum 0.9, weight decay 5e-4) where the encoder's learning rate
            follows a cosine over cycles, set once per cycle, and the
            predictors keep 0.03. A custom ``optim`` is not saved in the
            hyperparameters; pass it again to ``load_from_checkpoint``.

    Note:
        Distributed training needs
        ``Trainer(strategy="ddp_find_unused_parameters_true")``: each phase
        leaves half of the parameters without gradients. The training loss
        uses the global batch, so equal local batch sizes (``drop_last=True``)
        are required. Validation and test losses use each rank's local batch,
        so they are not directly comparable with the training loss under DDP.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "resnet18",
        projectors: Optional[Sequence[nn.Module]] = None,
        predictors: Optional[Sequence[nn.Module]] = None,
        num_projectors: int = 40,
        target_dim: int = 2048,
        predictor_hidden_dim: int = 256,
        predictor_epochs: int = 5,
        lambd: float = 0.01,
        candidate_ratio: int = 1,
        selection_sample: Optional[torch.Tensor] = None,
        low_resolution: bool = False,
        pretrained: bool = False,
        input_key: str = "image",
        **kwargs: Any,
    ) -> None:
        if predictor_epochs < 1:
            raise ValueError("predictor_epochs must be positive")
        if num_projectors < 1 or candidate_ratio < 1:
            raise ValueError("num_projectors and candidate_ratio must be positive")
        if predictors is not None:
            num_projectors = len(predictors)
            if num_projectors == 0:
                raise ValueError("provide at least one predictor")
        # Without ``optim``, LFR builds the authors' optimizer itself (see
        # ``configure_optimizers``); with it, the generic Module path applies.
        paper_schedule = "optim" not in kwargs
        super().__init__(forward=lfr, **kwargs)
        self._paper_schedule = paper_schedule

        if isinstance(encoder_name, str):
            self.backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            self.backbone = encoder_name

        if projectors is None:
            projectors = [
                random_image_projector(target_dim)
                for _ in range(num_projectors * candidate_ratio)
            ]
        pool_size = len(projectors)
        if len(projectors) < num_projectors:
            raise ValueError("provide at least as many projectors as predictors")
        if len(projectors) > num_projectors:
            if selection_sample is None:
                raise ValueError("selection_sample is required to select projectors")
            indices = select_diverse_projectors(
                projectors, selection_sample, num_projectors
            )
            projectors = [projectors[i] for i in indices]
        self.projectors = nn.ModuleList(projectors)
        self.projectors.requires_grad_(False)
        self.projectors.eval()

        if predictors is None:
            sample = selection_sample
            if sample is None:
                size = 32 if low_resolution else 224
                sample = torch.zeros(2, 3, size, size)
            with torch.no_grad():
                was_training = self.backbone.training
                self.backbone.eval()
                embed_dim = _output_dim(self.backbone, sample[:2])
                self.backbone.train(was_training)
                target_dims = [_output_dim(p, sample[:2]) for p in self.projectors]
            predictors = [
                predictor_mlp(embed_dim, dim, predictor_hidden_dim)
                for dim in target_dims
            ]
        self.predictors = nn.ModuleList(predictors)

        self.lfr_loss = BatchwiseBarlowTwinsLoss(lambd=lambd)
        self.predictor_epochs = predictor_epochs
        self.input_key = input_key
        self.save_hyperparameters(
            dict(
                num_projectors=num_projectors,
                target_dim=target_dim,
                predictor_hidden_dim=predictor_hidden_dim,
                predictor_epochs=predictor_epochs,
                lambd=lambd,
                # Selection has already happened and the checkpoint stores the
                # chosen projectors, so reloading must not draw a new pool.
                candidate_ratio=1,
                projector_pool_size=pool_size,
                low_resolution=low_resolution,
                pretrained=pretrained,
                input_key=input_key,
            )
        )
        if isinstance(encoder_name, str):
            self.save_hyperparameters({"encoder_name": encoder_name})

    def max_epochs_for(self, cycles: int) -> int:
        """Number of Trainer epochs covering ``cycles`` encoder epochs.

        Args:
            cycles: Encoder epochs, each followed by ``predictor_epochs``
                predictor epochs.

        Returns:
            The value to pass as ``Trainer(max_epochs=...)``.
        """
        return cycles * (1 + self.predictor_epochs)

    def _lr_factors(self) -> tuple[float, float]:
        """Learning-rate multipliers for the (encoder, predictor) groups.

        The encoder follows the authors' ``adjust_learning_rate``: a cosine over
        cycles, constant within a cycle. The predictors keep the base rate.
        """
        cycles = 0
        if self._trainer is not None and (self.trainer.max_epochs or 0) > 0:
            # A trailing partial cycle still trains, so round up.
            cycles = math.ceil(self.trainer.max_epochs / (1 + self.predictor_epochs))
        if cycles < 1:
            return 1.0, 1.0
        cycle = self.current_epoch // (1 + self.predictor_epochs)
        return 0.5 * (1 + math.cos(math.pi * cycle / cycles)), 1.0

    def configure_optimizers(self):
        """Build the authors' optimizer, unless a custom ``optim`` was given."""
        if not self._paper_schedule:
            return super().configure_optimizers()
        optimizer = torch.optim.SGD(
            [
                {"params": list(self.backbone.parameters())},
                {"params": list(self.predictors.parameters())},
            ],
            lr=0.03,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [lambda _: self._lr_factors()[0], lambda _: self._lr_factors()[1]],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_train_epoch_start(self) -> None:
        """Apply the new cycle's learning rates before its first step."""
        super().on_train_epoch_start()
        if not self._paper_schedule:
            return
        optimizers = self.optimizers()
        optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
        for group, factor in zip(optimizer.param_groups, self._lr_factors()):
            group["lr"] = group["initial_lr"] * factor

    @property
    def is_encoder_epoch(self) -> bool:
        """Whether the current epoch trains the encoder (else the predictors)."""
        return self.current_epoch % (1 + self.predictor_epochs) == 0

    def train(self, mode: bool = True) -> "LFR":
        """Switch training mode while keeping the frozen projectors in eval mode."""
        super().train(mode)
        self.projectors.eval()
        return self

    def setup(self, stage: str) -> None:
        """Reject DDP fitting that cannot handle phase-wise frozen weights."""
        super().setup(stage)
        from lightning.pytorch.strategies import DDPStrategy

        strategy = getattr(self.trainer, "strategy", None)
        ddp_kwargs = getattr(strategy, "_ddp_kwargs", None) or {}
        if (
            stage == "fit"
            and isinstance(strategy, DDPStrategy)
            and not ddp_kwargs.get("find_unused_parameters", False)
        ):
            raise RuntimeError(
                "LFR alternates which parameters receive gradients, which plain "
                "DDP rejects. Use Trainer(strategy='ddp_find_unused_parameters_true') "
                "or DDPStrategy(find_unused_parameters=True)."
            )

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """Run :meth:`Module.training_step` with the inactive half frozen."""
        encoder_epoch = self.is_encoder_epoch
        # Freezing parameters does not freeze BatchNorm running statistics.
        self.backbone.train(encoder_epoch)
        inactive = self.predictors if encoder_epoch else self.backbone
        # With gradient accumulation, a partial window at the end of the last
        # phase can leave gradients behind; the shared optimizer would apply
        # them to the half that is now frozen.
        for parameter in inactive.parameters():
            parameter.grad = None
        with _frozen(inactive):
            return super().training_step(batch, batch_idx)
