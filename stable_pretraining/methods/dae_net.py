"""DAE-Net: Deforming Auto-Encoder for Fine-Grained Shape Co-Segmentation.

Self supervised learning for fine-grained shape co-segmentation.

Reference:
    https://arxiv.org/abs/2311.13125
"""

import math

from ..backbone import DAENetBackbone
from ..forward import dae_net_forward
from ..losses import DAENetLoss
from ..module import Module
from ..callbacks import DAENetRevival


class DAENet(Module):
    """Assemble DAE-Net components for native Module training.

    Args:
        backbone: DAE-Net network to train.
        sparse_weight: Sparsity coefficient; .002 for Chair, .02 for Mug.
        lr: Fixed Adam learning rate.
        regularization_weight: Initial regularization multiplier before the
            training controller applies its schedule.
        era_length: Number of training epochs per era. Defaults to 10.

    Note:
        The method automatically registers a controller for era-based
        regularization warmup, branch revival, Adam resets, acceptance/rollback,
        and checkpoint persistence including RNG. For Linux single-node DDP,
        use FP32 ranks, a DAENetDistributedBatchSampler, zero loader workers,
        no gradient accumulation, and Trainer(use_distributed_sampler=False).
        Set strategy="ddp_find_unused_parameters_true" to keep gradient buckets
        stable across restarts. Batch size is global. Exact resumption targets
        completed epochs with deterministic loading and the same world size,
        data and environment.
    """

    def __init__(
        self,
        backbone: DAENetBackbone,
        *,
        sparse_weight: float = 0.002,
        lr: float = 2e-4,
        regularization_weight: float = 0.001,
        era_length: int = 10,
    ) -> None:
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be a finite and positive.")

        if not math.isfinite(regularization_weight) or regularization_weight < 0:
            raise ValueError("Regularization weight must be a finite and non-negative.")

        super().__init__(
            forward=dae_net_forward,
            backbone=backbone,
            dae_net_loss=DAENetLoss(sparse_weight=sparse_weight),
            regularization_weight=regularization_weight,
            hparams={
                "sparse_weight": sparse_weight,
                "lr": lr,
                "regularization_weight": regularization_weight,
                "era_length": era_length,
            },
            optim={
                "optimizer": {
                    "type": "Adam",
                    "lr": lr,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
                "scheduler": {
                    "type": "ConstantLR",
                    "factor": 1.0,
                },
            },
        )

        self._era_controller = DAENetRevival(era_length=era_length)

    def configure_callbacks(self) -> list[DAENetRevival]:
        """Provide the DAE-Net era controller to Lightning training infrastructure.

        Returns:
            The controller owned by this method instance.

        Note:
            The same controller instance owns era state across training hooks.
        """
        return [self._era_controller]
