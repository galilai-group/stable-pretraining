"""TS2Vec timestamp representations with cumulative encoder averaging."""

from typing import Any

from torch.optim.swa_utils import AveragedModel

from stable_pretraining import Module
from stable_pretraining.backbone.ts2vec import TS2VecEncoder
from stable_pretraining.callbacks.ts2vec import TS2VecAveraging
from stable_pretraining.forward import ts2vec
from stable_pretraining.losses.ts2vec import TS2VecLoss


class TS2Vec(Module):
    """Learn time-series representations with hierarchical contrastive learning.

    Args:
        input_dims: Number of input channels in ``batch['series']`` [B, T, C].
        output_dims: Number of features per timestamp.
        hidden_dims: Hidden convolution width.
        depth: Number of hidden residual blocks before the output block.
        alpha: Weight of instance contrast; temporal contrast uses 1 - alpha.
        temporal_unit: First hierarchy level with temporal contrast.
        optim: Module optimizer configuration; defaults to reference AdamW.

    Note:
        Evaluation uses the cumulative average of the initial encoder and each
        completed optimizer update. Both encoders and the averaging counter are
        saved in Lightning checkpoints. This is not an EMA teacher objective.
        Instance negatives are local to each rank, as in the single-device
        reference. DDP averages gradients without gathering representations;
        increasing world size does not increase the instance negative pool.
        See :class:`~stable_pretraining.losses.TS2VecLoss` for batch semantics.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        alpha: float = 0.5,
        temporal_unit: int = 0,
        optim: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            forward=ts2vec,
            optim=optim
            if optim is not None
            else {"optimizer": {"type": "AdamW", "lr": 0.001, "weight_decay": 0.01}},
            hparams=dict(
                input_dims=input_dims,
                output_dims=output_dims,
                hidden_dims=hidden_dims,
                depth=depth,
                alpha=alpha,
                temporal_unit=temporal_unit,
                optim=optim,
            ),
        )
        self.backbone = TS2VecEncoder(input_dims, output_dims, hidden_dims, depth)
        self.ts2vec_loss = TS2VecLoss(alpha=alpha, temporal_unit=temporal_unit)
        self.averaged_backbone = AveragedModel(self.backbone)
        self.averaged_backbone.update_parameters(self.backbone)
        self.averaged_backbone.requires_grad_(False)
        self.averaged_backbone.eval()

    def configure_callbacks(self) -> list[TS2VecAveraging]:
        """Install weight averaging after completed backbone optimizer steps."""
        return [TS2VecAveraging()]

    def train(self, mode: bool = True) -> "TS2Vec":
        """Keep the inference encoder deterministic while training the backbone."""
        super().train(mode)
        if hasattr(self, "averaged_backbone"):
            self.averaged_backbone.eval()
        return self
