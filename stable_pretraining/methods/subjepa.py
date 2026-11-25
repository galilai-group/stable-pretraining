"""Sub-JEPA: frozen-subspace regularization for action-conditioned world models.

Reference: https://arxiv.org/abs/2605.09241 (Zhao et al., 2026).

Example::

    import torch
    import stable_pretraining as spt

    model = spt.SubJEPA(backbone=torch.nn.Linear(4, 192), action_dim=2)
    batch = {"image": torch.randn(8, 4, 4), "action": torch.randn(8, 3, 2)}
    output = model(batch, "train")
    output["loss"].backward()
"""

from typing import Any

import torch
from torch import nn

from stable_pretraining import Module
from stable_pretraining.forward import subjepa
from stable_pretraining.losses.subjepa import SubJEPALoss


class SubJEPAPredictor(nn.Module):
    """Causal action-conditioned transformer predicting the next latent per step.

    Args:
        embed_dim: Encoder output dimension.
        action_dim: Dimension of each continuous action.
        depth: Number of transformer layers.
        num_heads: Attention heads; must divide embed_dim.
        max_history: Maximum context length.

    Note:
        This compact default implements causal latent prediction. Exact LeWM
        architecture reproductions can supply their own predictor to SubJEPA.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        depth: int = 3,
        num_heads: int = 6,
        max_history: int = 16,
    ) -> None:
        super().__init__()
        if min(embed_dim, action_dim, depth, num_heads, max_history) < 1:
            raise ValueError("predictor dimensions must be positive")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.position = nn.Parameter(torch.zeros(1, max_history, embed_dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    embed_dim,
                    num_heads,
                    dim_feedforward=4 * embed_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next states from aligned latent/action histories.

        Args:
            embedding: Current latents [batch, history, embed_dim].
            action: Actions [batch, history, action_dim].

        Returns:
            Next-state predictions [batch, history, embed_dim].
        """
        length = embedding.shape[1]
        if length > self.position.shape[1]:
            raise ValueError("trajectory context exceeds max_history")
        x = embedding + self.action_encoder(action) + self.position[:, :length]
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        return self.head(x)


class SubJEPA(Module):
    """Train an encoder and latent world model with the Sub-JEPA objective.

    Args:
        backbone: Observation encoder returning [batch, embed_dim]. May include
            the projection head; no pretrained weights or EMA teacher required.
        action_dim: Continuous action dimension.
        embed_dim: Backbone output dimension (192 in the paper).
        predictor: Optional module taking (latent_history, action_history).
        num_subspaces: Frozen random subspaces (32; use 16 for PushT).
        subspace_dim: Optional explicit dimension of each subspace.
        regularization_weight: Gaussian loss weight; tune on validation data.
        num_projections: Random directions per subspace.
        num_knots: Gaussian quadrature nodes.
        lr: Default AdamW learning rate.
        weight_decay: Default AdamW weight decay.
        optim: Optional standard Module optimizer configuration.
        predictor_depth: Default causal transformer depth.
        num_heads: Default causal transformer attention heads.
        max_history: Maximum default predictor context length.

    Note:
        Batches contain image [B, T, ...] and action [B, T-1, A], with T >= 2.
        A trailing unused action [B, T, A] is also accepted. A single observation
        per sample can be embedded with image [B, ...] and no action at eval.
        Run training through ``spt.Manager`` as with any other ``spt.Module``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        embed_dim: int = 192,
        predictor: nn.Module | None = None,
        num_subspaces: int = 32,
        subspace_dim: int | None = None,
        regularization_weight: float = 0.1,
        num_projections: int = 1024,
        num_knots: int = 17,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        optim: dict[str, Any] | None = None,
        predictor_depth: int = 3,
        num_heads: int = 6,
        max_history: int = 16,
    ) -> None:
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        super().__init__(
            forward=subjepa,
            backbone=backbone,
            predictor=predictor
            if predictor is not None
            else SubJEPAPredictor(
                embed_dim, action_dim, predictor_depth, num_heads, max_history
            ),
            subjepa_loss=SubJEPALoss(
                embed_dim,
                num_subspaces,
                subspace_dim,
                regularization_weight,
                num_projections,
                num_knots,
            ),
            action_dim=action_dim,
            embed_dim=embed_dim,
            optim=optim
            if optim is not None
            else {
                "optimizer": {"type": "AdamW", "lr": lr, "weight_decay": weight_decay}
            },
            hparams={
                "method": "SubJEPA",
                "embed_dim": embed_dim,
                "action_dim": action_dim,
                "num_subspaces": num_subspaces,
                "subspace_dim": subspace_dim,
                "regularization_weight": regularization_weight,
                "num_projections": num_projections,
                "num_knots": num_knots,
                "predictor_depth": predictor_depth if predictor is None else None,
                "num_heads": num_heads if predictor is None else None,
                "max_history": max_history if predictor is None else None,
            },
        )
