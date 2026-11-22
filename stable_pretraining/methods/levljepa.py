"""LeVLJEPA: end-to-end vision-language pretraining without negatives.

Reference and architecture: https://github.com/MLO-lab/LeVLJEPA.
Encoders are injected so the method also supports small, offline experiments.
"""

from collections import OrderedDict
from collections.abc import Sequence

from stable_pretraining import Module
from stable_pretraining.backbone import MLP
from stable_pretraining.forward import levljepa
from stable_pretraining.losses import LeVLJEPALoss
from torch import nn


def _head(
    in_dim: int, hidden_dims: Sequence[int], out_dim: int, dropout: float = 0.0
) -> nn.Module:
    """Build reference BN/GELU hidden layers with an unnormalized output."""
    dims = [in_dim, *hidden_dims, out_dim]
    if any(dim < 1 for dim in dims):
        raise ValueError(
            "all encoder, hidden and embedding dimensions must be positive"
        )
    head = MLP(
        in_channels=in_dim,
        hidden_channels=[*hidden_dims, out_dim],
        norm_layer="batch_norm",
        activation_layer=nn.GELU,
        dropout=dropout,
    )
    # Keep parameter keys stable when dropout is enabled on existing checkpoints.
    layers = OrderedDict()
    index = 0
    for layer in head:
        if isinstance(layer, nn.Dropout):
            if dropout:
                layers[f"dropout_{index}"] = layer
        else:
            layers[str(index)] = layer
            index += 1
    return nn.Sequential(layers)


class LeVLJEPA(Module):
    """Trainable image/text encoders with bidirectional prediction and SIGReg.

    Args:
        vision_encoder: Image encoder returning pooled [B, vision_dim] features.
        text_encoder: Accepts ``input_ids``/``attention_mask`` keywords, returns
            an HF-style ``last_hidden_state`` or a [B, L, text_dim] tensor.
        vision_dim: Output width of the vision encoder.
        text_dim: Hidden width of the text encoder.
        embedding_dim: Common width of the two regularized marginal spaces.
        pre_projection_dims: Hidden widths before the marginal output layer.
        predictor_dims: Hidden widths of each cross-modal predictor.
        lambda_vision: Weight of vision SIGReg.
        lambda_text: Weight of text SIGReg.
        alignment: ``"mse"`` or ``"cosine"`` prediction loss.
        num_projections: SIGReg random directions per modality.
        num_knots: SIGReg quadrature nodes.
        distributed: Use global-batch characteristic functions under DDP.
        seed: Seed of checkpointed regularizer slice sequences.
        optim: Standard ``spt.Module`` optimizer/scheduler configuration.
        predictor_dropout: Dropout probability in predictor heads, in [0, 1).
            Defaults to zero; 0.1 supports the paper training configuration.

    Note:
        Input is a dict with ``image``, ``input_ids`` and ``attention_mask``.
        The final unmasked text token is pooled; callers should append EOS.
        Default heads use batch normalization, so training needs at least two
        examples per device (or SyncBatchNorm). Empty hidden-width tuples
        produce linear heads suitable for very small experiments.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        vision_dim: int,
        text_dim: int,
        embedding_dim: int = 768,
        pre_projection_dims: Sequence[int] = (2048,),
        predictor_dims: Sequence[int] = (2048, 2048, 2048, 2048),
        lambda_vision: float = 0.02,
        lambda_text: float = 0.02,
        alignment: str = "mse",
        num_projections: int = 4096,
        num_knots: int = 17,
        distributed: bool = True,
        seed: int = 0,
        optim: dict | None = None,
        predictor_dropout: float = 0.0,
    ) -> None:
        if not 0.0 <= predictor_dropout < 1.0:
            raise ValueError("predictor_dropout must be in [0, 1)")
        super().__init__(
            forward=levljepa,
            optim=optim
            if optim is not None
            else {"optimizer": {"type": "AdamW", "lr": 2e-4, "weight_decay": 0.01}},
        )
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_pre_proj = _head(vision_dim, pre_projection_dims, embedding_dim)
        self.text_pre_proj = _head(text_dim, pre_projection_dims, embedding_dim)
        self.projector_vision = _head(
            embedding_dim, predictor_dims, embedding_dim, predictor_dropout
        )
        self.projector_text = _head(
            embedding_dim, predictor_dims, embedding_dim, predictor_dropout
        )
        self.levljepa_loss = LeVLJEPALoss(
            lambda_vision=lambda_vision,
            lambda_text=lambda_text,
            alignment=alignment,
            num_projections=num_projections,
            num_knots=num_knots,
            distributed=distributed,
            seed=seed,
        )
