"""COSTAR self-supervised pretraining for longitudinal histories."""

from copy import deepcopy
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..backbone.mlp import MLP
from ..forward import costar_forward
from ..losses.costar import COSTARLoss
from ..module import Module


class COSTAR(Module):
    """Train the online encoder on all valid prefixes (COSTAR Eqs. (2)-(6)).

    Args:
        covariate_dim: Nonnegative X; zero omits the covariate contrast.
        treatment_dim: Positive treatment count A.
        outcome_dim: Positive observed-outcome count Y.
        static_dim: Nonnegative static count V.
        d_model: Representation width D.
        num_heads: Attention heads dividing D.
        num_layers: Temporal/feature block pairs.
        max_relative_distance: Relative-position clipping radius.
        ffn_hidden_dim: FFN width; None uses 4*D.
        predictor_hidden_dim: SSL head width; None uses 2*D.
        dropout: Encoder and pooling dropout probability.
        temperature: Positive cosine-logit divisor.
        momentum: EMA coefficient in [0,1).
        lr: Positive Adam learning rate.
        weight_decay: Nonnegative Adam weight decay.
        scale_std: Gaussian multiplicative-noise standard deviation.
        shift_std: Gaussian feature-offset standard deviation.
        jitter_std: Gaussian elementwise-noise standard deviation.
        augmentation_probability: Activation probability per operation/view.

    Note:
        Inputs contain observed X/A_prev/Y, static features and prefix masks.
        Online receives gradients; the frozen teacher follows train/eval dropout.
        EMA runs only after successful optimizer steps. Future labels are ignored.
        Negatives are process-local; optimizer-owning callbacks are unsupported.
    """

    def __init__(
        self,
        covariate_dim: int,
        treatment_dim: int,
        outcome_dim: int,
        static_dim: int,
        d_model: int = 36,
        num_heads: int = 3,
        num_layers: int = 1,
        max_relative_distance: int = 15,
        ffn_hidden_dim: int | None = None,
        predictor_hidden_dim: int | None = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        momentum: float = 0.99,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scale_std: float = 0.5,
        shift_std: float = 0.5,
        jitter_std: float = 0.5,
        augmentation_probability: float = 0.5,
    ) -> None:
        super().__init__(forward=costar_forward, hparams={})
        self.save_hyperparameters()
        if not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be finite and in [0,1).")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive.")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        dimensions = (covariate_dim, treatment_dim, outcome_dim)
        strengths = (scale_std, shift_std, jitter_std)
        self.automatic_optimization = True
        self.encoder = COSTAREncoder(
            *dimensions,
            static_dim,
            d_model,
            num_heads,
            num_layers,
            max_relative_distance,
            ffn_hidden_dim,
            dropout,
        )
        self.pooling = COSTARPooling(
            *dimensions,
            dropout,
            allow_missing_covariates=covariate_dim == 0,
        )
        hidden = 2 * d_model if predictor_hidden_dim is None else predictor_hidden_dim
        _feature_counts(hidden)
        if hidden < 1:
            raise ValueError("Predictor hidden width must be positive.")
        self.predictor = MLP(d_model, [hidden, d_model], norm_layer=None, dropout=0.0)
        self.augmentation = COSTARAugmentation(
            *dimensions,
            *strengths,
            probability=augmentation_probability,
            allow_missing_covariates=covariate_dim == 0,
        )
        self.costar_loss = COSTARLoss(temperature=temperature)
        self.momentum_encoder = deepcopy(self.encoder).requires_grad_(False)
        self.momentum_pooling = deepcopy(self.pooling)
        self.momentum, self.lr, self.weight_decay = momentum, lr, weight_decay
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))

    def encode(
        self,
        dynamic_history: torch.Tensor,
        static_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract online endpoints with gradients.

        Args:
            dynamic_history: Floating [B,T,X+A+Y], already using A_prev.
            static_features: Floating [B,V].
            valid_mask: Optional boolean prefix mask [B,T].

        Returns:
            Available component and global representations [B,D].
        """
        tokens, _ = self.encoder(dynamic_history, static_features, valid_mask)
        return select_last_valid_representations(
            self.pooling(tokens, valid_mask), valid_mask
        )

    @torch.no_grad()
    def encode_momentum(
        self,
        dynamic_history: torch.Tensor,
        static_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return detached teacher endpoints; inputs and streams match ``encode``."""
        tokens, _ = self.momentum_encoder(dynamic_history, static_features, valid_mask)
        return select_last_valid_representations(
            self.momentum_pooling(tokens, valid_mask), valid_mask
        )

    @torch.no_grad()
    def update_momentum_encoder(self) -> None:
        """Apply EMA and copy buffers; called once after each successful step."""
        for target, online in zip(
            self.momentum_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            target.lerp_(online.detach(), 1.0 - self.momentum)
        for target, online in zip(
            self.momentum_encoder.buffers(), self.encoder.buffers(), strict=True
        ):
            target.copy_(online)
        self.ema_updates.add_(1)

    def _after_optimizer_step(self, *_: object) -> None:
        """A skipped GradScaler step never invokes this optimizer hook."""
        self.update_momentum_encoder()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return Adam over online parameters with one post-step EMA hook."""
        if self.callbacks_modules:
            raise RuntimeError(
                "COSTAR uses one optimizer; optimizer-owning callbacks are unsupported."
            )
        parameters = list(self.encoder.parameters()) + list(self.predictor.parameters())
        optimizer = torch.optim.Adam(
            parameters, lr=self.lr, weight_decay=self.weight_decay
        )
        optimizer.register_step_post_hook(self._after_optimizer_step)
        return optimizer

    def training_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Log losses weighted by valid-prefix count and return callback state."""
        state = self(batch, stage="fit")
        losses = {f"train/{k}": v for k, v in state.items() if k.startswith("loss")}
        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            batch_size=state["prediction"].shape[0] // 2,
        )
        return state


def _valid_prefix_mask(
    valid_mask: torch.Tensor | None,
    batch_size: int,
    time_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if time_steps < 1:
        raise ValueError("time_steps must be positive.")
    if valid_mask is None:
        return torch.ones(batch_size, time_steps, dtype=torch.bool, device=device)
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean.")
    if valid_mask.shape != (batch_size, time_steps) or valid_mask.device != device:
        raise ValueError("valid_mask must match the batch/time shape and device.")
    if not valid_mask[:, 0].all():
        raise ValueError("Each sample must have a nonempty valid prefix.")
    if (valid_mask[:, 1:] & ~valid_mask[:, :-1]).any():
        raise ValueError("valid_mask must describe right-padded valid prefixes.")
    return valid_mask


def _float_tensor(value: torch.Tensor, rank: int, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor.")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")


def _mask_padding(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values
    padding = ~mask.reshape(*mask.shape, *([1] * (values.ndim - 2)))
    return values.masked_fill(padding, 0.0)


def _attention_dimensions(width: int, heads: int, dropout: float) -> None:
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (width, heads)
    ):
        raise ValueError("Embedding width and head count must be positive integers.")
    if width % heads:
        raise ValueError("d_model must be divisible by num_heads.")
    _dropout_probability(dropout)


def _dropout_probability(dropout: float) -> None:
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and in [0,1).")


def _feature_counts(*counts: int) -> None:
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
        raise ValueError("Feature counts must be nonnegative integers.")


def _history_dimensions(
    covariate_dim: int,
    treatment_dim: int,
    outcome_dim: int,
    allow_missing_covariates: bool = False,
) -> int:
    _feature_counts(covariate_dim, treatment_dim, outcome_dim)
    if not isinstance(allow_missing_covariates, bool):
        raise TypeError("allow_missing_covariates must be boolean.")
    if (
        treatment_dim < 1
        or outcome_dim < 1
        or (covariate_dim == 0 and not allow_missing_covariates)
    ):
        raise ValueError(
            "Positive A/Y and nonempty X are required unless X=0 is enabled."
        )
    return covariate_dim + treatment_dim + outcome_dim


def _ffn_hidden(width: int, hidden: int | None) -> int:
    hidden = 4 * width if hidden is None else hidden
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
        raise ValueError("FFN hidden width must be a positive integer.")
    return hidden


class _COSTARAttentionProjection(nn.Module):
    """Biased Q/K/V projections shared by both attention implementations."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        _attention_dimensions(d_model, num_heads, dropout)
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Identity()
        self.attention_dropout = nn.Dropout(dropout)

    def _qkv(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        shape = (*values.shape[:2], self.num_heads, self.head_dim)
        return tuple(
            layer(values).reshape(shape).transpose(1, 2)
            for layer in (self.q_projection, self.k_projection, self.v_projection)
        )


class _COSTARSampleNorm(nn.Module):
    """Released correction=1 standard deviation with epsilon after sqrt."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width < 2:
            raise ValueError("Sample normalization requires d_model >= 2.")
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(-1, keepdim=True)
        # std has finite backward at constant inputs; sqrt(var) alone does not.
        return (
            centered / (values.std(-1, correction=1, keepdim=True) + 1e-6) * self.weight
            + self.bias
        )


class COSTARTemporalAttention(_COSTARAttentionProjection):
    """Causal Shaw relative K/V attention; clipped offsets are shared by heads."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_relative_distance: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(d_model, num_heads, dropout)
        if (
            isinstance(max_relative_distance, bool)
            or not isinstance(max_relative_distance, int)
            or max_relative_distance < 0
        ):
            raise ValueError("Relative-position radius must be a nonnegative integer.")
        self.max_relative_distance = max_relative_distance
        self.relative_key_embedding = nn.Embedding(
            2 * max_relative_distance + 1, self.head_dim
        )
        self.relative_value_embedding = nn.Embedding(
            2 * max_relative_distance + 1, self.head_dim
        )

    def _relative_position_ids(
        self, time_steps: int, device: torch.device
    ) -> torch.Tensor:
        positions = torch.arange(time_steps, device=device)
        radius = self.max_relative_distance
        return (positions[None, :] - positions[:, None]).clamp(-radius, radius) + radius

    def _attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        times = queries.shape[-2]
        relative_ids = self._relative_position_ids(times, queries.device)
        relative_keys = self.relative_key_embedding(relative_ids)
        relative_values = self.relative_value_embedding(relative_ids)
        scores = (
            queries @ keys.transpose(-2, -1)
            + torch.einsum("shid,ijd->shij", queries, relative_keys)
        ) * self.scale
        allowed = torch.ones(
            times, times, dtype=torch.bool, device=queries.device
        ).tril()
        if valid_mask is not None:
            allowed = allowed & valid_mask[:, None, None, :]
        weights = self.attention_dropout(
            scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        )
        return weights @ values + torch.einsum(
            "shij,ijd->shid", weights, relative_values
        )

    def forward(
        self, dynamic_embeddings: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Attend within each feature's [B,T,F,D] history; padded outputs are zero."""
        batch, times, features, width = dynamic_embeddings.shape
        mask = valid_mask
        sequences = (
            _mask_padding(dynamic_embeddings, mask)
            .permute(0, 2, 1, 3)
            .reshape(batch * features, times, width)
        )
        queries, keys, values = self._qkv(sequences)
        context = self._attention(
            queries,
            keys,
            values,
            None if mask is None else mask.repeat_interleave(features, dim=0),
        )
        output = self.output_projection(
            context.transpose(1, 2).reshape(batch * features, times, width)
        )
        return _mask_padding(
            output.reshape(batch, features, times, width).permute(0, 2, 1, 3), mask
        )


class COSTARTemporalBlock(nn.Module):
    """Temporal attention with a shared dynamic/static pointwise FFN and norm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_relative_distance: int,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.temporal_attention = COSTARTemporalAttention(
            d_model, num_heads, max_relative_distance, dropout
        )
        self.attention_norm = _COSTARSampleNorm(d_model)
        self.dynamic_ffn_norm = _COSTARSampleNorm(d_model)
        self.static_ffn_norm = self.dynamic_ffn_norm
        self.dynamic_ffn = MLP(
            d_model, [_ffn_hidden(d_model, ffn_hidden_dim), d_model], dropout=dropout
        )
        self.static_ffn = self.dynamic_ffn
        self.attention_residual_dropout = nn.Identity()

    def forward(
        self,
        dynamic_embeddings: torch.Tensor,
        static_embeddings: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update [B,T,F,D] and [B,V,D] without static history leakage."""
        mask = valid_mask
        clean = _mask_padding(dynamic_embeddings, mask)
        hidden = _mask_padding(
            self.attention_norm(clean + self.temporal_attention(clean, mask)), mask
        )
        dynamic = self.dynamic_ffn_norm(hidden + self.dynamic_ffn(hidden))
        static = self.static_ffn_norm(
            static_embeddings + self.static_ffn(static_embeddings)
        )
        return _mask_padding(dynamic, mask), static


class COSTARFeatureBlock(_COSTARAttentionProjection):
    """Per-time feature attention and a static-only pass with shared weights."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(d_model, num_heads, dropout)
        self.attention_residual_dropout = nn.Identity()
        self.attention_norm = _COSTARSampleNorm(d_model)
        self.ffn_norm = _COSTARSampleNorm(d_model)
        self.ffn = MLP(
            d_model, [_ffn_hidden(d_model, ffn_hidden_dim), d_model], dropout=dropout
        )

    def _feature_attention(self, feature_sequences: torch.Tensor) -> torch.Tensor:
        queries, keys, values = self._qkv(feature_sequences)
        context = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            scale=self.scale,
        )
        return self.output_projection(
            context.transpose(1, 2).reshape_as(feature_sequences)
        )

    def _encode_features(
        self,
        feature_sequences: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        output_features: int | None = None,
    ) -> torch.Tensor:
        if padding_mask is not None:
            feature_sequences = feature_sequences.masked_fill(padding_mask, 0.0)
        hidden = self.attention_norm(
            feature_sequences + self._feature_attention(feature_sequences)
        )
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask, 0.0)
        # Slice before FFN dropout so discarded static tokens consume no draws.
        if output_features is not None:
            hidden = hidden[:, :output_features]
        output = self.ffn_norm(hidden + self.ffn(hidden))
        return output if padding_mask is None else output.masked_fill(padding_mask, 0.0)

    def forward(
        self,
        dynamic_embeddings: torch.Tensor,
        static_embeddings: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Eqs. (16)-(17) to [B,T,F,D] and original [B,V,D] static tokens."""
        mask = valid_mask
        batch, times, features, width = dynamic_embeddings.shape
        static_over_time = static_embeddings[:, None].expand(-1, times, -1, -1)
        joint = torch.cat((dynamic_embeddings, static_over_time), dim=2).reshape(
            batch * times, -1, width
        )
        dynamic = self._encode_features(
            joint,
            None if mask is None else ~mask.reshape(batch * times, 1, 1),
            features,
        )
        # Joint static outputs contain history; only the static-only pass propagates.
        static = (
            self._encode_features(static_embeddings)
            if static_embeddings.shape[1]
            else static_embeddings
        )
        return dynamic.reshape(batch, times, features, width), static


class COSTAREncoder(nn.Module):
    """Alternate temporal/feature blocks (Appendix A, Eqs. (11)-(17)).

    X/A/Y/V are nonnegative, with X+A+Y>0. Projection and identities are applied
    once; relative K/V tables are shared across layers.
    """

    def __init__(
        self,
        covariate_dim: int,
        treatment_dim: int,
        outcome_dim: int,
        static_dim: int,
        d_model: int = 36,
        num_heads: int = 3,
        num_layers: int = 1,
        max_relative_distance: int = 15,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        counts = (covariate_dim, treatment_dim, outcome_dim, static_dim)
        _feature_counts(*counts)
        _attention_dimensions(d_model, num_heads, dropout)
        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers < 1
        ):
            raise ValueError("num_layers must be a positive integer.")
        self.covariate_dim, self.treatment_dim = covariate_dim, treatment_dim
        self.outcome_dim, self.static_dim = outcome_dim, static_dim
        self.dynamic_dim = sum(counts[:3])
        self.d_model, self.num_layers = d_model, num_layers
        if not self.dynamic_dim:
            raise ValueError("At least one dynamic feature is required.")
        self.input_projection = nn.Linear(1, d_model)
        self.group_embedding = nn.Embedding(4, d_model)
        self.index_embedding = nn.Embedding(max(counts), d_model)
        self.register_buffer(
            "feature_group_ids",
            torch.repeat_interleave(torch.arange(4), torch.tensor(counts)),
        )
        self.register_buffer(
            "feature_index_ids",
            torch.cat([torch.arange(count, dtype=torch.long) for count in counts]),
        )
        self.temporal_blocks = nn.ModuleList(
            COSTARTemporalBlock(
                d_model, num_heads, max_relative_distance, ffn_hidden_dim, dropout
            )
            for _ in range(num_layers)
        )
        self.feature_blocks = nn.ModuleList(
            COSTARFeatureBlock(d_model, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_layers)
        )
        first = self.temporal_blocks[0].temporal_attention
        for block in self.temporal_blocks[1:]:
            attention = block.temporal_attention
            attention.relative_key_embedding = first.relative_key_embedding
            attention.relative_value_embedding = first.relative_value_embedding

    def forward(
        self,
        dynamic_history: torch.Tensor,
        static_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prepared histories into causal dynamic/static feature tokens.

        Args:
            dynamic_history: Floating [B,T,X+A+Y], with treatments already shifted.
            static_features: Floating [B,V] on the same device.
            valid_mask: Optional boolean [B,T] of nonempty valid prefixes.

        Returns:
            [B,T,X+A+Y,D] and [B,V,D]; padded dynamic values/gradients are zero.
        """
        _float_tensor(dynamic_history, 3, "Dynamic history")
        _float_tensor(static_features, 2, "Static features")
        batch, times, features = dynamic_history.shape
        if (
            times < 1
            or features != self.dynamic_dim
            or static_features.shape != (batch, self.static_dim)
        ):
            raise ValueError("Input shapes must match the encoder feature counts.")
        if static_features.device != dynamic_history.device:
            raise ValueError("Dynamic and static inputs must share a device.")
        mask = _valid_prefix_mask(valid_mask, batch, times, dynamic_history.device)
        identity = self.group_embedding(self.feature_group_ids) + self.index_embedding(
            self.feature_index_ids
        )
        # Sanitize raw scalars before Linear, including its weight-gradient path.
        dynamic = self.input_projection(_mask_padding(dynamic_history, mask)[..., None])
        dynamic = _mask_padding(dynamic + identity[: self.dynamic_dim], mask)
        static = (
            self.input_projection(static_features[..., None])
            + identity[self.dynamic_dim :]
        )
        for temporal, feature in zip(
            self.temporal_blocks, self.feature_blocks, strict=True
        ):
            dynamic, static = temporal(dynamic, static, mask)
            dynamic, static = feature(dynamic, static, mask)
        return dynamic, static


class COSTARPooling(nn.Module):
    """Eq. (1) feature means per time; global mean weights features equally.

    Dropout applies only to history. A/Y must be positive; X=0 needs opt-in.
    """

    def __init__(
        self,
        covariate_dim: int,
        treatment_dim: int,
        outcome_dim: int,
        dropout: float = 0.0,
        *,
        allow_missing_covariates: bool = False,
    ) -> None:
        super().__init__()
        self.dynamic_dim = _history_dimensions(
            covariate_dim, treatment_dim, outcome_dim, allow_missing_covariates
        )
        _dropout_probability(dropout)
        self.covariate_dim, self.treatment_dim = covariate_dim, treatment_dim
        self.outcome_dim = outcome_dim
        self.history_dropout = nn.Dropout(dropout)

    def forward(
        self, dynamic_tokens: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Return available component/history [B,T,D] means with zero padding."""
        tokens = _mask_padding(dynamic_tokens, valid_mask)
        groups = tokens.split(
            (self.covariate_dim, self.treatment_dim, self.outcome_dim), dim=2
        )
        result = {
            name: group.mean(2)
            for name, group in zip(("covariate", "treatment", "outcome"), groups)
            if group.shape[2]
        }
        result["history"] = self.history_dropout(tokens.mean(2))
        return result


def select_last_valid_representations(
    representations: dict[str, torch.Tensor], valid_mask: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Select [B,D] endpoints from pooled [B,T,D] streams and a prefix mask."""
    streams = ("covariate", "treatment", "outcome", "history")
    reference = representations["history"]
    rows = torch.arange(reference.shape[0], device=reference.device)
    last = -1 if valid_mask is None else valid_mask.sum(1, dtype=torch.long) - 1
    return {
        name: representations[name][rows, last]
        for name in streams
        if name in representations
    }


class COSTARAugmentation(nn.Module):
    """Batch-shared Gaussian scale/shift and pointwise jitter (Section 4.2).

    Draw in scale/shift/jitter order with independent gates, including treatments
    and their initial placeholders. Zero std/probability skips RNG; p=1 skips gates.
    Evaluation remains stochastic. A/Y>0; X=0 requires opt-in.
    """

    def __init__(
        self,
        covariate_dim: int,
        treatment_dim: int,
        outcome_dim: int,
        scale_std: float = 0.5,
        shift_std: float = 0.5,
        jitter_std: float = 0.5,
        probability: float = 0.5,
        *,
        allow_missing_covariates: bool = False,
    ) -> None:
        super().__init__()
        self.dynamic_dim = _history_dimensions(
            covariate_dim,
            treatment_dim,
            outcome_dim,
            allow_missing_covariates,
        )
        strengths = (scale_std, shift_std, jitter_std)
        if any(not math.isfinite(value) or value < 0 for value in strengths):
            raise ValueError("Noise strengths must be finite and nonnegative.")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError("probability must be a real number.")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability must be finite and in [0,1].")
        self.covariate_dim = covariate_dim
        self.treatment_dim, self.outcome_dim = treatment_dim, outcome_dim
        self.scale_std, self.shift_std, self.jitter_std = map(float, strengths)
        self.probability = float(probability)

    def _enabled(self, std: float, device: torch.device) -> bool:
        if std == 0 or self.probability == 0:
            return False
        return self.probability == 1 or bool(
            torch.rand((), device=device) < self.probability
        )

    def forward(
        self,
        dynamic_history: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return an augmented history copy with matching dtype/device and zero padding."""
        history = dynamic_history
        _float_tensor(history, 3, "History")
        if history.shape[2] != self.dynamic_dim:
            raise ValueError("History width must match X+A+Y.")
        mask = _valid_prefix_mask(valid_mask, *history.shape[:2], history.device)
        result = _mask_padding(history, mask)
        noise = dict(device=history.device, dtype=history.dtype)
        shape = (self.dynamic_dim,)
        if self._enabled(self.scale_std, history.device):
            result = result * (1.0 + self.scale_std * torch.randn(shape, **noise))
        if self._enabled(self.shift_std, history.device):
            result = result + self.shift_std * torch.randn(shape, **noise)
        if self._enabled(self.jitter_std, history.device):
            result = result + self.jitter_std * torch.randn(history.shape, **noise)
        return _mask_padding(result, mask)
