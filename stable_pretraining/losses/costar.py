import math

import torch
import torch.nn.functional as F
from torch import nn


class COSTARLoss(nn.Module):
    """COSTAR Eq. (2): history loss plus the mean of available components.

    Args:
        temperature: Positive finite cosine-logit divisor.

    Note:
        Eqs. (4),(6) sum cross-view directions with detached targets.
        Positives are batch-diagonal; negatives stay within the local batch.
        All dictionaries share treatment/outcome/history keys; X=0 omits covariate.
        Values must be finite floating [B,D], B,D>0, same shape/device; dtypes may differ.
        Norm floor 1e-12; compute in float64 if present, else float32; autocast disabled.
    """

    _STREAMS = ("covariate", "treatment", "outcome", "history")

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        self.temperature = float(temperature)

    @staticmethod
    def _validate_embeddings(embeddings: torch.Tensor, name: str) -> None:
        if (
            not isinstance(embeddings, torch.Tensor)
            or not embeddings.is_floating_point()
        ):
            raise TypeError(f"{name} must be a floating-point tensor.")
        if embeddings.ndim != 2 or min(embeddings.shape) < 1:
            raise ValueError(f"{name} must have nonempty shape (batch, embedding).")

    @staticmethod
    def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize vectors without overflowing the norm at large magnitudes."""
        eps = 1e-12
        # The scale cancels analytically, so its gradient is unnecessary.
        scale = embeddings.abs().amax(dim=-1, keepdim=True).clamp_min(eps).detach()
        scaled = embeddings / scale
        norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        return scaled / norm.clamp_min(eps / scale)

    def info_nce(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Compute one-direction InfoNCE (Eq. (5)); see the class input contract.

        Args:
            queries: Online vectors.
            keys: Target vectors.

        Returns:
            Mean negative log-probability of diagonal positives.
        """
        self._validate_embeddings(queries, "queries")
        self._validate_embeddings(keys, "keys")
        if queries.shape != keys.shape:
            raise ValueError("queries and keys must have the same shape.")
        if queries.device != keys.device:
            raise ValueError("queries and keys must be on the same device.")
        return self._info_nce(queries, keys)

    def _info_nce(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        compute_dtype = (
            torch.float64
            if torch.float64 in (queries.dtype, keys.dtype)
            else torch.float32
        )
        # Explicit casting alone does not prevent autocast from lowering matmul.
        with torch.autocast(device_type=queries.device.type, enabled=False):
            normalized_queries = self._normalize(queries.to(dtype=compute_dtype))
            normalized_keys = self._normalize(keys.detach().to(dtype=compute_dtype))
            logits = (normalized_queries @ normalized_keys.T) / self.temperature
            positives = torch.arange(queries.shape[0], device=queries.device)
            return F.cross_entropy(logits, positives)

    @classmethod
    def _validate_streams(
        cls,
        streams: tuple[tuple[str, dict[str, torch.Tensor]], ...],
    ) -> tuple[str, ...]:
        active_streams: tuple[str, ...] = ()
        reference: torch.Tensor | None = None
        allowed_keys = (set(cls._STREAMS), set(cls._STREAMS[1:]))
        for stream_name, representations in streams:
            if not isinstance(representations, dict):
                raise TypeError(f"{stream_name} must be a dictionary.")
            keys = set(representations)
            if keys not in allowed_keys:
                raise ValueError(
                    f"{stream_name} must contain treatment, outcome, and history, "
                    "with only covariate optional."
                )
            if active_streams and keys != set(active_streams):
                raise ValueError("All dictionaries must contain the same streams.")
            active_streams = tuple(name for name in cls._STREAMS if name in keys)
            for name in active_streams:
                value = representations[name]
                cls._validate_embeddings(value, f"{stream_name}[{name}]")
                if reference is None:
                    reference = value
                elif value.shape != reference.shape:
                    raise ValueError("All representations must have the same shape.")
                elif value.device != reference.device:
                    raise ValueError("All representations must be on the same device.")
        return active_streams

    def forward(
        self,
        predictions1: dict[str, torch.Tensor],
        predictions2: dict[str, torch.Tensor],
        targets1: dict[str, torch.Tensor],
        targets2: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Combine the views according to the class contract.

        Args:
            predictions1: First-view online codes.
            predictions2: Second-view online codes.
            targets1: First-view momentum codes.
            targets2: Second-view momentum codes.

        Returns:
            Scalar loss, loss_history and available component losses.
        """
        active_streams = self._validate_streams(
            (
                ("predictions1", predictions1),
                ("predictions2", predictions2),
                ("targets1", targets1),
                ("targets2", targets2),
            )
        )
        stream_losses = {
            name: self._info_nce(predictions1[name], targets2[name])
            + self._info_nce(predictions2[name], targets1[name])
            for name in active_streams
        }
        components = active_streams[:-1]
        component_loss = stream_losses[components[0]]
        for name in components[1:]:
            component_loss = component_loss + stream_losses[name]
        component_loss = component_loss / len(components)
        return {
            "loss": stream_losses["history"] + component_loss,
            "loss_history": stream_losses["history"],
            **{f"loss_{name}": stream_losses[name] for name in components},
        }
