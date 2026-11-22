"""ADNCE with the dictionary-based Module training and evaluation interface."""

from typing import Any

from torch import nn

from stable_pretraining.forward import adnce
from stable_pretraining.losses import ADNCELoss
from stable_pretraining.module import Module


class ADNCE(Module):
    """Contrastive learning with adaptive, detached negative weights.

    Args:
        backbone: Feature extractor returning a ``[B, D]`` tensor.
        projector: Projection network mapping ``[B, D]`` to ``[B, P]``.
        temperature: Positive contrastive temperature.
        mu: Center of the Gaussian weighting curve on cosine similarities.
        sigma: Positive width of the weighting curve.
        optim: Module optimizer/scheduler configuration. Defaults to Adam with
            learning rate ``1e-3``. See :class:`stable_pretraining.Module` for
            supported single- and multiple-optimizer configurations.
        **kwargs: Additional Module options, including ``hparams`` and
            ``parallelize_fn``.

    Note:
        Call ``model(batch, stage="fit")`` with two view dictionaries during
        training. Evaluation accepts ``{"image": images}`` and returns encoder
        features without the projector. No dataset labels enter the SSL loss.

    Example:
        ::

            import torch
            import stable_pretraining as spt

            model = spt.ADNCE(
                backbone=torch.nn.Linear(32, 16),
                projector=torch.nn.Linear(16, 8),
                optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
            )
            batch = {
                "views": [
                    {"image": torch.randn(4, 32)},
                    {"image": torch.randn(4, 32)},
                ]
            }
            output = model(batch, stage="fit")
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module,
        temperature: float = 0.4,
        mu: float = 0.7,
        sigma: float = 0.5,
        optim: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        hparams = dict(kwargs.pop("hparams", None) or {})
        hparams.update(temperature=temperature, mu=mu, sigma=sigma)
        super().__init__(
            forward=adnce,
            backbone=backbone,
            projector=projector,
            adnce_loss=ADNCELoss(temperature=temperature, mu=mu, sigma=sigma),
            optim=optim
            if optim is not None
            else {"optimizer": {"type": "Adam", "lr": 1e-3}},
            hparams=hparams,
            **kwargs,
        )
