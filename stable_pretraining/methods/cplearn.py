"""CPLearn: Collapse-Proof Non-Contrastive Self-Supervised Learning.

Projects backbone features onto a frozen dictionary of random bipolar codes
(hyperdimensional computing) and trains with an invariance term plus a
uniform-prior matching term. The paper shows that the minima of this
objective avoid representation, cluster, dimensional and intracluster
collapse without stop-gradient, momentum encoder, predictor or explicit
clustering, and that larger dictionaries yield decorrelated and clustered
embeddings.

The paper leaves several details ambiguous or contradicts itself, and the
official code differs from it in places. Each choice is documented where it
is made: ``epsilon`` and ``activation`` in :class:`CPLearnProjector`, the
prior term, symmetrization and multi-GPU batch mean in
:class:`~stable_pretraining.losses.CPLearnLoss`. Defaults follow the official
code, which produced the reported results.

References:
    Sansone, E., Lebailly, T., Tuytelaars, T. "Collapse-Proof
    Non-Contrastive Self-Supervised Learning." ICML 2025
    (PMLR 267:52833-52860).
    https://arxiv.org/abs/2410.04959
    Official code: https://github.com/emsansone/CPLearn
"""

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.losses import CPLearnLoss


@dataclass
class CPLearnOutput(ModelOutput):
    """Output from CPLearn forward pass."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    logits: Optional[torch.Tensor] = None
    invariance: Optional[torch.Tensor] = None
    prior: Optional[torch.Tensor] = None


class CPLearnProjector(nn.Module):
    """CPLearn projector (Eq. 1): embeddings matched against frozen random codes.

    Computes ``H = act(BN(Linear(Z)))`` and returns the code logits
    ``H @ W / tau``, where ``W`` is a frozen ``f x c`` matrix with i.i.d.
    Rademacher (+/-1) entries and::

        tau = f / (sqrt(n) * log((1 - eps * (c - 1)) / eps))

    with ``n`` the batch size. ``softmax`` of the output gives the code
    assignment probabilities ``P``; ``argmax`` gives the cluster assignment.

    Note:
        **epsilon.** Footnote 3 of the paper states ``eps = 1e-8``
        throughout, while the Algorithm 1 pseudocode writes ``eps: 1e-6``.
        The official code uses ``1e-8``, which is the default. ``eps`` is the
        probability given to non-assigned codes at the optimum (Lemma 1), so
        it must satisfy ``0 < eps < 1/c``; ``log((1 - eps * (c - 1)) / eps)``
        sets the logit scale (about 18.4 for ``1e-8``, 13.8 for ``1e-6``).

        **activation.** Eq. 1 uses ``H = sqrt(f / n) * L2-norm(.)``. Then
        ``sqrt(f / n)`` cancels against the ``sqrt(n)`` in ``tau`` and the
        logits reduce to ``log((1 - eps * (c - 1)) / eps) * cos(h, w_j)``,
        independent of the batch size. Section 3.3 says experiments replace
        the L2 normalization with ``tanh`` but does not say whether the
        ``sqrt(f / n)`` scale or ``tau`` change. The official code applies
        ``tanh`` with no scale and keeps ``tau``, so the logits become
        ``sqrt(n) * log((1 - eps * (c - 1)) / eps) * (tanh(h) . w_j) / f``:
        up to ``sqrt(n)`` times larger than with L2 and dependent on the
        (per-device) batch size. ``"tanh"`` (default) reproduces the official
        code and ``"l2"`` reproduces Eq. 1.

        **embedding size.** The paper sets ``f`` to the backbone output size,
        while the official ResNet-18 config first maps the 512-d features to
        ``f = 2048`` with the linear layer. ``projector_dim`` controls ``f``.

    The codes are a buffer: they are not trained, they are saved in
    checkpoints, and DDP broadcasts them from rank 0, so all ranks share the
    same dictionary even without ``seed``.

    Args:
        in_dim (int): Backbone feature dimension.
        projector_dim (int, optional): Embedding dimension ``f``. Default is 2048.
        dictionary_size (int, optional): Number of codes ``c``. The paper
            recommends ``c >> f``. Default is 16384.
        activation (str, optional): ``"tanh"`` or ``"l2"``. Default is ``"tanh"``.
        epsilon (float, optional): Probability of the non-assigned codes at
            the optimum; sets the logit scale (see Note). Default is 1e-8.
        seed (int, optional): Seed for sampling the codes. If None, the global
            torch RNG is used. Default is None.
    """

    def __init__(
        self,
        in_dim: int,
        projector_dim: int = 2048,
        dictionary_size: int = 16384,
        activation: str = "tanh",
        epsilon: float = 1e-8,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if activation not in ("tanh", "l2"):
            raise ValueError(f"activation must be 'tanh' or 'l2', got {activation!r}")
        if not 0 < epsilon < 1 / dictionary_size:
            raise ValueError(
                f"epsilon must lie in (0, 1/dictionary_size) = (0, {1 / dictionary_size:g}), "
                f"got {epsilon:g}"
            )
        self.projector_dim = projector_dim
        self.dictionary_size = dictionary_size
        self.activation = activation
        self.epsilon = epsilon
        self.log_odds = math.log((1 - epsilon * (dictionary_size - 1)) / epsilon)

        self.linear = nn.Linear(in_dim, projector_dim)
        self.bn = nn.BatchNorm1d(projector_dim)

        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        codes = torch.randint(
            0, 2, (projector_dim, dictionary_size), generator=generator
        )
        self.register_buffer("codes", 2.0 * codes.float() - 1.0)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embeddings ``H``, shape ``[N, f]``."""
        h = self.bn(self.linear(x))
        if self.activation == "l2":
            return F.normalize(h, dim=-1) * math.sqrt(self.projector_dim / h.size(0))
        return torch.tanh(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the code logits ``H @ W / tau``, shape ``[N, c]``."""
        h = self.embed(x)
        tau = self.projector_dim / (math.sqrt(h.size(0)) * self.log_odds)
        return (h @ self.codes) / tau


class CPLearn(Module):
    """CPLearn: collapse-proof non-contrastive SSL.

    The projector and loss defaults follow the official ResNet-18 CIFAR-10
    recipe (Appendix L): ``f = 2048``, ``c = 16384``, ``beta = 1``. See
    :class:`CPLearnProjector` and :class:`~stable_pretraining.losses.CPLearnLoss`
    for how each option relates to the paper.

    Note:
        The prior term uses the batch mean, so with ``n`` samples per batch
        and hard assignments it cannot go below ``log c - log n``; with the
        paper's batch of 64 and ``c = 16384`` training sits on that bound. In
        CIFAR-10 ablations a batch of 256 with ``dictionary_size=2048``
        trained clearly better than the paper's setting (see
        ``benchmarks/cifar10/cplearn-resnet18.py``).

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dim: Embedding dimension ``f`` of the projector.
    :param dictionary_size: Number of frozen codes ``c``.
    :param beta: Weight of the invariance term.
    :param activation: Projector activation, ``"tanh"`` or ``"l2"``.
    :param epsilon: Probability of non-assigned codes at the optimum.
    :param prior: ``"reverse_kl"`` or ``"ce"`` prior-matching term.
    :param symmetric: Average the loss over both view orders.
    :param gather_distributed: Compute the prior batch mean over all ranks.
    :param codes_seed: Seed for the code dictionary (None uses the global RNG).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dim: int = 2048,
        dictionary_size: int = 16384,
        beta: float = 1.0,
        activation: str = "tanh",
        epsilon: float = 1e-8,
        prior: str = "reverse_kl",
        symmetric: bool = True,
        gather_distributed: bool = False,
        codes_seed: Optional[int] = None,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            self.backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            self.backbone = encoder_name

        with torch.no_grad():
            embed_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim

        self.projector = CPLearnProjector(
            embed_dim,
            projector_dim=projector_dim,
            dictionary_size=dictionary_size,
            activation=activation,
            epsilon=epsilon,
            seed=codes_seed,
        )
        self.cplearn_loss = CPLearnLoss(
            beta=beta,
            prior=prior,
            symmetric=symmetric,
            gather_distributed=gather_distributed,
        )

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> CPLearnOutput:
        if view2 is None:
            embedding = self.backbone(view1)
            return CPLearnOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
                logits=None,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        logits1 = self.projector(h1)
        logits2 = self.projector(h2)
        terms = self.cplearn_loss.terms(logits1, logits2)
        return CPLearnOutput(
            loss=terms["loss"],
            embedding=torch.cat([h1, h2], dim=0),
            logits=torch.cat([logits1, logits2], dim=0),
            invariance=terms["invariance"].detach(),
            prior=terms["prior"].detach(),
        )
