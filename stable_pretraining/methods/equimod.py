"""EquiMod: an equivariance module on top of SimCLR.

Adds a second, *equivariant* latent space to SimCLR. An augmentation-conditioned
predictor maps the embedding of the un-augmented image to the embedding of each
augmented view, given the parameters of the augmentation that produced it. The
prediction is trained with a second NT-Xent loss, added to the usual invariance
loss, so the backbone keeps augmentation-related information that a purely
invariant objective would discard.

Two entry points, sharing one implementation:

- :func:`stable_pretraining.forward.equimod`, the forward function, which runs
  on any :class:`~stable_pretraining.Module` that owns a backbone, a projector
  and an :class:`EquivarianceModule`.
- :class:`EquiMod`, a :class:`~stable_pretraining.Module` with those components
  pre-wired and that function as its ``forward(batch, stage)``.

References:
    Devillers & Lefort. "EquiMod: An Equivariance Module to Improve
    Self-Supervised Learning." ICLR 2023. https://arxiv.org/abs/2211.01244

Example::

    import stable_pretraining as spt

    model = spt.EquiMod(encoder_name="resnet18", low_resolution=True)

    # ``batch`` comes from a MultiViewTransform with three pipelines,
    # [un-augmented, augmented, augmented]
    out = model(batch, "fit")
    out["loss"].backward()
"""

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.forward import equimod
from stable_pretraining.losses import NTXEntLoss
from stable_pretraining.methods.simclr import _build_projector


_Stats = Union[torch.Tensor, Sequence[float]]


class EquivarianceModule(nn.Module):
    """The EquiMod equivariance branch, everything after the backbone.

    Holds the equivariance projector, the augmentation-parameter projector and
    the predictor, and knows how to read the augmentation parameters that the
    library's transforms record in each view. It is the one extra attribute an
    :class:`~stable_pretraining.Module` needs to run
    :func:`stable_pretraining.forward.equimod`; :class:`EquiMod` owns one as
    ``self.equi``.

    Note:
        Under DDP the batch-norm standardisation of the augmentation
        parameters (``param_stats=None``) uses per-GPU statistics: each rank
        standardises with the mean and variance of its own local batch, as do
        the batch norms inside the projectors and the predictor. They are
        global only if the model is converted to ``SyncBatchNorm`` (e.g.
        ``Trainer(sync_batchnorm=True)``). A fixed ``param_stats`` applies the
        same values on every rank.

    Args:
        embed_dim: Backbone output dimension.
        projector_dims: Hidden + output dimensions of the equivariance
            projector.
        param_dim: Size of the augmentation-parameter vector (15 for
            crop + flip + colour jitter + grayscale, 17 with Gaussian blur).
        param_projector_dims: Hidden + output dimensions of the parameter
            projector.
        predictor_hidden_dims: Hidden dimensions of the predictor. Empty gives
            the single linear layer used in the paper.
        param_stats: Optional ``(mean, std)`` of the parameter vector, each a
            tensor or sequence of length ``param_dim``, used to standardise
            it. If ``None``, a non-affine batch norm standardises it instead,
            with the statistics of the local (per-GPU) batch.
        weight: Weight ``lambda`` of the equivariance loss in the total loss.
        param_keys: Optional keyword overrides for :meth:`augmentation_params`
            (e.g. ``{"blur": "PILGaussianBlur"}``), used by
            :meth:`params_from_view`.
    """

    def __init__(
        self,
        embed_dim: int,
        projector_dims: Sequence[int] = (2048, 2048, 128),
        param_dim: int = 15,
        param_projector_dims: Sequence[int] = (128,),
        predictor_hidden_dims: Sequence[int] = (),
        param_stats: Optional[Tuple[_Stats, _Stats]] = None,
        weight: float = 1.0,
        param_keys: Optional[Mapping[str, Optional[str]]] = None,
    ):
        super().__init__()
        self.weight = weight
        self.param_dim = param_dim
        self.param_keys = dict(param_keys or {})

        self.projector = _build_projector(embed_dim, list(projector_dims))
        out_dim = list(projector_dims)[-1]
        self.param_projector = _build_projector(
            param_dim, list(param_projector_dims), final_bn_no_bias=False
        )
        self.predictor = _build_projector(
            out_dim + list(param_projector_dims)[-1],
            [*predictor_hidden_dims, out_dim],
        )

        if param_stats is None:
            self.param_norm = nn.BatchNorm1d(param_dim, affine=False)
        else:
            mean, std = (torch.as_tensor(s, dtype=torch.float) for s in param_stats)
            if mean.shape != (param_dim,) or std.shape != (param_dim,):
                raise ValueError(
                    f"param_stats must be two tensors of shape [{param_dim}], "
                    f"got {tuple(mean.shape)} and {tuple(std.shape)}"
                )
            if not (std > 0).all():
                raise ValueError(
                    "param_stats std must be strictly positive; use 1.0 for "
                    "parameters that are constant under your augmentations"
                )
            self.param_norm = None
            self.register_buffer("param_mean", mean)
            self.register_buffer("param_std", std)

    @staticmethod
    def augmentation_params(
        view: Mapping[str, torch.Tensor],
        crop: Optional[str] = "RandomResizedCrop",
        flip: Optional[str] = "RandomHorizontalFlip",
        jitter: Optional[str] = "ColorJitter",
        grayscale: Optional[str] = "RandomGrayscale",
        blur: Optional[str] = None,
    ) -> torch.Tensor:
        """Assemble the augmentation-parameter vector of one collated view.

        The transforms in ``stable_pretraining.data.transforms`` record what
        they sampled under their class name in the sample dict. This gathers
        those records into the layout of the official implementation:

        - ``[0:4]`` crop: top, left, height, width
        - ``[4]`` flip: 1 if flipped
        - ``[5]`` colour jitter applied, ``[6:10]`` order of its four
          operations, ``[10:14]`` brightness, contrast, saturation, hue factors
        - ``[14]`` grayscale: 1 if applied
        - ``[15:17]`` blur applied, sigma (only when ``blur`` is given)

        A skipped colour jitter is recorded as zeros by the transform; it is
        re-encoded here as the identity (order ``0,1,2,3``, factors
        ``1,1,1,0``) so that "not applied" and "applied with neutral factors"
        are close in parameter space.

        Args:
            view: One entry of ``batch["views"]`` after collation.
            crop: Key of the ``RandomResizedCrop`` record, or ``None`` to skip.
            flip: Key of the ``RandomHorizontalFlip`` record, or ``None``.
            jitter: Key of the ``ColorJitter`` record, or ``None``.
            grayscale: Key of the ``RandomGrayscale`` record, or ``None``.
            blur: Key of the ``PILGaussianBlur`` or ``GaussianBlur`` record,
                or ``None``.

        Returns:
            Float tensor of shape ``[B, P]``.
        """

        def record_of(key: str) -> torch.Tensor:
            if key not in view:
                raise KeyError(
                    f"view carries no {key!r} parameter record (keys: "
                    f"{sorted(view)}); add that transform to the augmented "
                    "pipelines or pass None for it"
                )
            return view[key].float()

        parts = []
        if crop is not None:
            parts.append(record_of(crop))
        if flip is not None:
            parts.append(record_of(flip).unsqueeze(1))
        if jitter is not None:
            record = record_of(jitter)
            factors, order = record[:, :4], record[:, 4:]
            # ``order`` is a permutation of 0..3 when applied, all zeros when skipped
            applied = order.sum(dim=1, keepdim=True) > 0
            factors = torch.where(applied, factors, factors.new_tensor([1, 1, 1, 0]))
            order = torch.where(applied, order, order.new_tensor([0, 1, 2, 3]))
            parts += [applied.float(), order, factors]
        if grayscale is not None:
            parts.append(record_of(grayscale).unsqueeze(1))
        if blur is not None:
            # GaussianBlur records (sigma_x, sigma_y), always equal
            record = record_of(blur)
            sigma = record.reshape(record.shape[0], -1)[:, :1]
            parts += [(sigma > 0).float(), sigma]
        return torch.cat(parts, dim=1)

    def params_from_view(self, view: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """:meth:`augmentation_params` with this module's ``param_keys`` applied."""
        return self.augmentation_params(view, **self.param_keys)

    def normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Standardise the raw parameter vector, [N, param_dim]."""
        if params.shape[-1] != self.param_dim:
            raise ValueError(
                f"expected augmentation parameters of size {self.param_dim} "
                f"(param_dim), got {params.shape[-1]}"
            )
        params = params.to(self.param_projector[0].weight.dtype)
        if self.param_norm is not None:
            return self.param_norm(params)
        return (params - self.param_mean) / self.param_std

    def forward(
        self,
        h_original: torch.Tensor,
        h_views: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the augmented views' equivariant embeddings.

        Args:
            h_original: Backbone features of the un-augmented images, [B, D].
            h_views: Backbone features of the augmented views, [V*B, D],
                view-major (all of view 1, then all of view 2, ...).
            params: Augmentation parameters in the same order, [V*B, param_dim].

        Returns:
            ``(y_views, y_hat, gain)``: the views' equivariant embeddings
            [V*B, K], their predictions [V*B, K], and the detached mean of
            ``cos(y, y_hat) - cos(y, y_original)``.
        """
        B = h_original.shape[0]
        n_views, remainder = divmod(h_views.shape[0], B)
        if remainder:
            raise ValueError(
                f"h_views has {h_views.shape[0]} rows, not a multiple of the "
                f"{B} un-augmented images; expected view-major [V*B, D]"
            )
        # One projector pass over [original, views] so batch-norm statistics
        # are shared by all of them, as in the official implementation.
        y = self.projector(torch.cat([h_original, h_views], dim=0))
        y_original = y[:B].repeat(n_views, 1)
        y_views = y[B:]
        p = self.param_projector(self.normalize_params(params))
        y_hat = self.predictor(torch.cat([y_original, p], dim=1))
        with torch.no_grad():
            gain = (
                F.cosine_similarity(y_views, y_hat)
                - F.cosine_similarity(y_views, y_original)
            ).mean()
        return y_views, y_hat, gain


class EquiMod(Module):
    """EquiMod: SimCLR with an augmentation-conditioned equivariance module.

    A :class:`~stable_pretraining.Module` whose ``forward(batch, stage)`` is
    :func:`stable_pretraining.forward.equimod`. It takes the batch dict of a
    three-view ``MultiViewTransform`` (``[un-augmented, augmented,
    augmented]``) and returns a dict with ``"loss"``, ``"embedding"`` and
    ``"label"``, so it trains through :class:`~stable_pretraining.Manager`
    as is.

    Components (the attributes the forward function reads):
        - ``backbone``: any feature extractor producing a flat [B, D] embedding
          (timm ViT/ResNet with the head removed).
        - ``projector`` and ``simclr_loss``: SimCLR's invariance head and
          NT-Xent loss on the two augmented views.
        - ``equi``: an :class:`EquivarianceModule`. Its predictor maps
          ``[equivariant embedding of the un-augmented image, parameter
          embedding]`` to the predicted equivariant embedding of each view.
        - ``equi_loss``: NT-Xent loss between each view's equivariant
          embedding and its prediction.
        - **Loss**: ``NT-Xent(z1, z2) + equi_weight * NT-Xent(y, y_hat)``.

    Note:
        Follows the official implementation where it differs from the paper:
        the equivariance loss is a standard NT-Xent over ``[y, y_hat]``, so
        the other view of the same image and the other predictions act as
        negatives (Eq. 4 of the paper excludes the sibling view). The
        projectors and the predictor reuse SimCLR's projector builder, whose
        final batch norm has a learnable scale (official: ``affine=False``).

        Under DDP both NT-Xent terms, invariance and equivariance, score each
        rank's anchors against candidates gathered from all ranks, so their
        negatives are those of the global batch. Batch-norm statistics are
        per GPU: with ``param_stats=None`` each rank standardises the
        augmentation parameters with its own local batch, and the same holds
        for the batch norms of the backbone, the projectors and the
        predictor. They are global only if the model is converted to
        ``SyncBatchNorm`` (e.g. ``Trainer(sync_batchnorm=True)``, as in the
        official multi-GPU run). A fixed ``param_stats`` applies the same
        values on every rank.

    Args:
        encoder_name: timm model name or a pre-instantiated ``nn.Module``
            whose ``forward`` returns a ``[B, D]`` tensor.
        projector_dims: Hidden + output dimensions of the invariance projector.
        temperature: Temperature of the invariance NT-Xent loss.
        equi_projector_dims: Hidden + output dimensions of the equivariance
            projector.
        equi_temperature: Temperature of the equivariance NT-Xent loss.
        equi_weight: Weight ``lambda`` of the equivariance loss.
        param_dim: Size of the augmentation-parameter vector (15 for
            crop + flip + colour jitter + grayscale, 17 with Gaussian blur).
        param_projector_dims: Hidden + output dimensions of the parameter
            projector.
        predictor_hidden_dims: Hidden dimensions of the predictor. Empty gives
            the single linear layer used in the paper.
        param_stats: Optional ``(mean, std)`` of the parameter vector, each a
            tensor or sequence of length ``param_dim``, used to standardise
            it. If ``None``, a non-affine batch norm standardises it instead,
            with the statistics of the local (per-GPU) batch.
        param_keys: Optional keyword overrides for
            :meth:`EquivarianceModule.augmentation_params` (e.g.
            ``{"blur": "PILGaussianBlur"}`` with ``param_dim=17``).
        low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
        pretrained: Load pretrained timm weights for the encoder.
        optim: Optimizer configuration accepted by
            :class:`~stable_pretraining.Module`. ``None`` keeps the Module
            default; it can also be set afterwards as ``model.optim``.

    Example::

        model = EquiMod(
            encoder_name="resnet18",
            low_resolution=True,
            optim={"optimizer": {"type": "LARS", "lr": 4.0}},
        )

        # training batch: three views, the augmented ones carrying the
        # parameters their transforms sampled
        out = model(batch, "fit")
        out["loss"].backward()

        # evaluation: a single-view batch, no loss
        model.eval()
        features = model({"image": images}, "validate")["embedding"]
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 128),
        temperature: float = 0.5,
        equi_projector_dims: Sequence[int] = (2048, 2048, 128),
        equi_temperature: float = 0.2,
        equi_weight: float = 1.0,
        param_dim: int = 15,
        param_projector_dims: Sequence[int] = (128,),
        predictor_hidden_dims: Sequence[int] = (),
        param_stats: Optional[Tuple[_Stats, _Stats]] = None,
        param_keys: Optional[Mapping[str, Optional[str]]] = None,
        low_resolution: bool = False,
        pretrained: bool = False,
        optim: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if isinstance(encoder_name, str):
            backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            backbone = encoder_name

        # Detect embedding dimension by running a tiny dummy input
        with torch.no_grad():
            embed_dim = backbone(torch.zeros(1, 3, 224, 224)).shape[-1]

        super().__init__(
            forward=equimod,
            hparams={
                "temperature": temperature,
                "equi_temperature": equi_temperature,
                "equi_weight": equi_weight,
                "param_dim": param_dim,
            },
            backbone=backbone,
            projector=_build_projector(embed_dim, list(projector_dims)),
            simclr_loss=NTXEntLoss(temperature=temperature),
            equi=EquivarianceModule(
                embed_dim,
                projector_dims=equi_projector_dims,
                param_dim=param_dim,
                param_projector_dims=param_projector_dims,
                predictor_hidden_dims=predictor_hidden_dims,
                param_stats=param_stats,
                weight=equi_weight,
                param_keys=param_keys,
            ),
            equi_loss=NTXEntLoss(temperature=equi_temperature),
            # an explicit ``optim=None`` would disable optimization in Module
            **({} if optim is None else {"optim": optim}),
        )
        self.embed_dim = embed_dim
