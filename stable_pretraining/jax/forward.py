"""Forward functions for the JAX backend.

These mirror :mod:`stable_pretraining.forward` one-to-one in shape: a stateless
``forward(self, batch, stage)`` that reads submodules off ``self``, flows
everything through a ``dict``, and returns it. The only differences from the
torch versions are the array ops (``jnp`` instead of ``torch``) and that the
loss branch keys off ``self.training`` exactly as on the torch side.

Batch convention (matches ``MultiViewTransform`` output): a multi-view batch is
a dict with a ``"views"`` list of per-view dicts (each holding ``"image"`` and
optionally ``"label"``); a single-view batch is a flat dict with ``"image"``.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import optax


def _get_views(batch: dict) -> Optional[list]:
    """Return the list of view dicts for a multi-view batch, else ``None``."""
    views = batch.get("views")
    if isinstance(views, (list, tuple)) and len(views) > 0:
        return list(views)
    return None


def supervised(self, batch: dict[str, Any], stage: str) -> dict[str, jnp.ndarray]:
    """Supervised cross-entropy forward (sanity baseline, mirrors torch ``supervised``).

    Args:
        self: The bound :class:`stable_pretraining.jax.Module`; requires
            ``backbone`` and ``classifier`` submodules.
        batch: Batch dict with ``"image"`` and optionally ``"label"``.
        stage: Current stage (``"fit"``, ``"validate"``, …).

    Returns:
        dict: ``{"embedding", "logits", "label"?, "loss"?}``.
    """
    out: dict[str, jnp.ndarray] = {}
    embedding = self.backbone(batch["image"])
    logits = self.classifier(embedding)
    out["embedding"] = embedding
    out["logits"] = logits
    if "label" in batch:
        out["label"] = batch["label"]
        out["loss"] = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["label"]
        ).mean()
        self.log(f"{stage}/loss", out["loss"])
    return out


def simclr(self, batch: dict[str, Any], stage: str) -> dict[str, jnp.ndarray]:
    """SimCLR forward (JAX port of :func:`stable_pretraining.forward.simclr`).

    SimCLR maximizes agreement between two augmented views via the NT-Xent
    contrastive loss in projection space :cite:`chen2020simple`.

    Args:
        self: The bound :class:`stable_pretraining.jax.Module`; requires
            ``backbone``, ``projector`` and ``simclr_loss`` submodules.
        batch: Either a multi-view batch (``{"views": [v1, v2], ...}``) for
            training or a single-view batch (``{"image": ...}``) for eval.
        stage: Current stage.

    Returns:
        dict: ``{"embedding", "label"?, "loss"?}`` — ``"loss"`` only while
        training. Embeddings are concatenated across views (``[2B, D]``).
    """
    out: dict[str, jnp.ndarray] = {}
    views = _get_views(batch)
    if views is not None:
        if len(views) != 2:
            raise ValueError(
                f"SimCLR requires exactly 2 views, got {len(views)}. "
                "Implement a custom forward for other configurations."
            )
        # Single backbone/projector pass over both views concatenated, then
        # split. This halves kernel launches vs one call per view (much faster
        # on launch-bound models) and matches the standard SimCLR BN-over-2B.
        n = views[0]["image"].shape[0]
        x = jnp.concatenate([v["image"] for v in views], axis=0)
        embedding = self.backbone(x)
        out["embedding"] = embedding
        if "label" in views[0]:
            out["label"] = jnp.concatenate([v["label"] for v in views], axis=0)
        if self.training:
            z = self.projector(embedding)
            out["loss"] = self.simclr_loss(z[:n], z[n:])
            self.log(f"{stage}/loss", out["loss"])
    else:
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]
    return out


def _two_view_embed(self, batch, out):
    """Shared multi-view path: embed both views, stash embeddings + labels.

    Returns the per-view embedding list (or ``None`` for a single-view batch,
    in which case ``out`` is already populated for eval).
    """
    views = _get_views(batch)
    if views is None:
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]
        return None
    if len(views) != 2:
        raise ValueError(f"Expected exactly 2 views, got {len(views)}.")
    embeddings = [self.backbone(v["image"]) for v in views]
    out["embedding"] = jnp.concatenate(embeddings, axis=0)
    if "label" in views[0]:
        out["label"] = jnp.concatenate([v["label"] for v in views], axis=0)
    return embeddings


def byol(self, batch: dict[str, Any], stage: str) -> dict[str, jnp.ndarray]:
    """BYOL forward :cite:`grill2020bootstrap` (EMA teacher, online predictor).

    Args:
        self: Bound Module with ``backbone`` and ``projector`` as
            :class:`~stable_pretraining.jax.backbone.TeacherStudentWrapper`,
            a plain ``predictor`` MLP, and ``byol_loss``.
        batch: Multi-view (train) or single-view (eval) batch.
        stage: Current stage.

    Returns:
        dict: ``{"embedding", "label"?, "loss"?}``. Embedding is the teacher
        feature (gradient-stopped).
    """
    out: dict[str, jnp.ndarray] = {}
    views = _get_views(batch)
    if views is None:
        out["embedding"] = self.backbone.forward_teacher(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]
        return out
    if len(views) != 2:
        raise ValueError(f"BYOL requires exactly 2 views, got {len(views)}.")
    images = [v["image"] for v in views]
    if "label" in views[0]:
        out["label"] = jnp.concatenate([v["label"] for v in views], axis=0)

    online_feat = [self.backbone.forward_student(img) for img in images]
    if self.training:
        online_pred = [
            self.predictor(self.projector.forward_student(f)) for f in online_feat
        ]
        target_proj = [
            self.projector.forward_teacher(self.backbone.forward_teacher(img))
            for img in images
        ]
        out["loss"] = 0.5 * (
            self.byol_loss(online_pred[0], target_proj[1])
            + self.byol_loss(online_pred[1], target_proj[0])
        )
        out["embedding"] = jnp.concatenate(
            [self.backbone.forward_teacher(img) for img in images], axis=0
        )
        self.log(f"{stage}/loss", out["loss"])
    else:
        out["embedding"] = jnp.concatenate(
            [self.backbone.forward_teacher(img) for img in images], axis=0
        )
    return out


def vicreg(self, batch: dict[str, Any], stage: str) -> dict[str, jnp.ndarray]:
    """VICReg forward (JAX port of :func:`stable_pretraining.forward.vicreg`).

    Args:
        self: Bound Module with ``backbone``, ``projector`` and ``vicreg_loss``.
        batch: Multi-view (train) or single-view (eval) batch.
        stage: Current stage.

    Returns:
        dict: ``{"embedding", "label"?, "loss"?}``.
    """
    out: dict[str, jnp.ndarray] = {}
    embeddings = _two_view_embed(self, batch, out)
    if embeddings is not None and self.training:
        z = [self.projector(e) for e in embeddings]
        out["loss"] = self.vicreg_loss(z[0], z[1])
        self.log(f"{stage}/loss", out["loss"])
    return out


def barlow_twins(self, batch: dict[str, Any], stage: str) -> dict[str, jnp.ndarray]:
    """Barlow Twins forward (JAX port of :func:`stable_pretraining.forward.barlow_twins`).

    Args:
        self: Bound Module with ``backbone``, ``projector`` and ``barlow_loss``.
        batch: Multi-view (train) or single-view (eval) batch.
        stage: Current stage.

    Returns:
        dict: ``{"embedding", "label"?, "loss"?}``.
    """
    out: dict[str, jnp.ndarray] = {}
    embeddings = _two_view_embed(self, batch, out)
    if embeddings is not None and self.training:
        z = [self.projector(e) for e in embeddings]
        out["loss"] = self.barlow_loss(z[0], z[1])
        self.log(f"{stage}/loss", out["loss"])
    return out


def simsiam(self, batch: dict[str, Any], stage: str) -> dict[str, jnp.ndarray]:
    """SimSiam forward :cite:`chen2021exploring` (predictor + stop-gradient, no EMA).

    Args:
        self: Bound Module with ``backbone``, ``projector``, ``predictor`` and
            ``simsiam_loss`` (negative cosine similarity).
        batch: Multi-view (train) or single-view (eval) batch.
        stage: Current stage.

    Returns:
        dict: ``{"embedding", "label"?, "loss"?}``.
    """
    out: dict[str, jnp.ndarray] = {}
    embeddings = _two_view_embed(self, batch, out)
    if embeddings is not None and self.training:
        z = [self.projector(e) for e in embeddings]
        p = [self.predictor(zz) for zz in z]
        # Stop-gradient on the target branch is what prevents collapse.
        loss = 0.5 * (
            self.simsiam_loss(p[0], jax.lax.stop_gradient(z[1]))
            + self.simsiam_loss(p[1], jax.lax.stop_gradient(z[0]))
        )
        out["loss"] = loss
        self.log(f"{stage}/loss", out["loss"])
    return out
