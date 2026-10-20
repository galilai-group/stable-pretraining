"""Forward functions for self-supervised learning methods.

This module provides pre-defined forward functions for various SSL methods
that can be used with the Module class. These functions define the training
logic for each method and can be specified in YAML configs or Python code.

Available forward functions:
    - ``supervised`` — standard supervised training
    - ``simclr`` — SimCLR contrastive learning
    - ``byol`` — BYOL momentum self-distillation
    - ``vicreg`` — VICReg variance-invariance-covariance regularization
    - ``barlow_twins`` — Barlow Twins cross-correlation alignment
    - ``swav`` — SwAV online clustering
    - ``nnclr`` — NNCLR nearest-neighbor contrastive learning
    - ``dino`` — DINO self-distillation with multi-crop
    - ``dinov2`` — DINOv2 with iBOT masked patch prediction
    - ``rotnet`` — RotNet image-rotation prediction

These are the lightweight composable form of each method. For full
``LightningModule`` implementations of 30+ SSL methods (BYOL, DINO, MAE,
iBOT, MoCo, SimMIM, VICRegL, etc.), see ``stable_pretraining/methods/``
and the complete catalog in ``METHODS.md``.

Example:
    Using in a YAML config::

        module:
          _target_: stable_pretraining.Module
          forward: stable_pretraining.forward.simclr
          backbone: ...
          projector: ...

    Using in Python code::

        from stable_pretraining import Module
        from stable_pretraining.forward import simclr

        module = Module(forward=simclr, backbone=backbone, projector=projector)
"""

from typing import Any

import torch

from .callbacks.queue import find_or_create_queue_callback, OnlineQueue


def _get_views_list(batch: dict):
    """Convert multi-view batch to list of views."""
    if "views" in batch:
        return batch["views"]
    elif "image" not in batch:
        # Dict of named views - convert to list
        return list(batch.values())
    else:
        # Single view, return None as sentinel
        return None


def _get_views_by_prefix(
    batch: dict, global_prefix: str = "global", local_prefix: str = "local"
):
    """Separate views by key prefix into global and local views.

    Args:
        batch: Input batch, either:
            - {"views": {"global_1": ..., "local_1": ...}}
            - {"global_1": ..., "local_1": ...} (direct dict)
            - {"image": ...} (single view, returns (None, None))
        global_prefix: Prefix for global view keys (default: "global")
        local_prefix: Prefix for local view keys (default: "local")

    Returns:
        Tuple of (global_views_list, local_views_list, all_views_list), or (None, None, None) for single view.

    Raises:
        ValueError: If views are in list format or if no keys with global_prefix are found
    """
    # Get views dict
    if "views" in batch:
        views_data = batch["views"]
        if isinstance(views_data, list):
            raise ValueError(
                f"Expected dict of named views with '{global_prefix}' and '{local_prefix}' key prefixes, "
                f"but got a list of {len(views_data)} views. "
                "Please use MultiViewTransform with a dict of named views "
                f"(e.g., {{'{global_prefix}_1': ..., '{global_prefix}_2': ..., '{local_prefix}_1': ...}})."
            )
        views_dict = views_data
    elif "image" not in batch:
        # Direct dict of named views
        views_dict = batch
    else:
        # Single view
        return None, None, None

    # Separate views by prefix
    global_views = []
    local_views = []

    for key, view in views_dict.items():
        if key.startswith(global_prefix):
            global_views.append(view)
        elif key.startswith(local_prefix):
            local_views.append(view)

    # Validate that global views exist
    if len(global_views) == 0:
        raise ValueError(
            f"No views with '{global_prefix}' prefix found in keys: {list(views_dict.keys())}. "
            f"Please ensure keys contain '{global_prefix}' (e.g., '{global_prefix}_1', '{global_prefix}_2')."
        )

    all_views = global_views + local_views
    return global_views, local_views, all_views


def supervised(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    """Forward function for standard supervised training.

    This function implements traditional supervised learning with labels,
    useful for baseline comparisons and fine-tuning pre-trained models.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - classifier: Classification head (e.g., Linear layer)
            - supervised_loss: Loss function for supervised learning
        batch: Input batch dictionary containing:
            - 'image': Tensor of images [N, C, H, W]
            - 'label': Ground truth labels [N] (optional, for loss computation)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'logits': Classification predictions
            - 'loss': Supervised loss (if labels provided)

    Note:
        Unlike SSL methods, this function uses actual labels for training
        and is primarily used for evaluation or supervised baselines.

    Example:
        ::

            import torch
            import stable_pretraining as spt
            from stable_pretraining.forward import supervised

            backbone = spt.backbone.from_torchvision("resnet50")
            module = spt.Module(
                forward=supervised,
                backbone=backbone,
                classifier=torch.nn.Linear(2048, 10),
                supervised_loss=torch.nn.CrossEntropyLoss(),
                optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
            )
    """
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    out["logits"] = self.classifier(out["embedding"])

    if "label" in batch:
        if not hasattr(self, "supervised_loss"):
            raise ValueError(
                "supervised requires 'supervised_loss' to be provided (e.g., nn.CrossEntropyLoss()). "
                "Pass it when constructing the Module: Module(..., supervised_loss=nn.CrossEntropyLoss(), ...)"
            )
        out["loss"] = self.supervised_loss(out["logits"], batch["label"])
        self.log(
            f"{stage}/loss",
            out["loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

    return out


def simclr(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    """Forward function for SimCLR (Simple Contrastive Learning of Representations).

    SimCLR learns representations by maximizing agreement between differently
    augmented views of the same image via a contrastive loss in the latent space.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - projector: Projection head mapping features to latent space
            - simclr_loss: NT-Xent contrastive loss function
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': NT-Xent contrastive loss (during training only)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the SimCLR paper :cite:`chen2020simple`.

    Example:
        ::

            import torch
            import stable_pretraining as spt
            from stable_pretraining.forward import simclr

            backbone = spt.backbone.from_torchvision("resnet50")
            module = spt.Module(
                forward=simclr,
                backbone=backbone,
                projector=torch.nn.Linear(2048, 128),
                simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
                optim={"optimizer": {"type": "LARS", "lr": 5}},
            )
    """
    out = {}

    views = _get_views_list(batch)
    if views is not None:
        # Multi-view training - SimCLR requires exactly 2 views
        if len(views) != 2:
            raise ValueError(
                f"SimCLR requires exactly 2 views, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )

        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks (probes need this)
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]
            out["loss"] = self.simclr_loss(projections[0], projections[1])
            self.log(
                f"{stage}/loss",
                out["loss"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
    else:
        # Single-view validation
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


def byol(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    """Forward function for BYOL (Bootstrap Your Own Latent).

    BYOL learns representations without negative pairs by using a momentum-based
    target network and predicting target projections from online projections.

    Args:
        self: Module instance with required attributes:
            - backbone: TeacherStudentWrapper for feature extraction
            - projector: TeacherStudentWrapper for projection head
            - predictor: Online network predictor
            - byol_loss: BYOL loss function (optional, uses MSE if not provided)
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from teacher backbone (EMA target)
            - 'loss': BYOL loss between predictions and targets (during training)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the BYOL paper :cite:`grill2020bootstrap`.

    Example:
        ::

            import torch
            import stable_pretraining as spt
            from stable_pretraining.forward import byol

            backbone = spt.TeacherStudentWrapper(
                spt.backbone.from_torchvision("resnet50")
            )
            projector = spt.TeacherStudentWrapper(torch.nn.Linear(2048, 256))
            module = spt.Module(
                forward=byol,
                backbone=backbone,
                projector=projector,
                predictor=torch.nn.Linear(256, 256),
                byol_loss=spt.losses.BYOLLoss(),
                optim={"optimizer": {"type": "Adam", "lr": 3e-4}},
            )
    """
    out = {}

    views = _get_views_list(batch)
    if views is not None:
        # Multi-view training - BYOL requires exactly 2 views
        if len(views) != 2:
            raise ValueError(
                f"BYOL requires exactly 2 views, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )

        images = [view["image"] for view in views]

        # Concatenate labels for callbacks
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        # Get online embeddings for both views
        online_features = [self.backbone.forward_student(img) for img in images]

        # Return early if not training
        if not self.training:
            with torch.no_grad():
                all_images = torch.cat(images, dim=0)
                target_only_features = self.backbone.forward_teacher(all_images)
            return {"embedding": target_only_features.detach(), **out}

        # Process online network
        online_proj = [self.projector.forward_student(feat) for feat in online_features]
        online_pred = [self.predictor(proj) for proj in online_proj]

        # Process target network
        with torch.no_grad():
            target_features = [self.backbone.forward_teacher(img) for img in images]
            target_proj = [
                self.projector.forward_teacher(feat) for feat in target_features
            ]

        if not hasattr(self, "byol_loss"):
            raise ValueError(
                "byol requires 'byol_loss' to be provided (e.g., spt.losses.BYOLLoss()). "
                "Pass it when constructing the Module: Module(..., byol_loss=spt.losses.BYOLLoss(), ...)"
            )

        # Compute loss between view pairs
        loss = (
            self.byol_loss(online_pred[0], target_proj[1])
            + self.byol_loss(online_pred[1], target_proj[0])
        ) / 2

        out["embedding"] = torch.cat(target_features, dim=0).detach()
        out["loss"] = loss
        self.log(
            f"{stage}/loss",
            out["loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

    else:
        # Single-view validation
        images = batch["image"]

        if "label" in batch:
            out["label"] = batch["label"]

        # Just return embeddings for validation
        with torch.no_grad():
            target_only_features = self.backbone.forward_teacher(images)
        out["embedding"] = target_only_features.detach()

    return out


def vicreg(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    """Forward function for VICReg (Variance-Invariance-Covariance Regularization).

    VICReg learns representations using three criteria: variance (maintaining
    information), invariance (to augmentations), and covariance (decorrelating
    features).

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - projector: Projection head for embedding transformation
            - vicreg_loss: VICReg loss with variance, invariance, covariance terms
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': Combined VICReg loss (during training only)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the VICReg paper :cite:`bardes2021vicreg`.

    Example:
        ::

            import torch
            import stable_pretraining as spt
            from stable_pretraining.forward import vicreg

            backbone = spt.backbone.from_torchvision("resnet50")
            module = spt.Module(
                forward=vicreg,
                backbone=backbone,
                projector=torch.nn.Linear(2048, 2048),
                vicreg_loss=spt.losses.VICRegLoss(),
                optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
            )
    """
    out = {}

    views = _get_views_list(batch)
    if views is not None:
        # Multi-view training - VICReg requires exactly 2 views
        if len(views) != 2:
            raise ValueError(
                f"VICReg requires exactly 2 views, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )

        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]
            out["loss"] = self.vicreg_loss(projections[0], projections[1])
            self.log(
                f"{stage}/loss",
                out["loss"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
    else:
        # Single-view validation
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


def barlow_twins(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    """Forward function for Barlow Twins.

    Barlow Twins learns representations by making the cross-correlation matrix
    between embeddings of augmented views as close to the identity matrix as
    possible, reducing redundancy while maintaining invariance.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - projector: Projection head (typically with BN and high dimension)
            - barlow_loss: Barlow Twins loss function
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': Barlow Twins loss (during training only)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the Barlow Twins paper :cite:`zbontar2021barlow`.

    Example:
        ::

            import torch
            import stable_pretraining as spt
            from stable_pretraining.forward import barlow_twins

            backbone = spt.backbone.from_torchvision("resnet50")
            module = spt.Module(
                forward=barlow_twins,
                backbone=backbone,
                projector=torch.nn.Linear(2048, 8192),
                barlow_loss=spt.losses.BarlowTwinsLoss(lambda_coeff=5e-3),
                optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
            )
    """
    out = {}

    views = _get_views_list(batch)
    if views is not None:
        # Multi-view training - Barlow Twins requires exactly 2 views
        if len(views) != 2:
            raise ValueError(
                f"Barlow Twins requires exactly 2 views, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )

        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]
            out["loss"] = self.barlow_loss(projections[0], projections[1])
            self.log(
                f"{stage}/loss",
                out["loss"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
    else:
        # Single-view validation
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


def swav(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    """Forward function for SwAV (Swapping Assignments between Views).

    SwAV learns representations by predicting the cluster assignment (code) of one
    view from the representation of another view. For small-batch training, this function
    manages a feature queue to stabilize the training process.

    Example:
        ::

            import torch
            import stable_pretraining as spt
            from stable_pretraining.forward import swav

            backbone = spt.backbone.from_torchvision("resnet50")
            prototypes = torch.nn.Linear(128, 3000, bias=False)
            module = spt.Module(
                forward=swav,
                backbone=backbone,
                projector=torch.nn.Linear(2048, 128),
                prototypes=prototypes,
                swav_loss=spt.losses.SwAVLoss(),
                optim={"optimizer": {"type": "LARS", "lr": 5}},
            )
    """
    out = {}
    views = _get_views_list(batch)

    if views is not None:
        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)

        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]
            proj1, proj2 = projections[0], projections[1]

            queue_feats = None
            use_queue_logic = hasattr(self, "use_queue") and self.use_queue

            if use_queue_logic:
                if not hasattr(self, "_swav_queue_callback"):
                    self._swav_queue_callback = find_or_create_queue_callback(
                        self.trainer,
                        key="swav_queue",
                        queue_length=self.queue_length,
                        dim=self.projection_dim,
                    )
                queue = OnlineQueue._shared_queues.get(self._swav_queue_callback.key)

                if (
                    queue is not None
                    and len(queue.get()) > 0
                    and self.trainer.current_epoch >= self.start_queue_at_epoch
                ):
                    queue_feats = queue.get().clone().detach().to(proj1.device)

                out["swav_queue"] = torch.cat(projections).detach()

            out["loss"] = self.swav_loss(proj1, proj2, self.prototypes, queue_feats)
            self.log(
                f"{stage}/loss",
                out["loss"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

    else:
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


def _find_nearest_neighbors(query, support_set):
    """Find the nearest neighbor for each query embedding in the support set."""
    query_norm = torch.nn.functional.normalize(query, dim=1)
    support_norm = torch.nn.functional.normalize(support_set, dim=1)
    similarity = torch.mm(query_norm, support_norm.t())
    _, indices = similarity.max(dim=1)
    return support_set[indices]


def nnclr(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    """Forward function for NNCLR (Nearest-Neighbor Contrastive Learning).

    NNCLR learns representations by using the nearest neighbor of an augmented
    view from a support set of past embeddings as a positive pair. This encourages
    the model to learn representations that are similar for semantically similar
    instances, not just for different augmentations of the same instance.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - projector: Projection head for embedding transformation
            - predictor: Prediction head used for the online view
            - nnclr_loss: NTXent contrastive loss function
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': NTXent contrastive loss (during training only)
            - 'nnclr_support_set': Projections to be added to the support set queue
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the NNCLR paper :cite:`dwibedi2021little`.

    Example:
        ::

            import torch
            import stable_pretraining as spt
            from stable_pretraining.forward import nnclr

            backbone = spt.backbone.from_torchvision("resnet50")
            module = spt.Module(
                forward=nnclr,
                backbone=backbone,
                projector=torch.nn.Linear(2048, 256),
                predictor=torch.nn.Linear(256, 256),
                nnclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
                support_set_size=16384,
                projection_dim=256,
                optim={"optimizer": {"type": "LARS", "lr": 5}},
            )
    """
    out = {}

    views = _get_views_list(batch)
    if views is not None:
        # Multi-view training - NNCLR requires exactly 2 views
        if len(views) != 2:
            raise ValueError(
                f"NNCLR requires exactly 2 views, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )

        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        if self.training:
            if not hasattr(self, "_nnclr_queue_callback"):
                self._nnclr_queue_callback = find_or_create_queue_callback(
                    self.trainer,
                    key="nnclr_support_set",
                    queue_length=self.hparams.support_set_size,
                    dim=self.hparams.projection_dim,
                )
            queue_callback = self._nnclr_queue_callback

            projections = [self.projector(emb) for emb in embeddings]
            proj_q, proj_k = projections[0], projections[1]

            support_set = OnlineQueue._shared_queues.get(queue_callback.key).get()

            if support_set is not None and len(support_set) > 0:
                pred_q = self.predictor(proj_q)
                pred_k = self.predictor(proj_k)

                nn_k = _find_nearest_neighbors(proj_k, support_set).detach()
                nn_q = _find_nearest_neighbors(proj_q, support_set).detach()

                loss_a = self.nnclr_loss(pred_q, nn_k)
                loss_b = self.nnclr_loss(pred_k, nn_q)
                out["loss"] = (loss_a + loss_b) / 2.0
            else:
                # Fallback to SimCLR style loss if queue is empty
                out["loss"] = self.nnclr_loss(proj_q, proj_k)

            self.log(
                f"{stage}/loss",
                out["loss"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

            # The key here must match the `key` argument of the `OnlineQueue` callback
            out["nnclr_support_set"] = torch.cat(projections)

    else:
        # Single-view validation
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


def dino(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    """Forward function for DINO (self-DIstillation with NO labels).

    DINO learns representations through self-distillation where a student network
    is trained to match the output of a teacher network (EMA of student) on
    different augmented views. Global views are processed by both networks while
    local views are only processed by the student.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: TeacherStudentWrapper for feature extraction
            - projector: TeacherStudentWrapper for projection head
            - dino_loss: DINOv1Loss instance (required, pass spt.losses.DINOv1Loss())
            - warmup_temperature_teacher (float): Starting teacher temperature
            - temperature_teacher (float): Final teacher temperature
            - warmup_epochs_temperature_teacher (int): Epochs to warm up temperature
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view).
            For multi-crop: First 2 views should be global crops, rest are local crops
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from teacher backbone
            - 'loss': DINO distillation loss (during training only)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the DINO paper :cite:`caron2021emerging`.
        Requires TeacherStudentWrapper for both backbone and projector,
        and assumes first 2 views in batch are global views.

    Example:
        ::

            import stable_pretraining as spt
            from stable_pretraining.forward import dino

            backbone = spt.TeacherStudentWrapper(
                spt.backbone.vit_hf("google/vit-base-patch16-224")
            )
            projector = spt.TeacherStudentWrapper(spt.backbone.MLP(768, [2048], 65536))
            module = spt.Module(
                forward=dino,
                backbone=backbone,
                projector=projector,
                dino_loss=spt.losses.DINOv1Loss(out_dim=65536),
                warmup_temperature_teacher=0.04,
                temperature_teacher=0.07,
                warmup_epochs_temperature_teacher=30,
                optim={"optimizer": {"type": "Adam", "lr": 5e-4}},
            )
    """
    out = {}

    # Use prefix-based view separation for DINO's global/local distinction
    global_views, local_views, all_views = _get_views_by_prefix(
        batch, global_prefix="global", local_prefix="local"
    )

    if all_views is not None:
        # Multi-view training
        n_global = len(global_views)
        n_local = len(local_views)
    else:
        # Single view validation
        images = batch["image"]
        if "label" in batch:
            out["label"] = batch["label"]

        with torch.no_grad():
            teacher_features = self.backbone.forward_teacher(images)
        out["embedding"] = teacher_features.last_hidden_state[:, 0].detach()
        return out

    # Multi-view processing
    batch_size = all_views[0]["image"].shape[0]

    # Concatenate labels for callbacks - only from global views for probes
    if "label" in all_views[0]:
        # For training, only use labels from global views since we only return global embeddings
        if self.training:
            out["label"] = torch.cat([view["label"] for view in global_views], dim=0)
        else:
            # For validation, use all labels since we return all embeddings
            out["label"] = torch.cat([view["label"] for view in all_views], dim=0)

    if not self.training:
        # During validation, just process all views through teacher
        all_images = torch.cat([view["image"] for view in all_views], dim=0)
        with torch.no_grad():
            teacher_features = self.backbone.forward_teacher(all_images)
        out["embedding"] = teacher_features.last_hidden_state[:, 0].detach()
        return out

    # Training: separate processing for global and local views
    global_images = torch.cat([view["image"] for view in global_views], dim=0)

    # Teacher processes only global views
    with torch.no_grad():
        teacher_features = self.backbone.forward_teacher(global_images)
        # Extract CLS token from HF output
        if hasattr(teacher_features, "last_hidden_state"):
            teacher_cls_features = teacher_features.last_hidden_state[:, 0, :]
        else:
            teacher_cls_features = (
                teacher_features[:, 0, :]
                if teacher_features.ndim == 3
                else teacher_features
            )
        teacher_logits = self.projector.forward_teacher(teacher_cls_features)
        teacher_logits = teacher_logits.view(n_global, batch_size, -1)

    # Student processes all views
    student_logits_list = []

    # Process global views through student
    student_features = self.backbone.forward_student(global_images)
    # Extract CLS token from HF output
    if hasattr(student_features, "last_hidden_state"):
        student_cls_features = student_features.last_hidden_state[:, 0, :]
    else:
        student_cls_features = (
            student_features[:, 0, :]
            if student_features.ndim == 3
            else student_features
        )
    student_global_logits = self.projector.forward_student(student_cls_features)
    student_global_logits = student_global_logits.view(n_global, batch_size, -1)
    student_logits_list.append(student_global_logits)

    # Process local views through student (if any)
    if n_local > 0:
        local_images = torch.cat([view["image"] for view in local_views], dim=0)
        student_features = self.backbone.forward_student(
            local_images, interpolate_pos_encoding=True
        )
        # Extract CLS token from HF output
        if hasattr(student_features, "last_hidden_state"):
            student_local_cls_features = student_features.last_hidden_state[:, 0, :]
        else:
            student_local_cls_features = (
                student_features[:, 0, :]
                if student_features.ndim == 3
                else student_features
            )
        student_local_logits = self.projector.forward_student(
            student_local_cls_features
        )
        student_local_logits = student_local_logits.view(n_local, batch_size, -1)
        student_logits_list.append(student_local_logits)

    # Concatenate student logits along the view dimension
    student_logits = torch.cat(student_logits_list, dim=0)

    if not hasattr(self, "dino_loss"):
        raise ValueError(
            "dino requires 'dino_loss' to be provided (e.g., spt.losses.DINOv1Loss()). "
            "Pass it when constructing the Module: Module(..., dino_loss=spt.losses.DINOv1Loss(), ...)"
        )

    if not hasattr(self, "projector"):
        raise ValueError(
            "dino requires 'projector' to be provided. "
            "This should be a TeacherStudentWrapper containing the projector (MLP + normalize + prototypes). "
            "Pass it when constructing the Module: Module(..., projector=wrapped_projector, ...)"
        )

    # Temperature scheduling for teacher
    if (
        hasattr(self, "warmup_epochs_temperature_teacher")
        and hasattr(self, "warmup_temperature_teacher")
        and hasattr(self, "temperature_teacher")
    ):
        if self.current_epoch < self.warmup_epochs_temperature_teacher:
            # Linear warmup from warmup_temperature_teacher to temperature_teacher
            progress = self.current_epoch / self.warmup_epochs_temperature_teacher
            temperature_teacher = self.warmup_temperature_teacher + progress * (
                self.temperature_teacher - self.warmup_temperature_teacher
            )
        else:
            temperature_teacher = self.temperature_teacher
    else:
        # Default temperature if attributes not set
        temperature_teacher = getattr(self, "temperature_teacher", 0.07)

    # Process teacher logits with classical centering (DINOv1 approach)
    teacher_probs = self.dino_loss.softmax_center_teacher(
        teacher_logits, teacher_temp=temperature_teacher
    )

    # Compute DINO loss
    loss = self.dino_loss(student_logits, teacher_probs)

    # Queue async center update (will be applied next iteration)
    self.dino_loss.update_center(teacher_logits)

    out["embedding"] = teacher_cls_features.detach()
    out["loss"] = loss
    self.log(
        f"{stage}/loss",
        out["loss"],
        on_step=True,
        on_epoch=True,
        sync_dist=True,
        prog_bar=True,
    )

    return out


def dinov2(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    """Forward function for DINOv2 with iBOT.

    DINOv2 combines two self-supervised losses:
    - DINO: CLS token distillation between global views
    - iBOT: Masked patch prediction

    Both losses use Sinkhorn-Knopp normalization for optimal transport.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: TeacherStudentWrapper for feature extraction (ViT)
            - projector: TeacherStudentWrapper for CLS token projection head
            - patch_projector: TeacherStudentWrapper for patch projection head
            - dinov2_loss: DINOv2Loss instance combining DINO + iBOT
            - warmup_temperature_teacher (float): Starting teacher temperature
            - temperature_teacher (float): Final teacher temperature
            - warmup_epochs_temperature_teacher (int): Epochs to warm up temperature
            - mask_ratio (float): Ratio of patches to mask for iBOT (default: 0.3)
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view).
            For multi-crop: First 2 views should be global crops, rest are local crops
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from teacher backbone
            - 'loss': Combined DINOv2 loss (DINO + iBOT)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the DINOv2 paper.
        Requires TeacherStudentWrapper for backbone, projector, and patch_projector.
        Assumes first 2 views in batch are global views.

    Example:
        ::

            import stable_pretraining as spt
            from stable_pretraining.forward import dinov2

            vit = spt.backbone.vit_hf(
                "google/vit-base-patch16-224", use_mask_token=True
            )
            backbone = spt.TeacherStudentWrapper(vit)
            projector = spt.TeacherStudentWrapper(spt.backbone.MLP(768, [2048], 65536))
            patch_projector = spt.TeacherStudentWrapper(
                spt.backbone.MLP(768, [2048], 65536)
            )
            module = spt.Module(
                forward=dinov2,
                backbone=backbone,
                projector=projector,
                patch_projector=patch_projector,
                dinov2_loss=spt.losses.DINOv2Loss(out_dim=65536),
                warmup_temperature_teacher=0.04,
                temperature_teacher=0.07,
                warmup_epochs_temperature_teacher=30,
                mask_ratio=0.3,
                optim={"optimizer": {"type": "Adam", "lr": 5e-4}},
            )
    """
    out = {}

    # Use prefix-based view separation for DINOv2's global/local distinction
    global_views, local_views, all_views = _get_views_by_prefix(
        batch, global_prefix="global", local_prefix="local"
    )

    if all_views is not None:
        # Multi-view training
        n_global = len(global_views)
        n_local = len(local_views)
    else:
        # Single view validation
        images = batch["image"]
        if "label" in batch:
            out["label"] = batch["label"]

        with torch.no_grad():
            teacher_features = self.backbone.forward_teacher(images)
        out["embedding"] = teacher_features.last_hidden_state[:, 0].detach()
        return out

    # Multi-view processing
    batch_size = all_views[0]["image"].shape[0]

    # Concatenate labels for callbacks - only from global views for probes
    if "label" in all_views[0]:
        if self.training:
            out["label"] = torch.cat([view["label"] for view in global_views], dim=0)
        else:
            out["label"] = torch.cat([view["label"] for view in all_views], dim=0)

    if not self.training:
        # During validation, just process all views through teacher
        all_images = torch.cat([view["image"] for view in all_views], dim=0)
        with torch.no_grad():
            teacher_features = self.backbone.forward_teacher(all_images)
        out["embedding"] = teacher_features.last_hidden_state[:, 0].detach()
        return out

    # Training: separate processing for global and local views
    global_images = torch.cat([view["image"] for view in global_views], dim=0)

    # Temperature scheduling for teacher
    if (
        hasattr(self, "warmup_epochs_temperature_teacher")
        and hasattr(self, "warmup_temperature_teacher")
        and hasattr(self, "temperature_teacher")
    ):
        if self.current_epoch < self.warmup_epochs_temperature_teacher:
            progress = self.current_epoch / self.warmup_epochs_temperature_teacher
            temperature_teacher = self.warmup_temperature_teacher + progress * (
                self.temperature_teacher - self.warmup_temperature_teacher
            )
        else:
            temperature_teacher = self.temperature_teacher
    else:
        temperature_teacher = getattr(self, "temperature_teacher", 0.04)

    # Get underlying backbones (handle TeacherStudentWrapper)
    teacher_backbone = getattr(self.backbone, "teacher", self.backbone)
    student_backbone = getattr(self.backbone, "student", self.backbone)

    # ===== Extract tokens ONCE from teacher (used for both CLS and patches) =====

    with torch.no_grad():
        # Extract teacher tokens for global views
        # Support both HuggingFace ViT (returns dict with last_hidden_state) and timm ViT
        teacher_output = teacher_backbone(global_images)

        if hasattr(teacher_output, "last_hidden_state"):
            # HuggingFace ViT: output.last_hidden_state is [B, num_patches+1, hidden_size]
            teacher_cls_features = teacher_output.last_hidden_state[
                :, 0, :
            ]  # CLS token
            teacher_patches_all = teacher_output.last_hidden_state[
                :, 1:, :
            ]  # Patch tokens
        else:
            # Fallback to timm or custom ViT (assumes returns tensor [B, num_patches+1, hidden_size])
            teacher_cls_features = teacher_output[:, 0, :]
            teacher_patches_all = teacher_output[:, 1:, :]

        # Reshape CLS features for multi-view: [n_global * batch_size, dim] -> [n_global, batch_size, dim]
        teacher_cls_features = teacher_cls_features.view(n_global, batch_size, -1)

        # CLS token projection
        teacher_cls_logits = self.projector.forward_teacher(
            teacher_cls_features.view(n_global * batch_size, -1)
        )
        teacher_cls_logits = teacher_cls_logits.view(n_global, batch_size, -1)

        # Apply Sinkhorn-Knopp to CLS tokens
        flat_cls_logits = teacher_cls_logits.view(-1, teacher_cls_logits.shape[-1])
        teacher_cls_probs = self.dinov2_loss.dino_loss.sinkhorn_knopp_teacher(
            flat_cls_logits,
            teacher_temp=temperature_teacher,
            num_samples=n_global * batch_size,
        )
        teacher_cls_probs = teacher_cls_probs.view(n_global, batch_size, -1)

    # Student processes all views for CLS tokens
    student_logits_list = []

    # Process global views through student
    student_output = self.backbone.forward_student(global_images)
    if hasattr(student_output, "last_hidden_state"):
        student_cls_features = student_output.last_hidden_state[:, 0, :]
    else:
        student_cls_features = (
            student_output[:, 0, :] if student_output.ndim == 3 else student_output
        )

    student_global_cls_logits = self.projector.forward_student(student_cls_features)
    student_global_cls_logits = student_global_cls_logits.view(n_global, batch_size, -1)
    student_logits_list.append(student_global_cls_logits)

    # Process local views through student (if any)
    if n_local > 0:
        local_images = torch.cat([view["image"] for view in local_views], dim=0)
        # For HF ViT: need to pass interpolate_pos_encoding=True for different sized images
        student_local_output = self.backbone.forward_student(
            local_images, interpolate_pos_encoding=True
        )
        if hasattr(student_local_output, "last_hidden_state"):
            student_local_cls_features = student_local_output.last_hidden_state[:, 0, :]
        else:
            student_local_cls_features = (
                student_local_output[:, 0, :]
                if student_local_output.ndim == 3
                else student_local_output
            )

        student_local_cls_logits = self.projector.forward_student(
            student_local_cls_features
        )
        student_local_cls_logits = student_local_cls_logits.view(
            n_local, batch_size, -1
        )
        student_logits_list.append(student_local_cls_logits)

    # Concatenate student CLS logits along the view dimension
    student_cls_logits = torch.cat(student_logits_list, dim=0)

    # ===== Patch Processing (iBOT) =====

    if hasattr(self, "patch_projector"):
        mask_ratio = getattr(self, "mask_ratio", 0.3)

        try:
            # Process first global view only for iBOT
            first_view = global_images[:batch_size]

            with torch.no_grad():
                # Extract teacher patches from first view (already computed above)
                teacher_patches = teacher_patches_all[
                    :batch_size
                ]  # [batch_size, n_patches, dim]

                # Create random mask: True = masked, False = visible (vectorized exact-k per sample)
                B, N, C = teacher_patches.shape
                n_masked = int(N * mask_ratio)

                bool_masked_pos = torch.zeros(
                    B, N, dtype=torch.bool, device=teacher_patches.device
                )
                # Efficient: topk on random values (O(N) vs argsort O(N log N))
                _, mask_indices = torch.topk(
                    torch.rand(B, N, device=teacher_patches.device), n_masked, dim=1
                )
                bool_masked_pos.scatter_(1, mask_indices, True)

                # Extract patches that WERE masked (these are visible to teacher, masked for student)
                teacher_masked_patches = teacher_patches[
                    bool_masked_pos
                ]  # [B*n_masked, C]
                teacher_patch_logits = self.patch_projector.forward_teacher(
                    teacher_masked_patches
                )

                # Apply Sinkhorn-Knopp to masked patches
                n_masked_total = teacher_masked_patches.shape[0]
                teacher_patch_probs = self.dinov2_loss.ibot_loss.sinkhorn_knopp_teacher(
                    teacher_patch_logits,
                    teacher_temp=temperature_teacher,
                    num_samples=n_masked_total,
                )

            # Student sees MASKED input (mask tokens injected at masked positions)
            # HuggingFace ViT handles masking natively via bool_masked_pos parameter
            student_output_masked = student_backbone(
                first_view, bool_masked_pos=bool_masked_pos
            )

            if hasattr(student_output_masked, "last_hidden_state"):
                student_patches = student_output_masked.last_hidden_state[
                    :, 1:, :
                ]  # Skip CLS
            else:
                # If not HF ViT, assume it's a raw tensor
                student_patches = (
                    student_output_masked[:, 1:, :]
                    if student_output_masked.ndim == 3
                    else student_output_masked
                )

            # Extract the MASK TOKEN OUTPUTS from student (at masked positions)
            student_masked_patches = student_patches[bool_masked_pos]  # [B*n_masked, C]
            student_patch_logits = self.patch_projector.forward_student(
                student_masked_patches
            )

            if not hasattr(self, "dinov2_loss"):
                raise ValueError(
                    "dinov2 requires 'dinov2_loss' to be provided (e.g., spt.losses.DINOv2Loss()). "
                    "Pass it when constructing the Module: Module(..., dinov2_loss=spt.losses.DINOv2Loss(), ...)"
                )

            loss = self.dinov2_loss(
                student_cls_logits=student_cls_logits,
                teacher_cls_probs=teacher_cls_probs,
                student_patch_logits=student_patch_logits,
                teacher_patch_probs=teacher_patch_probs,
            )

        except (RuntimeError, TypeError) as e:
            # Backbone doesn't support HF ViT masking - provide clear error message
            raise RuntimeError(
                f"DINOv2 with iBOT (patch_projector) requires a HuggingFace ViT backbone that supports "
                f"the 'bool_masked_pos' parameter for masking. "
                f"\n\nPlease use: backbone = spt.backbone.vit_hf('tiny', use_mask_token=True)"
                f"\n\nOriginal error: {e}"
            ) from e
    else:
        # No patch_projector provided - raise error instead of falling back to DINO-only
        raise ValueError(
            "dinov2 requires 'patch_projector' for iBOT loss. "
            "Either provide a patch_projector: Module(..., patch_projector=...) "
            "or use dino instead if you only want the DINO (CLS token) loss without iBOT."
        )

    # Return flattened CLS features for callbacks (probes expect [n_global * batch_size, dim])
    out["embedding"] = teacher_cls_features.view(n_global * batch_size, -1).detach()
    out["loss"] = loss
    self.log(
        f"{stage}/loss",
        out["loss"],
        on_step=True,
        on_epoch=True,
        sync_dist=True,
        prog_bar=True,
    )

    return out


ROTNET_ROTATIONS = (0, 90, 180, 270)


def _rotate_batch(
    images: torch.Tensor, rotations: tuple[int, ...] = ROTNET_ROTATIONS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack every rotation of a batch along the batch dimension.

    Args:
        images: Square images ``[B, C, H, W]``.
        rotations: Counter-clockwise angles in degrees, each a multiple of 90.

    Returns:
        ``(rotated, labels)``: ``rotated`` is ``[R*B, C, H, W]`` where block
        ``r`` holds ``images`` rotated by ``rotations[r]``, and ``labels`` is
        ``[R*B]`` with value ``r`` for that block.
    """
    if images.shape[-1] != images.shape[-2]:
        raise ValueError(
            f"RotNet needs square images, got {tuple(images.shape[-2:])}: a 90 "
            "degree rotation would change the input shape."
        )
    bad = [r for r in rotations if r % 90 != 0]
    if bad:
        raise ValueError(f"rotations must be multiples of 90 degrees, got {bad}")
    rotated = torch.cat(
        [torch.rot90(images, k=(r // 90) % 4, dims=(2, 3)) for r in rotations]
    )
    labels = torch.arange(len(rotations), device=images.device)
    return rotated, labels.repeat_interleave(images.shape[0])


def _rotnet_step(
    backbone: torch.nn.Module,
    rotation_head: torch.nn.Module,
    images: torch.Tensor,
    rotations: tuple[int, ...],
    loss_fn: Any,
) -> dict[str, torch.Tensor]:
    """Rotation-prediction loss shared by :func:`rotnet` and ``methods.RotNet``.

    Args:
        backbone: Feature extractor returning ``[N, C, H, W]`` or ``[N, D]``.
        rotation_head: Maps backbone features to one logit per rotation.
        images: Square images ``[B, C, H, W]``.
        rotations: Angles in degrees, each a multiple of 90.
        loss_fn: Callable on ``(logits, rotation_labels)``.

    Returns:
        Dictionary with 'loss', 'embedding' (unrotated images, pooled to
        ``[B, D]``), 'rotation_logits' and 'rotation_accuracy'.
    """
    rotated, rotation_labels = _rotate_batch(images, rotations)
    features = backbone(rotated)
    logits = rotation_head(features)
    loss = loss_fn(logits, rotation_labels)

    batch_size = images.shape[0]
    if 0 in rotations:
        start = rotations.index(0) * batch_size
        embedding = features[start : start + batch_size]
    else:
        # Eval mode so this probe-only pass leaves batch-norm statistics untouched.
        was_training = backbone.training
        backbone.eval()
        with torch.no_grad():
            embedding = backbone(images)
        backbone.train(was_training)
    if embedding.ndim == 4:
        embedding = embedding.mean(dim=(2, 3))

    with torch.no_grad():
        accuracy = (logits.argmax(dim=1) == rotation_labels).float().mean()
    return {
        "loss": loss,
        "embedding": embedding,
        "rotation_logits": logits,
        "rotation_accuracy": accuracy,
    }


def rotnet(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    """Forward function for RotNet (rotation-prediction pretext task).

    Every image is rotated by each angle in ``self.rotations`` (default 0, 90,
    180 and 270 degrees) and the network classifies which rotation it sees.
    All rotations go through the backbone in one pass, so they share
    batch-norm statistics as in the official implementation.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extractor returning ``[N, C, H, W]`` or ``[N, D]``
            - rotation_head: Maps backbone features to one logit per rotation
            - rotnet_loss: Rotation loss, e.g. ``nn.CrossEntropyLoss()``
            - rotations (optional): Angles; defaults to ``(0, 90, 180, 270)``
        batch: Input batch dictionary containing:
            - 'image': Square images ``[B, C, H, W]``, already cropped and flipped
            - 'label': Class labels ``[B]`` (optional, passed through for probes)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Features of the unrotated images ``[B, D]``,
              globally average-pooled when the backbone returns a feature map
            - 'loss': Rotation-classification loss, in every stage so the
              rotation accuracy can be tracked on validation data
            - 'rotation_logits': Logits ``[R*B, R]``
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in Gidaris et al., "Unsupervised Representation Learning by
        Predicting Image Rotations" (ICLR 2018). The loss has no cross-sample
        terms, so under DDP each rank computes it on its local batch and only
        gradients are averaged. Probes see only the unrotated images. If
        ``rotations`` excludes 0, the embedding comes from an extra no-grad,
        eval-mode pass over the unrotated images, so batch-norm statistics
        still come only from the rotations being trained on.

    Example:
        ::

            import torch
            import stable_pretraining as spt
            from stable_pretraining.forward import rotnet

            nin = spt.backbone.NetworkInNetwork(num_blocks=4, pool=False)
            module = spt.Module(
                forward=rotnet,
                backbone=torch.nn.Sequential(*nin.blocks[:2]),
                rotation_head=torch.nn.Sequential(
                    *nin.blocks[2:],
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(192, 4),
                ),
                rotnet_loss=torch.nn.CrossEntropyLoss(),
                optim={"optimizer": {"type": "SGD", "lr": 0.1, "momentum": 0.9}},
            )
    """
    if not hasattr(self, "rotnet_loss"):
        raise ValueError(
            "rotnet requires 'rotnet_loss' to be provided (e.g., nn.CrossEntropyLoss()). "
            "Pass it when constructing the Module: Module(..., rotnet_loss=nn.CrossEntropyLoss(), ...)"
        )
    rotations = tuple(getattr(self, "rotations", ROTNET_ROTATIONS))
    step = _rotnet_step(
        self.backbone,
        self.rotation_head,
        batch["image"],
        rotations,
        self.rotnet_loss,
    )
    out = {
        "embedding": step["embedding"],
        "loss": step["loss"],
        "rotation_logits": step["rotation_logits"],
    }
    if "label" in batch:
        out["label"] = batch["label"]

    self.log(
        f"{stage}/loss",
        step["loss"],
        on_step=True,
        on_epoch=True,
        sync_dist=True,
        prog_bar=True,
    )
    self.log(
        f"{stage}/rotation_accuracy",
        step["rotation_accuracy"],
        on_step=False,
        on_epoch=True,
        sync_dist=True,
    )
    return out
