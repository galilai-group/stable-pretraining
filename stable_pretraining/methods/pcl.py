"""PCL: Prototypical Contrastive Learning (Li et al., ICLR 2021).

Reference: https://arxiv.org/abs/2005.04966
"""

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch.nn as nn
from transformers.utils import ModelOutput
from torch.utils.data import DataLoader

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper, from_timm
from stable_pretraining.callbacks.queues import UnsortedQueue

from stable_pretraining.losses.pcl import ProtoNCELoss, select_prototypes

from lightning.pytorch.callbacks import Callback


@torch.no_grad()
def kmeans(
    features: torch.Tensor,
    num_clusters: int,
    num_iterations: int = 20,
    seed: int = 0,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Euclidean k-means for PCL.

    Returns:
        assignments: Cluster IDs [N].
        prototypes: Normalized centroids [K, D].
        concentrations: Prototype temperatures [K].
    """
    if features.ndim != 2 or not features.is_floating_point():
        raise ValueError("features must be a floating-point [N, D] tensor")
    if not torch.isfinite(features).all().item():
        raise ValueError("features must be finite")
    if num_iterations < 1:
        raise ValueError("num_iterations must be positive")

    n, _ = features.shape

    if not 1 <= num_clusters <= n:
        raise ValueError("num_clusters must be between 1 and N")
    if num_clusters == n:
        raise ValueError(
            "PCL concentration estimation requires some non-singleton clusters"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # Repeated identical vectors cannot be separated reliably by
    # nearest-centroid assignment alone.
    if torch.unique(features, dim=0).shape[0] < num_clusters:
        raise ValueError("Not enough distinct features for clustering")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(n, generator=generator)[:num_clusters]
    centroids = features[indices.to(features.device)].clone()

    def repair_empty_clusters(assignments, centers):
        """Give empty clusters samples from clusters with >1 member."""
        counts = torch.bincount(assignments, minlength=num_clusters)
        empty_ids = torch.where(counts == 0)[0]

        for empty_id in empty_ids.tolist():
            # Choose the point furthest from its assigned center,
            # considering only clusters that can spare a member.
            distances = (features - centers[assignments]).square().sum(dim=1)

            eligible = counts[assignments] > 1
            distances[~eligible] = -1

            donor = int(distances.argmax().item())
            old_id = int(assignments[donor].item())

            assignments[donor] = empty_id
            counts[old_id] -= 1
            counts[empty_id] += 1

        return assignments

    def compute_centroids(assignments):
        centers = torch.zeros_like(centroids)
        centers.index_add_(0, assignments, features)

        counts = torch.bincount(assignments, minlength=num_clusters)
        centers /= counts[:, None]
        return centers

    # Lloyd iterations with empty-cluster repair.
    for _ in range(num_iterations):
        distances = torch.cdist(features, centroids).square()
        assignments = distances.argmin(dim=1)
        assignments = repair_empty_clusters(assignments, centroids)

        updated = compute_centroids(assignments)

        if torch.allclose(updated, centroids, atol=1e-6, rtol=1e-6):
            centroids = updated
            break

        centroids = updated

    # Final assignment using latest centroids.
    assignments = torch.cdist(features, centroids).square().argmin(dim=1)

    assignments = repair_empty_clusters(assignments, centroids)

    # Recompute means to match the returned assignments.
    raw_centroids = compute_centroids(assignments)

    concentrations = estimate_concentrations(
        features,
        assignments,
        raw_centroids,
        temperature=temperature,
    )

    prototypes = F.normalize(raw_centroids, dim=1)

    return assignments, prototypes, concentrations


@torch.no_grad()
def estimate_concentrations(
    features: torch.Tensor,
    assignments: torch.Tensor,
    raw_centroids: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Estimate PCL prototype-specific temperatures.

    Args:
        features: Embeddings [N, D].
        assignments: Cluster IDs [N].
        raw_centroids: Unnormalized k-means centers [K, D].
        temperature: Desired mean prototype temperature.

    Returns:
        Positive prototype temperatures [K].
    """
    if features.ndim != 2 or raw_centroids.ndim != 2:
        raise ValueError("Expected 2D feature and centroid matrices")
    if features.shape[1] != raw_centroids.shape[1]:
        raise ValueError("Feature and centroid dimensions must match")
    if assignments.shape != (features.shape[0],):
        raise ValueError("Invalid assignment shape")
    if assignments.dtype != torch.long:
        raise ValueError("Assignments must have dtype torch.long")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(features).all().item():
        raise ValueError("Features must be finite")
    if not torch.isfinite(raw_centroids).all().item():
        raise ValueError("Centroids must be finite")

    k = raw_centroids.shape[0]

    if not torch.all((assignments >= 0) & (assignments < k)).item():
        raise ValueError("Cluster assignment out of range")

    phi = torch.zeros(k, device=features.device, dtype=features.dtype)
    counts = torch.bincount(assignments, minlength=k)

    for cluster_id in range(k):
        count = int(counts[cluster_id].item())

        if count > 1:
            members = features[assignments == cluster_id]
            distances = torch.linalg.vector_norm(
                members - raw_centroids[cluster_id],
                dim=1,
            )

            phi[cluster_id] = distances.mean() / torch.log(
                torch.tensor(
                    count + 10.0,
                    device=features.device,
                    dtype=features.dtype,
                )
            )

    # Singleton and empty clusters follow the reference fallback.
    valid = counts > 1

    if valid.any():
        phi[~valid] = phi[valid].max()
    else:
        raise ValueError("At least one cluster must contain two or more samples")

    # Prevent degenerate zero concentrations.
    if not torch.isfinite(phi).all().item() or phi.mean() <= 0:
        raise ValueError("Degenerate cluster concentrations")

    lower = torch.quantile(phi, 0.10)
    upper = torch.quantile(phi, 0.90)
    phi = phi.clamp(min=lower, max=upper)

    phi = temperature * phi / phi.mean()
    return phi


@torch.no_grad()
def extract_clustering_features(
    model: "PCL",
    dataloader: DataLoader,
) -> torch.Tensor:
    """Extract one momentum-encoder representation per dataset sample.

    The dataloader must provide:
        image: Tensor [B, C, H, W]
        indices: Unique dataset indices [B]

    Returns:
        L2-normalized representations [N, D], ordered by index.
    """
    num_samples = len(dataloader.dataset)
    output_dim = model.cluster_prototypes.shape[1]
    device = next(model.parameters()).device

    features = torch.empty(num_samples, output_dim, device=device)
    visited = torch.zeros(num_samples, dtype=torch.bool, device=device)

    was_training = model.training

    try:
        model.eval()

        for batch in dataloader:
            images = batch["image"].to(device)
            indices = batch["indices"].to(device=device, dtype=torch.long)

            if indices.ndim != 1:
                raise ValueError("indices must have shape [B]")
            if indices.numel() != images.shape[0]:
                raise ValueError("indices and images batch sizes differ")
            if not torch.all((indices >= 0) & (indices < num_samples)).item():
                raise ValueError("Dataset index out of range")
            if torch.unique(indices).numel() != indices.numel():
                raise ValueError("Duplicate dataset indices in batch")
            if visited[indices].any().item():
                raise ValueError("Dataset index visited more than once")

            embeddings = model.backbone.forward_teacher(images)
            projections = model.projector.forward_teacher(embeddings)
            projections = F.normalize(projections, dim=-1)

            features[indices] = projections
            visited[indices] = True

        if not visited.all().item():
            raise ValueError("Feature extraction did not visit every dataset sample")

        return features

    finally:
        model.train(was_training)


@dataclass
class PCLOutput(ModelOutput):
    """Output of the Prototypical Contrastive Learning method."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    queries: Optional[torch.Tensor] = None
    keys: Optional[torch.Tensor] = None
    instance_loss: Optional[torch.Tensor] = None
    prototype_loss: Optional[torch.Tensor] = None


class PCLClusteringState(nn.Module):
    """Store one clustering resolution as non-trainable buffers."""

    def __init__(
        self,
        assignments: torch.Tensor,
        prototypes: torch.Tensor,
        concentrations: torch.Tensor,
    ):
        super().__init__()

        self.register_buffer("assignments", assignments.detach().clone())
        self.register_buffer("prototypes", prototypes.detach().clone())
        self.register_buffer("concentrations", concentrations.detach().clone())


class PCL(Module):
    """PCL model: initial instance-contrastive warm-up implementation.

    Prototype clustering and ProtoNCE integration will be added next.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "resnet18",
        projector_dims: Sequence[int] = (512, 128),
        queue_length: int = 4096,
        temperature: float = 0.2,
        ema_decay: float = 0.999,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if queue_length < 1:
            raise ValueError("queue_length must be positive")
        if len(projector_dims) != 2:
            raise ValueError("projector_dims must contain two dimensions")

        if isinstance(encoder_name, str):
            encoder = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            encoder = encoder_name

        # Infer embedding dimensionality.
        with torch.no_grad():
            embed_dim = encoder(torch.zeros(1, 3, 224, 224)).shape[-1]

        hidden_dim, output_dim = projector_dims
        self.embed_dim = embed_dim
        self.temperature = temperature

        projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        self.backbone = TeacherStudentWrapper(
            encoder,
            warm_init=True,
            base_ema_coefficient=ema_decay,
            final_ema_coefficient=ema_decay,
        )

        self.projector = TeacherStudentWrapper(
            projector,
            warm_init=True,
            base_ema_coefficient=ema_decay,
            final_ema_coefficient=ema_decay,
        )

        self.queue = UnsortedQueue(
            max_length=queue_length,
            shape=(output_dim,),
            dtype=torch.float32,
        )

        self.prototype_loss_fn = ProtoNCELoss()

        # Clustering state; initially empty during warm-up.
        self.register_buffer(
            "cluster_assignments",
            torch.empty(0, dtype=torch.long),
        )
        self.register_buffer(
            "cluster_prototypes",
            torch.empty(0, output_dim),
        )
        self.register_buffer(
            "cluster_concentrations",
            torch.empty(0),
        )

        self.num_proto_negatives = 2
        self.additional_clusterings = nn.ModuleList()

    @torch.no_grad()
    def set_clustering(
        self,
        assignments: torch.Tensor,
        prototypes: torch.Tensor,
        concentrations: torch.Tensor,
    ) -> None:
        """Store one clustering resolution for prototype training."""
        if assignments.ndim != 1 or assignments.dtype != torch.long:
            raise ValueError("assignments must be a 1D long tensor")

        if prototypes.ndim != 2:
            raise ValueError("prototypes must have shape [K, D]")

        if prototypes.shape[1] != self.cluster_prototypes.shape[1]:
            raise ValueError("Prototype dimension mismatch")

        k = prototypes.shape[0]

        if concentrations.shape != (k,):
            raise ValueError("concentrations must have shape [K]")

        if assignments.numel() == 0 or k == 0:
            raise ValueError("Clustering must not be empty")

        if not torch.all((assignments >= 0) & (assignments < k)).item():
            raise ValueError("Cluster assignment out of range")

        if not torch.isfinite(prototypes).all().item():
            raise ValueError("Prototypes must be finite")

        if not torch.isfinite(concentrations).all().item():
            raise ValueError("Concentrations must be finite and positive")

        if not torch.all(concentrations > 0).item():
            raise ValueError("Concentrations must be positive")

        device = self.cluster_prototypes.device

        self.cluster_assignments = assignments.detach().to(device=device).clone()
        self.cluster_prototypes = prototypes.detach().to(device=device).clone()
        self.cluster_concentrations = concentrations.detach().to(device=device).clone()

        self.additional_clusterings = nn.ModuleList()

    @torch.no_grad()
    def set_clusterings(
        self,
        clusterings: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Store multiple clustering resolutions.

        Each element contains:
            assignments: [N]
            prototypes: [K, D]
            concentrations: [K]
        """
        if not clusterings:
            raise ValueError("At least one clustering is required")

        num_samples = clusterings[0][0].numel()
        output_dim = self.cluster_prototypes.shape[1]
        device = self.cluster_prototypes.device

        # Validate every resolution before modifying model state.
        for assignments, prototypes, concentrations in clusterings:
            if assignments.ndim != 1 or assignments.dtype != torch.long:
                raise ValueError("Invalid clustering assignments")

            if assignments.numel() != num_samples or num_samples == 0:
                raise ValueError("All clusterings must have the same N")

            if prototypes.ndim != 2 or prototypes.shape[1] != output_dim:
                raise ValueError("Invalid prototype dimensions")

            k = prototypes.shape[0]
            if k == 0 or concentrations.shape != (k,):
                raise ValueError("Invalid concentration dimensions")

            if not torch.all((assignments >= 0) & (assignments < k)).item():
                raise ValueError("Cluster assignment out of range")

            if not torch.isfinite(prototypes).all().item():
                raise ValueError("Prototypes must be finite")

            if not torch.isfinite(concentrations).all().item():
                raise ValueError("Concentrations must be finite")

            if not torch.all(concentrations > 0).item():
                raise ValueError("Concentrations must be positive")

        prepared = [
            (
                assignments.to(device=device, dtype=torch.long),
                prototypes.to(device=device),
                concentrations.to(device=device),
            )
            for assignments, prototypes, concentrations in clusterings
        ]

        first = prepared[0]
        extra_states = nn.ModuleList(
            PCLClusteringState(*state) for state in prepared[1:]
        )

        # The first resolution preserves the existing interface.
        self.set_clustering(*first)

        # Additional resolutions.
        self.additional_clusterings = extra_states

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Restore dynamically sized clustering buffers before loading."""
        names = (
            "cluster_assignments",
            "cluster_prototypes",
            "cluster_concentrations",
        )

        # Resize primary clustering buffers to the checkpoint shapes.
        for name in names:
            key = prefix + name
            if key in state_dict:
                saved = state_dict[key]
                current = getattr(self, name)

                if current.shape != saved.shape:
                    setattr(
                        self,
                        name,
                        torch.empty(
                            saved.shape,
                            dtype=current.dtype,
                            device=current.device,
                        ),
                    )

        # Reconstruct additional clustering modules.
        base = prefix + "additional_clusterings."
        level_ids = set()

        for key in state_dict:
            if key.startswith(base):
                suffix = key[len(base) :]
                parts = suffix.split(".")

                if len(parts) == 2 and parts[0].isdigit():
                    level_ids.add(int(parts[0]))

        if level_ids:
            expected = set(range(max(level_ids) + 1))
            if level_ids != expected:
                error_msgs.append("Non-contiguous additional clustering indices")
            else:
                device = self.cluster_prototypes.device

                states = []
                for level in sorted(level_ids):
                    level_prefix = f"{base}{level}."

                    keys = [
                        level_prefix + "assignments",
                        level_prefix + "prototypes",
                        level_prefix + "concentrations",
                    ]

                    if not all(key in state_dict for key in keys):
                        error_msgs.append(
                            f"Incomplete clustering state at level {level}"
                        )
                        continue

                    states.append(
                        PCLClusteringState(
                            state_dict[keys[0]].to(device),
                            state_dict[keys[1]].to(device),
                            state_dict[keys[2]].to(device),
                        )
                    )

                if len(states) == len(level_ids):
                    self.additional_clusterings = nn.ModuleList(states)
        else:
            self.additional_clusterings = nn.ModuleList()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _forward_views(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> PCLOutput:
        # Representation extraction without training.
        if view2 is None:
            with torch.no_grad():
                embedding = self.backbone.forward_teacher(view1)

            return PCLOutput(
                loss=embedding.new_zeros(()),
                embedding=embedding.detach(),
            )

        # Student query.
        embedding_student = self.backbone.forward_student(view1)
        q = self.projector.forward_student(embedding_student)
        q = F.normalize(q, dim=-1)

        # Momentum teacher key.
        with torch.no_grad():
            embedding_teacher = self.backbone.forward_teacher(view2)
            k = self.projector.forward_teacher(embedding_teacher)
            k = F.normalize(k, dim=-1)

        # Retrieve existing negatives BEFORE enqueuing current keys.
        with torch.no_grad():
            negatives = (
                self.queue.get()
                .detach()
                .clone()
                .to(
                    device=q.device,
                    dtype=q.dtype,
                )
            )

        positive_logits = (q * k).sum(dim=-1, keepdim=True)

        if negatives.shape[0] > 0:
            negative_logits = q @ negatives.T
            logits = torch.cat([positive_logits, negative_logits], dim=1)
        else:
            logits = positive_logits

        logits = logits / self.temperature

        targets = torch.zeros(
            q.shape[0],
            device=q.device,
            dtype=torch.long,
        )

        instance_loss = F.cross_entropy(logits, targets)

        prototype_loss = None
        total_loss = instance_loss

        # Enable prototype learning when clustering results exist.
        if self.cluster_assignments.numel() > 0:
            if indices is None:
                raise ValueError("Dataset indices are required after clustering")

            if indices.ndim != 1 or indices.shape[0] != q.shape[0]:
                raise ValueError("Invalid batch indices shape")

            if indices.dtype != torch.long:
                raise ValueError("indices must have dtype torch.long")

            indices = indices.to(self.cluster_assignments.device)

            if not torch.all(
                (indices >= 0) & (indices < self.cluster_assignments.numel())
            ).item():
                raise ValueError("Dataset index out of range")

            # First clustering resolution plus any additional levels.
            clusterings = [
                (
                    self.cluster_assignments,
                    self.cluster_prototypes,
                    self.cluster_concentrations,
                )
            ]

            for state in self.additional_clusterings:
                clusterings.append(
                    (
                        state.assignments,
                        state.prototypes,
                        state.concentrations,
                    )
                )

            losses = []

            for assignments, prototypes, concentrations in clusterings:
                batch_assignments = assignments[indices]

                pos, neg, pos_phi, neg_phi = select_prototypes(
                    prototypes=prototypes,
                    assignments=batch_assignments,
                    concentrations=concentrations,
                    num_negatives=self.num_proto_negatives,
                )

                level_loss = self.prototype_loss_fn(
                    queries=q,
                    positive_prototypes=pos.to(q.dtype),
                    negative_prototypes=neg.to(q.dtype),
                    positive_concentrations=pos_phi.to(q.dtype),
                    negative_concentrations=neg_phi.to(q.dtype),
                )

                losses.append(level_loss)

            # Average ProtoNCE across clustering resolutions.
            prototype_loss = torch.stack(losses).mean()
            total_loss = instance_loss + prototype_loss

        # Current keys become negatives for future batches.
        if self.training:
            with torch.no_grad():
                self.queue.append(k.detach().to(torch.float32))

        return PCLOutput(
            loss=total_loss,
            instance_loss=instance_loss,
            prototype_loss=prototype_loss,
            embedding=embedding_student.detach(),
            queries=q,
            keys=k.detach(),
        )

    def forward(
        self,
        view1,
        view2=None,
        indices=None,
        stage=None,
    ):
        """Support direct inference and stable-pretraining batches."""
        if isinstance(view1, dict):
            batch = view1

            if stage == "fit":
                output = self._forward_views(
                    view1=batch["view1"],
                    view2=batch["view2"],
                    indices=batch.get("indices"),
                )
            else:
                output = self._forward_views(
                    view1=batch["view1"],
                )

            return dict(output.items())

        return self._forward_views(
            view1=view1,
            view2=view2,
            indices=indices,
        )


class PCLClusteringCallback(Callback):
    """Refresh PCL clustering at configured epoch boundaries.

    Initial implementation supports single-process training.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        num_clusters: Union[int, Sequence[int]],
        warmup_epochs: int = 1,
        refresh_every: int = 1,
        num_iterations: int = 20,
        seed: int = 0,
    ):
        super().__init__()

        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be nonnegative")
        if refresh_every < 1:
            raise ValueError("refresh_every must be positive")
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive")

        # Support both a single K and multiple clustering resolutions.
        if isinstance(num_clusters, int):
            cluster_counts = (num_clusters,)
        else:
            cluster_counts = tuple(num_clusters)

        if not cluster_counts:
            raise ValueError("At least one cluster count is required")

        if any(not isinstance(k, int) or k < 2 for k in cluster_counts):
            raise ValueError("Cluster counts must be integers >= 2")

        self.dataloader = dataloader
        self.cluster_counts = cluster_counts
        self.warmup_epochs = warmup_epochs
        self.refresh_every = refresh_every
        self.num_iterations = num_iterations
        self.seed = seed

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch < self.warmup_epochs:
            return

        if (epoch - self.warmup_epochs) % self.refresh_every != 0:
            return

        if trainer.world_size != 1:
            raise RuntimeError(
                "Distributed PCL clustering refresh is not yet supported"
            )

        # Extract momentum-encoder features only once per refresh.
        features = extract_clustering_features(
            pl_module,
            self.dataloader,
        )

        clusterings = []

        for level, num_clusters in enumerate(self.cluster_counts):
            assignments, prototypes, concentrations = kmeans(
                features,
                num_clusters=num_clusters,
                num_iterations=self.num_iterations,
                seed=self.seed + epoch + level,
                temperature=pl_module.temperature,
            )

            clusterings.append((assignments, prototypes, concentrations))

        # Replace all clustering resolutions together.
        pl_module.set_clusterings(clusterings)
