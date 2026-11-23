import torch
import torch.nn as nn
import torch.nn.functional as F


# class DenseCL(L.LightningModule): # lightning module to organize training.
#     """Dense Contrastive Learning for self-supervised representation learning."""

#     def __init__(
#         self,
#         backbone,
#         feature_dim,
#         projection_dim=128,
#         temperature=0.2,
#         dense_loss_weight=0.5,
#         learning_rate=1e-3,
#         momentum=0.999, # for how weights are updated
#         weight_decay=1e-4, # regualarizer
#     ):
#         super().__init__()

#         self.backbone = backbone
#         self.momentum_backbone = copy.deepcopy(backbone)

#         # Freeze the target encoder; update it by EMA.
#         for parameter in self.momentum_backbone.parameters():
#             parameter.requires_grad = False

#         self.global_projector = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim, projection_dim),
#         )

#         self.local_projector = nn.Sequential(
#             nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(feature_dim, projection_dim, kernel_size=1),
#         )

#         self.temperature = temperature
#         self.dense_loss_weight = dense_loss_weight
#         self.momentum = momentum
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay

#     @torch.no_grad()
#     def update_momentum_encoder(self):
#         """Exponential moving average update of the target backbone."""
#         for online, target in zip(
#             self.backbone.parameters(),
#             self.momentum_backbone.parameters(),
#         ):
#             target.data.mul_(self.momentum).add_(online.data, alpha=1 - self.momentum)

#     def forward(self, batch, stage="train"):
#         if stage == "train":
#             self.update_momentum_encoder()

#         return densecl(self, batch, stage)

#     def configure_optimizers(self):
#         return torch.optim.AdamW(
#             (parameter for parameter in self.parameters() if parameter.requires_grad),
#             lr=self.learning_rate,
#             weight_decay=self.weight_decay,
#         )

# Helper functions for forward pass:


def info_nce(pos_similarity_score, neg_similarity_score):
    # Contrastive loss function
    all_logits = torch.cat(
        [pos_similarity_score.unsqueeze(-1), neg_similarity_score], dim=-1
    )  # put positive and negative similarity score into a tensor / torch.cat concatenates vectors
    info_nce_loss = torch.logsumexp(all_logits, dim=-1) - pos_similarity_score
    return info_nce_loss


@torch.no_grad()
def dense_correspondence(q, k):
    # find spatial feature (k) that best matches with q (the view corresponding with best positive correspondence)
    fq = F.normalize(q.flatten(2), dim=1)
    fk = F.normalize(k.flatten(2), dim=1)
    sim = torch.bmm(fq.transpose(1, 2), fk)
    return sim.argmax(dim=2)


def global_contrastive_loss(q, k_positive, k_negative, temperature, exclude_self=False):
    q = F.normalize(q, dim=-1)
    k_positive = F.normalize(k_positive, dim=-1).detach()
    k_negative = F.normalize(k_negative, dim=-1).detach()

    neg_logits = torch.einsum("bd,nd->bn", q, k_negative)
    if exclude_self:  # ensure that a query view (q) does not have a negative correspondence with itself
        own = torch.eye(q.size(0), dtype=torch.bool, device=q.device)
        neg_logits = neg_logits.masked_fill(own, float("-inf"))

    pos_logit = (q * k_positive).sum(dim=-1)  # (B,)
    return info_nce(pos_logit / temperature, neg_logits / temperature).mean()


def dense_contrastive_loss(r, t_all, idx, t_negative, temperature, exclude_self=False):
    # contrastive loss performed on individual spatial views of input image from augmentations of spatial views (goal is to learn features in spatial locations of image)
    r = F.normalize(r, dim=-1)  # input view
    t_all = F.normalize(t_all, dim=-1).detach()
    t_negative = F.normalize(
        t_negative, dim=-1
    ).detach()  # all views with negative correspondence

    positive_correspondence = torch.gather(
        t_all, 1, idx.unsqueeze(-1).expand(-1, -1, t_all.size(-1))
    )

    negative_similarity_score = torch.einsum("bsd,nd->bsn", r, t_negative)
    if exclude_self:
        self_r = torch.eye(r.size(0), dtype=torch.bool, device=r.device)
        negative_similarity_score = negative_similarity_score.masked_fill(
            self_r[:, None, :], float("-inf")
        )

    positive_similarity_score = (r * positive_correspondence).sum(dim=-1)
    return info_nce(
        positive_similarity_score / temperature, negative_similarity_score / temperature
    ).mean()


def feature_map_to_feature_vector(x):
    # Role/purpose: convert feature map into feature vector
    # input will be a feature map with dimensions B, D, H, W. We need dimensions B, HW, D to compare similarity of spatial locations
    return x.flatten(
        2
    ).transpose(
        1, 2
    )  # convert spatial feature map into feature vectors / flatten heit and width dimensions


class DenseCLLoss(nn.Module):
    """Compute global and dense contrastive losses for densecl."""

    def __init__(self, temperature: float = 0.2, dense_loss_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.dense_loss_weight = dense_loss_weight

    def forward(
        self,
        q,
        k,
        r,
        t,
        feat_q,
        feat_k,
        queue_global=None,
        queue_dense=None,
        return_parts=False,
    ):
        # q = global query
        # r = dense query

        idx = dense_correspondence(feat_q, feat_k)
        t_all = feature_map_to_feature_vector(t)

        in_batch = queue_global is None or queue_dense is None
        if in_batch:
            neg_global = k  # (B, D)
            neg_dense = t_all.mean(dim=1)  # (B, D) one pooled vector per image
        else:
            neg_global, neg_dense = queue_global, queue_dense

        loss_global = global_contrastive_loss(
            q, k, neg_global, self.temperature, exclude_self=in_batch
        )
        loss_dense = dense_contrastive_loss(
            feature_map_to_feature_vector(r),
            t_all,
            idx,
            neg_dense,
            self.temperature,
            exclude_self=in_batch,
        )
        lam = self.dense_loss_weight
        loss = (1 - lam) * loss_global + lam * loss_dense
        if return_parts:
            return loss, {
                "loss_global": loss_global.detach(),
                "loss_dense": loss_dense.detach(),
            }
        return loss
