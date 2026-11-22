import torch
import torch.nn as nn
import torch.nn.functional as F

#GENERAL NOTES
#online encoder = student learns image representations using backprop
#momentum encoder = teacher produces stables representations using ema

#query = representation of the current images first augmented view (from online encoder)
#key = representation of the current images second augmented view (from momentum encoder)

#queue = collection of key representaitons saved from prev batches (from momentum encoder)

#Suppose we have:
#- queries: [8, 128] — eight current online representations
#- keys: [8, 128] — their corresponding momentum representations
#- queue: [100, 128] — 100 previously stored momentum representations
#The resulting logits tensor has shape [8, 101]

class SCELoss(nn.Module):
    """
    Similarity Contrastive Estimation loss.

    Reference:
    https://arxiv.org/abs/2111.14585
    """

    def __init__(
        self,
        temperature: float = 0.2,
        target_temperature: float = 0.05, #smaller = teacher favors the most similar examples
        lambda_weight: float = 0.5,
    ):
        super().__init__()

        self.temperature = temperature
        self.target_temperature = target_temperature
        self.lambda_weight = lambda_weight

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        if target_temperature <= 0:
            raise ValueError("target_temperature must be positive")

        if not 0 <= lambda_weight <= 1:
            raise ValueError("lambda_weight must be between 0 and 1")

    def forward(self, queries, keys, queue):
        #normalize to unit length
        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
        queue = F.normalize(queue, dim=-1)

        #similarity to the matching positive key
        positive_logits = (queries * keys).sum(dim=-1, keepdim=True)

        #similarities to stored queue representations
        negative_logits = queries @ queue.T

        #put the positive match at column 0
        logits = torch.cat(
            [positive_logits, negative_logits],
            dim=1
        ) / self.temperature

        '''Rather than treating all queued representations as equally incorrect targets,
        the momentum encoder measures their relative similarity'''
        # Teacher similarities to stored queue representations
        relational_logits = keys @ queue.T

        # Algorithm 1: prepend a zero logit for the positive pair
        positive_teacher_logits = torch.zeros(
            (keys.shape[0], 1),
            device=keys.device,
            dtype=keys.dtype,
        )

        teacher_logits = torch.cat(
            [positive_teacher_logits, relational_logits],
            dim=1,
        )

        # Soft relational targets
        relational_probs = F.softmax(
            teacher_logits / self.target_temperature,
            dim=1,
        )

        # Hard target: the positive match is at index 0
        hard_targets = torch.zeros_like(relational_probs)
        hard_targets[:, 0] = 1.0

        #put into one target distribution
        targets = (
            self.lambda_weight * hard_targets
            + (1 - self.lambda_weight) * relational_probs
        )

        '''print("Target shape:", targets.shape)
        print("Target row sums:", targets.sum(dim=1))
        print("Positive target values:", targets[:, 0])'''

        '''calculating the sce loss'''
        # Numerical stability: compute log probabilities directly
        log_probs = F.log_softmax(logits, dim=1)

        # Soft-target cross-entropy
        loss = -(targets.detach() * log_probs).sum(dim=1).mean()

        return loss