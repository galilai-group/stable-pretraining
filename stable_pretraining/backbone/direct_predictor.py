"""Linear predictor of DirectPred, set in closed form from its input statistics."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lower bound on the predictor eigenvalues, for numerical safety.
_MIN_EIG = 1e-4


class DirectPredictor(nn.Module):
    r"""DirectPred predictor :cite:`tian2021understanding`.

    Instead of training the BYOL predictor by gradient descent, DirectPred
    keeps a running estimate of the uncentered correlation matrix of the
    predictor's inputs, the online projections :math:`f` (Eq. 19),

    .. math::

        \hat F \leftarrow \rho \hat F + (1 - \rho)\,\mathbb{E}_B[f f^\top],

    and sets the linear predictor from its eigendecomposition
    :math:`\hat F = U \operatorname{diag}(s) U^\top` (Eq. 18), with the
    eigenvalues normalized by their maximum:

    .. math::

        p_j = \sqrt{s_j / \max_k s_k} + \epsilon, \qquad
        W_p = U \operatorname{diag}(p) U^\top .

    The normalization makes :math:`W_p` independent of the scale of the
    projections, in line with the paper's remark that the magnitude of
    :math:`s_j` does not matter; the overall scale of :math:`W_p` does not
    affect the normalized BYOL loss. The default :math:`\epsilon = 0.1` is the
    value of Table 8 (Table 6 shows that a smaller value such as 0.01 is
    weaker). Predictor eigenvalues are floored at ``1e-4``.

    ``forward`` only applies :math:`W_p`. Call :meth:`update` once per training
    step with the online projections of all views, after the predictions of
    that step were computed: the refreshed weight is used from the next step
    on. The weight is replaced rather than modified in place, so the current
    step's backward pass is unaffected, and it is never trained by gradient
    descent. Under DDP the statistics are computed on the projections gathered
    from all processes, so every process holds the same predictor.

    Args:
        dim (int): Dimension of the projections (input and output size).
        rho (float, optional): Weight of the previous estimate in the running
            average (:math:`\rho`). Default is 0.3.
        eps (float, optional): Boost of the small eigenvalues
            (:math:`\epsilon`). Default is 0.1.
    """

    def __init__(self, dim: int, rho: float = 0.3, eps: float = 0.1):
        super().__init__()
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        self.dim = dim
        self.rho = rho
        self.eps = eps
        self.register_buffer("weight", torch.eye(dim))
        self.register_buffer("corr", torch.zeros(dim, dim))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear predictor :math:`W_p` to ``x`` of shape ``[N, dim]``."""
        return F.linear(x, self.weight.to(x.dtype))

    def closed_form_weight(self, corr: torch.Tensor) -> torch.Tensor:
        r"""Predictor weight :math:`U \operatorname{diag}(p) U^\top` for ``corr``."""
        s, u = torch.linalg.eigh(corr.double())
        s = s.clamp(min=0)  # remove round-off negatives of a PSD matrix
        p = (s / s.max().clamp(min=torch.finfo(s.dtype).tiny)).sqrt() + self.eps
        p = p.clamp(min=_MIN_EIG)
        return ((u * p) @ u.T).to(corr.dtype)

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        """Fold the projections ``z`` of shape ``[N, dim]`` into the estimate.

        Args:
            z (torch.Tensor): Online projections of all views of the current
                step, concatenated along the batch dimension.
        """
        if z.size(-1) != self.dim:
            raise ValueError(
                f"DirectPredictor was built for dim={self.dim}, "
                f"got inputs of dimension {z.size(-1)}."
            )
        # Imported here: stable_pretraining.utils imports this package.
        from ..utils import all_gather

        z = torch.cat(all_gather(z.detach()), 0)
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            batch_corr = (z.T @ z) / z.size(0)
            # The first update initializes the estimate (no host sync).
            corr = torch.where(
                self.num_updates == 0,
                batch_corr,
                self.rho * self.corr + (1 - self.rho) * batch_corr,
            )
            self.corr = corr
            self.weight = self.closed_form_weight(corr)
        self.num_updates = self.num_updates + 1
