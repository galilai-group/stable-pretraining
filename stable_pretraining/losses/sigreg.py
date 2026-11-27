"""Shared Epps-Pulley regularization losses."""

import torch
import torch.nn as nn
from torch.distributed.nn import all_reduce


class EppsPulley(nn.Module):
    """Epps-Pulley goodness-of-fit test for univariate normality.

    Projects data onto a grid of points and computes the Epps-Pulley statistic.

    :param t_max: Integration upper bound.
    :param n_points: Number of integration points.
    """

    def __init__(
        self, t_max: float = 3.0, n_points: int = 17, distributed: bool = True
    ):
        super().__init__()
        assert n_points % 2 == 1

        self._is_ddp = (
            distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        self.world_size = torch.distributed.get_world_size() if self._is_ddp else 1

        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        self.register_buffer("t", t)

        phi = (-0.5 * t**2).exp()
        self.register_buffer("phi", phi)

        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Samples [N, S] (N samples, S slices).

        :return: Per-slice statistic [S].
        """
        # N = x.size(0)
        N = x.size(-2)

        x_t = x.unsqueeze(-1) * self.t
        # cos_mean = x_t.cos().mean(0)
        # sin_mean = x_t.sin().mean(0)
        cos_mean = x_t.cos().mean(-3)
        sin_mean = x_t.sin().mean(-3)

        if self._is_ddp:
            all_reduce(cos_mean, op=torch.distributed.ReduceOp.AVG)
            all_reduce(sin_mean, op=torch.distributed.ReduceOp.AVG)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N * self.world_size


class SlicedEppsPulley(nn.Module):
    """Sliced Epps-Pulley goodness-of-fit test for multivariate normality.

    Projects data onto random 1-D directions and averages the univariate
    Epps-Pulley statistics.  A synchronised step counter seeds the random
    projections so all DDP ranks sample identical directions.

    :param num_slices: Number of random 1-D projections.
    :param t_max: EP integration upper bound.
    :param n_points: EP quadrature nodes.
    """

    def __init__(
        self,
        num_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
        distributed: bool = True,
        seeded: bool = True,
    ):
        super().__init__()
        self._is_ddp = (
            distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        self.num_slices = num_slices
        # self.ep = EppsPulley(t_max=t_max, n_points=n_points)
        self.seeded = seeded

        self.ep = EppsPulley(
            t_max=t_max,
            n_points=n_points,
            distributed=distributed,
        )

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg for [B, D] or [T, B, D] embeddings."""
        with torch.no_grad():
            g = None

            if self.seeded:
                step = self.global_step.clone()

                if self._is_ddp:
                    torch.distributed.broadcast(step, src=0)

                g = torch.Generator(device=x.device).manual_seed(step.item())
                self.global_step.add_(1)

            A = torch.randn(
                x.size(-1),
                self.num_slices,
                device=x.device,
                generator=g,
            )
            A = A / A.norm(p=2, dim=0)

        proj = x @ A
        return self.ep(proj).mean()
