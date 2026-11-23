"""DAE-Net implicit part autoencoder.

Source and license: THIRD_PARTY_NOTICES.md (DAE-Net).
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _reset_group(layer: nn.Conv1d, branch: int, width: int) -> None:
    section = slice(branch * width, (branch + 1) * width)
    nn.init.xavier_uniform_(layer.weight[section])
    nn.init.zeros_(layer.bias[section])


class _Template(nn.Module):
    """Independent templates with a shared-across-shapes coordinate warp."""

    def __init__(self, width: int, branches: int) -> None:
        super().__init__()
        self.width = width
        self.linear_d1 = nn.Conv1d(3 * branches, width * branches, 1, groups=branches)
        self.linear_d2 = nn.Conv1d(
            width * branches, width * branches, 1, groups=branches
        )
        self.linear_d3 = nn.Conv1d(width * branches, 3 * branches, 1, groups=branches)
        self.linear_1 = nn.Conv1d(3 * branches, width * branches, 1, groups=branches)
        self.linear_2 = nn.Conv1d(
            width * branches, width * branches, 1, groups=branches
        )
        self.linear_3 = nn.Conv1d(width * branches, branches, 1, groups=branches)

    def reset_branch(self, branch: int) -> None:
        for layer, width in (
            (self.linear_d1, self.width),
            (self.linear_d2, self.width),
            (self.linear_d3, 3),
        ):
            _reset_group(layer, branch, width)
        section = slice(branch * self.width, (branch + 1) * self.width)
        for layer in (self.linear_1, self.linear_2):
            nn.init.normal_(layer.weight[section], std=(2 / self.width) ** 0.5)
            nn.init.zeros_(layer.bias[section])
        # Keep the upstream approximation to pi for initialization parity.
        nn.init.constant_(
            self.linear_3.weight[branch : branch + 1], -((3.14 * 4 / self.width) ** 0.5)
        )
        nn.init.zeros_(self.linear_3.bias[branch : branch + 1])

    def forward(self, points: Tensor) -> Tensor:
        warped = F.leaky_relu(self.linear_d1(points), 0.02, inplace=True)
        warped = F.leaky_relu(self.linear_d2(warped), 0.02, inplace=True)
        warped = self.linear_d3(warped) + points
        hidden = F.relu(self.linear_1(warped))
        hidden = F.relu(self.linear_2(hidden))
        return torch.sigmoid(self.linear_3(hidden))


class _Deformer(nn.Module):
    """Shape-specific residual affine map and deformation."""

    def __init__(self, z_dim: int, width: int, branches: int) -> None:
        super().__init__()
        self.z_dim, self.width, self.branches = z_dim, width, branches
        self.linear_1 = nn.Conv1d(
            (3 + z_dim) * branches, width * branches, 1, groups=branches
        )
        self.linear_2 = nn.Conv1d(
            width * branches, width * branches, 1, groups=branches
        )
        self.linear_3 = nn.Conv1d(width * branches, 3 * branches, 1, groups=branches)

    def reset_branch(self, branch: int) -> None:
        for layer, width in (
            (self.linear_1, self.width),
            (self.linear_2, self.width),
            (self.linear_3, 3),
        ):
            _reset_group(layer, branch, width)

    def forward(self, codes: Tensor, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        affine = codes[:, :, self.z_dim :] - 0.5
        xyz = points.transpose(1, 2)
        homogeneous = torch.cat(
            (xyz, xyz.new_ones(xyz.shape[0], 1, xyz.shape[2])), dim=1
        )
        affine_points, inputs = [], []
        for branch in range(self.branches):
            local = affine[:, branch].reshape(-1, 3, 4) @ homogeneous + xyz
            affine_points.append(local)
            inputs.extend(
                (
                    local,
                    codes[:, branch, : self.z_dim]
                    .unsqueeze(-1)
                    .repeat(1, 1, points.shape[1]),
                )
            )
        hidden = F.leaky_relu(
            self.linear_1(torch.cat(inputs, dim=1)), 0.02, inplace=True
        )
        hidden = F.leaky_relu(self.linear_2(hidden), 0.02, inplace=True)
        return affine, torch.cat(affine_points, dim=1), self.linear_3(hidden)


class _Encoder(nn.Module):
    """64-cubed voxel encoder preserving the upstream parameter layout."""

    def __init__(self, z_dim: int, width: int, branches: int) -> None:
        super().__init__()
        self.code_dim, self.width, self.branches = z_dim + 12, width, branches
        for index in range(1, 5):
            input_channels = 1 if index == 1 else width * 2 ** (index - 2)
            channels = width * 2 ** (index - 1)
            setattr(
                self,
                f"conv_{index}",
                nn.Conv3d(input_channels, channels, 4, 2, 1, bias=False),
            )
            setattr(self, f"norm_{index}", nn.InstanceNorm3d(channels))
        self.conv_5 = nn.Conv3d(width * 8, width * 16, 4)
        self.linear_1 = nn.Linear(width * 16, branches)
        self.linear_2 = nn.Linear(width * 16, branches * self.code_dim)
        for layer in (self.linear_1, self.linear_2):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def reset_branch(self, branch: int) -> None:
        nn.init.normal_(self.linear_1.weight[branch : branch + 1], std=0.02)
        nn.init.zeros_(self.linear_1.bias[branch : branch + 1])
        section = slice(branch * self.code_dim, (branch + 1) * self.code_dim)
        nn.init.normal_(self.linear_2.weight[section], std=0.02)
        nn.init.normal_(self.linear_2.bias[section], std=0.1)
        # The upstream's overlapping prefix slices ultimately zero nine entries.
        nn.init.zeros_(
            self.linear_2.bias[branch * self.code_dim : branch * self.code_dim + 9]
        )

    def forward(self, voxels: Tensor) -> tuple[Tensor, Tensor]:
        hidden = voxels.to(dtype=self.conv_1.weight.dtype)
        for index in range(1, 5):
            hidden = F.leaky_relu(
                getattr(self, f"norm_{index}")(getattr(self, f"conv_{index}")(hidden)),
                0.02,
                inplace=True,
            )
        hidden = F.leaky_relu(self.conv_5(hidden), 0.02, inplace=True).reshape(
            -1, self.width * 16
        )
        existence = torch.sigmoid(
            self.linear_1(hidden).reshape(-1, self.branches, 1) - 8
        )
        codes = torch.sigmoid(
            self.linear_2(hidden).reshape(-1, self.branches, self.code_dim)
        )
        return existence, codes


class DAENetBackbone(nn.Module):
    """Encode voxels and query deformable implicit part templates.

    Args:
        branch_num: Number of independent part templates.
        z_dim: Shape-specific deformation code size per branch.
        ef_dim: Base voxel encoder width.
        df_dim: Shape-specific deformation MLP width.
        gf_dim: Template MLP width.

    Note:
        Branches are initialized as at the start of upstream training. This
        module supplies the network only; era revival, regularization schedules,
        optimizer resets, and acceptance/rollback belong to the training method.
        Parameter names under encoder/generator match the upstream network.
    """

    def __init__(
        self,
        branch_num: int = 16,
        z_dim: int = 4,
        ef_dim: int = 16,
        df_dim: int = 64,
        gf_dim: int = 64,
    ) -> None:
        super().__init__()
        for name, value in dict(
            branch_num=branch_num,
            z_dim=z_dim,
            ef_dim=ef_dim,
            df_dim=df_dim,
            gf_dim=gf_dim,
        ).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.branch_num, self.z_dim = branch_num, z_dim
        self.encoder = _Encoder(z_dim, ef_dim, branch_num)
        self.generator = nn.Module()
        self.generator.deformers = _Deformer(z_dim, df_dim, branch_num)
        self.generator.generators = _Template(gf_dim, branch_num)
        for branch in range(branch_num):
            self.reset_branch(branch)

    def reset_branch(self, branch_id: int) -> None:
        """Reinitialize one branch without changing shared encoder convolutions.

        Args:
            branch_id: Zero-based branch index.

        Raises:
            ValueError: If the branch index is invalid.

        Note:
            Parameter objects are preserved. Optimizer state is not reset here.
        """
        if (
            isinstance(branch_id, bool)
            or not isinstance(branch_id, int)
            or not 0 <= branch_id < self.branch_num
        ):
            raise ValueError("branch_id must be an integer in [0, branch_num)")
        self.encoder.reset_branch(branch_id)
        self.generator.deformers.reset_branch(branch_id)
        self.generator.generators.reset_branch(branch_id)

    def encode(self, voxels: Tensor) -> dict[str, Tensor]:
        """Encode a batch of occupancy grids for repeated decoding.

        Args:
            voxels: Occupancy grids with shape ``[B, 1, 64, 64, 64]``.

        Returns:
            Existence probabilities ``[B, K, 1]`` and codes ``[B, K, z_dim+12]``.
        """
        if (
            voxels.ndim != 5
            or tuple(voxels.shape[1:]) != (1, 64, 64, 64)
            or voxels.shape[0] == 0
        ):
            raise ValueError("voxels must have shape [B, 1, 64, 64, 64] with B > 0")
        existence, codes = self.encoder(voxels)
        return {"existence": existence, "codes": codes}

    def _check_points(self, points: Tensor) -> None:
        if (
            points.ndim != 3
            or points.shape[-1] != 3
            or points.shape[0] == 0
            or points.shape[1] == 0
        ):
            raise ValueError("points must have shape [B, Q, 3] with B, Q > 0")

    def decode(
        self, points: Tensor, encoded: dict[str, Tensor], *, affine_only: bool = False
    ) -> dict[str, Tensor]:
        """Query encoded shapes in object coordinates.

        Args:
            points: Query coordinates ``[B, Q, 3]``, normally in [-0.5, 0.5].
            encoded: Dictionary returned by :meth:`encode`.
            affine_only: Omit the shape-specific displacement when querying.

        Returns:
            Raw and existence-weighted occupancies ``[B,Q,K]``, sum/max
            occupancies ``[B,Q,1]``, residual affine matrices ``[B,K,12]``,
            unmasked deformation offsets ``[B,K,3,Q]``, and encoded tensors.
            Offsets remain available in affine-only mode for diagnostics.
        """
        self._check_points(points)
        existence, codes = encoded["existence"], encoded["codes"]
        batch, queries, _ = points.shape
        if existence.shape != (batch, self.branch_num, 1) or codes.shape != (
            batch,
            self.branch_num,
            self.z_dim + 12,
        ):
            raise ValueError(
                "encoded tensors must match the query batch and backbone dimensions"
            )
        affine, local, offsets = self.generator.deformers(codes, points)
        raw = self.generator.generators(
            local if affine_only else local + offsets
        ).transpose(1, 2)
        parts = raw * existence.transpose(1, 2)
        return {
            "existence": existence,
            "codes": codes,
            "raw_occupancy": raw,
            "part_occupancy": parts,
            "occupancy_sum": parts.sum(dim=-1, keepdim=True),
            "occupancy_max": parts.max(dim=-1, keepdim=True).values,
            "affine": affine,
            "deformation": offsets.reshape(batch, self.branch_num, 3, queries),
        }

    def templates(self, points: Tensor) -> Tensor:
        """Query canonical templates without shape-specific transformations.

        Args:
            points: Canonical query coordinates of shape ``[B,Q,3]``.

        Returns:
            Raw template occupancies of shape ``[B,Q,K]``.
        """
        self._check_points(points)
        return self.generator.generators(
            points.transpose(1, 2).repeat(1, self.branch_num, 1)
        ).transpose(1, 2)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Encode and decode a voxel/query batch.

        Args:
            batch: Dictionary containing ``voxels`` and ``points``.

        Returns:
            The dictionary returned by :meth:`decode`.
        """
        return self.decode(batch["points"], self.encode(batch["voxels"]))
