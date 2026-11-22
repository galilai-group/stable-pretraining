"""SimDINOv2 loss matches independent reference implementations of the paper math.

Reference implementations here are written with explicit Python loops rather
than the vectorized ops used by ``SimDINOv2Loss``, so a bug in one is unlikely
to be mirrored by the other.
"""

import pytest
import torch

from stable_pretraining.losses.simdinov2 import SimDINOv2Loss

pytestmark = pytest.mark.unit


def _unit(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(
        torch.randn(*shape, generator=generator), dim=-1
    )


def _reference_expansion(feats, eps, world_size=1):
    """Coding rate via ``slogdet`` instead of Cholesky."""
    n_views, m, p = feats.shape
    scalar = p / (m * world_size * eps)
    total = 0.0
    for view in range(n_views):
        cov = torch.zeros(p, p)
        for sample in range(m):
            z = feats[view, sample]
            cov = cov + torch.outer(z, z)
        sign, logabsdet = torch.linalg.slogdet(torch.eye(p) + scalar * cov)
        assert sign > 0
        total = total + 0.5 * logabsdet
    total = total / n_views
    return total * (p + world_size * m) / (p * world_size * m)


def _masked_only(features, patch_mask):
    """Select the masked patch positions in the row-major order the loss expects."""
    flat = features.flatten(0, 1)
    return flat[patch_mask.flatten()]


def _reference_compression(student_feats, teacher_feats, no_diag=True):
    total, count = 0.0, 0
    for t in range(teacher_feats.shape[0]):
        for s in range(student_feats.shape[0]):
            if no_diag and t == s:
                continue
            cos = torch.nn.functional.cosine_similarity(
                teacher_feats[t], student_feats[s], dim=-1
            )
            total = total + cos.mean()
            count += 1
    return total / count


def _reference_masked_patch(student, teacher, patch_mask, masks_weight=None):
    n_rows = patch_mask.shape[0]
    rows = torch.arange(n_rows).repeat_interleave(patch_mask.sum(-1))
    assert rows.shape[0] == student.shape[0], "one feature row per masked patch"
    if masks_weight is None:
        masks_weight = (1.0 / patch_mask.sum(-1).clamp(min=1.0))[rows]
    total = 0.0
    for index in range(student.shape[0]):
        cos = torch.nn.functional.cosine_similarity(
            teacher[index], student[index], dim=-1
        )
        total = total - cos * masks_weight[index]
    return total / n_rows


# --- expansion / coding rate ------------------------------------------------


@pytest.mark.parametrize("expa_type", [0, 1])
def test_expansion_matches_slogdet_reference(expa_type):
    torch.manual_seed(0)
    loss_fn = SimDINOv2Loss(eps=0.05, expa_type=expa_type)
    student = _unit((5, 8, 16), seed=1)
    teacher = _unit((2, 8, 16), seed=2)

    expansion_feats = (student[:2] + teacher) / 2
    if expa_type == 0:
        expansion_feats = student[:2]

    torch.testing.assert_close(
        loss_fn.expansion_loss(expansion_feats),
        _reference_expansion(expansion_feats, eps=0.05),
    )


def test_expansion_penalises_collapsed_features():
    loss_fn = SimDINOv2Loss()
    spread = _unit((2, 64, 32), seed=3)
    collapsed = spread[:, :1].expand(2, 64, 32).contiguous()
    assert loss_fn.expansion_loss(collapsed) < loss_fn.expansion_loss(spread)


def test_expansion_increases_with_eps_inverse_scaling():
    """The denominator is linear in ``1/eps``, so a smaller eps penalises more."""
    feats = _unit((2, 32, 16), seed=4)
    assert SimDINOv2Loss(eps=0.01).expansion_loss(feats) > SimDINOv2Loss(
        eps=0.5
    ).expansion_loss(feats)


def test_expansion_uses_world_size_in_scaling():
    feats = _unit((2, 8, 16), seed=5)
    loss_fn = SimDINOv2Loss(eps=0.05)
    torch.testing.assert_close(
        loss_fn.expansion_loss(feats),
        _reference_expansion(feats, eps=0.05, world_size=1),
    )


# --- compression ------------------------------------------------------------


def test_compression_matches_loop_reference_with_and_without_diagonal():
    student = _unit((12, 8, 16), seed=6)
    teacher = _unit((2, 8, 16), seed=7)
    loss_fn = SimDINOv2Loss()

    for no_diag in (True, False):
        got, _ = loss_fn.compression_loss(student, teacher, no_diag=no_diag)
        torch.testing.assert_close(
            got, _reference_compression(student, teacher, no_diag)
        )


def test_compression_of_identical_features_is_one():
    """All similarities are 1, so the mean is 1 whether or not pairs are dropped."""
    feats = torch.ones(2, 4, 8)
    loss_fn = SimDINOv2Loss()
    for no_diag in (True, False):
        got, _ = loss_fn.compression_loss(feats, feats, no_diag=no_diag)
        torch.testing.assert_close(got, torch.tensor(1.0))


def test_compression_drops_exactly_min_n_teacher_n_student_pairs():
    """Only the ``min(n_teacher, n_student)`` same-index pairs are excluded.

    Layout: teacher ``t`` is all-ones, student view 0 is anti-parallel to it.
    Pair values are ``[t0] = [-1, +1, +1]`` and ``[t1] = [-1, +1, +1]``, so the
    full mean is ``2 / 6`` while dropping the two same-index pairs ``(0, 0)`` and
    ``(1, 1)`` -- whose values are ``-1`` and ``+1`` -- leaves ``2 / 4``.
    """
    teacher = torch.ones(2, 1, 4)
    student = torch.ones(3, 1, 4)
    student[0] = -1.0
    loss_fn = SimDINOv2Loss()

    with_diag_kept, _ = loss_fn.compression_loss(student, teacher, no_diag=False)
    with_diag_removed, _ = loss_fn.compression_loss(student, teacher, no_diag=True)
    torch.testing.assert_close(with_diag_kept, torch.tensor(2.0 / 6.0))
    torch.testing.assert_close(with_diag_removed, torch.tensor(2.0 / 4.0))
    assert with_diag_removed > with_diag_kept


def test_compression_reports_diagnostic_over_first_n_teacher_student_views():
    """The diagnostic spans the first ``n_teacher`` student views, not just the first.

    Upstream slices ``sim[:, :len(teacher_feat_list)]``, so with locals-first
    ordering this is the pair of *local* crops, and it is read before the diagonal
    is zeroed.
    """
    student = _unit((4, 6, 16), seed=9)
    teacher = _unit((2, 6, 16), seed=10)
    _, global_mean = SimDINOv2Loss().compression_loss(student, teacher)
    expected = torch.stack(
        [
            torch.nn.functional.cosine_similarity(teacher[t], student[s], dim=-1).mean()
            for t in range(2)
            for s in range(2)
        ]
    ).mean()
    torch.testing.assert_close(global_mean, expected)


# --- masked patches ---------------------------------------------------------


def test_masked_patch_matches_loop_reference():
    torch.manual_seed(3)
    patch_mask = torch.zeros(4, 9).bool()
    patch_mask[0, [1, 4]] = True
    patch_mask[2, 7] = True
    n_masked = int(patch_mask.sum())
    student = _unit((n_masked, 8), seed=11)
    teacher = _unit((n_masked, 8), seed=12)

    got = SimDINOv2Loss().masked_patch_loss(student, teacher, patch_mask)
    torch.testing.assert_close(
        got, _reference_masked_patch(student, teacher, patch_mask)
    )


def test_masked_patch_normalizes_by_rows_not_masked_patches():
    """With perfectly aligned features the loss is exactly ``-masked_rows / rows``.

    That is what makes ``mask_sample_probability`` an implicit scale factor on the
    patch term: upstream masks only half of the rows, so its optimum sits at -0.5
    rather than -1.
    """
    loss_fn = SimDINOv2Loss()

    every_row = torch.zeros(4, 2).bool()
    every_row[:, 0] = True
    aligned = torch.ones(4, 2, 4)
    torch.testing.assert_close(
        loss_fn.masked_patch_loss(
            _masked_only(aligned, every_row),
            _masked_only(aligned, every_row),
            every_row,
        ),
        torch.tensor(-1.0),
    )

    half_rows = torch.zeros(4, 3).bool()
    half_rows[0] = True
    half_rows[1, 0] = True
    features = torch.ones(4, 3, 4)
    torch.testing.assert_close(
        loss_fn.masked_patch_loss(
            _masked_only(features, half_rows),
            _masked_only(features, half_rows),
            half_rows,
        ),
        torch.tensor(-0.5),
    )


def test_masked_patch_rejects_ungathered_features():
    """Unmasked positions must be gathered away before the loss sees the features.

    The loss weights are indexed by masked position, so passing a full
    ``[n_rows * n_patches, D]`` tensor has to raise rather than silently score the
    wrong positions.
    """
    loss_fn = SimDINOv2Loss()
    patch_mask = torch.zeros(4, 3).bool()
    patch_mask[0] = True
    patch_mask[1, 0] = True
    features = _unit((4, 3, 8), seed=21)
    gathered = _masked_only(features, patch_mask)
    assert gathered.shape[0] == int(patch_mask.sum())

    torch.testing.assert_close(
        loss_fn.masked_patch_loss(gathered, gathered, patch_mask),
        _reference_masked_patch(gathered, gathered, patch_mask),
    )
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        loss_fn.masked_patch_loss(
            features.flatten(0, 1), features.flatten(0, 1), patch_mask
        )


def test_masked_patch_accepts_explicit_masks_weight():
    torch.manual_seed(4)
    patch_mask = torch.zeros(3, 6).bool()
    patch_mask[0, [0, 2]] = True
    patch_mask[1, 5] = True
    n_masked = int(patch_mask.sum())
    student = _unit((n_masked, 8), seed=13)
    teacher = _unit((n_masked, 8), seed=14)
    weights = (
        (1.0 / patch_mask.flatten(1).sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(patch_mask)[patch_mask]
    )

    loss_fn = SimDINOv2Loss()
    torch.testing.assert_close(
        loss_fn.masked_patch_loss(student, teacher, patch_mask, weights),
        _reference_masked_patch(student, teacher, patch_mask, weights),
    )


# --- forward ----------------------------------------------------------------


def _forward_inputs(with_patches=True, seed=15):
    student = _unit((12, 8, 16), seed=seed)
    teacher = _unit((2, 8, 16), seed=seed + 1)
    if not with_patches:
        return student, teacher, None, None, None
    patch_mask = torch.zeros(16, 9).bool()
    patch_mask[::3, [0, 1]] = True
    n_masked = int(patch_mask.sum())
    return (
        student,
        teacher,
        _unit((n_masked, 16), seed=seed + 2),
        _unit((n_masked, 16), seed=seed + 3),
        patch_mask,
    )


@pytest.mark.parametrize(
    "cls_loss_weight,coeff,patch_loss_weight",
    [
        (1.0, 1.0, 1.0),
        (0.5, 2.0, 0.25),
    ],
)
def test_forward_total_is_weighted_sum_of_terms(
    cls_loss_weight, coeff, patch_loss_weight
):
    loss_fn = SimDINOv2Loss(
        eps=0.05,
        coeff=coeff,
        cls_loss_weight=cls_loss_weight,
        patch_loss_weight=patch_loss_weight,
    )
    student, teacher, s_patch, t_patch, patch_mask = _forward_inputs()

    loss, metrics = loss_fn(
        student, teacher, s_patch, t_patch, patch_mask, no_diag=True
    )
    expected_cls = -coeff * _reference_compression(
        student, teacher
    ) - _reference_expansion((student[:2] + teacher) / 2, eps=0.05)
    expected = (
        cls_loss_weight * expected_cls
        + patch_loss_weight * _reference_masked_patch(s_patch, t_patch, patch_mask)
    )
    torch.testing.assert_close(loss, expected)
    assert set(metrics) == {
        "loss",
        "loss_cls",
        "comp_loss",
        "expa_loss",
        "global_comp_loss",
        "loss_patch",
    }
    assert all(isinstance(v, torch.Tensor) and v.ndim == 0 for v in metrics.values())
    assert all(not v.requires_grad for v in metrics.values())


def test_forward_without_patches_omits_patch_metrics():
    loss_fn = SimDINOv2Loss()
    student, teacher, *_ = _forward_inputs(with_patches=False)
    loss, metrics = loss_fn(student, teacher)
    assert "loss_patch" not in metrics
    torch.testing.assert_close(loss, metrics["loss_cls"])


def test_forward_normalises_cls_features():
    """Head outputs are already L2-normalised, so the loss must ignore their scale."""
    loss_fn = SimDINOv2Loss()
    student, teacher, s_patch, t_patch, patch_mask = _forward_inputs()
    loss, _ = loss_fn(student, teacher, s_patch, t_patch, patch_mask)
    scaled, _ = loss_fn(student * 7.0, teacher * 0.1, s_patch, t_patch, patch_mask)
    torch.testing.assert_close(loss, scaled)


def test_forward_backward_reaches_student_features_only():
    loss_fn = SimDINOv2Loss()
    student = _unit((12, 8, 16), seed=16).requires_grad_()
    teacher = _unit((2, 8, 16), seed=17)
    patch_mask = torch.zeros(16, 9).bool()
    patch_mask[::2, 0] = True
    n_masked = int(patch_mask.sum())
    s_patch = _unit((n_masked, 16), seed=18).requires_grad_()
    t_patch = _unit((n_masked, 16), seed=19)

    loss, _ = loss_fn(student, teacher, s_patch, t_patch, patch_mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert s_patch.grad is not None and s_patch.grad.abs().sum() > 0


def test_forward_rejects_partial_patch_inputs():
    loss_fn = SimDINOv2Loss()
    student, teacher, s_patch, t_patch, patch_mask = _forward_inputs()
    with pytest.raises(ValueError, match="together"):
        loss_fn(student, teacher, s_patch, None, patch_mask)
    with pytest.raises(ValueError, match="patch_mask"):
        loss_fn(student, teacher, s_patch, t_patch, None)


@pytest.mark.parametrize("kwargs", [{"eps": 0.0}, {"eps": -1.0}, {"expa_type": 2}])
def test_invalid_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SimDINOv2Loss(**kwargs)
