"""Single-process unit tests for :class:`PVCLoss` and :class:`SufficientStatisticsLoss`."""

import math

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses import NTXEntLoss, PVCLoss, SufficientStatisticsLoss

pytestmark = pytest.mark.unit

K, D = 4, 8
LARGE_NUM = 1e9

LOSS_FNS = [
    pytest.param(lambda t: PVCLoss(t, aggregation="geometric"), id="geometric"),
    pytest.param(lambda t: PVCLoss(t, aggregation="arithmetic"), id="arithmetic"),
    pytest.param(lambda t: SufficientStatisticsLoss(t), id="suff_stats"),
]


def _views(m, k=K, d=D, seed=0):
    """``m`` un-normalized views of shape ``[k, d]`` in float64."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(k, d, generator=g, dtype=torch.float64) for _ in range(m)]


def _get_mask(k, m, beta):
    """``get_mask`` (Algs. 1-2): ``[k, m, k]`` self-sample mask excluding view ``beta``."""
    mask_sample = torch.eye(k, dtype=torch.float64)[:, None, :]  # ka kb -> ka 1 kb
    mask_beta = torch.ones(m, dtype=torch.float64)[None, :, None].clone()  # m -> 1 m 1
    mask_beta[:, beta] = 0
    return mask_beta * mask_sample  # [k, m, k]


def _algorithm_1_geometric(views, tau):
    """Algorithm 1 (geometric branch), transcribed literally.

    The pseudocode reuses ``losses_alpha`` for both the per-beta stack and the
    per-alpha list (and stacks an undefined ``loss_alpha``); those names are
    split here so it runs. Nothing else is changed.
    """
    Z = F.normalize(torch.stack(views, dim=1), dim=-1)  # [k, m, d]
    k, m, _ = Z.shape
    scores = torch.einsum("jmd,knd->jmnk", Z, Z) / tau  # [k, m, m, k]
    losses_alpha = []
    for alpha in range(m):
        losses_alpha_beta = []
        for beta in range(m):
            # skip on-diagonal terms
            if alpha != beta:
                logits = scores[:, alpha]  # [k, m, k]
                labels = torch.arange(k) + beta * k  # [k]
                mask = _get_mask(k, m, beta)  # [k, m, k]
                logits = (logits - mask * LARGE_NUM).flatten(1)  # [k, m * k]
                loss_alpha_beta = F.cross_entropy(logits, labels, reduction="none")
                losses_alpha_beta.append(loss_alpha_beta)
        stacked = torch.stack(losses_alpha_beta, dim=-1)  # [k, m-1]
        loss_alpha = stacked.mean(dim=-1)  # [k]
        losses_alpha.append(loss_alpha)
    losses = torch.stack(losses_alpha, dim=-1)  # [k, m]
    sample_losses = losses.mean(dim=-1)  # [k]
    return sample_losses.mean()


def _algorithm_2(views, tau):
    """Algorithm 2 (sufficient statistics), transcribed literally.

    The same ``losses_alpha`` / ``loss_alpha`` naming fix as Algorithm 1 is
    applied. The per-alpha stack is ``[k, 1]`` (only ``alpha == beta`` is kept),
    not ``[k, m-1]`` as its comment says.
    """
    Z = F.normalize(torch.stack(views, dim=1), dim=-1)  # [k, m, d]
    k, m, d = Z.shape
    # step 1: repeat the features M times
    Z_tilde = Z[:, :, None, :].expand(k, m, m, d)  # k m1 d -> k m1 m2 d
    # step 2: replace the effect of alpha-th view by zero
    # and correct the bias coefficient of mean
    diagonal_one = torch.eye(m, dtype=torch.float64)[None, :, :, None]  # 1 m1 m2 1
    diagonal_zero = torch.ones(k, m, m, d, dtype=torch.float64) - diagonal_one
    Z_tilde = m / (m - 1) * Z_tilde * diagonal_zero  # [k, m, m, d]
    # step 3: getting the average of rest-set and normalize
    Z_tilde = Z_tilde.mean(dim=1)  # [k, m, d]
    Z_tilde = F.normalize(Z_tilde, dim=-1)  # [k, m, d]
    scores = torch.einsum("jmd,knd->jmnk", Z, Z_tilde) / tau  # [k, m, m, k]
    losses_alpha = []
    for alpha in range(m):
        losses_alpha_beta = []
        for beta in range(m):
            # skip non-diagonal terms
            if alpha == beta:
                logits = scores[:, alpha]  # [k, m, k]
                labels = torch.arange(k) + beta * k  # [k]
                mask = _get_mask(k, m, beta)  # [k, m, k]
                logits = (logits - mask * LARGE_NUM).flatten(1)  # [k, m * k]
                loss_alpha_beta = F.cross_entropy(logits, labels, reduction="none")
                losses_alpha_beta.append(loss_alpha_beta)
        stacked = torch.stack(losses_alpha_beta, dim=-1)  # [k, 1]
        loss_alpha = stacked.mean(dim=-1)  # [k]
        losses_alpha.append(loss_alpha)
    losses = torch.stack(losses_alpha, dim=-1)  # [k, m]
    sample_losses = losses.mean(dim=-1)  # [k]
    return sample_losses.mean()


def _log_ell_eq14(z, i, alpha, beta, tau):
    """``log l_{i, alpha, beta}`` from Eq. 14, one term at a time.

    The numerator pairs views ``alpha`` and ``beta`` of sample ``i``; the
    negatives are every view ``gamma`` of every other sample ``j``, scored
    against ``beta``. ``z`` is normalized, shape ``[k, m, d]``.
    """
    k, m, _ = z.shape
    pos = (z[i, alpha] @ z[i, beta]) / tau
    terms = [pos]
    for j in range(k):
        if j == i:
            continue
        for gamma in range(m):
            terms.append((z[j, gamma] @ z[i, beta]) / tau)
    return pos - torch.logsumexp(torch.stack(terms), dim=0)


def _eq13_geometric_loop(views, tau):
    """Geometric PVC (Eq. 13) with Eq. 14 likelihoods, averaged over alpha."""
    z = F.normalize(torch.stack(views, dim=1), dim=-1)
    k, m, _ = z.shape
    total = 0.0
    for i in range(k):
        for alpha in range(m):
            for beta in range(m):
                if beta != alpha:
                    total += _log_ell_eq14(z, i, alpha, beta, tau) / (m - 1)
    return -total / (k * m)


def _eq12_arithmetic_loop(views, tau):
    """Arithmetic PVC (Eq. 12) with Eq. 14 likelihoods, averaged over alpha."""
    z = F.normalize(torch.stack(views, dim=1), dim=-1)
    k, m, _ = z.shape
    total = 0.0
    for i in range(k):
        for alpha in range(m):
            mean_ell = sum(
                torch.exp(_log_ell_eq14(z, i, alpha, beta, tau))
                for beta in range(m)
                if beta != alpha
            ) / (m - 1)
            total += torch.log(mean_ell)
    return -total / (k * m)


def _eq22_suff_stats_loop(views, tau):
    """Sufficient statistics loss (Eq. 22) with ``Q`` from Eq. 24, one term at a time.

    ``T_{i, alpha}`` is view ``alpha`` of sample ``i`` and ``Q_{i, alpha~}`` is
    the mean of that sample's other ``M - 1`` views. Two choices follow
    Algorithm 2 where the printed equations differ: ``Q`` is L2-normalized
    (Eq. 24 leaves it unnormalized), and the negatives run over ``j != i`` as
    in Eq. 21 (Eq. 22 prints ``j = 1..K``, which would count the positive
    twice).
    """
    z = F.normalize(torch.stack(views, dim=1), dim=-1)  # [k, m, d]
    k, m, _ = z.shape
    q = torch.empty_like(z)
    for i in range(k):
        for alpha in range(m):
            rest = sum(z[i, beta] for beta in range(m) if beta != alpha) / (m - 1)
            q[i, alpha] = F.normalize(rest, dim=-1)
    total = 0.0
    for i in range(k):
        for alpha in range(m):
            pos = (z[i, alpha] @ q[i, alpha]) / tau
            terms = [pos]
            for j in range(k):
                if j == i:
                    continue
                for gamma in range(m):
                    terms.append((z[i, alpha] @ q[j, gamma]) / tau)
            total += pos - torch.logsumexp(torch.stack(terms), dim=0)
    return -total / (k * m)


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("loss_fn", LOSS_FNS)
def test_two_views_equal_ntxent(loss_fn, temperature):
    z1, z2 = _views(2)
    expected = NTXEntLoss(temperature=temperature)(z1, z2)
    torch.testing.assert_close(loss_fn(temperature)([z1, z2]), expected)


@pytest.mark.parametrize("loss_fn", LOSS_FNS)
def test_view_order_invariant(loss_fn):
    # Every loss treats the M views symmetrically, so permuting the list of
    # views (the same permutation for every sample) must not change it.
    views = _views(4)
    perm = [2, 0, 3, 1]
    loss = loss_fn(0.5)
    torch.testing.assert_close(loss([views[p] for p in perm]), loss(views))


@pytest.mark.parametrize("loss_fn", LOSS_FNS)
def test_collapse_known_value(loss_fn):
    # With every embedding identical all scores tie, so each likelihood is one
    # positive over 1 + M(K - 1) equal terms and every loss equals
    # c(B, M) = log(B - M + 1) = log(1 + M(K - 1)): log(17) for M = 4, K = 5.
    m, k = 4, 5
    v = torch.randn(
        1, D, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    views = [v.expand(k, D) for _ in range(m)]
    expected = torch.tensor(math.log(1 + m * (k - 1)), dtype=torch.float64)
    torch.testing.assert_close(loss_fn(0.5)(views), expected)


@pytest.mark.parametrize("loss_fn", LOSS_FNS)
def test_perfect_separation_near_zero(loss_fn):
    # Every view of sample i is the basis vector e_i, so positives score 1 / tau
    # and negatives 0. The loss is log(1 + M(K - 1) e^{-1/tau}), ~1e-42 here.
    m = 4
    views = [torch.eye(K, D, dtype=torch.float64) for _ in range(m)]
    assert loss_fn(0.01)(views) < 1e-6


@pytest.mark.parametrize("loss_fn", LOSS_FNS)
def test_row_scale_invariant(loss_fn):
    # The inputs are L2-normalized, so scaling each row of each view by its own
    # positive constant must not change the loss.
    g = torch.Generator().manual_seed(1)
    views = _views(4)
    scaled = [
        v * torch.exp(3 * torch.randn(K, 1, generator=g, dtype=torch.float64))
        for v in views
    ]
    loss = loss_fn(0.5)
    torch.testing.assert_close(loss(scaled), loss(views))


@pytest.mark.parametrize("m", [2, 4, 8])
def test_geometric_matches_algorithm_1(m):
    views = _views(m)
    expected = _algorithm_1_geometric(views, tau=0.5)
    torch.testing.assert_close(PVCLoss(0.5, aggregation="geometric")(views), expected)


@pytest.mark.parametrize("m", [2, 4, 8])
def test_geometric_matches_eq14_loop(m):
    # Algorithm 1 scores the negatives against the anchor view alpha, while
    # Eq. 14 scores them against beta. Geometric averages over every ordered
    # (alpha, beta) pair, so the two conventions must give the same loss.
    views = _views(m)
    expected = _eq13_geometric_loop(views, tau=0.5)
    torch.testing.assert_close(PVCLoss(0.5, aggregation="geometric")(views), expected)


@pytest.mark.parametrize("m", [2, 4, 8])
def test_arithmetic_matches_eq12_loop(m):
    # PVCLoss follows Eq. 14: alpha is the target and the M - 1 likelihoods
    # averaged inside the log vary the query beta, whose view scores the
    # negatives. Not tested against Algorithm 1's arithmetic line,
    #     loss_alpha = logsumexp(losses_alpha, dim=-1) - log(k)
    # which has a typo: ``losses_alpha`` are negative log-likelihoods, so the
    # sign inside logsumexp is flipped, and the normalizer should be M - 1,
    # not k. The intended form is
    #     log(m - 1) - logsumexp(-losses_alpha, dim=-1).
    views = _views(m)
    expected = _eq12_arithmetic_loop(views, tau=0.5)
    torch.testing.assert_close(PVCLoss(0.5, aggregation="arithmetic")(views), expected)


@pytest.mark.parametrize("m", [2, 4, 8])
def test_suff_stats_matches_algorithm_2(m):
    views = _views(m)
    expected = _algorithm_2(views, tau=0.5)
    torch.testing.assert_close(SufficientStatisticsLoss(0.5)(views), expected)


@pytest.mark.parametrize("m", [2, 4, 8])
def test_suff_stats_matches_eq22_loop(m):
    views = _views(m)
    expected = _eq22_suff_stats_loop(views, tau=0.5)
    torch.testing.assert_close(SufficientStatisticsLoss(0.5)(views), expected)


@pytest.mark.parametrize("loss_fn", LOSS_FNS)
def test_gradients_reach_every_view(loss_fn):
    views = [v.requires_grad_() for v in _views(4)]
    loss_fn(0.5)(views).backward()
    for idx, v in enumerate(views):
        assert v.grad is not None, f"view {idx} received no gradient"
        assert torch.isfinite(v.grad).all(), f"view {idx} has a non-finite gradient"
        assert v.grad.abs().sum() > 0, f"view {idx} has an all-zero gradient"


@pytest.mark.parametrize("loss_fn", LOSS_FNS)
def test_gradcheck(loss_fn):
    # Compares analytic gradients with finite differences, including the
    # index_copy that files the arithmetic losses under their target view.
    loss = loss_fn(0.5)
    views = tuple(v.requires_grad_() for v in _views(3, k=3, d=4))
    assert torch.autograd.gradcheck(lambda *vs: loss(list(vs)), views)
