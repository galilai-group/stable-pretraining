"""Unit tests for TWIST (Self-Supervised Learning by Estimating Twin Class Distributions)."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

import stable_pretraining as spt
from stable_pretraining import forward as forward_module
from stable_pretraining.losses import TWISTLoss
from stable_pretraining.methods.twist import TWIST, TWISTOutput, _build_twist_projector

pytestmark = pytest.mark.unit

B, C = 16, 32


def _logits(seed: int = 0, scale: float = 1.0, batch: int = B, n_classes: int = C):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, n_classes, generator=g) * scale,
        torch.randn(batch, n_classes, generator=g) * scale,
    )


def _kl(p, q):
    return (p * (p.log() - q.log())).sum(dim=-1).mean()


def _entropy(p):
    return -(p * p.log()).sum(dim=-1)


def _paper_twist_loss(p1, p2, alpha, beta):
    """Algorithm 1 of the paper, transcribed literally (one direction)."""
    kl_div = ((p2 * p2.log()).sum(dim=1) - (p2 * p1.log()).sum(dim=1)).mean()
    mean_ent = -(p1 * p1.log()).sum(dim=1).mean()
    mean_prob = p1.mean(dim=0)
    ent_mean = -(mean_prob * mean_prob.log()).sum()
    return kl_div + alpha * mean_ent - beta * ent_mean


def _official_ent_loss(feat1, feat2, lam1, lam2, eps):
    """``EntLoss`` from the official ``objective.py`` on one process with ``tau=1``."""
    probs1, probs2 = F.softmax(feat1, dim=-1), F.softmax(feat2, dim=-1)

    def kl(p, q):
        return (p * (p + eps).log() - p * (q + eps).log()).sum(dim=1).mean()

    def eh(p):
        return (-(p * (p + eps).log()).sum(dim=1)).mean()

    def he(p):
        mean = p.mean(dim=0)
        return -(mean * (mean + eps).log()).sum()

    kl_term = 0.5 * (kl(probs1, probs2) + kl(probs2, probs1))
    eh_term = 0.5 * (eh(probs1) + eh(probs2))
    he_term = 0.5 * (he(probs1) + he(probs2))
    return kl_term + (1 + lam1) * eh_term - lam2 * he_term


# --- TWISTLoss ------------------------------------------------------------


def test_twist_loss_matches_paper_pseudocode():
    z1, z2 = _logits(scale=2.0)
    alpha, beta = 0.7, 1.3
    loss = TWISTLoss(sharpness_weight=alpha, diversity_weight=beta)(z1, z2)
    p1, p2 = F.softmax(z1, dim=-1), F.softmax(z2, dim=-1)
    expected = 0.5 * (
        _paper_twist_loss(p1, p2, alpha, beta) + _paper_twist_loss(p2, p1, alpha, beta)
    )
    assert loss.ndim == 0
    assert torch.allclose(loss, expected, atol=1e-5)


def test_twist_loss_terms_match_closed_forms():
    z1, z2 = _logits(seed=1, scale=2.0)
    terms = TWISTLoss().terms(z1, z2)
    p1, p2 = F.softmax(z1, dim=-1), F.softmax(z2, dim=-1)
    assert set(terms) == {"consistency", "sharpness", "diversity", "loss"}
    assert all(v.ndim == 0 for v in terms.values())
    assert torch.allclose(
        terms["consistency"], 0.5 * (_kl(p1, p2) + _kl(p2, p1)), atol=1e-5
    )
    assert torch.allclose(
        terms["sharpness"],
        0.5 * (_entropy(p1).mean() + _entropy(p2).mean()),
        atol=1e-5,
    )
    assert torch.allclose(
        terms["diversity"],
        0.5 * (_entropy(p1.mean(0)) + _entropy(p2.mean(0))),
        atol=1e-5,
    )
    assert torch.allclose(
        terms["loss"],
        terms["consistency"] + terms["sharpness"] - terms["diversity"],
        atol=1e-6,
    )


def test_twist_loss_converges_to_official_formula_as_eps_shrinks():
    """The official code adds ``EPS`` inside every log; ours is the ``EPS -> 0`` limit."""
    z1, z2 = (z.double() for z in _logits(seed=2, scale=2.0))
    lam1, lam2 = -0.6, 1.0  # the official ViT setting: alpha = 1 + lam1 = 0.4
    loss = TWISTLoss(sharpness_weight=1 + lam1, diversity_weight=lam2)(z1, z2)
    assert loss.dtype == torch.float64
    gaps = [
        abs(loss - _official_ent_loss(z1, z2, lam1, lam2, eps)).item()
        for eps in (1e-2, 1e-5, 1e-8)
    ]
    # The bias is first order in EPS: a 1000x smaller EPS shrinks it ~1000x.
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-2 * gaps[1]
    assert gaps[2] < 1e-4


def test_twist_loss_uniform_prediction_has_closed_form():
    """Collapsed (uniform) output: zero consistency error, sharpness == diversity == log C."""
    z = torch.zeros(B, C)
    terms = TWISTLoss().terms(z, z)
    log_c = torch.tensor(math.log(C))
    assert torch.allclose(terms["consistency"], torch.zeros(()), atol=1e-6)
    assert torch.allclose(terms["sharpness"], log_c, atol=1e-5)
    assert torch.allclose(terms["diversity"], log_c, atol=1e-5)
    # With alpha == beta the collapsed solution scores exactly 0 ...
    assert torch.allclose(terms["loss"], torch.zeros(()), atol=1e-5)
    # ... and a larger sharpness weight penalises it by (alpha - beta) * log C.
    loss = TWISTLoss(sharpness_weight=2.0, diversity_weight=1.0)(z, z)
    assert torch.allclose(loss, log_c, atol=1e-5)


def test_twist_loss_confident_balanced_prediction_reaches_lower_bound():
    """One-hot, class-balanced, view-consistent predictions attain ``-beta * log C``."""
    n = 2 * C
    z = 50.0 * F.one_hot(torch.arange(n) % C, C).float()
    beta = 1.5
    loss_fn = TWISTLoss(diversity_weight=beta)
    terms = loss_fn.terms(z, z)
    lower_bound = torch.tensor(-beta * math.log(C))
    assert torch.allclose(terms["consistency"], torch.zeros(()), atol=1e-6)
    assert torch.allclose(terms["sharpness"], torch.zeros(()), atol=1e-6)
    assert torch.allclose(terms["diversity"], torch.tensor(math.log(C)), atol=1e-5)
    assert torch.allclose(terms["loss"], lower_bound, atol=1e-5)
    # Random predictions sit strictly above the bound.
    z1, z2 = _logits(seed=3, batch=n)
    assert loss_fn(z1, z2) > lower_bound


def test_twist_loss_is_symmetric_in_the_two_views():
    z1, z2 = _logits(seed=4)
    loss_fn = TWISTLoss(sharpness_weight=0.5, diversity_weight=2.0)
    forward, backward = loss_fn.terms(z1, z2), loss_fn.terms(z2, z1)
    for key in forward:
        assert torch.allclose(forward[key], backward[key], atol=1e-6), key


def test_twist_loss_single_sample_has_zero_mutual_information():
    """With one sample the batch mean is that sample, so sharpness == diversity."""
    z1, z2 = _logits(seed=5, batch=1)
    terms = TWISTLoss().terms(z1, z2)
    assert torch.allclose(terms["sharpness"], terms["diversity"], atol=1e-6)
    assert torch.allclose(terms["loss"], terms["consistency"], atol=1e-6)


def test_twist_loss_weights_scale_their_terms():
    z1, z2 = _logits(seed=6)
    base = TWISTLoss().terms(z1, z2)
    assert torch.allclose(
        TWISTLoss(sharpness_weight=0.0, diversity_weight=0.0)(z1, z2),
        base["consistency"],
        atol=1e-6,
    )
    assert torch.allclose(
        TWISTLoss(sharpness_weight=3.0, diversity_weight=0.0)(z1, z2),
        base["consistency"] + 3.0 * base["sharpness"],
        atol=1e-6,
    )
    assert torch.allclose(
        TWISTLoss(sharpness_weight=0.0, diversity_weight=2.0)(z1, z2),
        base["consistency"] - 2.0 * base["diversity"],
        atol=1e-6,
    )
    assert torch.allclose(TWISTLoss()(z1, z2), base["loss"], atol=1e-6)


def test_twist_loss_every_term_is_differentiable():
    z1, z2 = _logits(seed=7)
    z1.requires_grad_(True)
    z2.requires_grad_(True)
    terms = TWISTLoss().terms(z1, z2)
    for key, value in terms.items():
        g1, g2 = torch.autograd.grad(value, (z1, z2), retain_graph=True)
        assert torch.isfinite(g1).all() and torch.isfinite(g2).all(), key
        assert g1.abs().sum() > 0 and g2.abs().sum() > 0, key


@pytest.mark.parametrize("term", ["consistency", "sharpness", "diversity", "loss"])
def test_twist_loss_gradients_match_finite_differences(term):
    # Finite-difference check of the analytic gradient of each term, including
    # the eps-free ``xlogy`` path in the diversity entropy. float64 so the
    # numerical derivative is accurate to the gradcheck tolerance.
    torch.manual_seed(9)
    z1 = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    z2 = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    loss = TWISTLoss(sharpness_weight=1.5, diversity_weight=0.75)
    assert torch.autograd.gradcheck(lambda a, b: loss.terms(a, b)[term], (z1, z2))


def test_twist_loss_is_finite_for_extreme_logits_and_low_precision_inputs():
    # At this scale the softmax underflows to exact zeros in float32, which
    # would turn a naive ``p * log(p)`` into ``0 * -inf = nan``.
    z1, z2 = _logits(seed=8, scale=1e4)
    terms = TWISTLoss().terms(z1, z2)
    assert all(torch.isfinite(v) for v in terms.values())
    assert terms["sharpness"] < 1e-3

    for dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = TWISTLoss()(z1.to(dtype), z2.to(dtype))
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "z1, z2",
    [
        ([[100.0, -100.0, -100.0]], [[100.0, -100.0, -100.0]]),
        ([[100.0, -100.0, -100.0]], [[-100.0, 100.0, -100.0]]),
        ([[100.0, -100.0, -100.0], [-100.0, 100.0, -100.0]], None),
    ],
)
def test_twist_loss_gradients_are_finite_for_saturated_logits(z1, z2):
    # Regression: logits this far apart make the softmax underflow to exact
    # zeros in float32. The loss values were already finite, but the
    # derivative of the diversity entropy at a zero mean probability used to
    # be -inf, which turned the gradient into NaN.
    z1 = torch.tensor(z1, requires_grad=True)
    z2 = (
        z1.detach().clone().requires_grad_(True)
        if z2 is None
        else torch.tensor(z2, requires_grad=True)
    )
    for autocast in (False, True):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
            terms = TWISTLoss().terms(z1, z2)
        for key, value in terms.items():
            assert torch.isfinite(value), key
            g1, g2 = torch.autograd.grad(value, (z1, z2), retain_graph=True)
            assert torch.isfinite(g1).all() and torch.isfinite(g2).all(), key


def test_twist_loss_is_exported():
    assert spt.losses.TWISTLoss is TWISTLoss
    assert "TWISTLoss" in spt.losses.__all__


# --- twist forward function -----------------------------------------------

EMBED_DIM = 8


class _CountingBackbone(torch.nn.Module):
    """Tiny backbone with a batch norm that records every batch size it sees."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 4 * 4, EMBED_DIM),
            torch.nn.BatchNorm1d(EMBED_DIM),
        )
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(x.shape[0])
        return self.net(x)


def _twist_module(training: bool = True):
    torch.manual_seed(0)
    return SimpleNamespace(
        backbone=_CountingBackbone(),
        projector=torch.nn.Sequential(
            torch.nn.Linear(EMBED_DIM, C), torch.nn.BatchNorm1d(C, affine=False)
        ),
        twist_loss=TWISTLoss(sharpness_weight=0.4),
        training=training,
        log=Mock(),
    )


def _two_view_batch(n_views: int = 2, with_labels: bool = True):
    g = torch.Generator().manual_seed(1)
    views = []
    for _ in range(n_views):
        view = {"image": torch.randn(B, 3, 4, 4, generator=g)}
        if with_labels:
            view["label"] = torch.arange(B)
        views.append(view)
    return {"views": views}


def test_twist_forward_runs_both_views_in_one_pass_and_matches_loss():
    module = _twist_module()
    batch = _two_view_batch()
    out = forward_module.twist(module, batch, "fit")

    # One concatenated pass: batch-norm statistics span both views.
    assert module.backbone.batch_sizes == [2 * B]
    assert out["embedding"].shape == (2 * B, EMBED_DIM)
    assert torch.equal(out["label"], torch.cat([torch.arange(B)] * 2))

    logits_1, logits_2 = module.projector(out["embedding"]).chunk(2)
    expected = module.twist_loss.terms(logits_1, logits_2)
    assert torch.allclose(out["loss"], expected["loss"], atol=1e-6)

    logged = {call.args[0]: call.args[1] for call in module.log.call_args_list}
    assert set(logged) == {
        "fit/loss",
        "fit/twist_consistency",
        "fit/twist_sharpness",
        "fit/twist_diversity",
    }
    for name in ("consistency", "sharpness", "diversity"):
        assert torch.allclose(logged[f"fit/twist_{name}"], expected[name], atol=1e-6)

    out["loss"].backward()
    grads = [p.grad for p in module.backbone.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_twist_forward_embedding_keeps_view_order():
    module = _twist_module()
    module.backbone.eval()  # running statistics, so per-view passes are comparable
    batch = _two_view_batch(with_labels=False)
    out = forward_module.twist(module, batch, "fit")
    assert "label" not in out
    for i, chunk in enumerate(out["embedding"].chunk(2)):
        expected = module.backbone(batch["views"][i]["image"])
        assert torch.allclose(chunk, expected, atol=1e-6)


def test_twist_forward_skips_loss_outside_training():
    module = _twist_module(training=False)
    out = forward_module.twist(module, _two_view_batch(), "validate")
    assert "loss" not in out
    assert out["embedding"].shape == (2 * B, EMBED_DIM)
    module.log.assert_not_called()


def test_twist_forward_single_view_validation():
    module = _twist_module(training=False)
    batch = {"image": torch.randn(B, 3, 4, 4), "label": torch.arange(B)}
    out = forward_module.twist(module, batch, "validate")
    assert set(out) == {"embedding", "label"}
    assert out["embedding"].shape == (B, EMBED_DIM)
    assert torch.equal(out["label"], batch["label"])

    unlabeled = forward_module.twist(module, {"image": batch["image"]}, "validate")
    assert set(unlabeled) == {"embedding"}


@pytest.mark.parametrize("n_views", [1, 3])
def test_twist_forward_requires_exactly_two_views(n_views):
    with pytest.raises(ValueError, match="exactly 2 views"):
        forward_module.twist(_twist_module(), _two_view_batch(n_views), "fit")


# --- TWIST method class ---------------------------------------------------

VIT_DIM = 24


def _encoder():
    return VisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=VIT_DIM,
        depth=2,
        num_heads=3,
        num_classes=0,
        dynamic_img_size=True,
    )


def _views(seed: int = 0, batch: int = 4):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, 3, 32, 32, generator=g),
        torch.randn(batch, 3, 32, 32, generator=g),
    )


def test_twist_projector_ends_with_non_affine_batch_norm():
    head = _build_twist_projector(VIT_DIM, [16, 16, 8])
    kinds = [type(layer).__name__ for layer in head]
    assert kinds == [
        "Linear",
        "BatchNorm1d",
        "ReLU",
        "Linear",
        "BatchNorm1d",
        "ReLU",
        "Linear",
        "BatchNorm1d",
    ]
    final_bn = head[-1]
    assert final_bn.num_features == 8
    assert not final_bn.affine and final_bn.weight is None
    assert head[1].affine

    # Train-mode logits are standardised per class across the batch.
    logits = head(torch.randn(64, VIT_DIM))
    assert torch.allclose(logits.mean(dim=0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(logits.std(dim=0, unbiased=False), torch.ones(8), atol=1e-3)


def test_twist_projector_without_norm_before_softmax_ends_with_linear():
    head = _build_twist_projector(VIT_DIM, [16, 8], norm_before_softmax=False)
    assert isinstance(head[-1], torch.nn.Linear) and head[-1].out_features == 8
    assert head[-1].bias is not None
    assert _build_twist_projector(VIT_DIM, [16, 8])[-2].bias is None
    model = TWIST(_encoder(), projector_dims=(16, 8), norm_before_softmax=False)
    assert isinstance(model.projector[-1], torch.nn.Linear)


def test_twist_training_forward_matches_loss_on_concatenated_pass():
    torch.manual_seed(0)
    model = TWIST(
        _encoder(), projector_dims=(16, 8), sharpness_weight=0.4, diversity_weight=1.2
    )
    model.train()
    assert model.embed_dim == VIT_DIM and model.n_classes == 8
    assert model.twist_loss.sharpness_weight == 0.4
    assert model.twist_loss.diversity_weight == 1.2

    v1, v2 = _views()
    out = model(v1, v2)
    assert isinstance(out, TWISTOutput)
    assert out.embedding.shape == (8, VIT_DIM)
    assert out.logits.shape == (8, 8)

    # The ViT has no batch statistics, so per-view passes give the same rows.
    assert torch.allclose(out.embedding[:4], model.backbone(v1), atol=1e-5)
    assert torch.allclose(out.embedding[4:], model.backbone(v2), atol=1e-5)

    expected = model.twist_loss.terms(*model.projector(out.embedding).chunk(2))
    assert torch.allclose(out.loss, expected["loss"], atol=1e-6)
    for name in ("consistency", "sharpness", "diversity"):
        assert torch.allclose(out[name], expected[name], atol=1e-6)
        assert not out[name].requires_grad

    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_twist_eval_forward_returns_embedding_and_zero_loss():
    model = TWIST(_encoder(), projector_dims=(16, 8)).eval()
    v1, _ = _views()
    with torch.no_grad():
        out = model(v1)
    assert out.embedding.shape == (4, VIT_DIM)
    assert out.loss.item() == 0
    assert out.logits is None and out.consistency is None
    assert out.sharpness is None and out.diversity is None


def test_twist_class_accepts_module_batch_and_stage():
    # Regression: ``Module.training_step`` calls ``forward(batch, stage="fit")``
    # with a batch dict, which the tensor-only signature used to reject.
    torch.manual_seed(0)
    model = TWIST(_encoder(), projector_dims=(16, 8))
    v1, v2 = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    batch = {
        "views": [{"image": v1, "label": labels}, {"image": v2, "label": labels}],
        "batch_idx": 0,
    }
    out = model(batch, stage="fit")
    assert isinstance(out, dict)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert out["embedding"].shape == (8, VIT_DIM)
    assert out["label"].tolist() == labels.tolist() * 2
    torch.manual_seed(0)
    torch.testing.assert_close(out["loss"], model(v1, v2).loss)

    named = {"view_a": {"image": v1}, "view_b": {"image": v2}, "batch_idx": 3}
    assert "label" not in model(named, stage="fit")

    model.eval()
    out = model({"image": v1, "label": labels}, stage="validate")
    assert out["embedding"].shape == (4, VIT_DIM)
    assert out["label"] is labels and out["loss"] == 0

    with pytest.raises(ValueError, match="exactly 2"):
        model({"views": [{"image": v1}]}, stage="fit")


def test_twist_class_trains_with_a_lightning_trainer(tmp_path):
    # Regression: the class must be usable directly as the LightningModule.
    import lightning as pl

    class _TwoViewDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return {
                "views": [
                    {"image": torch.randn(3, 32, 32), "label": idx % 4},
                    {"image": torch.randn(3, 32, 32), "label": idx % 4},
                ]
            }

    model = TWIST(_encoder(), projector_dims=(16, 8))
    model.optim = {"optimizer": {"type": "SGD", "lr": 0.01}}
    before = [p.detach().clone() for p in model.projector.parameters()]
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, torch.utils.data.DataLoader(_TwoViewDataset(), batch_size=4))
    after = list(model.projector.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_twist_is_exported_at_top_level():
    assert spt.TWIST is TWIST
    assert spt.methods.TWIST is TWIST
    assert "TWIST" in spt.__all__ and "TWIST" in spt.methods.__all__
