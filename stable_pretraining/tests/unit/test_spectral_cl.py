"""Unit tests for the spectral contrastive loss and the SpectralCL method."""

import lightning as pl
import pytest
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

import stable_pretraining as spt
from stable_pretraining.losses import SpectralContrastiveLoss
from stable_pretraining.methods import SpectralCL

pytestmark = pytest.mark.unit


def project_to_ball(z, mu):
    """Keep rows inside the ball of radius sqrt(mu); pull the others onto it."""
    radius = mu**0.5
    rows = [r if r.norm() < radius else F.normalize(r, dim=0) * radius for r in z]
    return torch.stack(rows)


def reference_loss(z1, z2, mu):
    """Algorithm 1 of HaoChen et al. (NeurIPS 2021) as plain loops.

    Includes the authors' ball projection and their division by mu.
    """
    z1, z2 = z1.double(), z2.double()
    if mu is not None:
        z1, z2 = project_to_ball(z1, mu), project_to_ball(z2, mu)
    n = z1.size(0)
    positive = sum(z1[i] @ z2[i] for i in range(n)) / n
    negative = sum(
        (z1[i] @ z2[j]) ** 2 for i in range(n) for j in range(n) if i != j
    ) / (n * (n - 1))
    loss = -2 * positive + negative
    return loss if mu is None else loss / mu


def encoder():
    return VisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=24,
        depth=2,
        num_heads=3,
        num_classes=0,
        dynamic_img_size=True,
    )


@pytest.fixture
def views():
    """Two views with row norms spanning roughly 0.4 to 16.

    For mu in {1, 10, 100} some rows then fall inside the ball and some outside.
    """
    g = torch.Generator().manual_seed(0)
    scale = torch.tensor([0.1, 0.3, 0.6, 1.0, 2.0, 4.0]).unsqueeze(1)
    return (
        scale * torch.randn(6, 16, generator=g),
        scale * torch.randn(6, 16, generator=g),
    )


class TestSpectralContrastiveLoss:
    """The loss matches the paper's estimator and handles edge cases."""

    @pytest.mark.parametrize("mu", [None, 0.5, 1.0, 10.0, 100.0])
    def test_matches_algorithm_1_reference(self, views, mu):
        z1, z2 = views
        loss = SpectralContrastiveLoss(mu=mu)(z1, z2)
        torch.testing.assert_close(
            loss.double(), reference_loss(z1, z2, mu), rtol=1e-4, atol=1e-4
        )

    def test_is_exported_from_losses_and_preserves_the_method_import(self):
        from stable_pretraining.methods.spectral_cl import (
            SpectralContrastiveLoss as MethodLoss,
        )

        assert spt.losses.SpectralContrastiveLoss is SpectralContrastiveLoss
        assert "SpectralContrastiveLoss" in spt.losses.__all__
        assert MethodLoss is SpectralContrastiveLoss

    def test_default_mu_is_one(self):
        assert SpectralContrastiveLoss().mu == 1.0

    def test_orthonormal_aligned_embeddings_have_expected_loss(self):
        z = torch.eye(5, 8)
        assert SpectralContrastiveLoss(mu=None)(z, z).item() == pytest.approx(-2.0)

    def test_aligned_orthogonal_embeddings_reach_lower_bound_on_ball_boundary(self):
        mu = 3.0
        z = mu**0.5 * torch.eye(5, 8)
        assert SpectralContrastiveLoss(mu=mu)(z, z).item() == pytest.approx(-2.0)

    def test_rows_outside_the_ball_lose_their_scale(self, views):
        z1, z2 = views
        loss = SpectralContrastiveLoss(mu=0.01)
        torch.testing.assert_close(loss(z1, z2), loss(3.0 * z1, 2.0 * z2))

    def test_rows_inside_the_ball_are_left_untouched(self, views):
        z1, z2 = views
        mu = 1e4
        torch.testing.assert_close(
            SpectralContrastiveLoss(mu=mu)(z1, z2),
            SpectralContrastiveLoss(mu=None)(z1, z2) / mu,
        )

    def test_without_mu_the_scale_matters(self, views):
        z1, z2 = views
        loss = SpectralContrastiveLoss(mu=None)
        assert not torch.allclose(loss(z1, z2), loss(3.0 * z1, 3.0 * z2))

    def test_symmetric_in_the_two_views(self, views):
        z1, z2 = views
        loss = SpectralContrastiveLoss()
        torch.testing.assert_close(loss(z1, z2), loss(z2, z1))

    def test_backward_gives_finite_nonzero_gradients(self, views):
        z1, z2 = (v.clone().requires_grad_() for v in views)
        SpectralContrastiveLoss()(z1, z2).backward()
        for z in (z1, z2):
            assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0

    @pytest.mark.parametrize("small_value", [0.0, 1e-20])
    @pytest.mark.parametrize("mu", [0.5, 1.0, 10.0])
    def test_tiny_rows_have_the_unprojected_gradient_inside_the_ball(
        self, small_value, mu
    ):
        z1 = torch.full((2, 2), small_value, requires_grad=True)
        z2 = torch.tensor([[0.25, -0.5], [-0.125, 0.25]], requires_grad=True)
        loss = SpectralContrastiveLoss(mu=mu)(z1, z2)
        expected = SpectralContrastiveLoss(mu=None)(z1, z2) / mu
        actual_grads = torch.autograd.grad(loss, (z1, z2))
        expected_grads = torch.autograd.grad(expected, (z1, z2))

        assert torch.isfinite(loss)
        torch.testing.assert_close(loss, expected, rtol=1e-6, atol=0)
        for actual, reference in zip(actual_grads, expected_grads):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=0)

    def test_loss_is_float32_under_autocast(self, views):
        z1, z2 = (v.bfloat16() for v in views)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = SpectralContrastiveLoss(mu=10.0)(z1, z2)
        assert loss.dtype == torch.float32
        torch.testing.assert_close(
            loss.double(), reference_loss(z1, z2, 10.0), rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize("mu", [0.0, -1.0])
    def test_rejects_non_positive_mu(self, mu):
        with pytest.raises(ValueError, match="mu"):
            SpectralContrastiveLoss(mu=mu)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [((4, 8), (4, 9)), ((4, 8), (3, 8)), ((4, 8, 1), (4, 8, 1))],
    )
    def test_rejects_mismatched_or_non_2d_inputs(self, shape1, shape2):
        with pytest.raises(ValueError, match="2-D"):
            SpectralContrastiveLoss()(torch.randn(*shape1), torch.randn(*shape2))

    def test_rejects_a_single_sample(self):
        with pytest.raises(ValueError, match="at least 2"):
            SpectralContrastiveLoss()(torch.randn(1, 8), torch.randn(1, 8))


class TestSpectralCL:
    """The method class wires backbone, projector, and loss together."""

    def test_train_forward_shapes_and_backward(self):
        model = SpectralCL(
            encoder_name=encoder(), projector_dims=(16, 8), mu=7.0
        ).train()
        v1, v2 = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
        out = model(v1, v2)

        assert model.spectral_loss.mu == 7.0
        assert out.embedding.shape == (8, 24)
        assert out.projection.shape == (8, 8)
        assert out.loss.ndim == 0 and torch.isfinite(out.loss)
        out.loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_eval_returns_embedding_without_a_loss(self):
        model = SpectralCL(encoder_name=encoder(), projector_dims=(16, 8)).eval()
        with torch.no_grad():
            out = model(torch.randn(3, 3, 32, 32))
        assert out.embedding.shape == (3, 24)
        assert out.projection is None and out.loss.item() == 0

    def test_projector_needs_an_output_dimension(self):
        with pytest.raises(ValueError, match="at least one"):
            SpectralCL(encoder_name=encoder(), projector_dims=())

    def test_is_exported_at_the_top_level(self):
        assert spt.SpectralCL is SpectralCL
        assert "SpectralCL" in spt.__all__


def two_view_batch(n=4, named=False, labels=True):
    """A training batch as ``MultiViewTransform`` yields it, after collation."""
    views = [
        {"image": torch.randn(n, 3, 32, 32)}
        | ({"label": torch.arange(n)} if labels else {})
        for _ in range(2)
    ]
    return {"view_0": views[0], "view_1": views[1]} if named else {"views": views}


class TestSpectralCLBatchContract:
    """``model(batch, stage=...)`` follows the contract ``Module.training_step`` uses."""

    @staticmethod
    def make_model(**kwargs):
        return SpectralCL(encoder_name=encoder(), projector_dims=(16, 8), **kwargs)

    @pytest.mark.parametrize("named", [False, True])
    def test_training_returns_loss_embedding_and_label(self, named):
        model = self.make_model().train()
        batch = two_view_batch(named=named) | {"batch_idx": 3}
        out = model(batch, stage="fit")

        assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
        assert out["embedding"].shape == (8, 24)
        assert out["label"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
        out["loss"].backward()

    def test_training_matches_the_tensor_interface(self):
        model = self.make_model().train()
        batch = two_view_batch()
        torch.manual_seed(0)
        from_batch = model(batch, stage="fit")["loss"]
        torch.manual_seed(0)
        views = batch["views"]
        from_tensors = model(views[0]["image"], views[1]["image"]).loss
        torch.testing.assert_close(from_batch, from_tensors)

    def test_training_without_labels_returns_no_label(self):
        out = self.make_model().train()(two_view_batch(labels=False), stage="fit")
        assert "label" not in out and "loss" in out

    @pytest.mark.parametrize("n_views", [1, 3])
    def test_training_needs_exactly_two_views(self, n_views):
        batch = {"views": two_view_batch()["views"][:1] * n_views}
        with pytest.raises(ValueError, match="exactly 2 views"):
            self.make_model().train()(batch, stage="fit")

    def test_training_on_a_single_image_batch_is_rejected(self):
        batch = {"image": torch.randn(4, 3, 32, 32), "label": torch.arange(4)}
        with pytest.raises(ValueError, match="exactly 2 views"):
            self.make_model().train()(batch, stage="fit")

    def test_eval_single_image_returns_embedding_and_label_without_a_loss(self):
        model = self.make_model().eval()
        batch = {"image": torch.randn(3, 3, 32, 32), "label": torch.arange(3)}
        with torch.no_grad():
            out = model(batch, stage="validate")
        assert "loss" not in out
        assert out["embedding"].shape == (3, 24) and out["label"].tolist() == [0, 1, 2]

    def test_eval_embeds_every_view_of_a_multi_view_batch(self):
        model = self.make_model().eval()
        with torch.no_grad():
            out = model(two_view_batch(), stage="validate")
        assert "loss" not in out
        assert out["embedding"].shape == (8, 24) and out["label"].shape == (8,)

    def test_trainer_fit_updates_the_weights_from_batch_dicts(self):
        torch.manual_seed(0)
        model = self.make_model(optim={"optimizer": {"type": "SGD", "lr": 0.1}})
        before = [p.detach().clone() for p in model.parameters()]
        trainer = pl.Trainer(
            accelerator="cpu",
            max_steps=2,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        # batch_size=None yields each prepared dict unchanged; a bare list would
        # be read by Lightning as several dataloaders.
        batches = torch.utils.data.DataLoader(
            [two_view_batch(), two_view_batch()], batch_size=None
        )
        trainer.fit(model, train_dataloaders=batches)

        assert trainer.global_step == 2
        assert any(
            not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())
        )
