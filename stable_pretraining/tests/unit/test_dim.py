"""Unit tests for the Deep InfoMax (DIM) components."""

import math
from collections import OrderedDict
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import stable_pretraining as spt
from stable_pretraining.backbone import DIMEmbedding, LocalGlobalCritic
from stable_pretraining.forward import local_dim
from stable_pretraining.losses import DIMPriorLoss, LocalDIMLoss

LOG_2 = math.log(2.0)


def _reference_fenchel_dual_jsd(l_enc: torch.Tensor, m_enc: torch.Tensor):
    """Port of ``fenchel_dual_loss(l, m, measure="JSD")`` from rdevon/DIM.

    ``l_enc`` is ``[N, units, n_locals]`` and ``m_enc`` is ``[N, units, 1]``.
    """
    n, units, n_locals = l_enc.size()
    l_flat = l_enc.permute(0, 2, 1).reshape(-1, units)
    m_flat = m_enc.permute(0, 2, 1).reshape(-1, units)
    u = torch.mm(m_flat, l_flat.t()).reshape(n, 1, n, n_locals).permute(0, 2, 3, 1)
    mask = torch.eye(n)
    e_pos = (LOG_2 - F.softplus(-u)).mean(2).mean(2)
    e_neg = (F.softplus(-u) + u - LOG_2).mean(2).mean(2)
    e_pos = (e_pos * mask).sum() / mask.sum()
    e_neg = (e_neg * (1 - mask)).sum() / (1 - mask).sum()
    return e_neg - e_pos


@pytest.mark.unit
class TestLocalDIMLoss:
    """Unit tests for :class:`LocalDIMLoss`."""

    def test_matches_reference_implementation(self):
        """Scores built as global-local dot products reproduce the author's loss."""
        torch.manual_seed(0)
        batch_size, units, n_locals = 5, 16, 9
        l_enc = torch.randn(batch_size, units, n_locals)
        m_enc = torch.randn(batch_size, units, 1)
        scores = torch.einsum("ad,bdi->abi", m_enc.squeeze(-1), l_enc)

        loss = LocalDIMLoss()(scores)

        expected = _reference_fenchel_dual_jsd(l_enc, m_enc)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_matches_closed_form(self):
        """Loss equals mean softplus over negatives plus over negated positives."""
        torch.manual_seed(1)
        scores = torch.randn(4, 4, 3)
        eye = torch.eye(4, dtype=torch.bool)

        loss = LocalDIMLoss()(scores)

        expected = (
            F.softplus(-scores[eye]).mean()
            + F.softplus(scores[~eye]).mean()
            - 2 * LOG_2
        )
        assert loss.ndim == 0
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_uninformative_critic_gives_zero(self):
        """A critic scoring every pair 0 cannot separate them: the estimate is 0."""
        loss = LocalDIMLoss()(torch.zeros(3, 3, 4))
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_perfect_critic_reaches_lower_bound(self):
        """Confident, correct scores drive the loss to its minimum ``-2 log 2``."""
        batch_size = 4
        eye = torch.eye(batch_size).unsqueeze(-1).expand(-1, -1, 6)
        scores = 50.0 * (2 * eye - 1)

        loss = LocalDIMLoss()(scores)

        assert torch.allclose(loss, torch.tensor(-2 * LOG_2), atol=1e-6)

    def test_estimate_is_bounded(self):
        """``-loss`` never exceeds ``2 log 2`` regardless of the scores."""
        torch.manual_seed(2)
        for scale in (0.1, 1.0, 100.0):
            loss = LocalDIMLoss()(scale * torch.randn(6, 6, 5))
            assert -loss.item() <= 2 * LOG_2 + 1e-6

    def test_gradient_direction(self):
        """Gradient descent raises positive scores and lowers negative ones."""
        scores = torch.zeros(3, 3, 2, requires_grad=True)
        LocalDIMLoss()(scores).backward()

        eye = torch.eye(3, dtype=torch.bool)
        assert (scores.grad[eye] < 0).all()
        assert (scores.grad[~eye] > 0).all()

    @pytest.mark.parametrize("shape", [(4, 4), (4, 3, 2), (4, 4, 2, 1)])
    def test_rejects_bad_shape(self, shape):
        """Scores must be a square-batch ``[B, B, M]`` tensor."""
        with pytest.raises(ValueError, match=r"\[B, B, M\]"):
            LocalDIMLoss()(torch.zeros(shape))

    def test_rejects_single_sample_batch(self):
        """A batch of one has no negative pairs."""
        with pytest.raises(ValueError, match="batch size >= 2"):
            LocalDIMLoss()(torch.zeros(1, 1, 4))


@pytest.mark.unit
class TestDIMEmbedding:
    """Unit tests for :class:`DIMEmbedding`."""

    def test_output_shape(self):
        """Every location is embedded independently: spatial size is kept."""
        emb = DIMEmbedding(in_channels=8, out_channels=32)
        out = emb(torch.randn(4, 8, 5, 5))
        assert out.shape == (4, 32, 5, 5)

    def test_output_is_layer_normalized(self):
        """Each location's embedding has zero mean and unit variance over channels."""
        torch.manual_seed(0)
        emb = DIMEmbedding(in_channels=8, out_channels=32)
        out = emb(torch.randn(4, 8, 3, 3))
        assert torch.allclose(out.mean(dim=1), torch.zeros(4, 3, 3), atol=1e-5)
        assert torch.allclose(
            out.var(dim=1, unbiased=False), torch.ones(4, 3, 3), atol=1e-3
        )

    def test_shortcut_initialized_near_identity(self):
        """With ``out >= in`` the shortcut starts as an identity plus small noise."""
        emb = DIMEmbedding(in_channels=4, out_channels=6)
        weight = emb.linear_shortcut.weight[:, :, 0, 0]
        eye = torch.eye(6, 4, dtype=torch.bool)
        assert torch.equal(weight[eye], torch.ones(4))
        assert weight[~eye].abs().max() <= 0.01

    def test_shortcut_default_init_when_compressing(self):
        """With ``out < in`` no identity exists, so the default init is kept."""
        emb = DIMEmbedding(in_channels=6, out_channels=4)
        diag = torch.diagonal(emb.linear_shortcut.weight[:, :, 0, 0])
        assert not torch.equal(diag, torch.ones(4))


@pytest.mark.unit
class TestLocalGlobalCritic:
    """Unit tests for :class:`LocalGlobalCritic`."""

    @staticmethod
    def _inputs(batch_size=4, channels=8, size=3, global_dim=5, seed=0):
        g = torch.Generator().manual_seed(seed)
        local = torch.randn(batch_size, channels, size, size, generator=g)
        glob = torch.randn(batch_size, global_dim, generator=g)
        return local, glob

    def test_output_shape(self):
        """All global/local-location pairs of the batch are scored."""
        critic = LocalGlobalCritic(local_channels=8, global_dim=5, embed_dim=16)
        local, glob = self._inputs()
        assert critic(local, glob).shape == (4, 4, 9)

    def test_scores_are_embedding_dot_products(self):
        """``scores[a, b, i]`` is the dot product of the two embeddings."""
        critic = LocalGlobalCritic(local_channels=8, global_dim=5, embed_dim=16)
        local, glob = self._inputs()

        scores = critic(local, glob)

        local_emb = critic.local_net(local).flatten(2)
        global_emb = critic.global_net(glob[:, :, None, None]).flatten(1)
        a, b, i = 1, 3, 7
        expected = torch.dot(global_emb[a], local_emb[b, :, i])
        assert torch.allclose(scores[a, b, i], expected, atol=1e-5)

    def test_global_feature_used_as_is_when_dims_match(self):
        """As in the reference, no global embedding is learned if ``G == embed_dim``."""
        critic = LocalGlobalCritic(local_channels=8, global_dim=16, embed_dim=16)
        local, glob = self._inputs(global_dim=16)

        scores = critic(local, glob)

        assert isinstance(critic.global_net, nn.Identity)
        local_emb = critic.local_net(local).flatten(2)
        assert torch.allclose(scores[0, 1, 2], torch.dot(glob[0], local_emb[1, :, 2]))

    def test_axes_follow_global_then_local(self):
        """Changing image ``b``'s local map only changes column ``b`` of the scores.

        In eval mode BatchNorm uses running statistics, so samples do not
        interact and the effect of each input is isolated.
        """
        critic = LocalGlobalCritic(local_channels=8, global_dim=5, embed_dim=16).eval()
        local, glob = self._inputs()
        base = critic(local, glob)

        local_changed = local.clone()
        local_changed[2] += 1.0
        diff = (critic(local_changed, glob) - base).abs().sum(dim=-1)
        assert diff[:, 2].gt(0).all()
        assert diff[:, [0, 1, 3]].eq(0).all()

        glob_changed = glob.clone()
        glob_changed[1] += 1.0
        diff = (critic(local, glob_changed) - base).abs().sum(dim=-1)
        assert diff[1].gt(0).all()
        assert diff[[0, 2, 3]].eq(0).all()

    def test_trains_with_local_dim_loss(self):
        """The loss backpropagates into both embedding networks and the features."""
        critic = LocalGlobalCritic(local_channels=8, global_dim=5, embed_dim=16)
        local, glob = self._inputs()
        local.requires_grad_(True)
        glob.requires_grad_(True)

        loss = LocalDIMLoss()(critic(local, glob))
        loss.backward()

        assert torch.isfinite(loss)
        for net in (critic.local_net, critic.global_net):
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()
            )
        assert local.grad.abs().sum() > 0
        assert glob.grad.abs().sum() > 0

    @pytest.mark.parametrize(
        "local_shape,global_shape",
        [((4, 8, 9), (4, 5)), ((4, 8, 3, 3), (4, 5, 1)), ((4, 8, 3, 3), (3, 5))],
    )
    def test_rejects_bad_inputs(self, local_shape, global_shape):
        """Inputs must be ``[B, C, H, W]`` and ``[B, G]`` with matching ``B``."""
        critic = LocalGlobalCritic(local_channels=8, global_dim=5, embed_dim=16)
        with pytest.raises(ValueError):
            critic(torch.randn(local_shape), torch.randn(global_shape))


@pytest.mark.unit
class TestDIMPriorLoss:
    """Unit tests for :class:`DIMPriorLoss`."""

    @staticmethod
    def _grad_norm(module: nn.Module) -> float:
        return sum(
            p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None
        )

    def test_returns_finite_scalars(self):
        """Both terms are finite scalars."""
        prior = DIMPriorLoss(spt.backbone.MLP(6, [16, 8, 1]))
        terms = prior(torch.randn(5, 6))
        assert set(terms) == {"encoder", "discriminator"}
        for value in terms.values():
            assert value.ndim == 0 and torch.isfinite(value)

    def test_encoder_term_does_not_train_discriminator(self):
        """The generator loss reaches the embedding but not the discriminator."""
        prior = DIMPriorLoss(spt.backbone.MLP(6, [16, 8, 1]))
        embedding = torch.randn(5, 6, requires_grad=True)

        prior(embedding)["encoder"].backward()

        assert embedding.grad.abs().sum() > 0
        assert self._grad_norm(prior.discriminator) == 0

    def test_discriminator_term_does_not_train_encoder(self):
        """The discriminator loss and penalty never reach the embedding."""
        prior = DIMPriorLoss(spt.backbone.MLP(6, [16, 8, 1]))
        embedding = torch.randn(5, 6, requires_grad=True)

        prior(embedding)["discriminator"].backward()

        assert embedding.grad is None
        assert self._grad_norm(prior.discriminator) > 0

    def test_matches_gan_losses(self):
        """Terms equal the GAN discriminator and non-saturating generator losses."""
        disc = nn.Linear(4, 1)
        prior = DIMPriorLoss(disc, gradient_penalty=0.0)
        embedding = torch.randn(8, 4)

        torch.manual_seed(3)
        terms = prior(embedding)
        torch.manual_seed(3)
        real = torch.rand(8, 4)

        fake = torch.sigmoid(embedding)
        expected_disc = F.softplus(-disc(real)).mean() + F.softplus(disc(fake)).mean()
        expected_enc = F.softplus(-disc(fake)).mean()
        assert torch.allclose(terms["discriminator"], expected_disc, atol=1e-6)
        assert torch.allclose(terms["encoder"], expected_enc, atol=1e-6)

    def test_gradient_penalty_matches_reference(self):
        """Penalty follows the reference: only the last sample's gradient, divided by N.

        A linear discriminator has input gradient ``w``, so each of the two
        penalties is ``|w|^2 / N`` and so is their average.
        """
        disc = nn.Linear(4, 1)
        embedding = torch.randn(8, 4)

        torch.manual_seed(0)
        without = DIMPriorLoss(disc, gradient_penalty=0.0)(embedding)["discriminator"]
        torch.manual_seed(0)
        with_gp = DIMPriorLoss(disc, gradient_penalty=2.0)(embedding)["discriminator"]

        expected = 2.0 * disc.weight.pow(2).sum() / 8
        assert torch.allclose(with_gp - without, expected, atol=1e-6)


def _tiny_dim_module(prior_weight: float = 0.1) -> spt.Module:
    """Build a small local DIM ``Module`` on 16x16 inputs."""
    encoder = nn.Sequential(
        OrderedDict(
            layer0=nn.Sequential(
                nn.Conv2d(3, 8, 4, 2, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU()
            ),
            layer1=nn.Sequential(
                nn.Conv2d(8, 16, 4, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU()
            ),
            flatten=nn.Flatten(),
            layer2=nn.Linear(16 * 4 * 4, 6),
        )
    )
    module = spt.Module(
        forward=local_dim,
        backbone=spt.backbone.HiddenStateExtractor(encoder, ["layer1"]),
        local_layer="layer1",
        dim_critic=LocalGlobalCritic(local_channels=16, global_dim=6, embed_dim=32),
        dim_loss=LocalDIMLoss(),
        prior_loss=DIMPriorLoss(spt.backbone.MLP(6, [16, 8, 1])),
        prior_weight=prior_weight,
    )
    module.log = Mock()
    return module


def _batch():
    g = torch.Generator().manual_seed(0)
    return {
        "image": torch.randn(4, 3, 16, 16, generator=g),
        "label": torch.tensor([0, 1, 2, 3]),
    }


@pytest.mark.unit
class TestLocalDIMForward:
    """Unit tests for the :func:`~stable_pretraining.forward.local_dim` forward function."""

    def test_training_step_trains_every_network(self):
        """The joint loss reaches the encoder, the critic and the discriminator."""
        module = _tiny_dim_module(prior_weight=0.1).train()

        out = module(_batch(), stage="fit")
        out["loss"].backward()

        assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
        assert out["embedding"].shape == (4, 6)
        assert out["layer1"].shape == (4, 16, 4, 4)
        assert torch.equal(out["label"], torch.tensor([0, 1, 2, 3]))
        for net in (module.backbone, module.dim_critic, module.prior_loss):
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()
            )
        logged = {call.args[0] for call in module.log.call_args_list}
        assert {
            "fit/loss",
            "fit/local_dim_loss",
            "fit/prior_encoder_loss",
            "fit/prior_discriminator_loss",
        } <= logged

    def test_zero_prior_weight_is_local_only(self):
        """With ``prior_weight == 0`` the loss is exactly the local DIM loss."""
        module = _tiny_dim_module(prior_weight=0.0).train()
        batch = _batch()

        out = module(batch, stage="fit")

        features = module.backbone(batch["image"])
        scores = module.dim_critic(
            features.hidden_states["layer1"], features.last_hidden_state
        )
        assert torch.allclose(out["loss"], module.dim_loss(scores), atol=1e-6)
        assert all(p.grad is None for p in module.prior_loss.parameters())
        logged = {call.args[0] for call in module.log.call_args_list}
        assert "fit/prior_encoder_loss" not in logged

    def test_validation_returns_embedding_only(self):
        """In eval mode no loss is computed."""
        module = _tiny_dim_module().eval()

        out = module(_batch(), stage="validate")

        assert "loss" not in out
        assert out["embedding"].shape == (4, 6)
        assert out["layer1"].shape == (4, 16, 4, 4)
        module.log.assert_not_called()
