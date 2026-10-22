"""Reference and integration checks for the BayesNCL method."""

import lightning as pl
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, kl_divergence
from torch.utils.data import DataLoader

import stable_pretraining as spt
from stable_pretraining.forward import bayesncl

pytestmark = pytest.mark.unit


def _model():
    torch.manual_seed(0)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
    model = spt.BayesNCL(
        encoder_name=encoder,
        embed_dim=4,
        projector_dims=(8, 6),
        gate_temperature=0.7,
        gate_threshold=0.6,
    )
    with torch.no_grad():
        model.projector[-1].bias.copy_(torch.tensor([-0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
        model.gating_head[-1].bias.copy_(
            torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        )
    return model


def _views():
    generator = torch.Generator().manual_seed(1)
    return [torch.randn(3, 3, 2, 2, generator=generator) for _ in range(2)]


def _reference_head(head, features):
    hidden = F.relu(F.linear(features, head[0].weight, head[0].bias))
    return F.linear(hidden, head[2].weight, head[2].bias)


def test_model_matches_reference_values_and_gradients():
    model = _model()
    reference = _model()
    reference.load_state_dict(model.state_dict())
    view1, view2 = _views()
    actual = model(view1, view2)

    features = torch.cat([reference.backbone(view1), reference.backbone(view2)])
    raw = _reference_head(reference.projector, features)
    smooth = F.gelu(raw)
    nonnegative = smooth - smooth.detach() + F.relu(raw).detach()
    projected = F.normalize(nonnegative, dim=-1)
    probabilities = torch.sigmoid(
        _reference_head(reference.gating_head, features.detach())
        / reference.gate_temperature
    )
    hard = (probabilities > reference.gate_threshold).float()
    gates = hard.detach() + probabilities - probabilities.detach()
    projected = F.normalize(projected * gates, dim=-1)

    temperature = reference.bayesncl_loss.contrastive_loss.temperature
    similarities = torch.exp(projected @ projected.T / temperature)
    indices = torch.arange(6)
    positives = similarities[indices, (indices + 3) % 6]
    denominator = similarities.masked_fill(torch.eye(6, dtype=torch.bool), 0).sum(1)
    contrastive = -(positives / denominator).log().mean()
    prior = Bernoulli(
        probabilities.new_tensor(reference.bayesncl_loss.prior_probability)
    )
    regularization = kl_divergence(Bernoulli(probabilities.clamp(0.01, 0.99)), prior)
    expected_loss = (
        contrastive + reference.bayesncl_loss.kl_weight * regularization.sum(1).mean()
    )

    torch.testing.assert_close(actual.embedding, features)
    torch.testing.assert_close(actual.projection, projected)
    torch.testing.assert_close(actual.gate_probabilities, probabilities)
    torch.testing.assert_close(actual.loss, expected_loss)
    actual.loss.backward()
    expected_loss.backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert name == ref_name
        torch.testing.assert_close(
            parameter.grad, ref_parameter.grad, rtol=1e-4, atol=1e-6
        )


def test_gate_gradients_do_not_reach_encoder_or_projector():
    model = _model()
    output = model(*_views())
    output.gate_probabilities.sum().backward()

    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.projector.parameters())
    assert any(
        parameter.grad.abs().sum() > 0 for parameter in model.gating_head.parameters()
    )


def test_evaluation_returns_encoder_features_for_tensor_and_dict_inputs():
    model = _model().eval()
    images = _views()[0]
    labels = torch.arange(3)
    with torch.no_grad():
        tensor_output = model(images)
        dict_output = model({"image": images, "label": labels}, stage="validate")
        expected = model.backbone(images)

    torch.testing.assert_close(tensor_output.embedding, expected)
    torch.testing.assert_close(dict_output["embedding"], expected)
    torch.testing.assert_close(dict_output["label"], labels)
    assert tensor_output.loss.item() == 0
    assert tensor_output.projection is None and tensor_output.gate_probabilities is None
    assert "loss" not in dict_output


def test_composable_forward_matches_model_and_preserves_labels():
    model = _model()
    view1, view2 = _views()
    labels = torch.arange(3)
    batch = {"views": [{"image": view, "label": labels} for view in (view1, view2)]}
    module = spt.Module(
        forward=bayesncl,
        backbone=model.backbone,
        projector=model.projector,
        gating_head=model.gating_head,
        bayesncl_loss=model.bayesncl_loss,
        gate_temperature=model.gate_temperature,
        gate_threshold=model.gate_threshold,
    )
    actual = module(batch, stage="fit")
    paired = model(view1, view2)
    dictionary = model(batch, stage="fit")
    for key in ("embedding", "projection", "gate_probabilities", "loss"):
        torch.testing.assert_close(actual[key], paired[key])
        torch.testing.assert_close(actual[key], dictionary[key])
    torch.testing.assert_close(actual["label"], labels.repeat(2))


def test_model_trains_one_step_with_existing_module_loop(tmp_path):
    model = _model()
    model.optim = {"optimizer": {"type": "SGD", "lr": 0.01}}
    view1, view2 = _views()
    samples = [
        {"views": [{"image": view1[i]}, {"image": view2[i]}]} for i in range(len(view1))
    ]
    loader = DataLoader(samples, batch_size=3)
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, train_dataloaders=loader)

    assert trainer.global_step == 1
    for component in ("backbone", "projector", "gating_head"):
        assert any(
            not torch.equal(parameter, before[f"{component}.{name}"])
            for name, parameter in getattr(model, component).named_parameters()
        )


@pytest.mark.parametrize("unequal_batches", [False, True])
def test_training_requires_two_matching_view_batches(unequal_batches):
    model = _model()
    view1, view2 = _views()
    views = (
        [{"image": view1}, {"image": view2[:1]}]
        if unequal_batches
        else [{"image": view1}]
    )
    with pytest.raises(ValueError, match="matching batch sizes|exactly 2 views"):
        model({"views": views}, stage="fit")
