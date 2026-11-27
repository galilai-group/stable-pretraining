"""Tests for the LeWM method."""

import torch
from torch import nn
import pytest

from stable_pretraining.methods.lewm import LeWM

pytestmark = pytest.mark.unit


class TinyWorldModel(nn.Module):
    """Minimal world model for testing LeWM training."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 8)
        self.action_encoder = nn.Linear(3, 8)
        self.predictor = nn.Linear(8, 8)

    def encode(self, batch):
        return {
            "emb": self.encoder(batch["pixels"]),
            "act_emb": self.action_encoder(batch["action"]),
        }

    def predict(self, emb, act_emb):
        return self.predictor(emb + act_emb)


def test_lewm_forward_backward():
    """LeWM computes finite losses and propagates gradients."""
    model = TinyWorldModel()

    method = LeWM(
        model=model,
        history_size=3,
        num_preds=1,
        sigreg_weight=0.09,
    )

    batch = {
        "pixels": torch.randn(4, 4, 4),
        "action": torch.randn(4, 4, 3),
    }

    output = method.forward(batch, stage="train")

    for key in ("pred_loss", "sigreg_loss", "loss"):
        assert key in output
        assert torch.isfinite(output[key]).all()

    expected = output["pred_loss"] + 0.09 * output["sigreg_loss"]
    torch.testing.assert_close(output["loss"], expected)

    output["loss"].backward()

    assert model.encoder.weight.grad is not None
    assert model.predictor.weight.grad is not None
    assert model.action_encoder.weight.grad is not None


def test_lewm_public_api():
    """LeWM is accessible from the package root."""
    import stable_pretraining as spt

    assert spt.LeWM is LeWM


def test_lewm_prediction_loss_reference():
    """Check prediction MSE against a known numerical result."""
    from stable_pretraining.losses.world_model import LeWMLoss
    from stable_pretraining.methods.lewm_components.architecture import SIGReg

    loss_fn = LeWMLoss(sigreg=SIGReg())

    pred = torch.tensor([[[1.0, 2.0]]])
    target = torch.tensor([[[3.0, 4.0]]])
    embeddings = torch.randn(4, 2, 2)

    output = loss_fn(pred, target, embeddings)

    torch.testing.assert_close(output["pred_loss"], torch.tensor(4.0))


def test_sigreg_reference():
    """Compare SIGReg against the original LeWM formula."""
    from stable_pretraining.methods.lewm_components.architecture import SIGReg

    x = torch.randn(3, 8, 4)

    torch.manual_seed(42)
    actual = SIGReg(knots=17, num_proj=32)(x)

    torch.manual_seed(42)
    A = torch.randn(4, 32)
    A = A / A.norm(dim=0)

    t = torch.linspace(0, 3, 17)
    phi = torch.exp(-t.square() / 2)

    x_t = (x @ A).unsqueeze(-1) * t

    error = (x_t.cos().mean(dim=1) - phi).square() + x_t.sin().mean(dim=1).square()

    expected = (2 * torch.trapezoid(error * phi, t, dim=-1) * x.size(1)).mean()

    torch.testing.assert_close(actual, expected)
