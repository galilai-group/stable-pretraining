"""MEC forward function and method class: training path, warm-up, and logging."""

from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch
from torch import nn

from stable_pretraining import forward
from stable_pretraining.backbone.utils import TeacherStudentWrapper
from stable_pretraining.losses import MECLoss
from stable_pretraining.methods import MEC

pytestmark = pytest.mark.unit


def _forward_model(**kwargs):
    torch.manual_seed(0)
    return SimpleNamespace(
        training=True,
        current_epoch=0,
        backbone=TeacherStudentWrapper(nn.Linear(3, 8)),
        projector=TeacherStudentWrapper(nn.Linear(8, 4)),
        predictor=nn.Linear(4, 4),
        mec_loss=MECLoss(),
        log=Mock(),
        **kwargs,
    )


def _views():
    return {
        "views": [
            {"image": torch.randn(6, 3), "label": torch.arange(6)} for _ in range(2)
        ]
    }


def _logged(model):
    return {c.args[0]: c.args[1] for c in model.log.call_args_list}


def _tiny_mec(**kwargs):
    torch.manual_seed(0)
    encoder = nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=4), nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )
    return MEC(encoder_name=encoder, projector_dim=16, predictor_hidden_dim=8, **kwargs)


class TestMECForward:
    """Tests for the ``mec`` forward function."""

    def test_training_returns_loss_teacher_embedding_and_labels(self):
        model = _forward_model()
        batch = _views()
        result = forward.mec(model, batch, "fit")

        images = torch.cat([v["image"] for v in batch["views"]])
        torch.testing.assert_close(
            result["embedding"], model.backbone.forward_teacher(images)
        )
        assert result["label"].tolist() == list(range(6)) * 2
        assert result["loss"].ndim == 0
        assert torch.isfinite(result["loss"])

        result["loss"].backward()
        assert model.predictor.weight.grad is not None
        assert model.backbone.student.weight.grad is not None
        assert all(p.grad is None for p in model.backbone.teacher.parameters())

    def test_missing_loss_raises(self):
        model = _forward_model()
        del model.mec_loss
        with pytest.raises(ValueError, match="mec_loss"):
            forward.mec(model, _views(), "fit")

    def test_no_warmup_by_default(self):
        model = _forward_model(mec_warmup_epochs=0)
        forward.mec(model, _views(), "fit")
        assert _logged(model)["fit/mec_lambda_inv_scale"] == 1.0

    @pytest.mark.parametrize("epoch,expected", [(0, 8.0), (5, 4.5), (10, 1.0)])
    def test_warmup_decays_lambda_inv_scale(self, epoch, expected):
        model = _forward_model(mec_warmup_epochs=10)
        model.current_epoch = epoch
        forward.mec(model, _views(), "fit")
        assert _logged(model)["fit/mec_lambda_inv_scale"] == pytest.approx(expected)

    def test_logs_loss_and_entropy(self):
        model = _forward_model()
        result = forward.mec(model, _views(), "fit")
        logged = _logged(model)

        batch_size, dim = 6, 4
        lambda_inv = model.mec_loss.lambda_inv(batch_size, dim)
        expected_entropy = -result["loss"] * (batch_size + dim) / 2 / lambda_inv
        torch.testing.assert_close(logged["fit/loss"], result["loss"])
        torch.testing.assert_close(logged["fit/mec_entropy"], expected_entropy.detach())


class TestMECMethod:
    """Tests for the ``MEC`` method class."""

    @pytest.mark.parametrize(
        "epoch,expected", [(0, 8.0), (5, 4.5), (10, 1.0), (50, 1.0)]
    )
    def test_lambda_inv_scale_schedule(self, epoch, expected):
        model = _tiny_mec(warmup_epochs=10, warmup_lambda_scale=8.0)
        with patch.object(
            MEC, "current_epoch", new_callable=PropertyMock, return_value=epoch
        ):
            assert model._lambda_inv_scale() == pytest.approx(expected)

    def test_training_output(self):
        model = _tiny_mec().train()
        v1, v2 = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
        output = model(v1, v2)

        assert output.embedding.shape == (8, 8)
        assert output.prediction.shape == (8, 16)
        assert output.target.shape == (8, 16)
        assert not output.target.requires_grad
        assert not output.entropy.requires_grad

        lambda_inv = model.mec_loss.lambda_inv(4, 16) * model._lambda_inv_scale()
        expected_entropy = -output.loss.detach() * (4 + 16) / 2 / lambda_inv
        torch.testing.assert_close(output.entropy, expected_entropy)

    @pytest.mark.parametrize("symmetric", [False, True])
    def test_symmetric_option_reaches_loss(self, symmetric):
        assert _tiny_mec(symmetric=symmetric).mec_loss.symmetric is symmetric

    def test_warmup_changes_loss(self):
        """The 8x warm-up at epoch 0 must actually reach the loss."""
        v1, v2 = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
        warm = _tiny_mec(warmup_epochs=10).train()(v1, v2).loss
        cold = _tiny_mec(warmup_epochs=0).train()(v1, v2).loss
        assert not torch.allclose(warm, cold)
