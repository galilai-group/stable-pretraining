from unittest.mock import Mock, patch
import pytest
import torch
from stable_pretraining.data.transforms import MultiViewTransform
from stable_pretraining.forward import nnclr_forward


def create_batch():
    import stable_pretraining.data.transforms as transforms

    transfomrs_ = MultiViewTransform(
        transforms=[
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
        ]
    )
    return transfomrs_({"image": torch.randn(8, 3, 32, 32), "label": torch.tensor([0])})


@pytest.mark.unit
class TestNNCLRUnit:
    """Unit tests for NNCLR components, following the SimCLR test style."""

    @patch("stable_pretraining.forward.find_or_create_queue_callback")
    @patch("stable_pretraining.forward.OnlineQueue")
    def test_nnclr_forward_with_queue(self, mock_online_queue, mock_find_queue):
        """Test the main NNCLR forward pass logic with a populated support set."""
        # Mock the components of the spt.Module
        mock_module = Mock()
        mock_module.training = True
        mock_module.backbone.return_value = torch.randn(8, 16)
        mock_module.projector.return_value = torch.randn(8, 8)
        mock_module.predictor.return_value = torch.randn(8, 8)
        mock_module.nnclr_loss.return_value = torch.tensor(0.5)
        mock_module.hparams.support_set_size = 100
        mock_module.hparams.projection_dim = 8
        mock_module.trainer = Mock()

        # Mock the queue to return a fake support set
        mock_support_set = torch.randn(100, 8)
        mock_online_queue._shared_queues.get.return_value.get.return_value = (
            mock_support_set
        )

        nnclr_forward_bound = nnclr_forward.__get__(mock_module, type(mock_module))

        batch = create_batch()
        outputs = nnclr_forward_bound(batch, "train")

        assert mock_module.backbone.call_count == 2  # Called for both views
        assert mock_module.projector.call_count == 2  # Called for both views
        assert mock_module.predictor.call_count == 2  # Called for both views
        mock_module.nnclr_loss.assert_called()  # Called twice for symmetric loss
        assert "loss" in outputs
        assert "nnclr_support_set" in outputs

    @patch("stable_pretraining.forward.find_or_create_queue_callback")
    @patch("stable_pretraining.forward.OnlineQueue")
    def test_nnclr_forward_no_queue(self, mock_online_queue, mock_find_queue):
        """Test the fallback SimCLR logic when the support set is empty."""
        mock_module = Mock()
        mock_module.training = True
        mock_module.backbone.return_value = torch.randn(8, 16)
        mock_module.projector.return_value = torch.randn(8, 8)
        mock_module.nnclr_loss.return_value = torch.tensor(0.5)
        mock_module.hparams.support_set_size = 100
        mock_module.hparams.projection_dim = 8
        mock_module.trainer = Mock()

        # Mock the queue to return an empty support set
        mock_online_queue._shared_queues.get.return_value.get.return_value = (
            torch.tensor([])
        )

        nnclr_forward_bound = nnclr_forward.__get__(mock_module, type(mock_module))

        batch = create_batch()

        outputs = nnclr_forward_bound(batch, "train")

        mock_module.predictor.assert_not_called()
        # Assert the loss was called once (SimCLR style)
        mock_module.nnclr_loss.assert_called_once()
        assert "loss" in outputs
