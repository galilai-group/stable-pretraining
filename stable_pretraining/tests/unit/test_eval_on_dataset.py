"""Unit tests for EvalOnDataset callback and callback_to_evaluator adapter."""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torchmetrics

from stable_pretraining.callbacks.eval_on_dataset import (
    EvalOnDataset,
    EvalDatasetEntry,
    callback_to_evaluator,
    _move_to_device,
    _eval_hook_callback,
    _eval_trainable_callback,
)


@pytest.mark.unit
class TestEvalOnDataset:
    """Tests for EvalOnDataset callback scheduling and execution."""

    def _make_trainer(self, epoch=0, world_size=1, rank=0, logger=None):
        t = Mock()
        t.current_epoch = epoch
        t.world_size = world_size
        t.global_rank = rank
        t.is_global_zero = rank == 0
        t.global_step = 100
        t.logger = logger
        return t

    def _entry(self, name="test", evaluators=None):
        return EvalDatasetEntry(
            name=name,
            data=Mock(),
            evaluators=evaluators or [Mock(return_value={})],
        )

    def test_init_stores_params(self):
        entry = self._entry()
        cb = EvalOnDataset(
            datasets=[entry],
            every_n_epochs=3,
            start_epoch=2,
        )
        assert cb.datasets == [entry]
        assert cb.every_n_epochs == 3
        assert cb.start_epoch == 2

    def test_epoch_scheduling_skip_before_start(self):
        entry = self._entry()
        cb = EvalOnDataset(
            datasets=[entry],
            every_n_epochs=1,
            start_epoch=5,
        )
        trainer = self._make_trainer(epoch=3)
        cb.on_train_epoch_end(trainer, Mock())
        entry.evaluators[0].assert_not_called()

    def test_epoch_scheduling_runs_at_start(self):
        entry = self._entry(evaluators=[Mock(return_value={"acc": 0.9})])
        cb = EvalOnDataset(
            datasets=[entry],
            every_n_epochs=2,
            start_epoch=4,
        )
        trainer = self._make_trainer(epoch=4)
        cb.on_train_epoch_end(trainer, Mock())
        entry.evaluators[0].assert_called_once()

    def test_epoch_scheduling_every_n(self):
        entry = self._entry()
        cb = EvalOnDataset(
            datasets=[entry],
            every_n_epochs=3,
            start_epoch=0,
        )
        # epoch 0: run, epoch 1: skip, epoch 2: skip, epoch 3: run
        for epoch, should_call in [(0, True), (1, False), (2, False), (3, True)]:
            entry.evaluators[0].reset_mock()
            trainer = self._make_trainer(epoch=epoch)
            cb.on_train_epoch_end(trainer, Mock())
            assert entry.evaluators[0].called == should_call, f"epoch={epoch}"

    def test_metrics_logged_with_prefix(self):
        entry = self._entry(
            name="cifar",
            evaluators=[Mock(return_value={"top1": 0.85, "top5": 0.95})],
        )
        pl_module = Mock()
        cb = EvalOnDataset(datasets=[entry], every_n_epochs=1, start_epoch=0)
        trainer = self._make_trainer(epoch=0)
        cb.on_train_epoch_end(trainer, pl_module)
        pl_module.log_dict.assert_called_once()
        logged = pl_module.log_dict.call_args[0][0]
        assert "eval/cifar/top1" in logged
        assert "eval/cifar/top5" in logged
        assert logged["eval/cifar/top1"] == 0.85

    def test_multiple_datasets_single_log_call(self):
        e1 = self._entry(
            name="cifar", evaluators=[Mock(return_value={"zs_top1": 0.8})]
        )
        e2 = self._entry(
            name="imagenet", evaluators=[Mock(return_value={"zs_top1": 0.7})]
        )
        pl_module = Mock()
        cb = EvalOnDataset(datasets=[e1, e2], every_n_epochs=1, start_epoch=0)
        trainer = self._make_trainer(epoch=0)
        cb.on_train_epoch_end(trainer, pl_module)
        e1.evaluators[0].assert_called_once()
        e2.evaluators[0].assert_called_once()
        # Single log_dict call with both datasets
        pl_module.log_dict.assert_called_once()
        logged = pl_module.log_dict.call_args[0][0]
        assert "eval/cifar/zs_top1" in logged
        assert "eval/imagenet/zs_top1" in logged

    def test_multiple_evaluators_per_dataset(self):
        ev1 = Mock(return_value={"zs_top1": 0.8})
        ev2 = Mock(return_value={"knn_top1": 0.7})
        entry = self._entry(name="test", evaluators=[ev1, ev2])
        pl_module = Mock()
        cb = EvalOnDataset(datasets=[entry], every_n_epochs=1, start_epoch=0)
        trainer = self._make_trainer(epoch=0)
        cb.on_train_epoch_end(trainer, pl_module)
        ev1.assert_called_once()
        ev2.assert_called_once()
        logged = pl_module.log_dict.call_args[0][0]
        assert "eval/test/zs_top1" in logged
        assert "eval/test/knn_top1" in logged

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_no_barrier_without_distributed(self, mock_dist):
        entry = self._entry()
        cb = EvalOnDataset(datasets=[entry], every_n_epochs=1, start_epoch=0)
        trainer = self._make_trainer(epoch=0)
        with patch("torch.distributed.barrier") as mock_barrier:
            cb.on_train_epoch_end(trainer, Mock())
            mock_barrier.assert_not_called()


@pytest.mark.unit
class TestCallbackToEvaluator:
    """Tests for callback_to_evaluator adapter."""

    def test_hook_callback_resets_and_collects_metrics(self):
        """Test that _eval_hook_callback resets, iterates, and collects metrics."""
        metric = torchmetrics.classification.MulticlassAccuracy(num_classes=3)
        callback = Mock()
        callback.name = "test_cb"

        pl_module = Mock()
        pl_module.device = torch.device("cpu")
        pl_module.callbacks_metrics = {
            "test_cb": {"_val": nn.ModuleDict({"top1": metric})}
        }

        # Fake the on_validation_batch_end to update metric
        def fake_hook(trainer, pl_module, outputs, batch, batch_idx):
            preds = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
            labels = torch.tensor([0, 1])
            metric.update(preds, labels)

        callback.on_validation_batch_end = fake_hook

        # Create a simple dataloader
        dataset = [{"x": torch.randn(2, 3)} for _ in range(2)]
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        trainer = Mock()
        result = _eval_hook_callback(
            callback, trainer, pl_module, loader, torch.device("cpu")
        )
        assert "test_cb_top1" in result
        assert result["test_cb_top1"] == pytest.approx(1.0)

    def test_adapter_calls_setup_once(self):
        """Test that callback_to_evaluator only calls setup once."""
        callback = Mock()
        callback.name = "test"
        callback.setup = Mock()
        callback.on_validation_batch_end = Mock()

        eval_fn = callback_to_evaluator(callback)

        pl_module = Mock()
        pl_module.device = torch.device("cpu")
        pl_module.callbacks_metrics = {"test": {"_val": nn.ModuleDict()}}
        trainer = Mock()
        loader = torch.utils.data.DataLoader([torch.zeros(1)], batch_size=1)

        eval_fn(trainer, pl_module, loader)
        eval_fn(trainer, pl_module, loader)

        callback.setup.assert_called_once()


@pytest.mark.unit
class TestMoveToDevice:
    """Tests for _move_to_device helper."""

    def test_dict_batch(self):
        batch = {"x": torch.zeros(2), "label": torch.ones(2), "name": "test"}
        result = _move_to_device(batch, torch.device("cpu"))
        assert isinstance(result["x"], torch.Tensor)
        assert result["name"] == "test"

    def test_tuple_batch(self):
        batch = (torch.zeros(2), torch.ones(2))
        result = _move_to_device(batch, torch.device("cpu"))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_list_batch(self):
        batch = [torch.zeros(2), torch.ones(2)]
        result = _move_to_device(batch, torch.device("cpu"))
        assert isinstance(result, list)

    def test_tensor_batch(self):
        batch = torch.zeros(2)
        result = _move_to_device(batch, torch.device("cpu"))
        assert isinstance(result, torch.Tensor)

    def test_non_tensor_passthrough(self):
        result = _move_to_device("hello", torch.device("cpu"))
        assert result == "hello"
