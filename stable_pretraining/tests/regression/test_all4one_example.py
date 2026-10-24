"""All4One example trains, exports, and fits a frozen linear probe."""

import lightning as pl
import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.tests._all4one_helpers import load_all4one_example

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]


@pytest.fixture
def example():
    return load_all4one_example()


def test_one_batch_pretraining_export_and_frozen_probe(example, tmp_path):
    args = example.parse_args(["--epochs", "1", "--batch-size", "2"])
    model, data, callbacks, config = example.build_experiment(args)

    def fit(module, data_module, callbacks):
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
            callbacks=callbacks,
        )
        manager = spt.Manager(trainer=trainer, module=module, data=data_module, seed=42)
        manager()
        return trainer

    fit(model, data, callbacks)
    path = tmp_path / "encoder.pt"
    example.export_encoder(model, config, path)
    args = example.parse_args(
        [
            "--stage",
            "linear",
            "--encoder-path",
            str(path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ]
    )
    linear, data, callbacks, _ = example.build_experiment(args)
    expected = {
        key: value.clone() for key, value in model.backbone.student.state_dict().items()
    }
    initial_head = callbacks[0]._probe_config.weight.detach().clone()
    trainer = fit(linear, data, callbacks)
    assert len(trainer.optimizers) == 1
    assert trainer.global_step == 1
    assert not torch.equal(callbacks[0].module.weight, initial_head)
    assert all(
        not param.requires_grad and param.grad is None
        for param in linear.backbone.parameters()
    )
    for key, value in linear.backbone.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    for metric in ("eval/linear_accuracy", "eval/linear_top5"):
        assert torch.isfinite(trainer.callback_metrics[metric])
    metric_files = list(tmp_path.rglob("metrics.csv"))
    assert metric_files
    assert any("eval/linear_top5" in path.read_text() for path in metric_files)
