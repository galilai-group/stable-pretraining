"""Visualization edge cases must not interrupt training or corrupt geometry."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from stable_pretraining.callbacks import _viz_utils as viz
from stable_pretraining.callbacks.attention_visualizer import AttentionVisualizer

pytestmark = pytest.mark.unit


def test_grid_and_attention_validation_and_prefix_free_maps():
    assert viz.infer_grid_size(4, 2) == (2, 2)
    with pytest.raises(ValueError, match="cells"):
        viz.infer_grid_size(5, 2)
    with pytest.raises(ValueError, match="tokens"):
        viz.pca_tokens_to_rgb(torch.randn(1, 5, 4), (2, 2))
    with pytest.raises(ValueError, match="key tokens"):
        viz.process_cls_attention(torch.ones(1, 2, 7), (2, 2))
    maps, mask = viz.process_cls_attention(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]), (2, 2), threshold=None
    )
    torch.testing.assert_close(maps.flatten(), torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]))
    assert mask is None


def test_minmax_exact_bounds_and_degenerate_foreground_fallback():
    values = torch.tensor([[2.0, 4.0], [4.0, 4.0], [6.0, 4.0]])
    torch.testing.assert_close(
        viz.robust_minmax(values, quantile=0, dim=0),
        torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
    )
    features = torch.randn(2, 4, 8)
    torch.manual_seed(71)
    fallback = viz.pca_tokens_to_rgb(features, (2, 2), foreground_threshold=2)
    torch.manual_seed(71)
    ordinary = viz.pca_tokens_to_rgb(features, (2, 2))
    torch.testing.assert_close(fallback, ordinary)


def test_ragged_labeled_figure_survives_one_logger_failure():
    picture = viz.render_grid_figure(
        [[torch.ones(3, 4, 4), torch.zeros(3, 4, 4)], [torch.ones(3, 4, 4)]],
        row_titles=["first", "second"],
        col_titles=["a", "b"],
        title="example",
        dpi=40,
    )
    assert picture.ndim == 3 and picture.shape[-1] == 3
    broken = SimpleNamespace(log_image=Mock(side_effect=OSError("offline")))
    good = SimpleNamespace(log_image=Mock())
    trainer = SimpleNamespace(loggers=[object(), broken, good])
    assert viz.emit_figure(trainer, "figure", picture, step=7, caption="test")
    good.log_image.assert_called_once()
    np.testing.assert_array_equal(
        good.log_image.call_args.kwargs["images"][0], np.asarray(picture)
    )
    assert good.log_image.call_args.kwargs["caption"] == ["test"]


def test_attention_gating_missing_keys_and_head_selection(monkeypatch):
    with pytest.raises(ValueError, match="head_reduction"):
        AttentionVisualizer("attention", "attn", head_reduction="bad")
    callback = AttentionVisualizer(
        "attention",
        "attn",
        log_on=("train", "test"),
        head_reduction="mean",
        grid_size=2,
    )
    trainer = SimpleNamespace(current_epoch=0, sanity_checking=True, global_rank=0)
    assert not callback._should_fire(trainer, 0)
    trainer.sanity_checking = False
    assert not callback._should_fire(trainer, 1)
    callback.on_train_batch_end(trainer, None, {}, {}, 0)
    assert callback._warned_missing
    callback.on_test_batch_end(trainer, None, {}, {}, 0)
    maps = torch.randn(2, 3, 2, 2)
    torch.testing.assert_close(
        callback._select_head_maps(maps, 1)[0][1], maps[1].mean(0)
    )
    assert callback._resolve_grid(torch.ones(1, 2, 4), 8, 8) == (2, 2)
    render = Mock()
    monkeypatch.setattr(callback, "_render_and_log", render)
    batch = {
        "image": torch.rand(1, 3, 8, 8),
        callback.attention: torch.rand(1, 2, 5, 5),
    }
    callback.on_train_batch_end(trainer, None, {}, batch, 0)
    assert render.call_args.args[-1] == "train"
    callback.on_test_batch_end(trainer, None, {}, batch, 0)
    assert render.call_args.args[-1] == "test"
    callback.on_train_batch_end(trainer, None, {}, batch, 1)
    assert render.call_count == 2
