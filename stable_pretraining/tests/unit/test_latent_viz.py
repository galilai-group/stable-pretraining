"""Latent projections remain independent of encoder training and survive checkpoints."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import lightning as pl
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from stable_pretraining import Module
from stable_pretraining.callbacks.latent_viz import LatentViz
from stable_pretraining.callbacks.queue import OnlineQueue
from stable_pretraining.utils.distance_metrics import compute_pairwise_distances_chunked

pytestmark = pytest.mark.unit


def _forward(self, batch, stage):
    embedding = self.backbone(batch["image"])
    return {
        "embedding": embedding,
        "label": batch["label"],
        "loss": embedding.square().mean(),
    }


def _model(optim=True):
    return Module(
        forward=_forward,
        backbone=nn.Linear(3, 4),
        optim={
            "optimizer": {"type": "SGD", "lr": 0.01},
            "scheduler": {"type": "ConstantLR", "factor": 1.0},
        }
        if optim
        else None,
    )


def _callback(model, **kwargs):
    projection = kwargs.pop("projection", None)
    return LatentViz(
        model,
        name="map",
        input="embedding",
        target="label",
        projection=projection if projection is not None else nn.Linear(4, 2),
        queue_length=kwargs.pop("queue_length", 8),
        input_dim=kwargs.pop("input_dim", 4),
        **kwargs,
    )


def _loader():
    generator = torch.Generator().manual_seed(81)
    return DataLoader(
        [
            {"image": torch.randn(3, generator=generator), "label": i % 2}
            for i in range(16)
        ],
        batch_size=4,
    )


def _trainer(tmp_path, callbacks, **kwargs):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        **kwargs,
    )


@pytest.mark.parametrize("queue_first", [False, True])
@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed", "bf16-true"])
def test_real_fit_trains_projection_without_changing_encoder_gradient(
    tmp_path, queue_first, precision
):
    torch.manual_seed(15)
    model = _model()
    initial = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    cb = _callback(
        model, update_interval=1, k_neighbors=2, plot_interval=1, verbose=True
    )
    before = cb._projection_config.weight.detach().clone()
    queues = [
        OnlineQueue("embedding", 8, dim=4),
        OnlineQueue("label", 8, dtype=torch.long),
    ]
    trainer = _trainer(
        tmp_path,
        queues + [cb] if queue_first else [cb] + queues,
        max_epochs=1,
        precision=precision,
    )
    trainer.fit(model, _loader(), _loader())
    assert trainer.global_step == 4
    assert not torch.equal(before, cb.projection_module.weight)
    assert cb._scheduler.last_epoch == 3
    archive = np.load(tmp_path / "latent_viz_map" / "epoch_0000.npz")
    assert archive["coordinates"].shape == (8, 2)
    assert archive["labels"].reshape(-1).tolist() == [0, 1] * 4
    assert np.isfinite(archive["coordinates"]).all()
    baseline = _model()
    baseline.backbone.load_state_dict(initial)
    reference = _trainer(tmp_path / "baseline", [], max_epochs=1, precision=precision)
    reference.fit(baseline, _loader())
    for actual, expected in zip(
        model.backbone.parameters(), baseline.backbone.parameters()
    ):
        torch.testing.assert_close(actual, expected)
    checkpoint = tmp_path / "state.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert not any("map" in key for key in saved["state_dict"])
    assert len(saved["optimizer_states"]) == 1
    assert saved["callbacks"][cb.state_key]["optimizer"]["state"]
    resumed = _model()
    resumed_cb = _callback(resumed, update_interval=1, k_neighbors=2)
    restored = _trainer(
        tmp_path / "resume", [resumed_cb], max_steps=5, precision=precision
    )
    restored.fit(resumed, _loader(), ckpt_path=checkpoint)
    assert restored.global_step == 5
    assert int(resumed_cb._optimizer.state[resumed_cb.module.weight]["step"]) == 4


@pytest.mark.parametrize("main_optimizer", [False, True])
def test_inactive_projection_does_not_step_optimizer_or_scheduler(
    tmp_path, main_optimizer
):
    model = _model(main_optimizer)
    cb = _callback(model, warmup_epochs=10)
    before = cb._projection_config.weight.detach().clone()
    trainer = _trainer(tmp_path, [cb], max_epochs=1, limit_train_batches=3)
    trainer.fit(model, _loader())
    assert trainer.global_step == (3 if main_optimizer else 0)
    torch.testing.assert_close(cb.module.weight, before)
    assert not cb._optimizer.state
    assert cb._scheduler.last_epoch == 0


@pytest.mark.parametrize(
    "accumulation,interval,steps", [(1, 2, 1), (2, 1, 2), (2, 2, 1)]
)
def test_update_intervals_and_accumulation(tmp_path, accumulation, interval, steps):
    cb_model = _model()
    cb = _callback(
        cb_model, update_interval=interval, accumulate_grad_batches=accumulation
    )
    trainer = _trainer(tmp_path, [cb], max_steps=4)
    trainer.fit(cb_model, _loader())
    assert int(cb._optimizer.state[cb.module.weight]["step"]) == steps
    assert trainer.global_step == 4


@pytest.mark.parametrize(
    "projection",
    [
        lambda: nn.Linear(4, 2),
        {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 2},
    ],
)
def test_projection_factories_and_custom_optimizer(projection):
    model = _model()
    cb = _callback(
        model,
        projection=projection,
        input_dim=(2, 2),
        optimizer=partial(torch.optim.SGD, lr=0.2),
    )
    model.configure_model()
    opt = cb.setup_optimizer(model)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["lr"] == 0.2
    assert cb.input_dim == 4
    assert "map" not in model.callbacks_modules
    cb.setup(SimpleNamespace(), model, "validate")
    assert cb._input_queue is None


@pytest.mark.parametrize(
    "key,value",
    [
        (key, 0)
        for key in [
            "queue_length",
            "k_neighbors",
            "n_negatives",
            "update_interval",
            "plot_interval",
            "accumulate_grad_batches",
        ]
    ]
    + [("queue_length", 1), ("warmup_epochs", -1), ("distance_metric", "bad")],
)
def test_invalid_configuration_fails_before_wrapping_model(key, value):
    model = _model()
    original = model.forward
    with pytest.raises(ValueError, match=key):
        _callback(model, **{key: value})
    assert model.forward == original


@pytest.mark.parametrize("size", [0, 1, 2, 6, 1001])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loss_is_finite_and_differentiable(size, dtype):
    cb = _callback(_model(), k_neighbors=1)
    x = torch.arange(size * 4).reshape(size, 4).to(dtype)
    z = (torch.arange(size * 2).reshape(size, 2) / 10).to(dtype).requires_grad_()
    torch.manual_seed(10)
    loss = cb._compute_loss(x, z)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    if size < 2:
        assert loss.item() == 0
    if size == 2:
        assert cb._last_repulsion_loss == 0
        expected = torch.log(2 + (z.float()[0] - z.float()[1]).square().sum())
        torch.testing.assert_close(loss, expected)


def test_negative_samples_exclude_self_and_positive_neighbors(monkeypatch):
    cb = _callback(_model(), k_neighbors=1)
    x = torch.tensor([[0.0], [1.0], [10.0]])
    sampled = []
    original = torch.multinomial

    def capture(weights, count, replacement):
        sampled.append(weights.clone())
        return original(weights, count, replacement)

    monkeypatch.setattr(torch, "multinomial", capture)
    cb._compute_loss(x, torch.randn(3, 2, requires_grad=True))
    torch.testing.assert_close(
        sampled[0], torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    )


@pytest.mark.parametrize(
    "metric", ["euclidean", "squared_euclidean", "cosine", "manhattan"]
)
@pytest.mark.parametrize("chunk", [-1, 2, 100])
def test_chunked_distances_match_explicit_pairwise_reference(metric, chunk):
    x = torch.tensor([[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0]])
    y = x[:2] + 0.3
    difference = x[:, None] - y[None, :]
    expected = {
        "euclidean": difference.square().sum(-1).sqrt(),
        "squared_euclidean": difference.square().sum(-1),
        "manhattan": difference.abs().sum(-1),
        "cosine": 1 - nn.functional.normalize(x) @ nn.functional.normalize(y).T,
    }[metric]
    torch.testing.assert_close(
        compute_pairwise_distances_chunked(x, y, metric, chunk), expected
    )


def test_invalid_distance_metric_is_rejected():
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_pairwise_distances_chunked(torch.ones(2, 3), torch.ones(2, 3), "bad")


@pytest.mark.parametrize(
    "reason",
    [
        "sanity",
        "batch",
        "loader",
        "warmup",
        "interval",
        "no_queue",
        "empty",
        "none",
        "rank",
    ],
)
def test_validation_gates_do_not_write(tmp_path, reason):
    model = _model()
    cb = _callback(model, warmup_epochs=1, plot_interval=2)
    model.configure_model()
    trainer = SimpleNamespace(
        sanity_checking=reason == "sanity",
        current_epoch=0 if reason == "warmup" else 3 if reason == "interval" else 2,
        global_rank=1 if reason == "rank" else 0,
        default_root_dir=str(tmp_path),
    )
    cb._input_queue = (
        None
        if reason == "no_queue"
        else SimpleNamespace(
            data=None
            if reason == "none"
            else torch.empty(0, 4)
            if reason == "empty"
            else torch.ones(3, 4)
        )
    )
    cb._plot_2d_embeddings = Mock()
    cb.on_validation_batch_end(
        trainer, model, {}, {}, int(reason == "batch"), int(reason == "loader")
    )
    cb._plot_2d_embeddings.assert_not_called()
    assert cb.module.training


@pytest.mark.parametrize("labels", [None, torch.empty(0, dtype=torch.long)])
def test_validation_without_labels_and_cache_directory_fallback(
    tmp_path, monkeypatch, labels
):
    from stable_pretraining._config import get_config

    model = _model()
    cb = _callback(model)
    model.configure_model()
    cb._input_queue = SimpleNamespace(data=torch.ones(3, 4, dtype=torch.float64))
    cb._target_queue = None if labels is None else SimpleNamespace(data=labels)
    monkeypatch.chdir(tmp_path)
    get_config()._cache_dir = str(tmp_path / "cache")
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=0,
        global_rank=0,
        default_root_dir=str(tmp_path),
    )
    cb.on_validation_batch_end(trainer, model, {}, {}, 0)
    with np.load(tmp_path / "cache" / "latent_viz_map" / "epoch_0000.npz") as saved:
        assert saved.files == ["coordinates"]
        assert saved["coordinates"].shape == (3, 2)
    assert cb.module.training


def test_callback_does_not_modify_model_methods_or_loss():
    model = _model()
    methods = (model.forward, model.configure_model, model.configure_optimizers)
    parameters = dict(model.named_parameters())
    cb = _callback(model)
    model.configure_model()
    assert methods == (model.forward, model.configure_model, model.configure_optimizers)
    assert parameters == dict(model.named_parameters())
    result = model(
        {"image": torch.ones(2, 3), "label": torch.zeros(2), "batch_idx": 0}, "fit"
    )
    torch.testing.assert_close(result["loss"], result["embedding"].square().mean())
    result["loss"].backward()
    assert all(p.grad is None for p in cb.module.parameters())


def test_explicit_output_path_and_eval_mode_restored_on_write_error(
    tmp_path, monkeypatch
):
    model = _model()
    cb = _callback(model, save_dir=str(tmp_path / "custom"))
    model.configure_model()
    trainer = SimpleNamespace(sanity_checking=False, current_epoch=0, global_rank=0)
    cb._input_queue = SimpleNamespace(data=torch.ones(3, 4))
    cb.on_validation_batch_end(trainer, model, {}, {}, 0)
    assert (tmp_path / "custom" / "epoch_0000.npz").is_file()
    cb.module.eval()
    monkeypatch.setattr(
        cb, "_plot_2d_embeddings", Mock(side_effect=OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        cb.on_validation_batch_end(trainer, model, {}, {}, 0)
    assert not cb.module.training


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_coordinate_export_preserves_values_and_supported_dtypes(tmp_path, dtype):
    cb = _callback(_model(), save_dir=str(tmp_path))
    coordinates = torch.tensor(
        [[0.25, -1.5], [2.0, 3.5]], dtype=dtype, requires_grad=True
    )
    cb._plot_2d_embeddings(coordinates, None, 3, SimpleNamespace())
    expected = coordinates.detach()
    if dtype == torch.bfloat16:
        expected = expected.float()
    with np.load(tmp_path / "epoch_0003.npz") as saved:
        np.testing.assert_array_equal(saved["coordinates"], expected.numpy())
        assert saved["coordinates"].dtype == expected.numpy().dtype


class _DistributedAudit(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        from pathlib import Path

        cb = next(cb for cb in trainer.callbacks if isinstance(cb, LatentViz))
        assert len(trainer.optimizers) == 1
        assert "map" not in pl_module.callbacks_modules
        if trainer.is_global_zero:
            assert int(cb._optimizer.state[cb.module.weight]["step"]) == 1
        else:
            assert cb._optimizer is None
        projection = pl_module.backbone
        assert trainer.global_step == 4
        torch.save(
            projection.state_dict(),
            Path(trainer.default_root_dir) / f"rank-{trainer.global_rank}.pt",
        )


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo unavailable"
)
def test_two_rank_cpu_ddp_handles_warmup_and_idle_projection(tmp_path, monkeypatch):
    import os
    from lightning.pytorch.strategies import DDPStrategy

    for name in list(os.environ):
        if name.startswith("SLURM_"):
            monkeypatch.delenv(name)
    model = _model()
    cb = _callback(model, warmup_epochs=1, update_interval=2, plot_interval=1)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(
            start_method="fork",
            process_group_backend="gloo",
            find_unused_parameters=False,
        ),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        callbacks=[cb, _DistributedAudit()],
        max_steps=4,
    )
    trainer.fit(model, _loader(), _loader())
    with np.load(tmp_path / "latent_viz_map" / "epoch_0001.npz") as archive:
        assert archive["coordinates"].shape == (16, 2)
        assert np.isfinite(archive["coordinates"]).all()
        assert len(archive["labels"]) == 16
    left = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    right = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    for key in left:
        torch.testing.assert_close(left[key], right[key])


def test_cpu_amp_grad_scaler_skips_idle_optimizer_and_trains_on_active_batches(
    tmp_path,
):
    from lightning.pytorch.plugins.precision import MixedPrecision

    model = _model()
    cb = _callback(model, update_interval=2)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    trainer = _trainer(
        tmp_path,
        [cb],
        max_steps=4,
        plugins=[MixedPrecision("16-mixed", device="cpu", scaler=scaler)],
    )
    trainer.fit(model, _loader())
    assert trainer.precision_plugin.scaler is scaler
    assert int(cb._optimizer.state[cb.module.weight]["step"]) == 1
    assert cb._scheduler.last_epoch == 1
    assert trainer.global_step == 4


@pytest.mark.parametrize("interval", [1, 2])
def test_accumulated_projection_update_matches_explicit_sgd_reference(
    tmp_path, interval
):
    import copy

    model = _model()
    model.optim["optimizer"]["lr"] = 0.0
    cb = _callback(
        model,
        update_interval=interval,
        accumulate_grad_batches=2,
        k_neighbors=8,
        optimizer=partial(torch.optim.SGD, lr=0.2),
    )
    reference = copy.deepcopy(cb._projection_config)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.2)
    history = torch.empty(0, 4)
    for idx, batch in enumerate(_loader()):
        if len(history) >= 2 and idx % interval == 0:
            (cb._compute_loss(history, reference(history)) / 2).backward()
        if (idx + 1) % 2 == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            history = torch.cat([history, model.backbone(batch["image"])])[-8:]
    trainer = _trainer(tmp_path, [cb], max_steps=4)
    trainer.fit(model, _loader())
    for actual, expected in zip(cb.module.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def _spatial_forward(self, batch, stage):
    result = _forward(self, batch, stage)
    result["embedding"] = result["embedding"].reshape(-1, 2, 2)
    return result


@pytest.mark.parametrize("input_dim", [(2, 2), [2, 2], None])
def test_spatial_features_keep_queue_shape_and_flatten_for_projection(
    tmp_path, input_dim
):
    model = Module(
        forward=_spatial_forward,
        backbone=nn.Linear(3, 4),
        optim={
            "optimizer": {"type": "SGD", "lr": 0.01},
            "scheduler": {"type": "ConstantLR", "factor": 1.0},
        },
    )
    cb = _callback(model, input_dim=input_dim, update_interval=1)
    trainer = _trainer(tmp_path, [cb], max_epochs=1)
    trainer.fit(model, _loader(), _loader())
    assert OnlineQueue._shared_queues["embedding"].get().shape == (8, 2, 2)
    with np.load(tmp_path / "latent_viz_map" / "epoch_0000.npz") as archive:
        assert archive["coordinates"].shape == (8, 2)
        assert np.isfinite(archive["coordinates"]).all()


def test_shared_encoder_parameters_are_rejected():
    model = _model()
    with pytest.raises(ValueError, match="must not share parameters"):
        _callback(model, projection=model.backbone)


def test_projection_factory_preserves_random_stream():
    model = _model()
    rng = torch.get_rng_state().clone()
    _callback(model, projection=lambda: nn.Linear(4, 2))
    torch.testing.assert_close(torch.get_rng_state(), rng)


@pytest.mark.parametrize("use_callback", [False, True])
def test_stochastic_encoder_matches_baseline_with_projection_dropout(
    tmp_path, use_callback
):
    import copy

    model = _model()
    model.backbone = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.3))
    baseline = copy.deepcopy(model)
    callbacks = []
    if use_callback:
        callbacks.append(
            _callback(
                model,
                projection=nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 2)),
                update_interval=1,
                k_neighbors=1,
            )
        )
    torch.manual_seed(431)
    trainer = _trainer(tmp_path, callbacks, max_epochs=1)
    trainer.fit(model, _loader())
    after = torch.get_rng_state().clone()
    torch.manual_seed(431)
    reference = _trainer(tmp_path / "baseline", [], max_epochs=1)
    reference.fit(baseline, _loader())
    torch.testing.assert_close(torch.get_rng_state(), after)
    for actual, expected in zip(
        model.backbone.parameters(), baseline.backbone.parameters()
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint_after", [1, 3, 4])
def test_callback_checkpoint_preserves_optimizer_scheduler_and_partial_gradients(
    monkeypatch, checkpoint_after
):
    import copy

    model = _model()
    model.log = Mock()
    trainer = SimpleNamespace(is_global_zero=True, current_epoch=0)
    config = dict(
        update_interval=1,
        accumulate_grad_batches=3,
        k_neighbors=1,
        verbose=False,
        scheduler={"type": "StepLR", "step_size": 1, "gamma": 0.5},
    )
    cb = _callback(model, **config)
    cb.on_fit_start(trainer, model)
    features = torch.randn(6, 4, requires_grad=True)
    monkeypatch.setattr(
        OnlineQueue,
        "_shared_queues",
        {
            "embedding": SimpleNamespace(get=lambda: features),
        },
    )
    for idx in range(checkpoint_after):
        cb.on_train_batch_start(trainer, model, {}, idx)
    saved = copy.deepcopy(cb.state_dict())
    restored = _callback(model, **config)
    restored.load_state_dict(saved)
    assert restored.state_dict() is saved
    restored.on_fit_start(trainer, model)
    for idx in range(checkpoint_after, 6):
        cb.on_train_batch_start(trainer, model, {}, idx)
        restored.on_train_batch_start(trainer, model, {}, idx)
    for actual, expected in zip(cb.module.parameters(), restored.module.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cb._scheduler.state_dict() == restored._scheduler.state_dict()
    for p, q in zip(cb.module.parameters(), restored.module.parameters()):
        for key, value in cb._optimizer.state[p].items():
            torch.testing.assert_close(value, restored._optimizer.state[q][key])
    assert features.grad is None
    assert all(p.grad is None for p in model.parameters())


def test_checkpoint_before_optimizer_initialization_and_empty_queue(monkeypatch):
    model = _model()
    cb = _callback(model)
    state = cb.state_dict()
    assert state["optimizer"] is None
    restored = _callback(model)
    restored.load_state_dict(state)
    trainer = SimpleNamespace(is_global_zero=True, current_epoch=0)
    restored.on_fit_start(trainer, model)
    monkeypatch.setattr(OnlineQueue, "_shared_queues", {})
    restored.on_train_batch_start(trainer, model, {}, 0)
    assert not restored._optimizer.state
    assert restored._scheduler.last_epoch == 0


def test_projection_can_train_without_an_encoder_optimizer(tmp_path):
    model = _model(optim=False)
    cb = _callback(model, update_interval=1)
    trainer = _trainer(tmp_path, [cb], max_epochs=1)
    trainer.fit(model, _loader())
    assert trainer.global_step == 0
    assert int(cb._optimizer.state[cb.module.weight]["step"]) == 3


def test_in_place_projection_cannot_modify_shared_features(tmp_path, monkeypatch):
    model = _model()
    model.log = Mock()
    cb = _callback(
        model,
        projection=nn.Sequential(nn.ReLU(inplace=True), nn.Linear(4, 2)),
        update_interval=1,
        verbose=False,
    )
    trainer = SimpleNamespace(
        is_global_zero=True,
        global_rank=0,
        current_epoch=0,
        sanity_checking=False,
        default_root_dir=str(tmp_path),
    )
    cb.on_fit_start(trainer, model)
    features = torch.tensor([[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]])
    expected = features.clone()
    monkeypatch.setattr(
        OnlineQueue,
        "_shared_queues",
        {
            "embedding": SimpleNamespace(get=lambda: features),
        },
    )
    original_loss = cb._compute_loss

    def check_loss_inputs(high, low):
        torch.testing.assert_close(high, expected)
        return original_loss(high, low)

    cb._compute_loss = check_loss_inputs
    cb.on_train_batch_start(trainer, model, {}, 0)
    torch.testing.assert_close(features, expected)
    cb._input_queue = SimpleNamespace(data=features)
    cb.on_validation_batch_end(trainer, model, {}, {}, 0)
    torch.testing.assert_close(features, expected)


@pytest.mark.parametrize("batches_per_epoch", [1, 2, 4])
def test_accumulation_windows_continue_across_short_epochs(tmp_path, batches_per_epoch):
    model = _model()
    cb = _callback(model, update_interval=1, accumulate_grad_batches=3)
    trainer = _trainer(
        tmp_path,
        [cb],
        max_epochs=3,
        limit_train_batches=batches_per_epoch,
    )
    trainer.fit(model, _loader())
    assert trainer.global_step == 3 * batches_per_epoch
    assert int(cb._optimizer.state[cb.module.weight]["step"]) == batches_per_epoch
    assert cb._scheduler.last_epoch == batches_per_epoch
    assert cb._accumulated_batches == 0


def test_nonzero_rank_does_not_allocate_optimizer_or_train():
    model = _model()
    cb = _callback(model)
    cb.setup_optimizer = Mock(side_effect=AssertionError("rank must stay idle"))
    model.log = Mock()
    before = cb.module.weight.detach().clone()
    trainer = SimpleNamespace(is_global_zero=False, current_epoch=0)
    cb.on_fit_start(trainer, model)
    cb.on_train_batch_start(trainer, model, {}, 0)
    cb.setup_optimizer.assert_not_called()
    model.log.assert_not_called()
    assert cb._optimizer is None
    torch.testing.assert_close(cb.module.weight, before)
    assert all(p.grad is None for p in cb.module.parameters())
