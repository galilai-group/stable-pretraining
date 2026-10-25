"""Unit tests for Learning From Randomness (LFR)."""

import math
from functools import partial

import lightning as pl
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader

import stable_pretraining as spt
from stable_pretraining.forward import lfr
from stable_pretraining.losses import BatchwiseBarlowTwinsLoss
from stable_pretraining.methods.lfr import (
    LFR,
    beta_init_,
    random_image_projector,
    select_diverse_projectors,
    weight_dropout_,
)

pytestmark = pytest.mark.unit


class TinyEncoder(nn.Module):
    """Small image encoder with BatchNorm, so phase-wise BN freezing is visible."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def tiny_lfr(**kwargs) -> LFR:
    kwargs.setdefault("num_projectors", 2)
    kwargs.setdefault("target_dim", 6)
    kwargs.setdefault("predictor_hidden_dim", 8)
    kwargs.setdefault("predictor_epochs", 1)
    return LFR(encoder_name=TinyEncoder(), low_resolution=True, **kwargs)


def image_samples(count: int = 8) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "image": torch.randn(3, 32, 32, generator=generator),
            "label": torch.tensor(i % 2),
        }
        for i in range(count)
    ]


def tiny_trainer(**kwargs) -> pl.Trainer:
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        **kwargs,
    )


def reference_bt_loss_bs(p, z, lambd=0.01):
    """``bt_loss_bs`` from the authors' ``utils/trainers.py`` (layer6ai-labs/lfr)."""
    c = torch.matmul(F.normalize(p), F.normalize(z).T)
    n = c.shape[0]
    off_diagonal = c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    return torch.diagonal(c).add(-1).pow(2).sum() + lambd * off_diagonal.pow(2).sum()


def reference_fast_map_dpp(
    kernel: np.ndarray, max_length: int, first: int
) -> list[int]:
    """Greedy DPP from the authors' ``utils/dpp.py`` (fast-map-dpp).

    The kernel has a unit diagonal, so the authors' first ``argmax`` is a tie
    broken by rounding; ``first`` fixes that choice.
    """
    cis = np.zeros((max_length, kernel.shape[0]))
    di2s = np.copy(np.diag(kernel))
    selected = [first]
    while len(selected) < max_length:
        k = len(selected) - 1
        ci_optimal = cis[:k, selected[-1]]
        di_optimal = math.sqrt(di2s[selected[-1]])
        eis = (kernel[selected[-1], :] - ci_optimal @ cis[:k, :]) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected[-1]] = -np.inf
        selected.append(int(np.argmax(di2s)))
    return selected


# --- loss -----------------------------------------------------------------


def test_loss_matches_authors_reference() -> None:
    torch.manual_seed(0)
    prediction, target = torch.randn(16, 10), torch.randn(16, 10)
    expected = reference_bt_loss_bs(prediction, target, lambd=0.05)
    actual = BatchwiseBarlowTwinsLoss(lambd=0.05)(prediction, target)
    torch.testing.assert_close(actual, expected)


def test_loss_uses_samples_as_matrix_axes() -> None:
    vectors = torch.eye(3)
    criterion = BatchwiseBarlowTwinsLoss(lambd=0.5)
    assert criterion(vectors, vectors).item() == pytest.approx(0.0)
    # Every prediction now resembles every target equally: each diagonal
    # entry is 1/sqrt(3), and the six off-diagonal entries are as large.
    repeated = torch.ones_like(vectors)
    expected = 3 * (1 - 1 / 3**0.5) ** 2 + 0.5 * 6 / 3
    assert criterion(repeated, vectors).item() == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_loss_is_float32_and_stable_for_half_inputs(dtype) -> None:
    torch.manual_seed(1)
    # 512^2 squared similarities near one would overflow a float16 sum.
    prediction = (torch.ones(512, 4) + 1e-2 * torch.randn(512, 4)).to(dtype)
    target = torch.ones(512, 4).to(dtype)
    loss = BatchwiseBarlowTwinsLoss()(prediction, target)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    expected = BatchwiseBarlowTwinsLoss()(prediction.float(), target.float())
    torch.testing.assert_close(loss, expected)


def test_loss_has_finite_gradients_for_collapsed_predictions() -> None:
    prediction = torch.zeros(4, 3, requires_grad=True)
    loss = BatchwiseBarlowTwinsLoss()(prediction, torch.randn(4, 3))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize(
    "prediction,target,match",
    [
        (torch.randn(1, 3), torch.randn(1, 3), "at least 2"),
        (torch.randn(2, 3), torch.randn(2, 4), "matching"),
        (torch.randn(2, 3, 1), torch.randn(2, 3, 1), "matching"),
    ],
)
def test_loss_rejects_invalid_inputs(prediction, target, match) -> None:
    with pytest.raises(ValueError, match=match):
        BatchwiseBarlowTwinsLoss()(prediction, target)


def test_loss_rejects_negative_lambda() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BatchwiseBarlowTwinsLoss(lambd=-1)


# --- projector construction and selection ---------------------------------


def test_beta_init_pushes_conv_weights_to_the_unit_interval_edges() -> None:
    torch.manual_seed(0)
    conv = beta_init_(nn.Conv2d(16, 16, 3))
    weights = conv.weight.flatten()
    assert weights.min() >= -1 and weights.max() <= 1
    # Beta(0.5, 0.5) puts about half its mass in |2x - 1| > 0.7.
    assert (weights.abs() > 0.7).float().mean() > 0.4


def test_weight_dropout_zeroes_the_requested_fraction_per_layer() -> None:
    torch.manual_seed(0)
    model = weight_dropout_(nn.Sequential(nn.Conv2d(3, 8, 3), nn.Linear(10, 10)), 0.4)
    for layer in model:
        zeros = (layer.weight == 0).sum().item()
        assert zeros == round(0.4 * layer.weight.numel())
    with pytest.raises(ValueError, match="rate"):
        weight_dropout_(nn.Linear(2, 2), rate=1.0)


@pytest.mark.parametrize("size", [32, 64])
def test_random_image_projector_maps_images_to_targets(size: int) -> None:
    projector = random_image_projector(dim=12)
    assert projector(torch.randn(2, 3, size, size)).shape == (2, 12)


def test_projector_selection_retains_unique_candidates() -> None:
    first = nn.Linear(4, 2, bias=False)
    duplicate = nn.Linear(4, 2, bias=False)
    distinct = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        first.weight.copy_(torch.eye(4)[:2])
        duplicate.weight.copy_(first.weight)
        distinct.weight.copy_(torch.eye(4)[2:])
    candidates = [first, duplicate, distinct]
    sample = torch.randn(8, 4, generator=torch.Generator().manual_seed(3))
    selected = select_diverse_projectors(candidates, sample, count=2)
    assert 2 in selected
    assert len({0, 1}.intersection(selected)) == 1
    assert selected == select_diverse_projectors(candidates, sample, count=2)


def test_projector_selection_matches_authors_dpp() -> None:
    torch.manual_seed(4)
    candidates = [nn.Sequential(nn.Linear(5, 8), nn.ReLU()) for _ in range(12)]
    sample = torch.randn(16, 5)
    with torch.no_grad():
        signatures = []
        for candidate in candidates:
            rep = F.normalize(candidate(sample), dim=1)
            signatures.append((rep @ rep.T).flatten())
        signatures = F.normalize(torch.stack(signatures).double(), dim=1)
    kernel = (signatures @ signatures.T).numpy()
    np.testing.assert_allclose(np.diag(kernel), 1.0)
    expected = reference_fast_map_dpp(kernel, 4, first=0)
    assert select_diverse_projectors(candidates, sample, count=4) == expected


def test_projector_selection_rejects_invalid_arguments() -> None:
    candidates = [nn.Linear(3, 2)]
    with pytest.raises(ValueError, match="count"):
        select_diverse_projectors(candidates, torch.randn(4, 3), count=2)
    with pytest.raises(ValueError, match="two examples"):
        select_diverse_projectors(candidates, torch.randn(1, 3), count=1)


# --- method construction --------------------------------------------------


def test_default_construction_infers_widths_and_freezes_projectors() -> None:
    model = tiny_lfr()
    assert len(model.projectors) == len(model.predictors) == 2
    assert model.predictors[0][0].in_features == 8
    assert model.predictors[0][-1].out_features == 6
    assert not any(p.requires_grad for p in model.projectors.parameters())
    model.train()
    assert not any(module.training for module in model.projectors.modules())
    assert model.hparams.num_projectors == 2
    assert model.max_epochs_for(cycles=3) == 6


def test_timm_encoder_name_builds_low_resolution_resnet() -> None:
    model = LFR(
        encoder_name="resnet18",
        low_resolution=True,
        num_projectors=1,
        target_dim=4,
        predictor_hidden_dim=4,
    )
    assert model.predictors[0][0].in_features == 512
    assert model.hparams.encoder_name == "resnet18"


def test_candidate_pool_selects_requested_number_of_projectors() -> None:
    sample = torch.randn(4, 3, 32, 32)
    model = tiny_lfr(candidate_ratio=3, selection_sample=sample)
    assert len(model.projectors) == len(model.predictors) == 2
    with pytest.raises(ValueError, match="selection_sample"):
        tiny_lfr(candidate_ratio=3)


def test_custom_modules_set_the_number_of_projectors() -> None:
    model = LFR(
        encoder_name=nn.Linear(4, 3),
        projectors=[nn.Linear(4, 5), nn.Linear(4, 2)],
        predictors=[nn.Linear(3, 5), nn.Linear(3, 2)],
        input_key="x",
    )
    assert model.hparams.num_projectors == 2
    out = model({"x": torch.randn(4, 4)}, stage="fit")
    assert torch.isfinite(out["loss"])
    with pytest.raises(ValueError, match="as many projectors"):
        LFR(
            encoder_name=nn.Linear(4, 3),
            projectors=[nn.Linear(4, 5)],
            predictors=[nn.Linear(3, 5), nn.Linear(3, 5)],
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"predictor_epochs": 0}, "predictor_epochs"),
        ({"num_projectors": 0}, "positive"),
        ({"candidate_ratio": 0}, "positive"),
        ({"predictors": []}, "at least one"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        tiny_lfr(**kwargs)


# --- forward ----------------------------------------------------------------


def test_forward_freezes_targets_and_exposes_embedding_and_label() -> None:
    model = tiny_lfr().train()
    batch = {"image": torch.randn(4, 3, 32, 32), "label": torch.arange(4)}
    out = model(batch, stage="fit")
    assert out["embedding"].shape == (4, 8)
    assert torch.equal(out["label"], batch["label"])
    assert out["loss"].ndim == 0
    out["loss"].backward()
    assert all(p.grad is None for p in model.projectors.parameters())
    assert any(p.grad is not None for p in model.backbone.parameters())
    assert any(p.grad is not None for p in model.predictors.parameters())
    assert set(model(batch, stage="predict")) == {"embedding", "label"}


def test_forward_averages_the_loss_over_projectors() -> None:
    model = tiny_lfr().eval()
    images = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        embedding = model.backbone(images)
        expected = sum(
            model.lfr_loss(predictor(embedding), projector(images))
            for projector, predictor in zip(model.projectors, model.predictors)
        ) / len(model.projectors)
        actual = model({"image": images}, stage="validate")["loss"]
    torch.testing.assert_close(actual, expected)


def test_forward_requires_loss_module() -> None:
    module = spt.Module(
        forward=lfr,
        backbone=nn.Linear(4, 3),
        projectors=nn.ModuleList([nn.Linear(4, 5)]),
        predictors=nn.ModuleList([nn.Linear(3, 5)]),
    )
    with pytest.raises(ValueError, match="lfr_loss"):
        module({"image": torch.randn(4, 4)}, stage="fit")


def test_forward_rejects_non_vector_embeddings() -> None:
    model = LFR(
        encoder_name=nn.Identity(),
        projectors=[nn.Flatten()],
        predictors=[nn.Flatten()],
    )
    with pytest.raises(ValueError, match=r"\[B, E\]"):
        model({"image": torch.randn(2, 3, 4, 4)}, stage="fit")


# --- alternating optimization ----------------------------------------------


class _Snapshots(pl.Callback):
    """Record encoder, predictor, encoder-BN and probe state around each epoch."""

    def __init__(self) -> None:
        self.states = []

    def _record(self, module: LFR) -> None:
        self.states.append(
            {
                "encoder": module.backbone.net[0].weight.detach().clone(),
                "bn": module.backbone.net[1].running_mean.detach().clone(),
                "predictor": module.predictors[0][0].weight.detach().clone(),
                "probe": module.callbacks_modules["probe"].weight.detach().clone(),
            }
        )

    def on_train_start(self, trainer, pl_module) -> None:
        self._record(pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._record(pl_module)


def test_alternating_epochs_update_one_half_while_probes_keep_training() -> None:
    torch.manual_seed(0)
    model = tiny_lfr()
    probe = spt.callbacks.OnlineProbe(
        model,
        name="probe",
        input="embedding",
        target="label",
        probe=nn.Linear(8, 2),
        loss=nn.CrossEntropyLoss(),
        metrics={"acc": torchmetrics.classification.MulticlassAccuracy(2)},
    )
    snapshots = _Snapshots()
    trainer = tiny_trainer(
        max_epochs=model.max_epochs_for(1), callbacks=[probe, snapshots]
    )
    trainer.fit(model, DataLoader(image_samples(), batch_size=4, drop_last=True))

    start, encoder_epoch, predictor_epoch = snapshots.states
    assert not torch.equal(start["encoder"], encoder_epoch["encoder"])
    assert not torch.equal(start["bn"], encoder_epoch["bn"])
    assert torch.equal(start["predictor"], encoder_epoch["predictor"])
    assert torch.equal(encoder_epoch["encoder"], predictor_epoch["encoder"])
    assert torch.equal(encoder_epoch["bn"], predictor_epoch["bn"])
    assert not torch.equal(encoder_epoch["predictor"], predictor_epoch["predictor"])
    for before, after in [(start, encoder_epoch), (encoder_epoch, predictor_epoch)]:
        assert not torch.equal(before["probe"], after["probe"])
    assert trainer.global_step == 4
    trainable = [
        p for n, p in model.named_parameters() if not n.startswith("projectors")
    ]
    assert all(p.requires_grad for p in trainable)


def test_default_optimizer_follows_the_authors_schedule() -> None:
    model = tiny_lfr()
    seen = []

    class _Rates(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            groups = trainer.optimizers[0].param_groups
            seen.append((trainer.current_epoch, groups[0]["lr"], groups[1]["lr"]))

    # Three cycles of one encoder epoch and one predictor epoch, 2 batches each.
    trainer = tiny_trainer(max_epochs=model.max_epochs_for(3), callbacks=[_Rates()])
    trainer.fit(model, DataLoader(image_samples(), batch_size=4, drop_last=True))
    optimizer = trainer.optimizers[0]
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.9
    assert optimizer.defaults["weight_decay"] == 5e-4
    assert len(seen) == 12
    for epoch, encoder_lr, predictor_lr in seen:
        cycle = epoch // 2
        # Authors' adjust_learning_rate: cosine over cycles, fixed within one.
        expected = 0.03 * 0.5 * (1 + math.cos(math.pi * cycle / 3))
        assert encoder_lr == pytest.approx(expected)
        assert predictor_lr == pytest.approx(0.03)


def test_custom_optim_uses_the_generic_module_path() -> None:
    model = tiny_lfr(optim={"optimizer": {"type": "Adam", "lr": 1e-3}})
    trainer = tiny_trainer(max_epochs=1, limit_train_batches=1)
    trainer.fit(model, DataLoader(image_samples(), batch_size=4, drop_last=True))
    assert isinstance(trainer.optimizers[0], torch.optim.Adam)


def test_plain_ddp_strategy_is_rejected_with_a_fix() -> None:
    model = tiny_lfr()
    model.trainer = tiny_trainer(strategy=DDPStrategy())
    with pytest.raises(RuntimeError, match="ddp_find_unused_parameters_true"):
        model.setup("fit")
    model.trainer = tiny_trainer(strategy=DDPStrategy(find_unused_parameters=True))
    model.setup("fit")


# --- review regressions -----------------------------------------------------


def _epoch_weights(model: LFR, max_epochs: int, **kwargs) -> list[dict]:
    """Encoder and predictor weights at the start and after every epoch."""
    states = []

    class _Record(pl.Callback):
        def _snapshot(self) -> None:
            states.append(
                {
                    "encoder": model.backbone.net[-1].weight.detach().clone(),
                    "predictor": model.predictors[0][0].weight.detach().clone(),
                }
            )

        def on_train_start(self, trainer, pl_module) -> None:
            self._snapshot()

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            self._snapshot()

    trainer = tiny_trainer(max_epochs=max_epochs, callbacks=[_Record()], **kwargs)
    # 12 samples in batches of 4: three batches per epoch.
    trainer.fit(model, DataLoader(image_samples(12), batch_size=4, drop_last=True))
    return states


def test_accumulated_gradients_never_update_the_frozen_half() -> None:
    torch.manual_seed(0)
    # Accumulating over 2 of 3 batches leaves a partial window at each epoch end.
    model = tiny_lfr(
        optim={"optimizer": partial(torch.optim.SGD, lr=0.1), "frequency": 2}
    )
    start, encoder_epoch, predictor_epoch, encoder_again = _epoch_weights(model, 3)
    assert not torch.equal(start["encoder"], encoder_epoch["encoder"])
    assert torch.equal(start["predictor"], encoder_epoch["predictor"])
    assert torch.equal(encoder_epoch["encoder"], predictor_epoch["encoder"])
    assert not torch.equal(encoder_epoch["predictor"], predictor_epoch["predictor"])
    assert torch.equal(predictor_epoch["predictor"], encoder_again["predictor"])


def test_checkpoint_with_candidate_pool_reloads(tmp_path) -> None:
    torch.manual_seed(0)
    model = tiny_lfr(candidate_ratio=3, selection_sample=torch.randn(4, 3, 32, 32))
    trainer = tiny_trainer(max_epochs=1, limit_train_batches=1)
    trainer.fit(model, DataLoader(image_samples(), batch_size=4, drop_last=True))
    checkpoint = tmp_path / "lfr.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = LFR.load_from_checkpoint(
        checkpoint, encoder_name=TinyEncoder(), weights_only=False
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, msg=name)


def test_candidate_pool_keeps_the_diverse_selection() -> None:
    sample = torch.randn(6, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(5)
    TinyEncoder()  # tiny_lfr builds its encoder first; consume the same draws
    pool = [random_image_projector(6) for _ in range(2 * 4)]
    chosen = select_diverse_projectors(pool, sample, count=2)
    torch.manual_seed(5)
    model = tiny_lfr(candidate_ratio=4, selection_sample=sample)
    assert len(model.projectors) == 2
    for projector, index in zip(model.projectors, chosen):
        for kept, expected in zip(projector.parameters(), pool[index].parameters()):
            torch.testing.assert_close(kept, expected)


def test_projector_selection_caps_the_sample_size() -> None:
    torch.manual_seed(2)
    candidates = [nn.Sequential(nn.Linear(5, 8), nn.ReLU()) for _ in range(10)]
    sample = torch.randn(32, 5)
    subset = torch.randperm(32, generator=torch.Generator().manual_seed(0))[:8]
    capped = select_diverse_projectors(candidates, sample, count=4, max_samples=8)
    assert capped == select_diverse_projectors(candidates, sample[subset], count=4)


def test_loss_without_gather_uses_only_the_local_batch() -> None:
    torch.manual_seed(3)
    prediction, target = torch.randn(5, 4), torch.randn(5, 4)
    criterion = BatchwiseBarlowTwinsLoss()
    torch.testing.assert_close(
        criterion(prediction, target, gather=False), criterion(prediction, target)
    )


def test_loss_stays_float32_under_autocast() -> None:
    torch.manual_seed(4)
    target = torch.randn(64, 32)
    prediction = target + 0.01 * torch.randn_like(target)
    criterion = BatchwiseBarlowTwinsLoss()
    expected = criterion(prediction, target)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = criterion(prediction, target)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_projectors_use_the_authors_initialization_order() -> None:
    torch.manual_seed(5)
    projector = random_image_projector(dim=12)
    convs = [m for m in projector.modules() if isinstance(m, nn.Conv2d)]
    linear = projector[-1]
    # Beta init runs after dropout and redraws every conv weight.
    for conv in convs:
        assert (conv.weight == 0).float().mean() < 0.01
    assert (linear.weight == 0).sum().item() == round(0.4 * linear.weight.numel())


def test_forward_supports_true_half_precision() -> None:
    model = tiny_lfr().to(torch.bfloat16).eval()
    images = torch.randn(4, 3, 32, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model({"image": images}, stage="validate")
    assert torch.isfinite(out["loss"])


def test_plain_ddp_is_allowed_outside_fitting() -> None:
    model = tiny_lfr()
    model.trainer = tiny_trainer(strategy=DDPStrategy())
    for stage in ("validate", "test", "predict"):
        model.setup(stage)
    with pytest.raises(RuntimeError, match="ddp_find_unused_parameters_true"):
        model.setup("fit")


def test_singleton_evaluation_batch_has_a_loss() -> None:
    model = tiny_lfr().eval()
    with torch.no_grad():
        out = model({"image": torch.randn(1, 3, 32, 32)}, stage="validate")
    assert torch.isfinite(out["loss"])
    with pytest.raises(ValueError, match="at least 2"):
        BatchwiseBarlowTwinsLoss()(torch.randn(1, 3), torch.randn(1, 3))
