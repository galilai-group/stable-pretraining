"""Pinned EEG-JEPA parity, including stochastic forwards and Manager updates."""

import copy
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import lightning as pl
import pytest
import torch

from examples.eeg_jepa.config import Config
from examples.eeg_jepa.data import Collate, synthetic_samples
from examples.eeg_jepa.model import build_module
from examples.eeg_jepa.reference import load_reference
from examples.eeg_jepa.train import run

pytestmark = pytest.mark.integration


@pytest.fixture
def reference():
    root = os.environ.get("EEG_JEPA_REFERENCE")
    if not root:
        pytest.skip("Set EEG_JEPA_REFERENCE to the external pinned reference archive")
    torch.set_num_threads(1)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield load_reference(Path(root))
    finally:
        torch.use_deterministic_algorithms(previous)


def reference_trainer(reference, config, model):
    # Bypass file/logging setup; retain the authors' actual schedule methods.
    trainer = reference.trainer.EEGJEPATrainer.__new__(reference.trainer.EEGJEPATrainer)
    trainer.params = SimpleNamespace(
        lr=config.lr,
        final_lr=config.final_lr,
        lr_scheduler="WarmupCosine",
        warmup_epochs=config.lr_warmup_steps,
        ema_start=config.ema_start,
        ema_end=config.ema_end,
        warmup_lambda_epochs=config.context_warmup_steps,
        lambda_ctx=config.lambda_ctx,
        vcreg_warmup_epochs=config.vcreg_warmup_steps,
    )
    trainer.data_length = 1
    trainer.total_steps = config.total_steps
    trainer.global_step = 0
    trainer.optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters())
        + list(model.fusion_mlp.parameters())
        + list(model.predictor.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    trainer.optimizer_scheduler = trainer._build_lr_scheduler(trainer.params)
    return trainer


def reference_forward(reference, model, batch, trainer, config):
    x = batch["eeg"] / 100.0
    context, target = batch["context_indices"], batch["target_indices"]
    mask = reference.model.indices_to_binary_mask(
        target, x.shape[0], x.shape[1], config.patches
    )
    outputs = model(x, context, target, mask)
    valid = batch["valid_grid"].flatten(1)
    factor = trainer._vcreg_factor_at_step()
    result = reference.model.compute_structured_latent_loss(
        *outputs,
        context,
        target,
        patch_num=config.patches,
        lambda_ctx=trainer._lambda_at_step(),
        ctx_valid_mask=valid.gather(1, context),
        tgt_valid_mask=valid.gather(1, target),
        lambda_var=config.lambda_var * factor,
        lambda_cov=config.lambda_cov * factor,
        vcreg_gamma=config.vcreg_gamma,
        vcreg_sample_size=config.vcreg_sample_size,
        return_stats=True,
    )
    return outputs, result


def max_difference(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    return float((left.detach() - right.detach()).abs().max())


def compare_outputs(outputs, result, actual):
    for expected_list, key in zip(
        outputs,
        (
            "context_predictions",
            "target_predictions",
            "context_targets",
            "target_targets",
        ),
    ):
        for expected, adapted in zip(expected_list, actual[key], strict=True):
            max_difference(expected, adapted)
    max_difference(result[0], actual["loss"])
    assert result[1:] == (
        actual["prediction_loss"],
        actual["context_loss"],
        actual["stats"],
    )


def compare_gradients(reference_model, adapted_model):
    maximum = 0.0
    seen = set()
    for (name, expected), (other_name, actual) in zip(
        reference_model.named_parameters(),
        adapted_model.named_parameters(),
        strict=True,
    ):
        assert name == other_name
        assert (expected.grad is None) == (actual.grad is None), name
        if expected.grad is not None:
            maximum = max(maximum, max_difference(expected.grad, actual.grad))
            if expected.grad.count_nonzero():
                seen.add(name.split(".")[0])
        if name.startswith("target_encoder."):
            assert not actual.requires_grad and actual.grad is None
    assert seen == {"context_encoder", "fusion_mlp", "predictor"}
    return maximum


@pytest.mark.parametrize(
    "training,step,sample_size", [(False, 0, 8192), (True, 1, 8192), (True, 3, 8)]
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_forward_loss_and_all_gradients(reference, training, step, sample_size, device):
    config = replace(Config(), vcreg_sample_size=sample_size)
    torch.manual_seed(12)
    original = reference.model.EEGJEPA(**config.model_kwargs())
    adapted = build_module(reference, config)
    adapted.eeg.model.load_state_dict(original.state_dict())
    adapted.eeg.completed_steps.fill_(step)
    original.to(device).train(training)
    adapted.to(device).train(training)
    trainer = reference_trainer(reference, config, original)
    trainer.global_step = step
    batch = Collate(reference.masks.NMETTargetSupportCollator(**config.mask_kwargs()))(
        synthetic_samples(config)[: config.batch_size]
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    original_batch = {**batch, "eeg": batch["eeg"].clone().requires_grad_()}
    adapted_batch = {**batch, "eeg": batch["eeg"].clone().requires_grad_()}
    torch.manual_seed(47)
    outputs, result = reference_forward(
        reference, original, original_batch, trainer, config
    )
    result[0].backward()
    torch.manual_seed(47)
    actual = adapted(adapted_batch, stage="fit")
    actual["loss"].backward()
    compare_outputs(outputs, result, actual)
    gradient_difference = compare_gradients(original, adapted.eeg.model)
    input_difference = max_difference(
        original_batch["eeg"].grad, adapted_batch["eeg"].grad
    )
    assert original.target_encoder.training == adapted.eeg.teacher.training == training
    assert all(not t.requires_grad for t in actual["target_targets"])
    assert int(adapted.eeg.completed_steps) == step
    print(
        f"forward/loss/parameter-gradient max_abs=0; input-gradient={input_difference}; "
        f"training={training}, step={step}, VCReg samples={sample_size}; "
        f"gradient={gradient_difference}"
    )


def test_nmet_masks_and_validity_match(reference):
    config = Config()
    samples = synthetic_samples(config)[: config.batch_size]
    torch.manual_seed(5)
    expected = reference.masks.NMETTargetSupportCollator(**config.mask_kwargs())(
        samples
    )
    torch.manual_seed(5)
    actual = Collate(reference.masks.NMETTargetSupportCollator(**config.mask_kwargs()))(
        samples
    )
    for left, right in (
        (expected[0], actual["eeg"]),
        (expected[1][0], actual["context_indices"]),
        (expected[2][0], actual["target_indices"]),
        (expected[3], actual["valid_grid"]),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for key in ("context_indices", "target_indices"):
        assert actual["valid_grid"].flatten(1).gather(1, actual[key]).all()
    for context, target in zip(actual["context_indices"], actual["target_indices"]):
        assert not torch.isin(context, target).any()


def test_reference_step_schedules(reference):
    config = Config()
    trainer = reference_trainer(
        reference, config, reference.model.EEGJEPA(**config.model_kwargs())
    )
    module = build_module(reference, config)
    for step in range(config.total_steps):
        trainer.global_step = step
        assert config.momentum(step) == trainer._momentum_at_step()
        assert config.lr_factor(step) == pytest.approx(
            trainer.optimizer_scheduler.get_last_lr()[0] / config.lr, abs=1e-15
        )
        for parameter in module.eeg.model.context_encoder.parameters():
            with torch.no_grad():
                parameter.add_(0.01)
        expected = copy.deepcopy(module.eeg.model)
        expected.update_target_encoder(trainer._momentum_at_step())
        module.eeg.update_teacher()
        for name, value in expected.state_dict().items():
            max_difference(value, module.eeg.model.state_dict()[name])
        assert int(module.eeg.completed_steps) == step + 1
        trainer.optimizer.step()
        trainer.optimizer_scheduler.step()
    restored = build_module(reference, config)
    restored.load_state_dict(module.state_dict())
    assert int(restored.eeg.completed_steps) == config.total_steps


class CompareTraining(pl.Callback):
    """Observe the real Manager loop against independent reference updates."""

    def __init__(self, reference, config):
        self.reference, self.config = reference, config
        self.steps = 0

    def on_fit_start(self, trainer, pl_module):
        self.model = copy.deepcopy(pl_module.eeg.model)
        self.control = reference_trainer(self.reference, self.config, self.model)

    def check_state(self, module):
        for name, value in self.model.state_dict().items():
            max_difference(value, module.eeg.model.state_dict()[name])
        assert int(module.eeg.completed_steps) == self.steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.check_state(pl_module)
        self.rng = torch.random.get_rng_state()
        self.cuda_rng = (
            torch.cuda.get_rng_state() if pl_module.device.type == "cuda" else None
        )
        self.batch = batch
        assert pl_module.eeg.teacher.training

    def on_before_backward(self, trainer, pl_module, loss):
        state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if self.cuda_rng is not None else None
        try:
            torch.random.set_rng_state(self.rng)
            if self.cuda_rng is not None:
                torch.cuda.set_rng_state(self.cuda_rng)
            self.control.optimizer.zero_grad(set_to_none=True)
            self.expected_outputs, self.expected_result = reference_forward(
                self.reference, self.model, self.batch, self.control, self.config
            )
            self.expected_result[0].backward()
        finally:
            torch.random.set_rng_state(state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.grad is not None],
            self.config.clip_value,
        )
        compare_gradients(self.model, pl_module.eeg.model)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        compare_outputs(self.expected_outputs, self.expected_result, outputs)
        self.control.optimizer.step()
        self.control.optimizer_scheduler.step()
        self.model.update_target_encoder(self.control._momentum_at_step())
        self.control.global_step += 1
        self.steps += 1
        assert trainer.global_step == self.steps
        assert (
            trainer.optimizers[0].param_groups[0]["lr"]
            == (self.control.optimizer.param_groups[0]["lr"])
        )

    def on_train_end(self, trainer, pl_module):
        self.check_state(pl_module)


@pytest.mark.parametrize(
    "accelerator", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]
)
def test_manager_matches_reference_optimizer_and_teacher(
    reference, tmp_path, accelerator
):
    config = Config()
    observer = CompareTraining(reference, config)
    run(
        Path(os.environ["EEG_JEPA_REFERENCE"]),
        tmp_path / "manager",
        config,
        callbacks=[observer],
        accelerator=accelerator,
    )
    assert observer.steps == config.total_steps
    print(
        f"Manager: {observer.steps} steps; outputs/loss/clipped-gradients/"
        "online-weights/teacher-weights max_abs=0"
    )


def test_pin_rejects_modified_source_and_preserves_imports(reference, tmp_path):
    root = Path(os.environ["EEG_JEPA_REFERENCE"])
    shutil.copytree(root / "models", tmp_path / "models")
    (tmp_path / "models/eeg_jepa.py").write_text("# changed")
    with pytest.raises(ValueError, match="Reference differs"):
        load_reference(tmp_path)
    sentinel = object()
    previous = sys.modules.get("models")
    sys.modules["models"] = sentinel
    try:
        other = tmp_path / "complete"
        shutil.copytree(root, other)
        load_reference(other)
        assert sys.modules["models"] is sentinel
    finally:
        if previous is None:
            sys.modules.pop("models", None)
        else:
            sys.modules["models"] = previous


@pytest.mark.parametrize("custom_settings", [False, True])
@pytest.mark.parametrize("failure", [None, "trainer", "fit"])
def test_run_restores_global_settings(
    reference, tmp_path, monkeypatch, custom_settings, failure
):
    import stable_pretraining as spt
    from examples.eeg_jepa import train

    settings = spt.get_config()
    monkeypatch.setattr(
        settings,
        "_default_callbacks",
        {"env_dump": True, "registry": False} if custom_settings else {},
    )
    monkeypatch.setattr(
        settings,
        "_default_loggers",
        {"registry": False} if custom_settings else {},
    )
    previous = (
        settings.cache_dir,
        settings.default_callbacks,
        settings.default_loggers,
    )
    output = tmp_path / "restore"
    observed = []

    def check_active_settings():
        assert settings.cache_dir == str(output)
        assert settings.default_callbacks["env_dump"] is False
        assert settings.default_callbacks["hf_checkpoint"] is False
        assert settings.default_loggers == {"registry": True}
        observed.append(True)

    class InspectSettings(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            check_active_settings()
            if failure == "fit":
                raise RuntimeError("injected fit failure")

    def fail_trainer(**kwargs):
        check_active_settings()
        raise RuntimeError("injected trainer failure")

    if failure == "trainer":
        monkeypatch.setattr(train.pl, "Trainer", fail_trainer)

    def fit():
        return run(
            Path(os.environ["EEG_JEPA_REFERENCE"]),
            output,
            replace(Config(), total_steps=1),
            callbacks=[InspectSettings()],
        )

    if failure is None:
        fit()
    else:
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            fit()
    assert observed
    assert (
        settings.cache_dir,
        settings.default_callbacks,
        settings.default_loggers,
    ) == previous
