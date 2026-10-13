"""Offline DAE-Net parity and behavior tests against the frozen upstream network."""

import io
from types import SimpleNamespace
from lightning.pytorch.trainer.states import TrainerFn
from unittest.mock import Mock


import pytest


import torch


import stable_pretraining as spt


from stable_pretraining.tests.unit import _dae_net_reference as reference


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_torch():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(previous_threads)


def make_model(**kwargs):
    config = dict(branch_num=3, z_dim=4, ef_dim=2, df_dim=6, gf_dim=8)
    return spt.backbone.DAENetBackbone(**(config | kwargs))


def make_reference(**kwargs):
    config = dict(branch_num=3, z_dim=4, ef_dim=2, df_dim=6, gf_dim=8)
    model = reference.dae_network(**(config | kwargs))
    for branch in range(model.branch_num):
        model.encoder.init(branch)
        model.generator.init(branch)
    return model


def make_batch(batch_size=2):
    return {
        "voxels": torch.randint(0, 2, (batch_size, 1, 64, 64, 64)).to(torch.uint8),
        "points": torch.rand(batch_size, 11, 3) - 0.5,
    }


def reference_loss(summed, maximum, existence, affine, masked_offsets, target):
    # These expressions are transcribed from the pinned upstream training loop,
    # independent of the production loss and its output reshaping.
    terms = {
        "reconstruction_sum": torch.mean(
            torch.where(target > 0.5, 1 - summed, torch.clamp(summed, min=0.1)) ** 2
        ),
        "reconstruction_max": torch.mean(
            torch.where(target > 0.5, 1 - maximum, torch.clamp(maximum, min=0.1)) ** 2
        )
        * 0.1,
        "sparsity": torch.mean(-((torch.mean(existence, 0) - 1) ** 2)) * (0.02 * 0.37),
        "affine": torch.mean(affine**2) * (0.03 * 0.37),
        "deformation": torch.mean(masked_offsets**2) * (100.0 * 0.37),
    }
    return {"loss": sum(terms.values()), **terms}


@pytest.mark.parametrize("branches,z_dim", [(1, 4), (3, 2), (8, 4), (16, 4)])
def test_initialization_matches_upstream(branches, z_dim):
    torch.manual_seed(19)
    expected = make_reference(branch_num=branches, z_dim=z_dim)
    torch.manual_seed(19)
    actual = make_model(branch_num=branches, z_dim=z_dim)
    assert expected.state_dict().keys() == actual.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(
            actual.state_dict()[name], tensor, rtol=0, atol=0, msg=name
        )


@pytest.mark.parametrize("affine_only", [False, True])
def test_forward_matches_upstream(affine_only):
    model, official = make_model(), make_reference()
    official.load_state_dict(model.state_dict(), strict=True)
    batch = make_batch()
    with torch.no_grad():
        encoded = model.encode(batch["voxels"])
        existence, codes = official.encoder(batch["voxels"])
        torch.testing.assert_close(
            encoded["existence"], existence, atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(encoded["codes"], codes, atol=1e-6, rtol=1e-5)
        output = model.decode(batch["points"], encoded, affine_only=affine_only)
        summed, maximum, affine, masked = official.generator(
            batch["points"], codes, existence, out_sum=True, affine_only=affine_only
        )
        raw = official.generator(
            batch["points"],
            codes,
            None,
            out_sum=True,
            out_branch=True,
            affine_only=affine_only,
        )
        for key, expected in (
            ("occupancy_sum", summed),
            ("occupancy_max", maximum),
            ("affine", affine),
            ("raw_occupancy", raw),
            ("part_occupancy", raw * existence.transpose(1, 2)),
        ):
            torch.testing.assert_close(
                output[key], expected, atol=1e-6, rtol=1e-5, msg=key
            )
        weighted = output["deformation"] * output["raw_occupancy"].transpose(
            1, 2
        ).unsqueeze(2)
        torch.testing.assert_close(weighted, masked, atol=1e-6, rtol=1e-5)
        templates = official.generator(batch["points"], None, None, out_branch=True)
        torch.testing.assert_close(
            model.templates(batch["points"]), templates, atol=1e-6, rtol=1e-5
        )


def test_loss_gradients_and_adam_match_upstream():
    model, official = make_model(), make_reference()
    with torch.no_grad():
        # The default sigmoid(-8) existence can hide errors behind the .1 floor.
        model.encoder.linear_1.bias.copy_(torch.tensor([8.0, 8.5, 9.0]))
    official.load_state_dict(model.state_dict(), strict=True)
    batch = make_batch()
    target = torch.randint(0, 2, (2, 11, 1)).float()
    output = model(batch)
    actual = spt.losses.DAENetLoss(sparse_weight=0.02, affine_weight=0.03)(
        output, target, 0.37
    )
    existence, codes = official.encoder(batch["voxels"])
    summed, maximum, affine, offsets = official.generator(
        batch["points"], codes, existence, out_sum=True
    )
    expected = reference_loss(summed, maximum, existence, affine, offsets, target)
    for name in actual:
        torch.testing.assert_close(
            actual[name], expected[name], atol=1e-6, rtol=1e-5, msg=name
        )
    actual["loss"].backward()
    expected["loss"].backward()
    actual_parameters = dict(model.named_parameters())
    for name, parameter in official.named_parameters():
        gradient = actual_parameters[name].grad
        assert gradient is not None and parameter.grad is not None, name
        assert torch.isfinite(gradient).all(), name
        torch.testing.assert_close(
            gradient, parameter.grad, atol=1e-6, rtol=1e-5, msg=name
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    reference_optimizer = torch.optim.Adam(official.parameters(), lr=0.0002)
    optimizer.step()
    reference_optimizer.step()
    for name, parameter in official.named_parameters():
        actual_parameter = actual_parameters[name]
        torch.testing.assert_close(
            actual_parameter, parameter, atol=1e-6, rtol=1e-5, msg=name
        )
        for key, value in reference_optimizer.state[parameter].items():
            torch.testing.assert_close(
                optimizer.state[actual_parameter][key],
                value,
                atol=1e-6,
                rtol=1e-5,
                msg=f"{name}/{key}",
            )


@pytest.mark.parametrize("branch", [0, 1, 2])
def test_reset_matches_upstream_and_preserves_other_parameters(branch):
    model, official = make_model(), make_reference()
    official.load_state_dict(model.state_dict())
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    torch.manual_seed(44)
    official.encoder.init(branch)
    official.generator.init(branch)
    torch.manual_seed(44)
    model.reset_branch(branch)
    changed = []
    for name, parameter in model.named_parameters():
        assert id(parameter) == identities[name]
        torch.testing.assert_close(
            parameter, official.state_dict()[name], atol=0, rtol=0, msg=name
        )
        if name.startswith("encoder.conv_"):
            torch.testing.assert_close(
                parameter, before[name], atol=0, rtol=0, msg=name
            )
        else:
            width = parameter.shape[0] // model.branch_num
            for other in range(model.branch_num):
                if other != branch:
                    section = slice(other * width, (other + 1) * width)
                    torch.testing.assert_close(
                        parameter[section],
                        before[name][section],
                        atol=0,
                        rtol=0,
                        msg=name,
                    )
        changed.append(not torch.equal(parameter, before[name]))
    assert any(changed)


def test_serialization_and_chunked_queries():
    model, restored = make_model().eval(), make_model().eval()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    batch = make_batch()
    with torch.no_grad():
        expected, actual = model(batch), restored(batch)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        encoded = restored.encode(batch["voxels"])
        parts = [
            restored.decode(points, encoded)["part_occupancy"]
            for points in batch["points"].split(4, dim=1)
        ]
        torch.testing.assert_close(
            torch.cat(parts, dim=1), actual["part_occupancy"], atol=1e-6, rtol=1e-5
        )


def loss_input():
    return {
        "occupancy_sum": torch.zeros(2, 3, 1, dtype=torch.float64, requires_grad=True),
        "occupancy_max": torch.zeros(2, 3, 1, dtype=torch.float64, requires_grad=True),
        "raw_occupancy": torch.full(
            (2, 3, 2), 0.25, dtype=torch.float64, requires_grad=True
        ),
        "existence": torch.tensor(
            [[[0.2], [0.4]], [[0.8], [0.4]]], dtype=torch.float64, requires_grad=True
        ),
        "affine": torch.full((2, 2, 12), 0.5, dtype=torch.float64, requires_grad=True),
        "deformation": torch.full(
            (2, 2, 3, 3), 2.0, dtype=torch.float64, requires_grad=True
        ),
    }


def test_loss_values_and_detached_occupancy():
    inputs = loss_input()
    terms = spt.losses.DAENetLoss(sparse_weight=0.02, affine_weight=0.03)(
        inputs, torch.zeros(2, 3, 1), 0.37
    )
    expected = {
        "reconstruction_sum": 0.01,
        "reconstruction_max": 0.001,
        "sparsity": -0.305 * 0.02 * 0.37,
        "affine": 0.25 * 0.03 * 0.37,
        "deformation": 0.25 * 100 * 0.37,
    }
    for key, value in expected.items():
        assert terms[key].item() == pytest.approx(value, abs=1e-12)
    assert terms["loss"].item() == pytest.approx(sum(expected.values()), abs=1e-12)
    terms["loss"].backward()
    assert inputs["raw_occupancy"].grad is None
    torch.testing.assert_close(
        inputs["deformation"].grad,
        torch.full_like(inputs["deformation"], 2 * 2 * 0.25**2 / 36 * 100 * 0.37),
    )
    torch.testing.assert_close(
        inputs["existence"].grad,
        torch.tensor([[[0.25], [0.3]], [[0.25], [0.3]]], dtype=torch.float64)
        * 0.02
        * 0.37,
    )


def test_reconstruction_floor_threshold_and_gradient():
    inputs = loss_input()
    predictions = torch.tensor(
        [0.05, 0.2, 0.7, 1.2, -0.2, 0.3], dtype=torch.float64
    ).reshape(2, 3, 1)
    inputs["occupancy_sum"] = predictions.clone().requires_grad_()
    inputs["occupancy_max"] = predictions.clone().requires_grad_()
    targets = torch.tensor([0.0, 0.5, 1.0, 1.0, 0.0, 0.51]).reshape(2, 3, 1)
    terms = spt.losses.DAENetLoss(sparse_weight=0, deform_weight=0)(inputs, targets)
    assert terms["reconstruction_sum"].item() == pytest.approx(
        (0.01 + 0.04 + 0.09 + 0.04 + 0.01 + 0.49) / 6
    )
    terms["loss"].backward()
    gradient = (
        torch.tensor([0, 0.4, -0.6, 0.4, 0, -1.4], dtype=torch.float64).reshape(2, 3, 1)
        / 6
    )
    torch.testing.assert_close(inputs["occupancy_sum"].grad, gradient)
    torch.testing.assert_close(inputs["occupancy_max"].grad, gradient * 0.1)


def test_disabled_regularizers_do_not_read_nan_values():
    inputs = loss_input()
    for key in ("existence", "affine", "deformation", "raw_occupancy"):
        inputs[key] = torch.full_like(inputs[key], float("nan"))
    terms = spt.losses.DAENetLoss(sparse_weight=0, deform_weight=0, affine_weight=0)(
        inputs, torch.zeros(2, 3, 1)
    )
    assert terms["loss"].item() == pytest.approx(0.011)
    assert all(terms[key].item() == 0 for key in ("sparsity", "deformation", "affine"))


def test_zero_warmup_and_negative_total_loss():
    inputs = loss_input()
    loss = spt.losses.DAENetLoss(sparse_weight=1, deform_weight=0)
    target = torch.zeros(2, 3, 1)
    assert loss(inputs, target, 0)["loss"].item() == pytest.approx(0.011)
    assert loss(inputs, target)["loss"].item() == pytest.approx(0.011 - 0.305)


@pytest.fixture
def forward_case(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            backbone = spt.backbone.DAENetBackbone(
                branch_num=3, z_dim=4, ef_dim=2, df_dim=6, gf_dim=8
            )
            with torch.no_grad():
                # Nontrivial existence avoids masking errors behind the loss floor.
                backbone.encoder.linear_1.bias.copy_(torch.tensor([8.0, 8.5, 9.0]))
            module = spt.Module(
                forward=spt.forward.dae_net_forward,
                backbone=backbone,
                dae_net_loss=spt.losses.DAENetLoss(
                    sparse_weight=0.02, affine_weight=0.03
                ),
                regularization_weight=0.37,
                optim=None,
            )
            # These unit checks isolate logging arguments from Trainer lifecycle.
            monkeypatch.setattr(module, "log", Mock())
            batch = {
                "voxels": torch.randint(0, 2, (2, 1, 64, 64, 64)).float(),
                "points": torch.rand(2, 11, 3) - 0.5,
                "occupancy": (torch.arange(22).reshape(2, 11, 1) % 2).float(),
                "shape_id": ["shape-a", "shape-b"],
            }
            yield module, batch
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("stage", ["fit", "validate", "test"])
def test_forward_matches_direct_components_and_logging(forward_case, stage):
    module, batch = forward_case
    module.train(stage == "fit")
    with torch.set_grad_enabled(stage == "fit"):
        network_output = module.backbone(batch)
        expected = module.dae_net_loss(network_output, batch["occupancy"], 0.37)
        result = module(batch, stage=stage)

    diagnostics = {"occupancy_sum", "occupancy_max", "existence", "codes"}
    expected_keys = (
        diagnostics | {"loss"} | {f"loss_{name}" for name in expected if name != "loss"}
    )
    assert set(result) == expected_keys
    assert result["loss"].ndim == 0
    assert torch.isfinite(result["loss"])
    assert result["loss"].requires_grad == (stage == "fit")
    for name, value in expected.items():
        key = "loss" if name == "loss" else f"loss_{name}"
        assert result[key].ndim == 0
        torch.testing.assert_close(result[key], value, rtol=0, atol=0)
    for key in diagnostics:
        torch.testing.assert_close(result[key], network_output[key], rtol=0, atol=0)
    for key, value in result.items():
        if key != "loss":
            assert not value.requires_grad
            assert value.grad_fn is None

    module.log.assert_called_once()
    args, kwargs = module.log.call_args
    assert args[0] == f"{stage}/loss"
    assert not args[1].requires_grad
    torch.testing.assert_close(args[1], result["loss"].detach(), rtol=0, atol=0)
    assert kwargs == {"on_step": stage == "fit", "on_epoch": True, "batch_size": 2}


@pytest.mark.parametrize("with_targets", [False, True])
def test_prediction_needs_no_loss_or_regularization_weight(
    forward_case, monkeypatch, with_targets
):
    module, batch = forward_case
    if not with_targets:
        batch.pop("occupancy")
    loss_forward = Mock(side_effect=AssertionError("prediction must not compute loss"))
    monkeypatch.setattr(module.dae_net_loss, "forward", loss_forward)
    monkeypatch.delattr(module, "regularization_weight")
    module.eval()
    with torch.no_grad():
        expected = module.backbone(batch)
        result = module(batch, stage="predict")
    assert "loss" not in result
    assert result.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)
    loss_forward.assert_not_called()
    module.log.assert_not_called()


def test_ddp_epoch_logging_weights_full_batch_and_tail(forward_case, monkeypatch):
    module, batch = forward_case
    module.train()
    module._trainer = Mock(world_size=2)
    logged = []
    for global_size, report in ((3, 2.0), (2, 8.0)):
        losses = {
            "loss": torch.tensor(report),
            "loss_report": torch.tensor(report),
            "global_batch_size": torch.tensor(global_size),
        }
        monkeypatch.setattr(module.dae_net_loss, "forward", Mock(return_value=losses))
        result = module(batch, stage="fit")
        assert "loss_global_batch_size" not in result
        args, kwargs = module.log.call_args
        logged.append((args[1].item(), kwargs["batch_size"]))
    epoch_loss = sum(value * size for value, size in logged) / sum(
        size for _, size in logged
    )
    assert epoch_loss == pytest.approx((3 * 2.0 + 2 * 8.0) / 5)


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    with torch.random.fork_rng(devices=[]):
        yield spt.callbacks.DAENetRevival(era_length=2)


@pytest.mark.parametrize("completed", [None, 0, 1])
def test_checkpoint_load_requires_controller_epoch_end(controller, completed):
    trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.FITTING))
    checkpoint = {
        "epoch": 1,
        "callbacks": {controller.state_key: {"completed_epoch": completed}},
    }
    if completed == 1:
        controller.on_load_checkpoint(trainer, None, checkpoint)
    else:
        with pytest.raises(ValueError, match="completed-epoch"):
            controller.on_load_checkpoint(trainer, None, checkpoint)


@pytest.mark.parametrize("processed", [1, 2])
def test_legacy_checkpoint_checks_lightning_epoch_progress(controller, processed):
    trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.FITTING))
    checkpoint = {
        "epoch": 1,
        "callbacks": {controller.state_key: {}},
        "loops": {"fit_loop": {"epoch_progress": {"total": {"processed": processed}}}},
    }
    if processed == 2:
        controller.on_load_checkpoint(trainer, None, checkpoint)
    else:
        with pytest.raises(ValueError, match="completed-epoch"):
            controller.on_load_checkpoint(trainer, None, checkpoint)


@pytest.mark.parametrize(
    "stage", [TrainerFn.VALIDATING, TrainerFn.TESTING, TrainerFn.PREDICTING]
)
def test_evaluation_does_not_require_a_resumable_checkpoint(controller, stage):
    trainer = SimpleNamespace(state=SimpleNamespace(fn=stage))
    checkpoint = {
        "epoch": 1,
        "callbacks": {controller.state_key: {"completed_epoch": None}},
    }
    controller.on_load_checkpoint(trainer, None, checkpoint)


@pytest.mark.parametrize("completed", [None, 1])
def test_benchmark_checks_controller_finalization(monkeypatch, completed):
    import importlib.util
    import sys
    from pathlib import Path

    folder = Path(__file__).resolve().parents[3] / "benchmarks/dae_net"

    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"dae_review_{name}", folder / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.setitem(sys.modules, "protocol", load("protocol"))
    settings = dict(epochs=4, era_length=2, updates_per_epoch=2, accelerator="cpu")
    checkpoint = dict(
        epoch=1,
        global_step=4,
        hyper_parameters={"run_settings": settings},
        optimizer_states=[{}],
        lr_schedulers=[{}],
        callbacks={
            "DAENetRevival": dict(
                completed_epoch=completed,
                torch_rng_state=torch.get_rng_state(),
                accepted_weights={"weight": torch.ones(1)},
            )
        },
    )
    validate = load("train").validate_resume
    if completed is None:
        with pytest.raises(ValueError, match="completed-epoch"):
            validate(checkpoint, settings)
    else:
        assert validate(checkpoint, settings) == 2
