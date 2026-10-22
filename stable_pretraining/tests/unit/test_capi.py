"""Download-free CAPI numerical, geometry, gradient and API contracts.

Fixtures execute unmodified author functions from facebookresearch/capi commit
98b4fa17ee8eec8810c17022df9a27a44845368b, independently of this implementation.
Numerical fixture: seed 17, float32, B=4/P=3/K=5, temperatures 0.12/0.06,
three Sinkhorn iterations; author P/B/K outputs are stored as B/P/K.
Architecture fixture: seed 29, EncoderDecoder width 16, one encoder/predictor
block, four heads/registers, patch size eight, no stochastic depth. It records
full teacher patches and sampled predictions at explicit flattened coordinates.
"""

import json
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate

import stable_pretraining as spt
from stable_pretraining.backbone.capi import capi_mask
from stable_pretraining.losses.capi import CAPILoss, positionwise_sinkhorn

pytestmark = pytest.mark.unit


def tiny(**kwargs):
    return spt.CAPI(
        embed_dim=32,
        depth=2,
        num_heads=4,
        predictor_depth=2,
        patch_size=8,
        n_registers=4,
        num_clusters=8,
        drop_path_rate=0,
        total_steps=8,
        warmup_steps=2,
        **kwargs,
    )


def test_author_numerical_fixture():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/capi/numerical.json").read_text()
    )
    logits = torch.tensor(fixture["logits"], requires_grad=True)
    predictions = torch.tensor(fixture["predictions"], requires_grad=True)
    terms = CAPILoss()(predictions, logits, torch.tensor(fixture["indices"]))
    torch.testing.assert_close(
        terms["assignments"], torch.tensor(fixture["assignments"]), atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        terms["prediction_loss"], torch.tensor(fixture["prediction_loss"])
    )
    torch.testing.assert_close(
        terms["clustering_loss"], torch.tensor(fixture["clustering_loss"])
    )
    terms["loss"].backward()
    torch.testing.assert_close(
        logits.grad, torch.tensor(fixture["clustering_gradient"])
    )
    torch.testing.assert_close(
        predictions.grad, torch.tensor(fixture["prediction_gradient"])
    )


def test_balancing_is_positionwise_and_input_unchanged():
    logits = torch.randn(64, 3, 5) * 0.1
    logits[:, 0, 0] += 8
    before = logits.clone()
    assignments = positionwise_sinkhorn(logits)
    torch.testing.assert_close(logits, before)
    torch.testing.assert_close(assignments.sum(-1), torch.ones(64, 3))
    torch.testing.assert_close(
        assignments.mean(0), torch.full((3, 5), 0.2), atol=0.005, rtol=0
    )
    extreme = positionwise_sinkhorn(torch.tensor([[[1e20, -1e20]], [[-1e20, 1e20]]]))
    assert torch.isfinite(extreme).all()
    assert not assignments.requires_grad


def test_geometry_and_subsampling():
    torch.manual_seed(3)
    visible, predicted = capi_mask(3, (4, 4), torch.device("cpu"), 0.65, 0.5)
    assert visible.shape == (18,)
    assert predicted.shape == (15,)
    assert not torch.isin(visible, predicted).any()
    model = tiny()
    images = torch.randn(3, 3, 32, 32)
    tokens, coords, full = model.backbone.prepare_tokens_and_drop(images, visible)
    assert tokens.shape == (3, 10, 32)
    torch.testing.assert_close(
        coords[:, 4:], full.flatten(0, 1)[visible].reshape(3, 6, 2)
    )
    assert model.backbone.decoder.blocks[0].residual1.fn.k_proj.in_features == 32
    assert model.backbone.decoder.n_blocks == 2


def test_gradient_isolation_and_dict_contract():
    model = tiny().train()
    batch = {"image": torch.randn(3, 3, 32, 32), "label": torch.arange(3)}
    output = model(batch, "train")
    assert output["patch_features"].shape == (3, 16, 32)
    torch.testing.assert_close(output["embedding"], output["patch_features"].mean(1))
    torch.testing.assert_close(output["label"], batch["label"])
    output["prediction_loss"].backward(retain_graph=True)
    assert all(p.grad is None for p in model.clustering.parameters())
    assert any(p.grad is not None for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())
    model.zero_grad(set_to_none=True)
    output["clustering_loss"].backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.student_head.parameters())
    assert all(p.grad is not None for p in model.clustering.parameters())
    model.zero_grad(set_to_none=True)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    assert not model.teacher.training
    for stage in ("val", "test"):
        result = model(batch, stage)
        assert "loss" not in result
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, state[k])
    model.eval()
    assert "loss" not in model(batch, "fit")


@pytest.mark.parametrize("grid", [(2, 16), (16, 2), (1, 16), (16, 1)])
@pytest.mark.parametrize("mask_ratio", [0.25, 0.65, 0.95])
def test_mask_on_narrow_grids(grid, mask_ratio):
    torch.manual_seed(3)
    positions = grid[0] * grid[1]
    masked = int(positions * mask_ratio)
    visible, predicted = capi_mask(
        3, grid, torch.device("cpu"), mask_ratio, prediction_subsampling=1
    )
    assert visible.shape == (3 * (positions - masked),)
    assert predicted.shape == (3 * masked,)
    for batch_index in range(3):
        assert (visible // positions == batch_index).sum() == positions - masked
        assert (predicted // positions == batch_index).sum() == masked
    torch.testing.assert_close(
        torch.cat((visible, predicted)).sort().values, torch.arange(3 * positions)
    )


def test_example_dataset_config(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from PIL import Image

    image_dir = tmp_path / "class0"
    image_dir.mkdir()
    Image.new("RGB", (32, 48)).save(image_dir / "sample.png")
    monkeypatch.setenv("CAPI_TRAIN_DIRECTORY", str(tmp_path))
    config = OmegaConf.load(
        Path(__file__).resolve().parents[3] / "examples/capi_config.yaml"
    )
    dataset = instantiate(config.data.train.dataset)
    sample = dataset[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32
    assert torch.isfinite(sample["image"]).all()
    assert sample["label"] == 0


def test_labels_do_not_change_loss():
    model = tiny().train()
    images = torch.randn(3, 3, 32, 32)
    torch.manual_seed(5)
    first = model({"image": images, "label": torch.zeros(3)}, "train")
    torch.manual_seed(5)
    second = model({"image": images, "label": torch.ones(3)}, "train")
    torch.testing.assert_close(first["loss"], second["loss"])


def test_exports_and_hydra():
    assert spt.CAPI is spt.methods.CAPI
    assert callable(spt.forward.capi)
    config = dict(
        _target_="stable_pretraining.CAPI",
        embed_dim=32,
        depth=1,
        predictor_depth=1,
        num_heads=4,
        patch_size=8,
        n_registers=4,
        num_clusters=8,
        total_steps=8,
        warmup_steps=2,
    )
    assert isinstance(instantiate(config), spt.Module)


def test_optimizer_membership_and_reference_multipliers():
    model = tiny()
    optimizers, schedulers = model.configure_optimizers()
    student = {id(p) for g in optimizers[0].param_groups for p in g["params"]}
    cluster = {id(p) for g in optimizers[1].param_groups for p in g["params"]}
    expected = {
        id(p)
        for m in (model.backbone, model.student_head)
        for p in m.parameters()
        if p.requires_grad
    }
    assert student == expected
    assert cluster == {id(p) for p in model.clustering.parameters()}
    assert not student & cluster
    assert not student & {id(p) for p in model.teacher.parameters()}
    assert len(schedulers) == 2
    assert sorted(
        {g["weight_decay"] for g in optimizers[0].param_groups}
    ) == pytest.approx([0, 0.01, 0.1])
    initial_lrs = {g["initial_lr"] for g in optimizers[0].param_groups}
    assert initial_lrs == {0, 0.0002, 0.001}


def test_author_architecture_fixture():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/capi/architecture.json").read_text()
    )
    torch.manual_seed(fixture["seed"])
    model = spt.backbone.capi_backbone(
        embed_dim=16,
        depth=1,
        num_heads=4,
        predictor_depth=1,
        patch_size=8,
        n_registers=4,
        drop_path_rate=0,
    ).eval()
    images = torch.tensor(fixture["images"])
    patches, _ = model.forward_pretrain(images)
    _, predictions = model.forward_pretrain(
        images,
        torch.tensor(fixture["visible"]),
        torch.tensor(fixture["predicted"]),
        do_prediction=True,
    )
    torch.testing.assert_close(patches, torch.tensor(fixture["patches"]))
    torch.testing.assert_close(predictions, torch.tensor(fixture["predictions"]))
    pooled, registers, dense = model(images)
    torch.testing.assert_close(pooled, patches.mean(1))
    assert registers.shape == (2, 4, 16)
    assert dense.shape == (2, 2, 2, 16)


def test_ema_resume_schedule_continuity():
    model = tiny(ema_momentum=0.5)
    model.log = lambda *args, **kwargs: None
    callback = spt.CAPIEMA()
    with torch.no_grad():
        for p in model.backbone.parameters():
            p.add_(0.1)
    callback._update(model)
    callback._update(model)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    resumed = tiny(ema_momentum=0.5)
    resumed.load_state_dict(state)
    resumed.log = lambda *args, **kwargs: None
    callback._update(model)
    callback._update(resumed)
    assert int(resumed.capi_updates) == 3
    for a, b in zip(
        model.teacher.parameters(), resumed.teacher.parameters(), strict=True
    ):
        torch.testing.assert_close(a, b)


def test_bf16_stochastic_depth():
    model = spt.CAPI(
        embed_dim=32,
        depth=2,
        num_heads=4,
        predictor_depth=2,
        patch_size=8,
        n_registers=4,
        num_clusters=8,
        drop_path_rate=0.2,
        total_steps=8,
        warmup_steps=1,
    ).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model({"image": torch.randn(4, 3, 32, 32)}, "train")
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
