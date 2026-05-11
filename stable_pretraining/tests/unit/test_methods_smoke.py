"""Smoke tests for every method in ``stable_pretraining.methods``.

Each test instantiates the method with the smallest reasonable
configuration on CPU, runs a single forward + backward pass on dummy
input, and asserts:

* the loss is finite,
* the loss has a working ``.backward()``,
* at least one parameter received a non-zero gradient.

No optimisation, no real data, no GPU. Intended to catch wiring
regressions in the method classes (shape mismatches, missing kwargs,
broken loss imports, etc.).
"""

from typing import Tuple

import pytest
import torch

import stable_pretraining.methods as M

pytestmark = pytest.mark.unit


# --- helpers -------------------------------------------------------------

# A small ViT keeps CPU runtime under a few seconds per test. Most methods
# only need 2 examples to exercise the loss path.
TINY_VIT = "vit_tiny_patch16_224"  # 192-d, 12 layers, 5.5M params
B = 2
C, H, W = 3, 224, 224


def _two_views() -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(B, C, H, W, generator=g),
        torch.randn(B, C, H, W, generator=g),
    )


def _assert_loss_and_backward(model: torch.nn.Module, output) -> None:
    """Loss must be a finite scalar with a working ``.backward()``."""
    loss = output.loss
    assert loss is not None, "method returned None loss"
    assert loss.ndim == 0, f"loss must be scalar, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"loss must be finite, got {loss.item()}"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(
        g.abs().sum() > 0 for g in grads
    ), "all gradients are zero — backward path is broken"


# --- two-view methods (forward(view1, view2)) ----------------------------

TWO_VIEW_METHODS = [
    ("BarlowTwins", {"projector_dims": (256, 256, 256)}),
    ("BYOL", {"projector_dims": (512, 64), "predictor_dims": (512, 64)}),
    ("CMAE", {"projector_dim": 64}),
    ("MoCov2", {"projector_dims": (256, 64), "queue_length": 32}),
    ("MoCov3", {"projector_dims": (512, 512, 64), "predictor_hidden_dim": 512}),
    ("NNCLR", {"projector_dims": (256, 64), "queue_length": 32, "predictor_hidden_dim": 256}),
    ("PIRL", {"projector_dim": 32, "queue_length": 32, "jigsaw_grid": 4}),
    ("SimCLR", {"projector_dims": (256, 256, 64)}),
    ("SimSiam", {"projector_dim": 256, "predictor_hidden_dim": 64}),
    ("TiCO", {"projector_dims": (256, 64)}),
    ("VICReg", {"projector_dims": (256, 256, 256)}),
    ("VICRegL", {"projector_dim": 256}),
    ("WMSE", {"projector_dims": (256, 32), "eps": 1e-1}),
]


@pytest.mark.parametrize("method_name,kwargs", TWO_VIEW_METHODS)
def test_two_view_method_forward_backward(method_name: str, kwargs: dict) -> None:
    cls = getattr(M, method_name)
    model = cls(encoder_name=TINY_VIT, **kwargs)
    model.train()
    v1, v2 = _two_views()
    output = model(v1, v2)
    _assert_loss_and_backward(model, output)


# --- single-view masked methods (forward(images)) ------------------------

MASKED_METHODS = [
    ("BEiT", {"vocab_size": 256}),
    ("Data2Vec", {"top_k_blocks": 2, "mask_ratio": 0.5}),
    ("MaskFeat", {"mask_ratio": 0.5}),
    ("SimMIM", {"mask_ratio": 0.5}),
    ("iGPT", {}),
]


@pytest.mark.parametrize("method_name,kwargs", MASKED_METHODS)
def test_masked_method_forward_backward(method_name: str, kwargs: dict) -> None:
    cls = getattr(M, method_name)
    model = cls(encoder_name=TINY_VIT, **kwargs)
    model.train()
    v, _ = _two_views()
    output = model(v)
    _assert_loss_and_backward(model, output)


# --- multi-crop self-distillation methods --------------------------------

MULTICROP_METHODS = [
    ("DINO", {"n_prototypes": 256, "encoder_kwargs": {"dynamic_img_size": True}}),
    ("iBOT", {"n_cls_prototypes": 256, "n_patch_prototypes": 64, "mask_ratio": 0.3}),
    ("DINOv2", {"n_cls_prototypes": 256, "n_patch_prototypes": 64, "mask_ratio": 0.3}),
    ("DINOv3", {"n_cls_prototypes": 256, "n_patch_prototypes": 64, "mask_ratio": 0.3, "n_register_tokens": 2}),
]


@pytest.mark.parametrize("method_name,kwargs", MULTICROP_METHODS)
def test_multicrop_method_forward_backward(method_name: str, kwargs: dict) -> None:
    cls = getattr(M, method_name)
    model = cls(encoder_name=TINY_VIT, **kwargs)
    model.train()
    v1, v2 = _two_views()
    output = model(global_views=[v1, v2])
    _assert_loss_and_backward(model, output)


# --- methods needing a custom call signature -----------------------------


def test_msn_forward_backward():
    model = M.MSN(
        encoder_name=TINY_VIT,
        n_prototypes=256,
        mask_ratio=0.5,
    )
    model.train()
    v1, v2 = _two_views()
    output = model(v1, v2)
    _assert_loss_and_backward(model, output)


def test_swav_forward_backward_two_view():
    """SwAV's compatibility 2-view forward (no multi-crop)."""
    model = M.SwAV(
        encoder_name=TINY_VIT,
        projector_dims=(256, 64),
        n_prototypes=64,
        dynamic_img_size=True,
    )
    model.train()
    v1, v2 = _two_views()
    output = model(v1, v2)
    _assert_loss_and_backward(model, output)


def test_swav_forward_backward_multicrop():
    """SwAV's full multi-crop path with 2 global + 2 local views."""
    model = M.SwAV(
        encoder_name=TINY_VIT,
        projector_dims=(256, 64),
        n_prototypes=64,
        dynamic_img_size=True,
    )
    model.train()
    v1, v2 = _two_views()
    g = torch.Generator().manual_seed(1)
    local1 = torch.randn(B, C, 96, 96, generator=g)
    local2 = torch.randn(B, C, 96, 96, generator=g)
    output = model(global_views=[v1, v2], local_views=[local1, local2])
    _assert_loss_and_backward(model, output)


def test_ijepa_forward_backward():
    model = M.IJEPA(
        model_or_model_name=TINY_VIT,
        predictor_embed_dim=64,
        predictor_depth=2,
        num_targets=2,
    )
    model.train()
    v, _ = _two_views()
    output = model(v)
    _assert_loss_and_backward(model, output)


def test_mae_forward_backward():
    model = M.MAE(
        model_or_model_name=TINY_VIT,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_num_heads=2,
        mask_ratio=0.5,
    )
    model.train()
    v, _ = _two_views()
    output = model(v)
    _assert_loss_and_backward(model, output)


def test_lejepa_forward_backward():
    model = M.LeJEPA(
        encoder_name=TINY_VIT,
        n_slices=64,
        n_points=5,
    )
    model.train()
    v1, v2 = _two_views()
    g = torch.Generator().manual_seed(2)
    local1 = torch.randn(B, C, 96, 96, generator=g)
    output = model.forward(global_views=[v1, v2], local_views=[local1])
    _assert_loss_and_backward(model, output)


def test_mim_refiner_forward_backward():
    """MIM-Refiner takes a pre-built encoder; just pass a fresh timm ViT."""
    import timm

    encoder = timm.create_model(TINY_VIT, num_classes=0, pretrained=False)
    model = M.MIMRefiner(
        pretrained_encoder=encoder,
        n_cls_prototypes=128,
        n_patch_prototypes=32,
        mask_ratio=0.3,
    )
    model.train()
    v1, v2 = _two_views()
    output = model(global_views=[v1, v2])
    _assert_loss_and_backward(model, output)


# --- eval-mode passthrough ----------------------------------------------


@pytest.mark.parametrize(
    "method_name,kwargs",
    [
        ("SimCLR", {"projector_dims": (256, 256, 64)}),
        ("BYOL", {"projector_dims": (256, 64), "predictor_dims": (256, 64)}),
        ("VICReg", {"projector_dims": (256, 256, 256)}),
    ],
)
def test_eval_mode_no_loss(method_name: str, kwargs: dict) -> None:
    """In eval mode the methods must return a finite (zero) loss
    and a usable embedding tensor."""
    cls = getattr(M, method_name)
    model = cls(encoder_name=TINY_VIT, **kwargs)
    model.eval()
    v, _ = _two_views()
    with torch.no_grad():
        output = model(v)
    assert torch.isfinite(output.loss).all()
    assert output.embedding is not None and output.embedding.shape[0] == B
