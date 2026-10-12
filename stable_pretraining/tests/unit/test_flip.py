"""FLIP's masking, embedding, and contrastive-learning contracts."""

import math
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from torch import nn

import stable_pretraining as spt

pytestmark = pytest.mark.unit


class TinyTextEncoder(nn.Module):
    """Small trainable encoder with an explicit padding-mask contract."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)

    def forward(self, input_ids, attention_mask=None):
        features = self.embedding(input_ids)
        if attention_mask is None:
            return features.mean(1)
        weights = attention_mask.unsqueeze(-1)
        return (features * weights).sum(1) / weights.sum(1)


def tiny_flip(mask_ratio=0.5, **kwargs):
    return spt.FLIP(
        text_encoder=TinyTextEncoder(),
        text_dim=16,
        encoder_name=VisionTransformer(
            img_size=32,
            patch_size=8,
            embed_dim=24,
            depth=2,
            num_heads=3,
            num_classes=0,
        ),
        projection_dim=8,
        mask_ratio=mask_ratio,
        **kwargs,
    )


def test_named_encoder_records_architecture_for_checkpoint_reload():
    model = spt.FLIP(
        TinyTextEncoder(),
        text_dim=16,
        encoder_name="vit_tiny_patch16_224",
        projection_dim=8,
    )
    assert model.hparams.encoder_name == "vit_tiny_patch16_224"


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(7)
    return {
        "image": torch.randn(4, 3, 32, 32, generator=generator),
        "input_ids": torch.randint(0, 32, (4, 6), generator=generator),
    }


@pytest.mark.parametrize("ratio,visible", [(0.0, 16), (0.5, 8), (0.75, 4), (0.99, 1)])
def test_masking_removes_tokens_before_attention_and_preserves_positions(
    batch, ratio, visible
):
    model = tiny_flip(mask_ratio=ratio).train()
    captured = {}

    def capture_mask(module, inputs, output):
        captured["positioned"] = inputs[0]
        captured["ids"] = output.ids_keep

    def capture_attention(module, inputs):
        captured["attention"] = inputs[0]

    mask_hook = model.image_encoder.masking.register_forward_hook(capture_mask)
    block_hook = model.image_encoder.vit.blocks[0].register_forward_pre_hook(
        capture_attention
    )
    output = model(batch)
    mask_hook.remove()
    block_hook.remove()

    assert captured["attention"].shape == (4, 1 + visible, 24)
    expected = captured["positioned"].gather(
        1, captured["ids"].unsqueeze(-1).expand(-1, -1, 24)
    )
    torch.testing.assert_close(captured["attention"][:, 1:], expected)
    assert all(len(row.unique()) == visible for row in captured["ids"])
    assert output["embedding"].shape == (4, 24)
    assert output["image_embedding"].shape == output["text_embedding"].shape == (4, 8)


def test_loss_matches_unmasked_reference_and_updates_every_component(batch):
    torch.manual_seed(5)
    model = tiny_flip(mask_ratio=0).train()
    output = model(batch)
    image_features = model.image_encoder.vit.forward_features(batch["image"]).mean(1)
    image = F.normalize(model.image_projector(image_features), dim=-1)
    text = F.normalize(
        model.text_projector(model.text_encoder(batch["input_ids"])), dim=-1
    )
    logits = image @ text.T / 0.07
    targets = torch.arange(4)
    expected = (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
    ) / 2
    torch.testing.assert_close(output["embedding"], image_features)
    torch.testing.assert_close(output["loss"], expected)
    for key in ("image_embedding", "text_embedding"):
        torch.testing.assert_close(output[key].norm(dim=-1), torch.ones(4))

    output["loss"].backward()
    for component in (
        model.image_encoder,
        model.text_encoder,
        model.image_projector,
        model.text_projector,
    ):
        grads = [p.grad for p in component.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert model.logit_scale.grad is not None
    assert model.logit_scale.grad.abs() > 0


@pytest.mark.parametrize("scale", [-10.0, math.log(1000)])
def test_temperature_bounds_match_reference(batch, scale):
    model = tiny_flip().eval()
    with torch.no_grad():
        model.logit_scale.fill_(scale)
    out = model(batch, stage="validate")
    logits = out["image_embedding"] @ out["text_embedding"].T
    logits *= math.exp(min(max(scale, 0), math.log(100)))
    labels = torch.arange(4)
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    torch.testing.assert_close(out["loss"], expected)


def test_evaluation_encodes_both_or_either_modality_without_masking(batch):
    model = tiny_flip(mask_ratio=0.75).eval()
    lengths = []
    hook = model.image_encoder.vit.blocks[0].register_forward_pre_hook(
        lambda module, args: lengths.append(args[0].shape[1])
    )
    with torch.no_grad():
        paired = model(batch, stage="validate")
        image = model({"image": batch["image"]}, stage="predict")
        text = model({"input_ids": batch["input_ids"]}, stage="test")
        torch.testing.assert_close(
            model.encode_image(batch["image"]), image["image_embedding"]
        )
        torch.testing.assert_close(
            model.encode_text(batch["input_ids"]), text["text_embedding"]
        )
    hook.remove()
    assert lengths == [17, 17, 17]
    assert set(image) == {"embedding", "image_embedding"}
    assert set(text) == {"text_embedding"}
    torch.testing.assert_close(paired["image_embedding"], image["image_embedding"])
    torch.testing.assert_close(paired["text_embedding"], text["text_embedding"])


def test_padding_mask_reaches_text_encoder(batch):
    model = tiny_flip().eval()
    mask = torch.ones_like(batch["input_ids"])
    mask[:, -2:] = 0
    changed = batch["input_ids"].clone()
    changed[:, -2:] = (changed[:, -2:] + 1) % 32
    expected = model.encode_text(batch["input_ids"], mask)
    torch.testing.assert_close(model.encode_text(changed, mask), expected)
    out = model({"input_ids": changed, "attention_mask": mask}, stage="predict")
    torch.testing.assert_close(out["text_embedding"], expected)


def test_composable_forward_matches_method_class(batch):
    model = tiny_flip(mask_ratio=0)
    module = spt.Module(
        forward=spt.forward.flip,
        **{
            name: deepcopy(getattr(model, name))
            for name in (
                "image_encoder",
                "text_encoder",
                "image_projector",
                "text_projector",
                "logit_scale",
                "clip_loss",
            )
        },
    )
    actual, expected = module(batch, stage="fit"), model(batch)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize("ratio", [-0.1, 1.0, float("nan")])
def test_invalid_mask_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="mask_ratio"):
        tiny_flip(mask_ratio=ratio)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0},
        {"temperature": float("inf")},
        {"text_dim": 0},
        {"projection_dim": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    config = dict(text_encoder=TinyTextEncoder(), text_dim=16)
    config.update(kwargs)
    with pytest.raises(ValueError):
        spt.FLIP(**config)


@pytest.mark.parametrize("keys", [[], ["image"], ["input_ids"]])
def test_training_requires_pairs(batch, keys):
    with pytest.raises(ValueError, match="FLIP.*requires"):
        tiny_flip()({key: batch[key] for key in keys})


def test_mismatched_pair_count_is_rejected(batch):
    batch["input_ids"] = batch["input_ids"][:2]
    with pytest.raises(ValueError, match="equal image and text"):
        tiny_flip()(batch)


def test_unpooled_text_output_is_rejected(batch):
    model = tiny_flip()
    model.text_encoder.forward = lambda input_ids: model.text_encoder.embedding(
        input_ids
    )
    with pytest.raises(ValueError, match="pooled features"):
        model(batch)
