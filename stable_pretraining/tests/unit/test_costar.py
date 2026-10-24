"""Core COSTAR mathematics, causal histories, gradients and training."""

import math
from copy import deepcopy

import lightning as pl
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from stable_pretraining.methods.costar import (
    COSTAR,
    COSTAREncoder,
    COSTARFeatureBlock,
    COSTARPooling,
    COSTARTemporalAttention,
    select_last_valid_representations,
)

pytestmark = pytest.mark.unit


def _shaw_oracle(q, k, v, relative_keys, relative_values, radius, mask):
    """Sum Shaw's scalar terms without the implementation's matmul/einsum."""
    sequences, heads, times, width = q.shape
    result = []
    for sample in range(sequences):
        sample_heads = []
        for head in range(heads):
            rows = []
            for query in range(times):
                keys = [j for j in range(query + 1) if mask[sample, j]]
                ids = [min(max(j - query, -radius), radius) + radius for j in keys]
                scores = torch.stack(
                    [
                        sum(
                            q[sample, head, query, d]
                            * (k[sample, head, j, d] + relative_keys[r, d])
                            for d in range(width)
                        )
                        / math.sqrt(width)
                        for j, r in zip(keys, ids)
                    ]
                )
                weights = scores.softmax(0)
                rows.append(
                    torch.stack(
                        [
                            sum(
                                weights[i]
                                * (v[sample, head, j, d] + relative_values[r, d])
                                for i, (j, r) in enumerate(zip(keys, ids))
                            )
                            for d in range(width)
                        ]
                    )
                )
            sample_heads.append(torch.stack(rows))
        result.append(torch.stack(sample_heads))
    return torch.stack(result)


def _feature_oracle(block, values):
    """Independently evaluate full feature attention and both residual norms."""
    heads, width = block.num_heads, block.head_dim
    projected = [
        F.linear(values, layer.weight, layer.bias).reshape(
            values.shape[0], values.shape[1], heads, width
        )
        for layer in (block.q_projection, block.k_projection, block.v_projection)
    ]
    q, k, v = projected
    sequences = []
    for sample in range(values.shape[0]):
        tokens = []
        for query in range(values.shape[1]):
            contexts = []
            for head in range(heads):
                scores = torch.stack(
                    [
                        sum(
                            q[sample, query, head, d] * k[sample, key, head, d]
                            for d in range(width)
                        )
                        / math.sqrt(width)
                        for key in range(values.shape[1])
                    ]
                )
                weights = scores.softmax(0)
                contexts.append(
                    torch.stack(
                        [
                            sum(
                                weights[key] * v[sample, key, head, d]
                                for key in range(values.shape[1])
                            )
                            for d in range(width)
                        ]
                    )
                )
            tokens.append(torch.cat(contexts))
        sequences.append(torch.stack(tokens))
    attention = torch.stack(sequences)
    assert isinstance(block.output_projection, nn.Identity)

    def normalize(x, norm):
        centered = x - x.mean(-1, keepdim=True)
        denominator = (
            centered.square().sum(-1, keepdim=True) / (x.shape[-1] - 1)
        ).sqrt() + 1e-6
        return centered / denominator * norm.weight + norm.bias

    hidden = normalize(values + attention, block.attention_norm)
    update = F.relu(F.linear(hidden, block.ffn[0].weight, block.ffn[0].bias))
    update = F.linear(update, block.ffn[3].weight, block.ffn[3].bias)
    return normalize(hidden + update, block.ffn_norm)


def _encoder():
    return COSTAREncoder(
        2,
        1,
        1,
        1,
        d_model=4,
        num_heads=2,
        num_layers=2,
        max_relative_distance=1,
        ffn_hidden_dim=8,
        dropout=0.0,
    ).double()


def test_shaw_attention_matches_scalar_values_and_gradients():
    torch.manual_seed(17)
    device = torch.device("cpu")
    attention = COSTARTemporalAttention(4, 2, 1).to(device=device, dtype=torch.float64)
    reference = deepcopy(attention)
    inputs = [
        torch.randn(2, 2, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]
    copies = [value.detach().clone().requires_grad_() for value in inputs]
    mask = torch.tensor([[True] * 4, [True, True, False, False]], device=device)
    expected_ids = torch.tensor(
        [[1, 2, 2, 2], [0, 1, 2, 2], [0, 0, 1, 2], [0, 0, 0, 1]], device=device
    )
    assert torch.equal(attention._relative_position_ids(4, device), expected_ids)
    actual = attention._attention(*inputs, mask)
    expected = _shaw_oracle(
        *copies,
        reference.relative_key_embedding.weight,
        reference.relative_value_embedding.weight,
        1,
        mask,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    coefficient = torch.linspace(-1, 1, actual.numel(), device=device).reshape_as(
        actual
    )
    parameters = inputs + [
        attention.relative_key_embedding.weight,
        attention.relative_value_embedding.weight,
    ]
    targets = copies + [
        reference.relative_key_embedding.weight,
        reference.relative_value_embedding.weight,
    ]
    actual_gradients = torch.autograd.grad((actual * coefficient).sum(), parameters)
    expected_gradients = torch.autograd.grad((expected * coefficient).sum(), targets)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-9, atol=1e-10
        )


def test_feature_and_static_paths_match_independent_oracle():
    torch.manual_seed(23)
    block = COSTARFeatureBlock(4, 2, 8).double()
    dynamic = torch.randn(1, 2, 2, 4, dtype=torch.float64, requires_grad=True)
    static = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
    actual_dynamic, actual_static = block(dynamic, static)
    joint = torch.cat((dynamic, static[:, None].expand(-1, 2, -1, -1)), dim=2)
    expected_dynamic = _feature_oracle(block, joint.reshape(2, 4, 4))[:, :2].reshape_as(
        dynamic
    )
    expected_static = _feature_oracle(block, static)
    torch.testing.assert_close(actual_dynamic, expected_dynamic, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(actual_static, expected_static, rtol=1e-9, atol=1e-10)
    actual_loss = actual_dynamic[..., 0].sum() + actual_static[..., 1].sum()
    expected_loss = expected_dynamic[..., 0].sum() + expected_static[..., 1].sum()
    for actual, expected in zip(
        torch.autograd.grad(actual_loss, (dynamic, static)),
        torch.autograd.grad(expected_loss, (dynamic, static)),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)


def test_encoder_is_causal_and_nan_padding_has_zero_finite_gradients():
    torch.manual_seed(31)
    encoder = _encoder()
    mask = torch.tensor([[True] * 4, [True, True, False, False]])
    history = torch.randn(2, 4, 4, dtype=torch.float64)
    history[~mask] = float("nan")
    history.requires_grad_()
    static = torch.randn(2, 1, dtype=torch.float64, requires_grad=True)
    tokens, static_tokens = encoder(history, static, mask)
    changed = history.detach().clone()
    changed[0, 3] += torch.tensor([10.0, -15.0, 8.0, -6.0])
    changed[~mask] = 1e8
    perturbed, static_again = encoder(changed, static, mask)
    torch.testing.assert_close(perturbed[0, :3], tokens[0, :3])
    torch.testing.assert_close(perturbed[1], tokens[1])
    torch.testing.assert_close(static_again, static_tokens)
    assert torch.count_nonzero(tokens[~mask]) == 0
    assert torch.isfinite(tokens).all()
    coefficients = torch.randn_like(tokens)
    ((tokens * coefficients).sum() + static_tokens[..., 0].sum()).backward()
    assert torch.isfinite(history.grad).all() and torch.isfinite(static.grad).all()
    assert torch.count_nonzero(history.grad[~mask]) == 0
    assert history.grad[mask].abs().sum() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None
    )


def test_pooling_weights_features_and_endpoint_gradients_exactly():
    mask = torch.tensor([[True, True, True], [True, True, False]])
    tokens = torch.arange(48, dtype=torch.float64).reshape(2, 3, 4, 2)
    tokens[~mask] = float("nan")
    tokens.requires_grad_()
    pooled = COSTARPooling(2, 1, 1)(tokens, mask)
    torch.testing.assert_close(pooled["covariate"][mask], tokens[mask][:, :2].mean(1))
    torch.testing.assert_close(pooled["treatment"][mask], tokens[mask][:, 2])
    torch.testing.assert_close(pooled["outcome"][mask], tokens[mask][:, 3])
    torch.testing.assert_close(pooled["history"][mask], tokens[mask].mean(1))
    assert not torch.allclose(
        pooled["history"][mask],
        sum(pooled[k][mask] for k in ("covariate", "treatment", "outcome")) / 3,
    )
    endpoints = select_last_valid_representations(pooled, mask)
    for name in pooled:
        torch.testing.assert_close(endpoints[name], pooled[name][[0, 1], [2, 1]])
        assert torch.count_nonzero(pooled[name][~mask]) == 0
    sum(value.sum() for value in endpoints.values()).backward()
    expected = torch.zeros_like(tokens)
    expected[[0, 1], [2, 1], :2] = 0.75
    expected[[0, 1], [2, 1], 2:] = 1.25
    torch.testing.assert_close(tokens.grad, expected)


def _model() -> COSTAR:
    return COSTAR(
        covariate_dim=2,
        treatment_dim=1,
        outcome_dim=1,
        static_dim=1,
        d_model=6,
        num_heads=2,
        num_layers=1,
        ffn_hidden_dim=12,
        predictor_hidden_dim=9,
        dropout=0.0,
        momentum=0.75,
    )


def _batch() -> dict[str, torch.Tensor]:
    mask = torch.arange(4)[None] < torch.tensor([4, 3, 1])[:, None]
    history = torch.linspace(-2, 2, 48).reshape(3, 4, 4)
    return {
        "dynamic_history": history.masked_fill(~mask[:, :, None], float("nan")),
        "static_features": torch.tensor([[-1.0], [0.4], [1.1]]),
        "valid_mask": mask,
        "label": torch.full((3,), float("nan")),
        "future_outcomes": torch.full((3, 4, 1), float("nan")),
    }


def test_ssl_forward_backward_and_frozen_teacher() -> None:
    model, batch = _model(), _batch()
    for name, online in model.encoder.state_dict().items():
        target = model.momentum_encoder.state_dict()[name]
        torch.testing.assert_close(target, online, rtol=0, atol=0)
        assert target.data_ptr() != online.data_ptr()
    assert all(not p.requires_grad for p in model.momentum_encoder.parameters())
    history = batch["dynamic_history"].requires_grad_()
    batch["views"] = (history, history * 1.09)
    output = model(batch, stage="fit")
    count = int(batch["valid_mask"].sum())
    assert torch.isfinite(output["loss"])
    assert output["embedding"].shape == (3, 6)
    assert output["prediction"].shape == output["target"].shape == (2 * count, 6)
    assert output["prediction"].requires_grad and not output["target"].requires_grad
    output["loss"].backward()
    assert history.grad is not None and torch.isfinite(history.grad).all()
    assert not history.grad[~batch["valid_mask"]].count_nonzero()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in (*model.encoder.parameters(), *model.predictor.parameters())
    )
    assert all(p.grad is None for p in model.momentum_encoder.parameters())
    assert model.ema_updates.item() == 0


def test_ema_updates_after_adam_step_not_forward_or_validation() -> None:
    model, batch = _model(), _batch()
    optimizer = model.configure_optimizers()
    assert type(optimizer) is torch.optim.Adam
    before = deepcopy(model.momentum_encoder.state_dict())
    model(batch, stage="fit")["loss"].backward()
    model.eval()
    batch["views"] = "ignored during clean validation"
    output = model(batch, stage="val")
    expected = model.encode(
        batch["dynamic_history"], batch["static_features"], batch["valid_mask"]
    )
    torch.testing.assert_close(output["embedding"], expected["history"])
    assert output["loss"].item() == model.ema_updates.item() == 0
    for name, value in before.items():
        torch.testing.assert_close(
            model.momentum_encoder.state_dict()[name], value, rtol=0, atol=0
        )
    optimizer.step()
    assert model.ema_updates.item() == 1
    assert any(
        not torch.equal(before[name], online)
        for name, online in model.encoder.named_parameters()
    )
    for name, online in model.encoder.named_parameters():
        target = model.momentum_encoder.state_dict()[name]
        torch.testing.assert_close(target, before[name] * 0.75 + online.detach() * 0.25)
    for name, online in model.encoder.named_buffers():
        torch.testing.assert_close(
            model.momentum_encoder.state_dict()[name], online, rtol=0, atol=0
        )


def test_lightning_accumulation_updates_ema_once_per_optimizer_step(tmp_path) -> None:
    model, batch = _model(), _batch()
    before = [p.detach().clone() for p in model.encoder.parameters()]
    samples = [{key: value[i] for key, value in batch.items()} for i in range(3)] * 2
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, train_dataloaders=DataLoader(samples, batch_size=3))
    assert trainer.global_step == model.ema_updates.item() == 1
    assert any(
        not torch.equal(old, online)
        for old, online in zip(before, model.encoder.parameters())
    )
