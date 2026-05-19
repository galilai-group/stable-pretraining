"""Unit tests for the global ``exclude_bias_norm`` setting (#368)."""

import copy

import pytest
import torch
import torch.nn as nn

import stable_pretraining as spt
from stable_pretraining.optim.utils import (
    create_optimizer,
    is_bias_or_norm_param,
    split_params_for_weight_decay,
)


@pytest.fixture(autouse=True)
def _reset_global_config():
    """Each test starts with the default global config and restores it after."""
    cfg = spt.get_config()
    original = cfg.exclude_bias_norm
    yield
    cfg.exclude_bias_norm = original


def _toy_model() -> nn.Module:
    """A model that has both decay-eligible weights AND bias/norm params.

    Linear → BN → Linear gives us:
    - 2× weight (2D, should get weight_decay)
    - 2× bias (1D, should get weight_decay=0)
    - 2× BN weight/bias (norm-layer, should get weight_decay=0)
    """
    return nn.Sequential(nn.Linear(8, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))


@pytest.mark.unit
class TestGlobalExcludeBiasNormDefault:
    """The global default propagates when the per-config flag is unset."""

    def test_default_is_false_for_backward_compat(self):
        assert spt.get_config().exclude_bias_norm is False

    def test_global_off_no_split(self):
        """Default (global=False, no per-call): single param group, full weight decay."""
        model = _toy_model()
        opt = create_optimizer(
            model.parameters(),
            {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.01},
            named_params=model.named_parameters(),
        )
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["weight_decay"] == 0.01

    def test_global_on_applies_default(self):
        """spt.set(exclude_bias_norm=True) → optimizer splits params."""
        spt.set(exclude_bias_norm=True)
        model = _toy_model()
        opt = create_optimizer(
            model.parameters(),
            {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.01},
            named_params=model.named_parameters(),
        )
        # Two groups: decay-applied + decay-zero
        assert len(opt.param_groups) == 2
        wds = sorted(g["weight_decay"] for g in opt.param_groups)
        assert wds == [0.0, 0.01]

    def test_explicit_false_overrides_global_true(self):
        """Per-call ``exclude_bias_norm=False`` beats global True."""
        spt.set(exclude_bias_norm=True)
        model = _toy_model()
        opt = create_optimizer(
            model.parameters(),
            {
                "type": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.01,
                "exclude_bias_norm": False,
            },
            named_params=model.named_parameters(),
        )
        # Explicit False should produce a single group.
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["weight_decay"] == 0.01

    def test_explicit_true_works_when_global_false(self):
        """Per-call ``True`` works even when global is the default False."""
        model = _toy_model()
        opt = create_optimizer(
            model.parameters(),
            {
                "type": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.01,
                "exclude_bias_norm": True,
            },
            named_params=model.named_parameters(),
        )
        assert len(opt.param_groups) == 2

    def test_string_config_picks_up_global(self):
        """Even the string-only config (``"AdamW"``) reads the global flag."""
        spt.set(exclude_bias_norm=True)
        model = _toy_model()
        opt = create_optimizer(
            model.parameters(), "AdamW", named_params=model.named_parameters()
        )
        assert len(opt.param_groups) == 2


@pytest.mark.unit
class TestGlobalExcludeBiasNormValidation:
    """Setter validation + serialisation."""

    def test_setter_rejects_non_bool(self):
        cfg = spt.get_config()
        with pytest.raises(TypeError, match="exclude_bias_norm must be a bool"):
            cfg.exclude_bias_norm = 1
        with pytest.raises(TypeError, match="exclude_bias_norm must be a bool"):
            cfg.exclude_bias_norm = "true"

    def test_set_function_accepts_bool(self):
        spt.set(exclude_bias_norm=True)
        assert spt.get_config().exclude_bias_norm is True
        spt.set(exclude_bias_norm=False)
        assert spt.get_config().exclude_bias_norm is False

    def test_set_with_other_args_doesnt_disturb(self):
        """Mixing ``exclude_bias_norm`` with another setting works."""
        spt.set(exclude_bias_norm=True, requeue_checkpoint=False)
        cfg = spt.get_config()
        assert cfg.exclude_bias_norm is True
        assert cfg.requeue_checkpoint is False

    def test_repr_includes_flag(self):
        spt.set(exclude_bias_norm=True)
        assert "exclude_bias_norm=True" in repr(spt.get_config())


@pytest.mark.unit
class TestGlobalExcludeBiasNormCorrectParamSplit:
    """Verify the param-group split actually contains the right tensors."""

    def test_bias_and_bn_land_in_zero_decay_group(self):
        spt.set(exclude_bias_norm=True)
        model = _toy_model()
        opt = create_optimizer(
            model.parameters(),
            {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.05},
            named_params=model.named_parameters(),
        )
        # Identify groups
        decay_group = next(g for g in opt.param_groups if g["weight_decay"] > 0)
        zero_group = next(g for g in opt.param_groups if g["weight_decay"] == 0)

        # Map param ids to names for verification
        id_to_name = {id(p): n for n, p in model.named_parameters()}
        decay_names = sorted(id_to_name[id(p)] for p in decay_group["params"])
        zero_names = sorted(id_to_name[id(p)] for p in zero_group["params"])

        # The Linear 2D weights must be in the decay group
        assert "0.weight" in decay_names  # first Linear
        assert "2.weight" in decay_names  # second Linear

        # Biases and BN params must be in zero-decay group
        assert "0.bias" in zero_names
        assert "1.weight" in zero_names  # BatchNorm weight (1-D)
        assert "1.bias" in zero_names  # BatchNorm bias
        assert "2.bias" in zero_names


@pytest.mark.unit
class TestIsBiasOrNormParamHeuristic:
    """Coverage tests for the bias/norm classifier — including the 1-D fix."""

    def test_named_bias_caught(self):
        # Linear's bias is named '*.bias' and is 1-D — both rules trigger.
        m = nn.Linear(4, 2)
        for name, p in m.named_parameters():
            if name == "bias":
                assert is_bias_or_norm_param(name, p) is True

    def test_named_norm_substring_caught(self):
        p = nn.Parameter(torch.randn(3, 3))  # 2-D, would NOT be caught by dim rule
        assert is_bias_or_norm_param("encoder.layer_norm.weight", p) is True
        assert is_bias_or_norm_param("bn1.weight", p) is False  # no 'norm' substring
        assert is_bias_or_norm_param("LayerNorm.weight", p) is True

    def test_1d_param_caught_even_without_norm_in_name(self):
        """1-D BN weight inside ``Sequential`` must still be flagged.

        Named ``1.weight`` (no 'norm' substring), so only the 1-D rule
        catches it — this is the regression the heuristic fix targets.
        """
        m = nn.Sequential(nn.Linear(8, 4), nn.BatchNorm1d(4))
        for name, p in m.named_parameters():
            if name == "1.weight":  # BatchNorm weight, 1-D
                assert is_bias_or_norm_param(name, p) is True
            elif name == "0.weight":  # Linear weight, 2-D
                assert is_bias_or_norm_param(name, p) is False

    def test_2d_weight_not_caught(self):
        # A regular Conv/Linear weight (>=2-D) must NOT be flagged.
        p = nn.Parameter(torch.randn(4, 8))
        assert is_bias_or_norm_param("encoder.fc.weight", p) is False

    def test_split_helper_groups_all_expected_params(self):
        """End-to-end on a realistic mixed model (Conv + BN + Linear + LN)."""
        m = nn.Sequential(
            nn.Conv2d(3, 4, 3),
            nn.BatchNorm2d(4),
            nn.Flatten(),
            nn.Linear(4 * 100, 8),  # arbitrary input size — only shapes matter
            nn.LayerNorm(8),
            nn.Linear(8, 2),
        )
        groups = split_params_for_weight_decay(m.named_parameters(), weight_decay=0.1)
        decay_group = next(g for g in groups if g["weight_decay"] > 0)
        zero_group = next(g for g in groups if g["weight_decay"] == 0)

        id_to_name = {id(p): n for n, p in m.named_parameters()}
        decay_names = sorted(id_to_name[id(p)] for p in decay_group["params"])
        zero_names = sorted(id_to_name[id(p)] for p in zero_group["params"])

        # Conv weight (4-D), both Linear weights (2-D) → decay group.
        assert decay_names == ["0.weight", "3.weight", "5.weight"]
        # All biases + all 1-D norm scales → zero-decay group.
        assert zero_names == [
            "0.bias",
            "1.bias",
            "1.weight",
            "3.bias",
            "4.bias",
            "4.weight",
            "5.bias",
        ]


@pytest.mark.unit
class TestTrainingStepImpact:
    """Behavioural tests: actually step an optimizer and check parameter moves.

    Strategy: zero all gradients, then call ``optimizer.step()``. With zero
    gradients, the SGD update reduces to a pure weight-decay shrink:

        w_after = w_before * (1 - lr * weight_decay)

    Parameters in the decay group should shrink by exactly that factor;
    parameters in the zero-decay group should stay bit-identical.
    """

    @torch.no_grad()
    def _zero_grads_and_attach(self, model: nn.Module) -> None:
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

    def test_with_global_on_only_weights_shrink(self):
        spt.set(exclude_bias_norm=True)
        torch.manual_seed(0)
        model = _toy_model()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        lr, wd = 0.1, 0.5
        opt = torch.optim.SGD(
            split_params_for_weight_decay(model.named_parameters(), weight_decay=wd),
            lr=lr,
        )
        self._zero_grads_and_attach(model)
        opt.step()

        expected_shrink = 1.0 - lr * wd  # SGD's coupled decay form
        for name, p in model.named_parameters():
            b = before[name]
            if name in ("0.weight", "2.weight"):
                # 2-D Linear weights → should shrink
                torch.testing.assert_close(p, b * expected_shrink, rtol=1e-5, atol=1e-7)
            else:
                # Biases + 1-D BN params → must be exactly unchanged
                torch.testing.assert_close(p, b, rtol=0, atol=0)

    def test_with_global_off_everything_shrinks(self):
        """Sanity check: without the flag, weight decay touches every parameter."""
        spt.set(exclude_bias_norm=False)
        torch.manual_seed(0)
        model = _toy_model()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        lr, wd = 0.1, 0.5
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        self._zero_grads_and_attach(model)
        opt.step()

        expected_shrink = 1.0 - lr * wd
        for name, p in model.named_parameters():
            b = before[name]
            torch.testing.assert_close(p, b * expected_shrink, rtol=1e-5, atol=1e-7)

    def test_global_flag_matches_explicit_per_config(self):
        """Global True should produce the same trained state as explicit True."""
        torch.manual_seed(123)
        model_global = _toy_model()
        model_explicit = copy.deepcopy(model_global)

        # Global path
        spt.set(exclude_bias_norm=True)
        opt_a = create_optimizer(
            model_global.parameters(),
            {"type": "SGD", "lr": 0.05, "weight_decay": 0.1},
            named_params=model_global.named_parameters(),
        )

        # Explicit-per-config path, with the global default OFF to be sure
        # we're testing the per-config flag in isolation.
        spt.set(exclude_bias_norm=False)
        opt_b = create_optimizer(
            model_explicit.parameters(),
            {
                "type": "SGD",
                "lr": 0.05,
                "weight_decay": 0.1,
                "exclude_bias_norm": True,
            },
            named_params=model_explicit.named_parameters(),
        )

        # Same data → same gradients → same updates if grouping matches.
        torch.manual_seed(42)
        x = torch.randn(16, 8)
        target = torch.randn(16, 2)
        for _ in range(3):
            for m, opt in [(model_global, opt_a), (model_explicit, opt_b)]:
                opt.zero_grad()
                loss = ((m(x) - target) ** 2).mean()
                loss.backward()
                opt.step()

        # The two models should be bit-identical after training.
        for (n_a, p_a), (n_b, p_b) in zip(
            model_global.named_parameters(), model_explicit.named_parameters()
        ):
            assert n_a == n_b
            torch.testing.assert_close(p_a, p_b, rtol=1e-6, atol=1e-7)

    def test_global_flag_changes_actual_training_outcome(self):
        """Global on vs off produces different trained weights.

        Confirms the flag has a real effect on optimisation, not just a
        no-op that happens to leave some param shapes unchanged.
        """
        torch.manual_seed(0)
        model_on = _toy_model()
        model_off = copy.deepcopy(model_on)

        spt.set(exclude_bias_norm=True)
        opt_on = create_optimizer(
            model_on.parameters(),
            {"type": "SGD", "lr": 0.05, "weight_decay": 0.5},
            named_params=model_on.named_parameters(),
        )
        spt.set(exclude_bias_norm=False)
        opt_off = create_optimizer(
            model_off.parameters(),
            {"type": "SGD", "lr": 0.05, "weight_decay": 0.5},
            named_params=model_off.named_parameters(),
        )

        torch.manual_seed(7)
        x = torch.randn(16, 8)
        target = torch.randn(16, 2)
        for _ in range(3):
            for m, opt in [(model_on, opt_on), (model_off, opt_off)]:
                opt.zero_grad()
                loss = ((m(x) - target) ** 2).mean()
                loss.backward()
                opt.step()

        # Biases / BN params: should differ — decayed under OFF, not under ON.
        biases_match = []
        for name in ("0.bias", "1.weight", "1.bias", "2.bias"):
            p_on = dict(model_on.named_parameters())[name]
            p_off = dict(model_off.named_parameters())[name]
            biases_match.append(torch.allclose(p_on, p_off, rtol=1e-6, atol=1e-7))
        # At least one of the bias/norm tensors must differ between runs.
        # (Some may coincidentally match if they were zero, but BN.weight
        # starts at 1.0 and will diverge.)
        assert not all(biases_match), (
            "No bias/norm params differed between global on/off — the global "
            "flag may not actually be applied."
        )
