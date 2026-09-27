# Jet: invertible images, tokens, and exact logdet

Jet is a backbone, exposed as `spt.backbone.Jet`. It does not choose a training
objective or create its own training loop. This is an unreleased addition;
install the current source as described in [the quickstart](quickstart.md).

```python
import torch
import stable_pretraining as spt

encoder = spt.backbone.Jet(
    image_size=16, patch_size=2, in_channels=3,
    coupling_layers=2, hidden_dim=16, depth=1, num_heads=2,
    scale_parameterization="exp_floor", scale_eps=1e-4,
    capture_stats=True,
)
images = torch.randn(2, 3, 16, 16)
tokens, logdet = encoder(images)
reconstructed, inverse_logdet = encoder.inverse(tokens)
torch.testing.assert_close(reconstructed, images)
torch.testing.assert_close(logdet + inverse_logdet, torch.zeros_like(logdet), atol=1e-6, rtol=0)
assert tokens.shape == (2, 64, 12)
assert logdet.shape == (2,)
```

Inputs are BCHW. Patch dimensions must be even; spatial coupling also needs an
even number of patches. The conditioner uses transformers with zero-initialized
output projections, so initialization is identity in patch coordinates.
The port adapts btrude/jet-pytorch under Apache-2.0; see the repository's
`THIRD_PARTY_NOTICES.md` and packaged license. It has its own state-dict layout;
upstream/deepstats checkpoints need an explicit conversion.

The default scale is `eps + (1-eps)*exp(raw)`. Its logarithm uses `logaddexp`.
The scale itself is evaluated by adding the floor directly, since
`exp(log(eps))` can round below that floor. There is no upper clipping and no
`log(scale+eps)` determinant approximation. `jet_sigmoid` selects the original
`2*sigmoid(raw)` bounded ablation. Choose one parameterization for the entire
encoder; settings are included in its state dict and checked on reload.

The original pre-scale bias convention is preserved: `(x + bias)*scale`, with
inverse `y/scale - bias`. This equals `x*scale + shift` for `shift=bias*scale`.
Coupling determinants sum `log_scale` over transformed coordinates. Scale,
affine arithmetic, and logdet use float32 minimum; float64 is preserved.
Use the same autocast context for forward and inverse. Floating-point rounding
still limits reconstruction accuracy, particularly after large expansion.

With `capture_stats=True`, `encoder.diagnostics` contains detached tensors:

- `flow/layer_k/log_scale_{mean,std,min,max}`
- `flow/layer_k/scale_{mean,std,min,max}`
- `flow/logdet_per_dim_{mean,std,min,max}`

Pass them to `Module.log_dict` in the forward callable. Values with
`abs(log_scale)>10` warn; nonfinite scales, shifts, outputs, and determinants
raise. No warning changes the optimization or clamps a value. Checks synchronize
with the device, so this correctness-first implementation has a runtime cost.
`checkpoint_conditioner=True` optionally trades recomputation for activation memory.

## MSE plus entropy demonstration

```bash
python -m stable_pretraining.quickstart --method jet-entropy --cache-dir ./jet-demo
spt web ./jet-demo/runs
```

This smoke example predicts mean-pooled tokens across two augmented views,
using MSE minus 0.01 times mean full-token logdet per input dimension. It logs
semantic mean-token energy, residual-token energy, and flow diagnostics. It
reuses `spt.Module`, `spt.OnlineProbe`, `spt.data.DataModule`, and `spt.Manager`.
It is not a claim that the objective has a finite optimum or learns semantic
representations: residual directions can expand without being penalized by a
mean-token MSE. Inspect that behavior in a real experiment.

The exact determinant belongs to **all tokens**, not their mean or any reduced
projection. A floor on coupling scales also does not bound every singular value
of the full Jacobian. For a continuous input distribution, expected full-output
logdet gives its differential-entropy change under the invertible transform;
it does not by itself supply an absolute entropy estimate for discrete images.
