"""Torch <-> JAX parity regression tests.

These are the guardrails that keep the two backends from diverging: given the
*same parameters* and the *same inputs*, the torch and JAX paths must produce
the same forward outputs, the same loss, the same optimizer step, and the same
multi-step training dynamics (within float32 tolerance). Parameters are copied
torch -> JAX with :func:`stable_pretraining.jax.utils.copy_torch_params_`.
"""

import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

from stable_pretraining.jax import losses as jax_losses  # noqa: E402
from stable_pretraining.jax.backbone import MLP  # noqa: E402
from stable_pretraining.jax.losses import NTXEntLoss as JaxNTXEnt  # noqa: E402
from stable_pretraining.jax.utils import copy_linear_, copy_torch_params_  # noqa: E402
from stable_pretraining import losses as torch_losses  # noqa: E402
from stable_pretraining.losses import NTXEntLoss as TorchNTXEnt  # noqa: E402

# float32 round-trips through two linear algebra stacks; these tolerances are
# tight enough to catch real bugs but absorb backend-specific fp ordering.
ATOL, RTOL = 1e-5, 1e-4


def _torch_mlp(d, hidden):
    """Torch Sequential mirroring jax ``MLP(d, hidden)`` (ReLU between layers)."""
    layers, prev = [], d
    for i, h in enumerate(hidden):
        layers.append(torch.nn.Linear(prev, h))
        if i < len(hidden) - 1:
            layers.append(torch.nn.ReLU())
        prev = h
    return torch.nn.Sequential(*layers)


def _kernel(linear):
    return np.asarray(linear.kernel[...])


def test_mlp_forward_parity():
    d, hidden = 16, [32, 8]
    tmlp = _torch_mlp(d, hidden).eval()
    jmlp = MLP(d, hidden, rngs=nnx.Rngs(0))
    copy_torch_params_(jmlp, tmlp)

    x = np.random.RandomState(0).randn(4, d).astype("float32")
    with torch.no_grad():
        yt = tmlp(torch.from_numpy(x)).numpy()
    yj = np.asarray(jmlp(jnp.asarray(x)))
    np.testing.assert_allclose(yt, yj, atol=ATOL, rtol=RTOL)


def test_ntxent_loss_parity():
    rng = np.random.RandomState(0)
    z_i = rng.randn(8, 16).astype("float32")
    z_j = rng.randn(8, 16).astype("float32")
    lt = float(
        TorchNTXEnt(temperature=0.5)(torch.from_numpy(z_i), torch.from_numpy(z_j))
    )
    lj = float(JaxNTXEnt(temperature=0.5)(jnp.asarray(z_i), jnp.asarray(z_j)))
    assert abs(lt - lj) < 1e-4


def test_byol_loss_parity():
    rng = np.random.RandomState(3)
    p = rng.randn(8, 16).astype("float32")
    t = rng.randn(8, 16).astype("float32")
    lt = float(torch_losses.BYOLLoss()(torch.from_numpy(p), torch.from_numpy(t)))
    lj = float(jax_losses.BYOLLoss()(jnp.asarray(p), jnp.asarray(t)))
    assert abs(lt - lj) < 1e-4


def test_vicreg_loss_parity():
    rng = np.random.RandomState(4)
    z_i = rng.randn(16, 32).astype("float32")
    z_j = rng.randn(16, 32).astype("float32")
    lt = float(torch_losses.VICRegLoss()(torch.from_numpy(z_i), torch.from_numpy(z_j)))
    lj = float(jax_losses.VICRegLoss()(jnp.asarray(z_i), jnp.asarray(z_j)))
    np.testing.assert_allclose(lt, lj, rtol=1e-4, atol=1e-4)


def test_barlow_twins_loss_parity():
    rng = np.random.RandomState(5)
    z_i = rng.randn(16, 32).astype("float32")
    z_j = rng.randn(16, 32).astype("float32")
    lt = float(
        torch_losses.BarlowTwinsLoss()(torch.from_numpy(z_i), torch.from_numpy(z_j))
    )
    lj = float(jax_losses.BarlowTwinsLoss()(jnp.asarray(z_i), jnp.asarray(z_j)))
    np.testing.assert_allclose(lt, lj, rtol=1e-4, atol=1e-4)


def test_negative_cosine_similarity_parity():
    from stable_pretraining.losses.utils import NegativeCosineSimilarity as TorchNeg

    rng = np.random.RandomState(6)
    a = rng.randn(8, 16).astype("float32")
    b = rng.randn(8, 16).astype("float32")
    lt = float(TorchNeg()(torch.from_numpy(a), torch.from_numpy(b)))
    lj = float(jax_losses.NegativeCosineSimilarity()(jnp.asarray(a), jnp.asarray(b)))
    np.testing.assert_allclose(lt, lj, rtol=1e-4, atol=1e-5)


def test_infonce_loss_parity():
    from stable_pretraining.losses.joint_embedding import InfoNCELoss as TorchInfoNCE

    rng = np.random.RandomState(7)
    anchors = rng.randn(8, 16).astype("float32")
    candidates = rng.randn(8, 16).astype("float32")
    targets = np.arange(8)
    lt = float(
        TorchInfoNCE(temperature=0.1)(
            torch.from_numpy(anchors),
            torch.from_numpy(candidates),
            torch.from_numpy(targets),
        )
    )
    lj = float(
        jax_losses.InfoNCELoss(temperature=0.1)(
            jnp.asarray(anchors), jnp.asarray(candidates), jnp.asarray(targets)
        )
    )
    np.testing.assert_allclose(lt, lj, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", ["ntxent", "byol", "vicreg", "barlow", "negcos"])
def test_two_arg_loss_parity(name):
    """All two-argument losses match torch within fp32 tolerance, by name."""
    from stable_pretraining.losses.utils import NegativeCosineSimilarity as TorchNeg

    pairs = {
        "ntxent": (TorchNTXEnt(temperature=0.5), JaxNTXEnt(temperature=0.5)),
        "byol": (torch_losses.BYOLLoss(), jax_losses.BYOLLoss()),
        "vicreg": (torch_losses.VICRegLoss(), jax_losses.VICRegLoss()),
        "barlow": (torch_losses.BarlowTwinsLoss(), jax_losses.BarlowTwinsLoss()),
        "negcos": (TorchNeg(), jax_losses.NegativeCosineSimilarity()),
    }
    torch_loss, jax_loss = pairs[name]
    rng = np.random.RandomState(abs(hash(name)) % 1000)
    a = rng.randn(16, 16).astype("float32")
    b = rng.randn(16, 16).astype("float32")
    lt = float(torch_loss(torch.from_numpy(a), torch.from_numpy(b)))
    lj = float(jax_loss(jnp.asarray(a), jnp.asarray(b)))
    np.testing.assert_allclose(lt, lj, rtol=1e-4, atol=1e-4)


def test_sinkhorn_parity():
    """JAX Sinkhorn-Knopp must match torch SwAVLoss.sinkhorn for the same scores."""
    from stable_pretraining.jax.losses import sinkhorn as jax_sinkhorn
    from stable_pretraining.losses.joint_embedding import SwAVLoss as TorchSwAV

    rng = np.random.RandomState(8)
    scores = rng.randn(16, 10).astype("float32")
    torch_q = (
        TorchSwAV(epsilon=0.05, sinkhorn_iterations=3)
        .sinkhorn(torch.from_numpy(scores))
        .numpy()
    )
    jax_q = np.asarray(jax_sinkhorn(jnp.asarray(scores), epsilon=0.05, n_iterations=3))
    np.testing.assert_allclose(jax_q, torch_q, rtol=1e-4, atol=1e-5)


def test_single_sgd_step_parity():
    d, o = 16, 8
    tlin = torch.nn.Linear(d, o)
    jlin = nnx.Linear(d, o, rngs=nnx.Rngs(0))
    copy_linear_(jlin, tlin)

    x = np.random.RandomState(0).randn(4, d).astype("float32")
    topt = torch.optim.SGD(tlin.parameters(), lr=0.1)
    loss = (tlin(torch.from_numpy(x)) ** 2).mean()
    topt.zero_grad()
    loss.backward()
    topt.step()

    jopt = nnx.Optimizer(jlin, optax.sgd(0.1), wrt=nnx.Param)
    _, grads = nnx.value_and_grad(lambda m: (m(jnp.asarray(x)) ** 2).mean())(jlin)
    jopt.update(jlin, grads)

    np.testing.assert_allclose(
        tlin.weight.detach().numpy(), _kernel(jlin).T, atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        tlin.bias.detach().numpy(), np.asarray(jlin.bias[...]), atol=ATOL, rtol=RTOL
    )


def test_training_dynamics_parity():
    """20 SGD steps on a fixed batch must trace the same loss curve on both backends."""
    d, hidden, steps, lr = 16, [32, 8], 20, 0.1
    tmlp = _torch_mlp(d, hidden)
    jmlp = MLP(d, hidden, rngs=nnx.Rngs(0))
    copy_torch_params_(jmlp, tmlp)

    x = np.random.RandomState(0).randn(8, d).astype("float32")
    xt, xj = torch.from_numpy(x), jnp.asarray(x)

    topt = torch.optim.SGD(tmlp.parameters(), lr=lr)
    jopt = nnx.Optimizer(jmlp, optax.sgd(lr), wrt=nnx.Param)

    @nnx.jit
    def jstep(m, opt):
        loss, grads = nnx.value_and_grad(lambda mm: (mm(xj) ** 2).mean())(m)
        opt.update(m, grads)
        return loss

    t_losses, j_losses = [], []
    for _ in range(steps):
        tl = (tmlp(xt) ** 2).mean()
        topt.zero_grad()
        tl.backward()
        topt.step()
        t_losses.append(float(tl.detach()))
        j_losses.append(float(jstep(jmlp, jopt)))

    np.testing.assert_allclose(t_losses, j_losses, atol=1e-4, rtol=1e-3)
    assert t_losses[-1] < t_losses[0]  # both actually optimized


def test_simclr_forward_parity():
    """Full backbone+projector+NT-Xent pipeline agrees with the torch path."""
    d, e, p = 16, 12, [24, 8]
    t_bb, j_bb = _torch_mlp(d, [20, e]), MLP(d, [20, e], rngs=nnx.Rngs(0))
    t_proj, j_proj = _torch_mlp(e, p), MLP(e, p, rngs=nnx.Rngs(1))
    copy_torch_params_(j_bb, t_bb)
    copy_torch_params_(j_proj, t_proj)

    rng = np.random.RandomState(0)
    v1 = rng.randn(8, d).astype("float32")
    v2 = rng.randn(8, d).astype("float32")

    with torch.no_grad():
        z1 = t_proj(t_bb(torch.from_numpy(v1)))
        z2 = t_proj(t_bb(torch.from_numpy(v2)))
        lt = float(TorchNTXEnt(temperature=0.5)(z1, z2))

    jz1 = j_proj(j_bb(jnp.asarray(v1)))
    jz2 = j_proj(j_bb(jnp.asarray(v2)))
    lj = float(JaxNTXEnt(temperature=0.5)(jz1, jz2))

    assert abs(lt - lj) < 1e-4
