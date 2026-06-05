"""Minimal SimCLR run on the JAX / Flax-NNX backend.

This mirrors the torch ``examples/`` scripts but uses the parallel JAX backend
(``stable_pretraining.jax``). The forward-function shape, the dict-flow, the
callbacks, and the Lightning-style trainer hooks are all the same as on the
torch path — only the engine underneath is JAX-native (functional grads +
optax). Run with::

    python examples/jax_simclr_minimal.py

Install the backend first: ``pip install -e ".[jax]"``.
"""

import numpy as np
from flax import nnx

import stable_pretraining.jax as spj


def synthetic_two_view(n_batches=8, batch_size=64, dim=128, num_classes=10, seed=0):
    """Yield synthetic two-view training batches (stand-in for a real loader)."""
    rng = np.random.RandomState(seed)
    batches = []
    for _ in range(n_batches):
        labels = rng.randint(0, num_classes, size=batch_size)
        batches.append(
            {
                "views": [
                    {
                        "image": rng.randn(batch_size, dim).astype("float32"),
                        "label": labels,
                    },
                    {
                        "image": rng.randn(batch_size, dim).astype("float32"),
                        "label": labels,
                    },
                ]
            }
        )
    return batches


def synthetic_single_view(n_batches=4, batch_size=64, dim=128, num_classes=10, seed=1):
    """Yield synthetic single-view validation batches."""
    rng = np.random.RandomState(seed)
    return [
        {
            "image": rng.randn(batch_size, dim).astype("float32"),
            "label": rng.randint(0, num_classes, size=batch_size),
        }
        for _ in range(n_batches)
    ]


def main():
    dim, num_classes = 128, 10
    rngs = nnx.Rngs(0)

    # 1. Backbone (native Flax-NNX) producing [B, dim] embeddings.
    backbone = spj.backbone.MLP(dim, [256, dim], rngs=rngs)

    # 2. Method class wires backbone + projector + NT-Xent loss into a Module.
    model = spj.SimCLR(
        backbone=backbone,
        embed_dim=dim,
        rngs=rngs,
        projector_dims=(256, 128),
        temperature=0.5,
        optim={"type": "adamw", "learning_rate": 1e-3},
    )

    # 3. Evaluation callbacks read the forward-output dict — zero extra wiring.
    probe = spj.OnlineProbe(
        "linear_probe", probe=nnx.Linear(dim, num_classes, rngs=rngs)
    )
    rankme = spj.RankMe()

    # 4. Train via the Lightning-style trainer.
    trainer = spj.Trainer(max_epochs=10, callbacks=[probe, rankme])
    trainer.fit(model, synthetic_two_view(), synthetic_single_view())

    print("final SimCLR loss :", round(trainer.callback_metrics["fit/loss"], 4))
    print("linear probe acc  :", round(probe.accuracy, 4))
    print("RankMe            :", round(rankme.value, 3))


if __name__ == "__main__":
    main()
