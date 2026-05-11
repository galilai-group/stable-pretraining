Distributed Training with FSDP2
===============================

``stable-pretraining`` supports `PyTorch's FSDP2
<https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html>`_
(``torch.distributed.fsdp.fully_shard``) as a first-class training
strategy alongside DDP. This page covers when to reach for FSDP2, how to
opt in, why the integration exists in the form it does, and the known
limitations.


When to use FSDP2 vs DDP
------------------------

DDP replicates parameters, gradients, and optimizer state on every rank.
FSDP shards them. The trade-off is memory vs. communication.

**Use DDP when:**

- The full model + AdamW optimizer state comfortably fits on a single GPU.
  As a rough rule of thumb, anything up to ViT-Large with standard SSL
  setups runs fine on DDP and is faster than FSDP on small clusters.
- You're optimizing for throughput on a small number of GPUs.

**Reach for FSDP2 when:**

- Optimizer state or activations dominate memory and you'd otherwise OOM
  (e.g. large EMA-teacher methods like DINOv2 with a ViT-Large student).
- You want to fit a larger batch on the same hardware.
- You're scaling to many GPUs and DDP's full replication is wasteful.

When in doubt, start with DDP. Switch to FSDP2 only when memory is the
binding constraint.


Opting in
---------

The simplest way to use FSDP2 is via :func:`make_fsdp_strategy`:

.. code-block:: python

    import lightning as pl
    from stable_pretraining.utils.fsdp2 import make_fsdp_strategy

    strategy = make_fsdp_strategy()  # auto-detects num_gpus, sensible defaults

    trainer = pl.Trainer(
        strategy=strategy,
        accelerator="gpu",
        devices=num_gpus,
        precision="bf16-mixed",  # see "Mixed precision" below
    )

That's the entire user-facing surface. ``Module.configure_model`` does the
sharding work behind the scenes by reading the device mesh Lightning sets
on the module and dispatching to :func:`default_parallelize_fn`.

For a complete worked example, compare these scripts side by side:

- ``benchmarks/imagenet10/lejepa_vit_small.py`` (single-GPU)
- ``benchmarks/imagenet10/lejepa_vit_small_ddp.py`` (DDP, 2 GPUs)
- ``benchmarks/imagenet10/lejepa_vit_small_fsdp.py`` (FSDP2, 2 GPUs)

The diff between DDP and FSDP2 is one line: the strategy passed to
``pl.Trainer``. Everything else (model, data, callbacks, optimizer) is
identical.


Why FSDP2 and not FSDP1
-----------------------

Lightning's :class:`~lightning.pytorch.strategies.FSDPStrategy` (and the
``Trainer(strategy="fsdp")`` string shortcut that resolves to it) wraps
PyTorch's *FSDP1* (``FullyShardedDataParallel``). FSDP1 uses a flat
training-state machine and registers autograd hooks at wrap time. Methods
that perform **multiple forward passes per training step** through the
same wrapped unit confuse this state machine: the post-backward hook for
forward-1 fires before the pre-backward hook for forward-2 has
transitioned state, and FSDP1 asserts:

::

    Expects BACKWARD_PRE or BACKWARD_POST state but got HandleTrainingState.FORWARD

This is hit by exactly the methods this repo specializes in: LeJEPA, DINO,
IJEPA, BYOL with multi-view, MAE with multi-block-mask. For LeJEPA in
particular, the multi-forward pattern is intrinsic — global views are
224×224 (seq_len 197) and local views are 96×96 (seq_len 37), so they
*cannot* be merged into a single batched forward through a ViT.

FSDP2 (``fully_shard``) drops the flat state machine and uses standard
PyTorch hooks per call. Multi-forward is supported by design. This is the
principal reason the integration wires FSDP2 specifically.


Why ``stable_pretraining/utils/fsdp2.py`` exists
-----------------------------------------------

Lightning 2.x exposes FSDP2 through :class:`ModelParallelStrategy`, which
unlike :class:`FSDPStrategy`:

- has **no string-shortcut registration** (no ``Trainer(strategy="fsdp2")``),
- has **no** ``ignored_modules`` **kwarg** to exclude submodules from
  sharding,
- has **no** ``auto_wrap_policy`` **callable** for per-block wrapping,
- requires the user's ``LightningModule`` to override ``configure_model``
  (the strategy's ``setup`` raises ``TypeError`` otherwise).

FSDP1 had built-in equivalents for all four of these. FSDP2 doesn't yet,
so the wiring is on the user's side. ``utils/fsdp2.py`` is that wiring,
pre-baked correctly so every benchmark file doesn't reinvent it. Once
Lightning ships a string shortcut for FSDP2 and ``ModelParallelStrategy``
adds ``ignored_modules``, large chunks of this module simplify away.

The file has four substantive pieces:

1. **The factory** — :func:`make_fsdp_strategy` constructs
   ``ModelParallelStrategy`` with one footgun-fixed default:
   ``data_parallel_size`` auto-computes from ``LOCAL_WORLD_SIZE`` (set by
   ``torchrun``) or ``torch.cuda.device_count()``. Lightning's own
   ``"auto"`` resolves this to ``num_nodes`` (= 1 single-node), which
   would leave the data-parallel axis at size 1 and trip the
   ``data_parallel_size * tensor_parallel_size == world_size`` check
   several layers down.

2. **The parallelize_fn** — :func:`default_parallelize_fn` is what
   ``Module.configure_model`` dispatches to. It does the work
   ``ModelParallelStrategy`` does *not* do for you:

   - **Slices the data-parallel submesh** out of Lightning's 2-D
     ``(data_parallel × tensor_parallel)`` mesh. The strategy hands you a
     2-D mesh even at ``tensor_parallel_size=1`` (shape ``(N, 1)``); pass
     that to ``fully_shard`` and it interprets the second dim as an HSDP
     replicate axis (silently wrong, no error).
   - **Recognizes block classes from a curated registry** built lazily by
     try-importing from {``timm``, ``transformers``, ``torchvision``}. The
     registry currently covers ``timm.models.vision_transformer.Block``,
     ``transformers.models.vit.modeling_vit.ViTLayer``, and
     ``torchvision.models.resnet.{BasicBlock, Bottleneck}``. Models built
     from any of these classes get per-block ``fully_shard``; models that
     don't intersect the registry **raise**
     :class:`UnsupportedModelError` at strategy setup (instead of silently
     falling back to root-only sharding, which is the worst failure
     mode — training succeeds but throughput is mysteriously degraded).
     Users with out-of-scope architectures pass their own
     ``parallelize_fn`` (see :ref:`fsdp_custom_parallelize`).
   - **Detaches callback containers around the root sweep** —
     :class:`OnlineProbe`, :class:`OnlineKNN`, :class:`RankMe` and other
     :class:`TrainableCallback` subclasses register their probe / queue /
     metric modules in ``module.callbacks_modules`` and own their own
     standalone optimizers. Each callback's optimizer expects regular
     ``nn.Parameter``\\ s. If the root ``fully_shard(module)`` call
     converts those parameters to ``DTensor``\\ s, the callback's
     optimizer ends up with stale references that no longer exist in the
     form they were when the optimizer was constructed; updates either
     silently corrupt or visibly fail. The exclusion uses
     **detach-around-root**: pop the containers out of the module tree
     before ``fully_shard(model)`` and reattach after. Skipping them in
     the per-block loop alone is *not* sufficient — the root sweep claims
     every parameter reachable from ``model`` that isn't already in a
     nested FSDP unit. Detach/reattach is the standard FSDP2 way to
     exclude submodules from the root sweep without giving them their
     own FSDP unit.

3. **The alignment check** — :func:`assert_aligned_wrapping` is used by
   :class:`~stable_pretraining.TeacherStudentWrapper` to guarantee
   correct in-place EMA via ``zip(teacher.parameters(),
   student.parameters())``. Under FSDP2 each parameter is a ``DTensor``,
   and for the EMA to be correct each pair has to address the same
   logical region of the same parameter — that means matching shape,
   dtype, ``DTensor.placements``, *and* ``device_mesh``, not just shape.
   Two DTensors with equal local shapes can have different placements
   (``Shard(0)`` vs ``Shard(1)`` on equal-sized dims; ``Shard`` vs
   ``Replicate`` in degenerate cases); the check rejects mismatches
   early so you don't get a silent corruption budgeted at "fine for the
   first epoch" and then drift.

4. **Detection helpers** — :func:`is_fsdp_strategy` and
   :func:`describe_fsdp_strategy` for :class:`Manager`-side logging and
   tests.


Mixed precision
---------------

:class:`ModelParallelStrategy` does **not** accept ``precision="16-mixed"``
(fp16 mixed). Supported values are ``32-true``, ``bf16-mixed``,
``bf16-true``, and ``16-true``. ``bf16-mixed`` is the closest analogue of
DDP's ``"16-mixed"`` — same mixed-precision forward-in-half /
backward-in-full pattern, just with bfloat16 instead of float16 (no
loss-scaling needed). A10G, L4, and A100 all support bf16.

For finer-grained control (different param / reduce / output dtypes,
``cast_forward_inputs``, etc.), pass a :class:`MixedPrecisionPolicy` to
:func:`make_fsdp_strategy`:

.. code-block:: python

    from torch.distributed.fsdp import MixedPrecisionPolicy
    from stable_pretraining.utils.fsdp2 import make_fsdp_strategy

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    strategy = make_fsdp_strategy(mp_policy=policy)


.. _fsdp_custom_parallelize:

Customizing the parallelization
-------------------------------

Two escape hatches, in increasing order of power.

**Inline ``block_classes`` for "default behavior with my block class".**
The registry is curated and conservative; if your architecture has a
recognizable repeated block class that just isn't in the registry yet,
pass it via ``block_classes=`` to :func:`default_parallelize_fn`. The
cleanest way to do this is via a custom ``parallelize_fn`` passed to
``spt.Module``:

.. code-block:: python

    from functools import partial
    from stable_pretraining import Module
    from stable_pretraining.utils.fsdp2 import default_parallelize_fn
    from my_arch import MyBlock

    module = Module(
        model=...,
        forward=...,
        parallelize_fn=partial(default_parallelize_fn, block_classes={MyBlock}),
    )

**Full custom ``parallelize_fn`` for non-standard sharding.** For hybrid
sharding, tensor parallelism, ``reshard_after_forward=False``, custom
``MixedPrecisionPolicy``, or anything else the default doesn't cover,
write your own ``(model, mesh) -> None`` callable:

.. code-block:: python

    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from stable_pretraining import Module

    def my_parallelize(model, mesh):
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        for blk in model.backbone.transformer_blocks:
            fully_shard(blk, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=policy)

    module = Module(model=..., forward=..., parallelize_fn=my_parallelize)

Either escape hatch is preferable to silently shipping a misconfigured
default. Subclassing ``spt.Module`` and overriding ``configure_model``
directly works too — it's the same code path Lightning calls.


Compatibility notes
-------------------

A few stable-pretraining-side adjustments are required for FSDP2 to be
correct end-to-end. These are all already in place; this section is for
context if you're tracing a bug.

- ``Module.validation_step`` / ``test_step`` / ``predict_step`` route
  through ``self(batch, ...)`` (i.e. ``__call__``) rather than
  ``self.forward(batch, ...)``. PyTorch's forward pre/post hooks (which
  is how FSDP2 triggers the all-gather of sharded ``DTensor`` params
  before each forward) only fire on ``__call__``, not on a direct
  ``.forward()`` call. The old routing left ``DTensor``\\ s un-gathered
  during validation, and the very next op (e.g. patch-embed conv) raised
  ``aten.convolution.default: got mixed torch.Tensor and DTensor``.

- ``Module.named_parameters`` accepts and forwards ``remove_duplicate``.
  PyTorch's FSDP wrap path always calls ``model.named_parameters(
  remove_duplicate=False)``; without forwarding the kwarg, the override
  raises ``TypeError`` before ``fully_shard`` can look at the model.

- :class:`LogUnusedParametersOnce` self-disables under FSDP2. After
  ``fully_shard``, parameters are ``DTensor``\\ s and gradient flow goes
  through the unsharded gathered tensor that FSDP2 produces inside the
  forward all-gather. The detector's ``Tensor.register_hook`` on the
  ``DTensor`` never fires; without the self-disable, every parameter
  would falsely report as "unused" and flood the log.


See also
--------

- :func:`stable_pretraining.utils.fsdp2.make_fsdp_strategy`
- :func:`stable_pretraining.utils.fsdp2.default_parallelize_fn`
- :func:`stable_pretraining.utils.fsdp2.assert_aligned_wrapping`
- :class:`stable_pretraining.TeacherStudentWrapper`
- :class:`stable_pretraining.callbacks.OnlineProbe`
- ``benchmarks/imagenet10/lejepa_vit_small_fsdp.py`` — worked example
