FSDP2 (sharded training)
========================

``stable-pretraining`` supports `FSDP2 <https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html>`_
— PyTorch's per-parameter sharding API (``fully_shard``) — through Lightning's
:class:`~lightning.pytorch.strategies.ModelParallelStrategy`. FSDP2 shards model
parameters, gradients, and optimizer state across data-parallel ranks, so you
can train models that don't fit on one GPU with the same memory savings as
ZeRO-3.

Requirements: ``torch>=2.4`` and ``lightning>=2.4``.

Quick start
-----------

Asking for FSDP2 is exactly like asking for DDP — a single trainer-config
switch:

.. code-block:: yaml

    trainer:
      strategy: fsdp2          # registered automatically on `import stable_pretraining`
      precision: bf16-mixed    # recommended; ModelParallelStrategy rejects 16-mixed
      devices: auto
      accelerator: gpu

That's it. No code changes to your :class:`~stable_pretraining.Module`, forward
function, or method are required. When ``strategy="fsdp2"`` is active, the
:meth:`~stable_pretraining.Module.configure_model` hook shards the trainable
parts of your model before the optimizer is built.

What gets sharded
-----------------

By default, every **trainable child subtree** of your ``Module`` — the
``backbone``, ``projector``, ``predictor``, ``prototypes``, etc. — is sharded.
Blocks are detected generically (the parameter-owning elements of any
``nn.ModuleList`` / ``nn.Sequential``), so timm ViTs, torchvision ResNets, and
custom transformer stacks all shard per-block with no configuration.

The online-evaluation callbacks (``OnlineProbe``, ``OnlineKNN``, ``RankMe``,
...) own *separate* optimizers over plain parameters and are deliberately **not**
sharded, so probing keeps working unchanged.

Teacher/student methods
-----------------------

EMA methods (DINO, BYOL, MoCo, I-JEPA, ...) wrap their backbone in
:class:`~stable_pretraining.backbone.TeacherStudentWrapper`. Under FSDP2 the
student and teacher are sharded *identically* so the EMA update — a positional
zip over their parameters — stays correct on the sharded ``DTensor``\\ s. This
is handled automatically; an alignment assertion fails fast at setup if a custom
configuration would break it.

Mixed precision
---------------

Use ``precision: bf16-mixed`` (Lightning autocast). FSDP2's own
``MixedPrecisionPolicy`` (which controls the dtype of the sharded all-gather) is
optional and only needed for advanced setups — pass it via a strategy object
(below).

Customizing the sharding
------------------------

Advanced users can control exactly what is sharded by passing a
``parallelize_fn`` to the ``Module``. The contract is
``(module, device_mesh) -> None``; apply ``fully_shard`` to whatever you like:

.. code-block:: python

    import stable_pretraining as spt
    from torch.distributed.fsdp import fully_shard

    def my_parallelize_fn(module, device_mesh):
        dp_mesh = device_mesh["data_parallel"]
        for block in module.backbone.blocks:
            fully_shard(block, mesh=dp_mesh)
        fully_shard(module.backbone, mesh=dp_mesh)

    module = spt.Module(forward=..., backbone=..., parallelize_fn=my_parallelize_fn)

To pass a custom mixed-precision policy or explicit parallel sizes, build the
strategy object directly:

.. code-block:: python

    from torch.distributed.fsdp import MixedPrecisionPolicy
    from stable_pretraining.utils.fsdp2 import make_fsdp2_strategy

    strategy = make_fsdp2_strategy(
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )
    # trainer: {"strategy": strategy, ...}

Notes & limitations
-------------------

* **FSDP1 is not supported.** ``Trainer(strategy="fsdp")`` is rejected with a
  redirect to ``"fsdp2"`` — FSDP1's flat training-state machine breaks the
  multi-forward methods (DINO/I-JEPA/LeJEPA).
* Checkpointing uses Lightning's distributed checkpoint (DCP) path provided by
  ``ModelParallelStrategy`` (``save_distributed_checkpoint=True`` by default).
* FSDP2 + ``ModelParallelStrategy`` is marked experimental upstream; APIs may
  evolve with PyTorch/Lightning.
