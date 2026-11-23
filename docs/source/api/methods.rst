stable_pretraining.methods
==========================

.. module:: stable_pretraining.methods
.. currentmodule:: stable_pretraining.methods

The methods module provides ready-to-use ``LightningModule`` subclasses, one
per SSL algorithm. Each class pre-wires the backbone, loss function, optimizer,
and any required callbacks so you can start training with minimal boilerplate.

All method classes are importable from the top-level namespace::

    import stable_pretraining as spt

    model = spt.SimCLR(backbone=backbone, projector=projector, temperature=0.1)

Or directly from the sub-package::

    from stable_pretraining.methods import SimCLR, BYOL, DINO

See :doc:`forward` for the stateless forward-function equivalents and
``METHODS.md`` at the repository root for the complete method catalog.

Contrastive Methods
-------------------

Methods that learn representations by contrasting positive and negative pairs,
or by bootstrapping without explicit negatives.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   SimCLR
   BYOL
   NNCLR
   MoCov2
   MoCov3
   SimSiam
   PIRL
   TiCO

Feature Redundancy Reduction
-----------------------------

Methods that learn representations by reducing redundancy or decorrelating
feature dimensions rather than using explicit contrastive pairs.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   VICReg
   VICRegL
   BarlowTwins
   WMSE

Self-Distillation and Clustering
----------------------------------

Methods that use momentum-updated teacher networks, self-distillation, or
online clustering to learn representations without negative pairs.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   DINO
   DINOv2
   DINOv3
   iBOT
   SwAV
   MSN
   Data2Vec

Masked Image Modeling
----------------------

Methods that learn representations by reconstructing masked regions of the
input, either in pixel space or in a latent feature space.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   MAE
   BEiT
   CMAE
   MaskFeat
   SimMIM
   MIMRefiner
   iGPT
   IJEPA
   LeJEPA
   VISReg
   SALT

3D Shape Co-Segmentation
------------------------

DAE-Net :cite:p:`chen2023daenet` learns deformable part templates from unlabeled
shape geometry. Each shape is reconstructed from a selected subset of parts.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   DAENet

.. code-block:: python

   import lightning as pl
   import stable_pretraining as spt

   model = spt.DAENet(
       backbone=spt.backbone.DAENetBackbone(branch_num=8),
       sparse_weight=0.02,
       lr=2e-4,
       era_length=10,
   )
   trainer = pl.Trainer(accelerator="gpu", devices=1,
                        precision="32-true", max_epochs=160)
   manager = spt.Manager(trainer=trainer, module=model, data=data_module)
   manager()

Here ``data_module`` must supply dict batches containing ``voxels``, ``points``
and ``occupancy`` as described in :doc:`forward`, plus a
``usage_dataloader()`` that visits each training shape once for branch usage.
:class:`~stable_pretraining.data.DAENetDataModule` provides these interfaces.

The method registers :class:`~stable_pretraining.callbacks.DAENetRevival` for
regularization warmup, branch revival, Adam state resets and era
acceptance/rollback. Rejection restores the accepted network weights; it does
not rewind training progress or optimizer state. Completed-epoch checkpoints
also preserve controller state for continuation in the same environment.

The DDP path requires Linux, one node, FP32, zero DataLoader workers and
one Adam update per step. Configure ``DAENetDataModule(distributed=True)`` and
``Trainer(devices=N, strategy="ddp_find_unused_parameters_true",
use_distributed_sampler=False)`` to keep gradient buckets stable across restarts.
Batch size is global; ``N`` is the number of processes on the node. Tails are retained
without padding, and every rank must receive a nonempty shard. Rank zero computes
usage and revival, then broadcasts weights and ages. Checkpoints gather each rank's RNG after the era decision and before
checkpoint saving; ``state_dict()`` itself performs no communication. Resume
requires unchanged world size, data, global batch and device/software environment.

The ``benchmarks/dae_net/`` directory contains the full-category ShapeNet Part
recipes, data preparation and semantic scoring commands. Those experiments
include test geometry in unlabeled training and fit branch-to-label mappings
on test labels; their scores measure full-category co-segmentation.

Other
-----

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   NEPA
