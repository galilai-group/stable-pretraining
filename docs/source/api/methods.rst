stable_pretraining.methods
==========================

.. module:: stable_pretraining.methods
.. currentmodule:: stable_pretraining.methods

The methods module provides 30 ready-to-use ``LightningModule`` subclasses, one
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

Other
-----

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   NEPA

Latent World Models
-------------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   RCaux

RC-aux combines multi-horizon latent prediction with budget-conditioned
reachability. Supply a world model that returns ``embedding`` (B, T, D),
``prediction`` (B, K, D), and optionally ``regularization_loss``::

    import stable_pretraining as spt

    model = spt.RCaux(
        world_model=world_model,
        embedding_dim=256,
        max_horizon=5,
        context_index=1,
        loss_kwargs={"reachability_weight": 0.35},
        optim={"optimizer": {"type": "AdamW", "lr": 3e-4}},
    )
    spt.Manager(trainer=trainer, module=model, data=data, seed=0)()

The method delegates to :func:`stable_pretraining.forward.rcaux`. In evaluation
mode only ``embedding`` is required. Under DDP, cross-trajectory negatives come
from the local batch on each rank (per GPU); there is no global latent gather.
A local batch of one omits cross-trajectory negatives.

For full-state checkpoint resume, recreate the same model and loss configuration
and pass an absolute checkpoint path to ``Manager`` with ``weights_only=False``.
See ``benchmarks/rcaux/README.md`` for the complete prediction-only versus RC-aux
training comparison, dataset collection command, and measured results.
