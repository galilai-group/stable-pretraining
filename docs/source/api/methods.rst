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
   CAPI
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

CAPI usage
----------

CAPI :cite:`darcet2025capi` predicts position-wise balanced teacher clusters
from visible student patches. Use ``spt.CAPI`` with ``spt.Manager`` and a
single BCHW tensor under ``image``. Image height and width must be divisible
by the patch size, with enough patches to retain both visible and masked tokens.
The method class installs its EMA callback automatically.
Embeddings average teacher patches, excluding registers; select the student
with ``representation="student"``. Schedules count optimizer updates;
``accumulation_steps`` does not enlarge the assignment-balancing batch.

For an existing ImageFolder directory (one subdirectory per class), run from
the repository root:

.. code-block:: bash

   CAPI_TRAIN_DIRECTORY=/path/to/images spt examples/capi_config.yaml

The configuration supplies a single normalized 224-pixel crop per image;
labels are available to evaluation callbacks but are not used by the SSL loss.
Its architecture and schedules use author-scale defaults, which require
substantial GPU memory. For a reduced-scale example and frozen-feature
evaluation, see ``benchmarks/imagenet10/capi-vit-small.py``. That script runs
inside a Slurm allocation and downloads Imagenette and CIFAR-10 if needed.

.. autoclass:: stable_pretraining.callbacks.CAPIEMA
