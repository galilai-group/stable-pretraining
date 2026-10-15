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

Information Maximization
------------------------

CorInfoMax combines agreement between augmented views with covariance
log-determinants that discourage dimensional collapse.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   CorInfoMax

.. code-block:: python

    import stable_pretraining as spt

    model = spt.CorInfoMax(
        encoder_name="resnet18",
        low_resolution=True,
        projector_dims=(2048, 2048, 64),
    )

Custom encoders can be passed through ``encoder_name`` with an explicit
``embed_dim`` when they do not expose ``num_features``. They must return feature
vectors shaped ``[batch_size, embed_dim]``. See
``examples/corinfomax_cifar10.py`` for the complete pretraining and frozen linear
evaluation recipe, including transforms and optimizer settings.

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
