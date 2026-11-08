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

Latent World Models
-------------------

``SubJEPA`` trains a shared observation encoder and causal action-conditioned
predictor without an EMA teacher. It combines next-latent MSE with Gaussian
regularization in frozen row-orthonormal subspaces. Targets retain gradients.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   SubJEPA

Provide ``image`` with shape ``[batch, time, ...]`` and ``action`` with shape
``[batch, time - 1, action_dim]``. The backbone must return a flat latent vector
of dimension ``embed_dim`` for each observation. The default causal predictor
is compact; pass ``predictor=`` to reproduce a particular LeWM architecture.
Use 32 subspaces for the paper's default 192-dimensional latent, or 16 for
PushT. Tune ``regularization_weight`` on held-out data.

The regularizer follows the authors' released code by estimating characteristic
functions across the batch separately at each time step, rather than pooling
time and batch as described in the paper's equation 6. Under DDP, the statistic
uses differentiable global batch sums. Frozen projection matrices and the
random-direction step counter are checkpointed.

Run ``python examples/subjepa_digits.py --epochs 20 --output /tmp/subjepa-digits``
for a small CPU diagnostic on translated sklearn handwritten digits. This
reports held-out dynamics errors and representation diagnostics; it does not
reproduce the paper's Two-Room, Reacher, PushT, or OGB-Cube planning results.

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
