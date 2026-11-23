stable_pretraining.backbone
===========================
.. module:: stable_pretraining.backbone
.. currentmodule:: stable_pretraining.backbone

The backbone module provides neural network architectures and utilities for self-supervised learning.

Architectures
-------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   MLP
   Resnet9
   ConvMixer
   Jet
   DAENetBackbone

DAE-Net components
~~~~~~~~~~~~~~~~~~

``DAENetBackbone`` accepts occupancy grids shaped ``[B, 1, 64, 64, 64]``
and query coordinates shaped ``[B, Q, 3]`` in object coordinates, normally
between -0.5 and 0.5. Query resolution does not change the encoder input size.
The following synthetic example computes the component loss:

.. code-block:: python

   import torch
   import stable_pretraining as spt

   backbone = spt.backbone.DAENetBackbone(branch_num=8)
   criterion = spt.losses.DAENetLoss(sparse_weight=0.02)  # Mug recipe
   batch = {
       "voxels": torch.zeros(1, 1, 64, 64, 64),
       "points": torch.rand(1, 128, 3) - 0.5,
   }
   target = torch.zeros(1, 128, 1)
   output = backbone(batch)
   losses = criterion(output, target, regularization_weight=0.001)
   losses["loss"].backward()

For inference, encode once and decode successive query chunks. Each branch
has its own occupancy field; ``part_occupancy`` includes its predicted
existence probability, while ``raw_occupancy`` does not.

.. code-block:: python

   backbone.eval()
   with torch.no_grad():
       encoded = backbone.encode(batch["voxels"])
       chunks = [
           backbone.decode(points, encoded)["part_occupancy"]
           for points in batch["points"].split(64, dim=1)
       ]
       parts = torch.cat(chunks, dim=1)  # [B, Q, branch_num]
       templates = backbone.templates(batch["points"])

Construction initializes all branches as at the start of upstream training.
``reset_branch`` preserves parameter objects and shared encoder convolutions;
it does not reset optimizer state. For the complete training method, use
:class:`~stable_pretraining.methods.DAENet`, which registers the era controller
for regularization scheduling, branch revival, Adam resets and
acceptance/rollback. The loss can be negative because its sparsity term is
negative.

Utility Functions
-----------------

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst

   from_timm
   from_torchvision
   set_embedding_dim

Specialized Modules
-------------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   TeacherStudentWrapper
   EvalOnly
