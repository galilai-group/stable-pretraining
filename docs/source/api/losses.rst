stable_pretraining.losses
=========================

.. module:: stable_pretraining.losses
.. currentmodule:: stable_pretraining.losses

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   NTXEntLoss
   NegativeCosineSimilarity
   VICRegLoss
   BarlowTwinsLoss
   DAENetLoss

DAE-Net objective
-----------------

:class:`DAENetLoss` implements the occupancy and regularization objective of
DAE-Net :cite:p:`chen2023daenet`. It takes the backbone output, occupancy
targets shaped ``[B, Q, 1]`` and an era-dependent ``regularization_weight``.

.. code-block:: python

   import stable_pretraining as spt

   criterion = spt.losses.DAENetLoss(sparse_weight=0.02)
   losses = criterion(output, target, regularization_weight=0.001)
   losses["loss"].backward()

Here ``output`` is the result of ``DAENetBackbone`` for the corresponding
voxel/query batch, and ``target`` contains its occupancy labels.

The returned terms are ``reconstruction_sum``, ``reconstruction_max``,
``sparsity``, ``deformation`` and ``affine``, combined into ``loss``. Components
already include their configured weights. The max-occupancy weight defaults
to 0.1, sparsity to 0.002, deformation to 100, and affine regularization to 0.
The Mug benchmark overrides sparsity to 0.02.

Sparsity is negative and depends on the batch-mean part existence. Deformation
uses detached raw occupancy to weight displacements before squaring.
Consequently, total loss can be negative and should not be interpreted as an
IoU loss. :class:`~stable_pretraining.methods.DAENet` supplies the scheduled
regularization weight through its era controller.
