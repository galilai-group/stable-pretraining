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
