stable_pretraining.optim
========================
.. module:: stable_pretraining.optim
.. currentmodule:: stable_pretraining.optim

The optim module provides custom optimizers and learning rate schedulers for self-supervised learning.

Optimizers
----------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   LARS

Learning Rate Schedulers
------------------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   CosineDecayer
   LinearWarmup
   LinearWarmupCosineAnnealing
   LinearWarmupCyclicAnnealing
   LinearWarmupThreeStepsAnnealing
