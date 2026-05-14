stable_pretraining.data
=======================
.. module:: stable_pretraining.data
.. currentmodule:: stable_pretraining.data

The data module provides comprehensive tools for dataset handling, transforms, sampling, and data loading in self-supervised learning contexts.

Core Components
---------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   DataModule
   Collator
   Dataset

Real Data Wrappers
------------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   FromTorchDataset
   HFDataset
   Subset

Synthetic Data Generators
-------------------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   GMM
   MinariStepsDataset
   MinariEpisodeDataset

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst

   swiss_roll
   generate_perlin_noise_2d
   perlin_noise_3d

Noise Models
------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   Categorical
   ExponentialMixtureNoiseModel
   ExponentialNormalNoiseModel

Samplers
---------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   RepeatedRandomSampler
   SupervisedBatchSampler
   RandomBatchSampler

Utility Functions
-----------------

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst

   fold_views
   random_split
   download
   bulk_download

Modules
-------

.. autosummary::
   :toctree: gen_modules/

   transforms
   dataset_stats
   synthetic_data
