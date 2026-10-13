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

DAE-Net Data Loading
--------------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   DAENetDataModule
   DAENetDataset
   DAENetExposureSampler
   DAENetDistributedBatchSampler

DAE-Net owns manifest-backed HDF5 loading, deterministic query offsets and
exposure sampling. Shapes are read directly from HDF5; no open file handles
are shared between workers. The usage loader visits training records once
with voxels only, independently of repeated query training.

``DAENetDataModule`` constructs its own DataLoaders. General ``DataModule``
defaults for pinned memory and persistent workers apply to dictionary loader
configurations; prebuilt DataLoaders retain their existing settings.

For single-node DDP, set ``distributed=True`` and use
``Trainer(use_distributed_sampler=False)``. ``batch_size`` is global;
``DAENetDistributedBatchSampler`` partitions each ordered global exposure batch
without padding or dropping its tail. All ranks must receive at least one shape
per step. The current DDP controller requires ``num_workers=0`` and derives the
training loader generator state from the sampler seed and epoch. Usage remains
a full traversal performed by rank zero. See the full-category benchmark for
launch and completed-epoch recovery commands.

Offline surface-label loading and occupancy/semantic scoring live in
``benchmarks/dae_net/`` alongside the evaluation entry point.

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
