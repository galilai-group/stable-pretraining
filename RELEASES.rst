
Version 0.1
-----------

- Added `matmul_precision` config parameter to control TensorFloat-32 (TF32) precision on Ampere and newer GPUs.
- Base trainer offering the basic functionalities of stable-SSL (logging, checkpointing, data loading etc).
- Template trainers for supervised and self-supervised learning (general joint embedding, JEPA, and teacher student models).
- Examples of self-supervised learning methods : SimCLR, Barlow Twins, VicReg, DINO, MoCo, SimSiam.
- Classes to load templates neural networks (backbone, projector, etc).
- LARS optimizer.
- Linear warmup schedulers.
- Loss functions: NTXEnt, Barlow Twins, Negative Cosine Similarity, VICReg.
- Base classes for multi-view dataloaders.
- Functionalities to read the loggings and easily export the results.
- RankMe, LiDAR metrics to monitor training.
- Examples of extracting run data from WandB and utilizing it to create figures.
- Fixed a bug in the logging functionality.
- Fixed ``OrderedQueue`` and ``UnsortedQueue`` checkpoint resume failing on scalar-label buffers when using lazy initialization (``shape=None``).
