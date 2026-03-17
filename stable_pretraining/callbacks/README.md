# Callback guide

## Available callbacks

### Evaluation & Monitoring

| Callback | File | Description |
|----------|------|-------------|
| `OnlineProbe` | [probe.py](probe.py) | Trains a lightweight linear probe on frozen representations during pretraining to monitor downstream task performance in real time. Maintains its own optimizer and training loop, fully independent of the main model. |
| `OnlineKNN` | [knn.py](knn.py) | Non-parametric k-nearest neighbors evaluator. Uses a rolling queue of cached embeddings and labels to compute weighted KNN predictions during validation. |
| `RankMe` | [rankme.py](rankme.py) | Tracks the effective rank of feature representations by computing the exponential entropy of normalized singular values. A drop in rank signals dimensional collapse. |
| `LiDAR` | [lidar.py](lidar.py) | Monitors representation quality via Linear Discriminant Analysis Rank - the effective rank of the LDA eigenvalue distribution over surrogate classes of augmented views. |
| `CLIPZeroShot` | [clip_zero_shot.py](clip_zero_shot.py) | Zero-shot classification evaluator for CLIP-style models. Compares image embeddings against pre-encoded class text embeddings to produce predictions without any fine-tuning. |
| `ImageRetrieval` | [image_retrieval.py](image_retrieval.py) | Image retrieval evaluator (following the DINO protocol). Computes normalized embeddings, gathers across ranks, and evaluates retrieval metrics on query/gallery splits. |
| `LatentViz` | [latent_viz.py](latent_viz.py) | Online latent-space visualization. Learns a 2D projection that preserves neighborhood structure (contrastive loss on k-NN graphs) and periodically plots it during training. |
| `EpochMilestones` | [earlystop.py](earlystop.py) | Early-stops training if a monitored metric fails to reach a threshold by a given epoch (e.g., "accuracy ≥ 0.5 by epoch 20"). Supports both `max` and `min` directions. |

### Training Utilities

| Callback | File | Description |
|----------|------|-------------|
| `TeacherStudentCallback` | [teacher_student.py](teacher_student.py) | Automatically discovers `TeacherStudentWrapper` instances and performs EMA (exponential moving average) teacher updates at configurable frequency - after backward or after optimizer step. |
| `WeightDecayUpdater` | [wd_schedule.py](wd_schedule.py) | Updates the optimizer's weight decay on a per-batch schedule. Supports constant, linear, cosine, and exponential schedules with per-param-group targeting. |
| `EmbeddingCache` | [embedding_cache.py](embedding_cache.py) | Registers forward hooks on named submodules to cache their intermediate outputs, optionally merging them into the model's forward output dict. |


## Practical Considerations

### Dataset strategy

Parametric probes (e.g. OnlineProbe) are jointly optimized alongside the backbone model, i.e. each batch is passed through both modules. As a consequence, the benchmark / downstream data (with labels) used for the probe needs to be present in the pretraining dataset. If labels are not available for all datapoints, a filter can be applied in the forward pass to identify the batch indices with labels.

This setup currently doesn't allow for any advanced splitting strategies for the probes, as the splitting is directly linked with the pretraining data.

### Probe convergence

The features probes receive are continuously shifting during training. As a consequence, parametric probes constantly need to adapt to the representations, which slows down the learning. In practice, this is however no concern, as once the representations converge, the probe training does as well.


### Probe hyperparameter tuning

The proposed way of evaluating different probe hyperparameters (e.g. hidden dimension, number of layers, ...) is to create one probe for each configuration. As the probes are lightweight, its typically no problem to optimize multiple jointly with the backbone.
