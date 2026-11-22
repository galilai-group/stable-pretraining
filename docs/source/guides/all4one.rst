All4One training and reference audit
====================================

The runnable entry point is ``examples/all4one.py``. It supports CIFAR-10,
CIFAR-100, ImageNet-100, and a small synthetic smoke test. It uses ResNet-18
for all three real datasets. The generic ``spt.All4One`` class also accepts
custom backbones; this example does not implement the paper's ViT or full
ImageNet experiments.

Reference and interpretation
----------------------------

The implementation is compared against the authors' repository at commit
``00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08``:

* `Core model and objectives <https://github.com/ImaGonEs/all4one/blob/00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08/solo/methods/all4one.py>`_
* `CIFAR recipe <https://github.com/ImaGonEs/all4one/blob/00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08/scripts/pretrain/cifar/all4one.yaml>`_
* `Batch-size scaling <https://github.com/ImaGonEs/all4one/blob/00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08/solo/args/pretrain.py>`_
* `Backbone construction <https://github.com/ImaGonEs/all4one/blob/00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08/solo/methods/base.py>`_
* `Pretraining transforms <https://github.com/ImaGonEs/all4one/blob/00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08/solo/data/pretrain_dataloader.py>`_
* `Evaluation transforms <https://github.com/ImaGonEs/all4one/blob/00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08/solo/data/classification_dataloader.py>`_
* `Reported training durations <https://github.com/ImaGonEs/all4one/blob/00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08/README.md>`_
* `Paper <https://arxiv.org/abs/2303.09417>`_

The core defaults retain the released implementation's interpretation. This
is not identical to a literal implementation of every sentence in the paper:

.. list-table:: Paper/release differences
   :header-rows: 1
   :widths: 25 35 40

   * - Detail
     - Paper
     - Released code and current core defaults
   * - Final projector BatchNorm
     - Section 4.1 excludes final layers
     - Included
   * - Feature diagonal coefficient
     - Equations 3--4 imply 5.0
     - 2.5 (off-diagonal coefficient is also 2.5)
   * - Teacher features for feature contrast
     - Section 3.3 describes batch-axis normalization
     - Row normalization, then batch-axis normalization
   * - Attention learning rate
     - General batch-size scaling rule
     - Separate group starts at 0.1, without batch-size scaling

Do not describe this implementation as matching both definitions exactly.
Changing the diagonal coefficient alone does not switch the entire model to
the paper's literal formulation.

Training settings
-----------------

.. list-table:: Example defaults for real datasets
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Value
   * - Backbone
     - torchvision ResNet-18, no pretrained weights, zero_init_residual=False
   * - CIFAR input stem
     - 3x3 convolution, stride 1, padding 2; no max-pool
   * - Pretraining epochs
     - CIFAR-10/100: 1000; ImageNet-100: 400 (as reported in the repository)
   * - Batch size
     - 256 on one device; no gradient accumulation
   * - Representation base LR
     - 1.0, multiplied by batch_size / 256
   * - Attention LR
     - 0.1, unscaled, with the same warmup/cosine schedule
   * - Representation/attention optimizer
     - LARS: SGD momentum 0.9, weight decay 1e-5, eta 0.02, clip_lr=True;
       biases and normalization parameters excluded
   * - Schedule
     - 10 warmup epochs from 3e-5, then cosine decay to zero
   * - Projector / predictors
     - 2048-2048-256 / 4096-256; PyTorch default initialization
   * - Transformer
     - 3 layers, 8 heads, feed-forward width 512, dropout 0.1
   * - Queue / neighbors
     - 98304 normalized entries / 5 cosine-nearest neighbors
   * - Temperatures
     - Neighbor contrast 0.2; centroid contrast 0.1
   * - Teacher
     - Parameter-only EMA, cosine coefficient 0.99 to 1.0; independent BN buffers

The released LARS optimizer includes the SGD momentum update; it is not an
alternative to SGD in the sense of replacing it with AdamW. The example
uses LARS for representation and attention groups in the CNN experiment,
as does the released model's optimizer construction.

CIFAR pretraining normalization uses the release's dataset-specific values:
CIFAR-10 mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616);
CIFAR-100 mean (0.5071, 0.4865, 0.4409), std (0.2673, 0.2564, 0.2762).
Its classification loader uses CIFAR-10 mean and std (0.247, 0.243, 0.261)
for both CIFAR datasets; that distinction is recorded separately in the
example configuration. ImageNet-100 uses ImageNet normalization.

The repository contains an All4One CIFAR YAML but no dedicated All4One
ImageNet-100 YAML at this commit. The ImageNet-100 example combines the
reported 400-epoch duration, the shared All4One settings, and the repository's
ImageNet-100 asymmetric augmentation recipe; it is not a recovered complete
original experiment configuration. The exact original 100-class selection
must also be supplied by the user for benchmark comparisons.

Running and evaluating
----------------------

See ``benchmarks/all4one/README.md`` for setup, commands using the existing
example, recorded single-seed results, and available source/environment evidence.

Pretraining (dataset already present)::

    python examples/all4one.py --dataset cifar10 --data-root /path/to/cifar \
        --accelerator gpu --batch-size 256 --export-encoder /path/to/encoder.pt

Separate frozen linear evaluation::

    python examples/all4one.py --stage linear --dataset cifar10 \
        --data-root /path/to/cifar --accelerator gpu \
        --encoder-path /path/to/encoder.pt --epochs 100 --lr 0.1

The linear stage freezes the final backbone, trains a new classifier using
labels, and reports held-out accuracy. Its 100-epoch SGD/cosine settings are
a baseline, not a verified All4One-specific linear recipe from the release.
OnlineProbe and OnlineKNN during pretraining are diagnostics; online accuracy
must not be presented as a completed separate frozen linear evaluation.
Labels do not contribute gradients to the All4One representation objective.

``--lr`` means a base rate at batch size 256, including in the linear stage.
``--attention-lr`` is an unscaled group rate. Effective rates, preprocessing,
recipe identifier, and the reference commit are stored in run metadata.
Synthetic runs keep their small-model test settings and unscaled rates.
There is no dry-run option.

The ``spt.All4One`` class alone is a composable model, not the entire training
recipe. Supply ``optim`` when constructing it directly; otherwise it inherits
the framework's generic AdamW setup. Use the example for the audited CNN
recipe.

Compatibility and remaining limits
----------------------------------

Existing checkpoints and metrics are not rewritten. Earlier runs used timm
initialization and ImageNet normalization for CIFAR and cannot be relabeled
as runs of the revised recipe. Frozen evaluation reads saved preprocessing
metadata from exported encoders, including legacy exports. Match the saved
CIFAR stem padding when loading an encoder. Full training resume requires the
same model, optimizer, preprocessing, and scheduler configuration.

This port uses the framework's separate optimizers/callbacks rather than the
release's single optimizer and inline online classifier. The online probe
uses SGD rather than the release's LARS classifier group. Its warmup scheduler
reaches the peak at step W rather than the release's W-1 (which repeats the
peak at W). These differences must be considered before claiming identical
training trajectories. The release also repeats projector forwards inside its
training step; this port computes each branch once, so BatchNorm update counts
and random-number consumption are not guaranteed identical.

Tests cover objective values and gradients, queue ordering, attention,
initialization, batch-size scaling, dataset defaults, EMA timing, frozen
backbone state, and checkpoint resume. Passing them does not establish
reproduction of the paper's reported accuracies; that requires fresh complete
training and a matched evaluation protocol.
