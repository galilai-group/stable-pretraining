DirectCLR
=========

DirectCLR applies symmetric cross-view InfoNCE to a fixed leading subvector of
an encoder's output, without a trainable projector. It implements Jing et al.,
`Understanding Dimensional Collapse in Contrastive Self-supervised Learning
<https://arxiv.org/abs/2110.09348>`_ (ICLR 2022).
The complete encoder representation remains available for downstream evaluation.

Tensor interface
----------------

The method class follows the same two-view interface as SimCLR:

.. code-block:: python

   import torch
   import stable_pretraining as spt

   model = spt.DirectCLR(
       encoder_name="resnet50", subspace_dim=360, temperature=0.1
   )
   view1 = torch.randn(2, 3, 224, 224)
   view2 = torch.randn_like(view1)
   output = model(view1, view2)
   output.loss.backward()
   # output.embedding: [4, 2048]; output.projection: [4, 360]

   model.eval()
   with torch.no_grad():
       features = model(view1)
   # features.embedding: [2, 2048]; features.loss: scalar zero
   # features.projection is None

``encoder_name`` also accepts a prebuilt image encoder returning flat features.
``low_resolution`` and ``pretrained`` configure timm encoders. Construction
checks that ``subspace_dim`` is an integer between one and the encoder width;
shape discovery preserves BatchNorm buffers and the encoder's training modes.

Composable training interface
-----------------------------

For dict batches, callbacks and the Module/Manager lifecycle, use the composable
forward function. Each view contains ``image`` and optionally ``label``:

.. code-block:: python

   backbone = spt.backbone.from_timm("resnet50", num_classes=0, pretrained=False)
   module = spt.Module(
       forward=spt.forward.directclr,
       backbone=backbone,
       subspace_dim=360,
       directclr_loss=spt.losses.DirectCLRLoss(temperature=0.1),
       optim={"optimizer": {"type": "SGD", "lr": 0.03, "momentum": 0.9}},
   )

Training batches contain two view dictionaries under ``views``. Evaluation uses
one ``image`` batch and returns full ``embedding`` features. Labels only feed
callbacks, not the SSL loss. The benchmark uses this interface and configures
LARS, warmup and cosine decay explicitly. Distributed loss gathers candidates
with autograd, requires equal batch shapes across ranks and promotes reduced
precision similarities to float32.

Run a small CPU example
-----------------------

Install from the checkout with ``python -m pip install .``. Generated images
exercise the training/checkpoint path without downloading a dataset:

.. code-block:: bash

   python benchmarks/imagenet100/directclr.py train \
       --dataset fake --backbone resnet18 --dim 64 \
       --accelerator cpu --precision 32-true --workers 0 --cpu-threads 2 \
       --epochs 2 --batch-size 8 --max-steps 8 --output /tmp/directclr-example

The planned budget is eight updates. To exercise recovery, first stop at four
updates, read the checkpoint path from ``training_summary.json``, and rerun with
``--resume <checkpoint> --max-steps 8`` on the same two-epoch schedule.
Generated-image accuracy is an execution check rather than a representation score.

ImageNet-100 results, training curves, measured settings and fresh-run commands
are in `benchmarks/imagenet100/RESULTS.md
<https://github.com/galilai-group/stable-pretraining/blob/main/benchmarks/imagenet100/RESULTS.md>`_.
The reference defaults are 360 dimensions and temperature 0.1; the recorded
ImageNet-100 experiments fix 256 dimensions and select temperature on dev.
