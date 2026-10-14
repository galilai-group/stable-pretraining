MMCR: maximum manifold capacity representations
===============================================

MMCR (Yerxa et al., NeurIPS 2023) compresses representations of augmentations
of the same sample while expanding the nuclear norm of the sample centroids.
See the `paper <https://arxiv.org/abs/2303.03307>`_ and
`author implementation <https://github.com/ThomasYerxa/mmcr/blob/db5a25caca93a55583fba5f9647f6eb36fca2efe/mmcr/cifar_stl/loss_mmcr.py>`_.

Loss contract
-------------

``spt.losses.MMCRLoss(lmbda=0.0)`` accepts raw projections shaped ``[B, V, D]``:
sample, view, feature. Views must remain aligned by sample. The loss normalizes
each projection, averages views to obtain centroids, and computes

.. math::

   L = \frac{\lambda}{B}\sum_{i=1}^{B}\|Z_i\|_* - \|C\|_*,
   \qquad C_i = \frac{1}{V}\sum_v Z_{iv}.

The centroid matrix is neither renormalized nor divided by batch size. Negative
losses are expected; values from different batch sizes are not directly comparable.
``lmbda=0`` skips the local SVD and matches the authors' CIFAR/STL default.
FP16/BF16 input is promoted to FP32 for normalization and SVD; FP64 is preserved.
No noise or singular-value smoothing is added. At rank deficiency the nuclear
norm is nonsmooth and autograd selects a subgradient.

Under DDP, differentiable gathering computes the global-batch objective by
default. All ranks must provide equal nonempty ``[B, V, D]`` shapes and enter
the collective together. Use a distributed sampler and consistent view counts.
``gather_distributed=False`` explicitly uses rank-local batches and changes
the objective. DDP averages model gradients; do not additionally multiply or
divide this loss by world size. Gradient accumulation also does not turn
microbatch nuclear norms into the nuclear norm of one larger batch.

Framework integration
---------------------

The class and composable forward share the same dictionary contract::

    import torch
    import stable_pretraining as spt

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    model = spt.MMCR(backbone, torch.nn.Linear(512, 128), lmbda=0.0)
    # Equivalent: spt.Module(forward=spt.forward.mmcr, backbone=backbone,
    #     projector=..., mmcr_loss=spt.losses.MMCRLoss(), optim=...)
    batch = {"views": [{"image": view1, "label": labels},
                       {"image": view2, "label": labels}]}
    state = model(batch, "fit")
    state["loss"].backward()

The forward also accepts named view dictionaries. Each view is encoded
separately, so different crop resolutions are supported if the backbone
returns fixed-size feature vectors. Callback embeddings, raw projections,
and labels are concatenated in view-major order. Loss input is explicitly
stacked in sample-major ``[B, V, D]`` order. Evaluation uses ``model.eval()``
and ``{"image": images}``, returning backbone embeddings without a projector
or loss. Connect an ``OnlineProbe`` and train through ``spt.Manager`` as in
``examples/mmcr_cifar10.py``. Labels are used by the probe, not by MMCR.

Small reproducible experiment
-----------------------------

From the repository root::

    python examples/mmcr_cifar10.py --download --accelerator gpu --epochs 10
    python examples/mmcr_cifar10.py --method simclr --accelerator gpu --epochs 10 --cache-dir simclr-cifar10

The example uses a CIFAR-adapted ResNet-18, two views, a 512-to-512-to-128
projector, AdamW with the Module default cosine scheduler and a detached online
classification probe (micro-averaged top-1 accuracy). It reserves 5,000
training examples for validation using a seeded split; the CIFAR test set is
not used. The baseline uses the same split, augmentations, architecture, seed
and training budget. ``--train-samples`` and ``--val-samples`` optionally
select small subsets for development. ``--fake-data`` avoids downloads.

This is a small development recipe, not a reproduction of the authors'
ResNet-50 / 40-view / 500-epoch CIFAR setup. Online-probe accuracy is not frozen
linear evaluation. A short run checks integration; meaningful performance
claims need a longer matched baseline, frozen evaluation and multiple seeds.
Every run saves an ``experiment-summary.json`` alongside its checkpoints,
including configuration, steps, runtime, peak GPU memory and metrics.

On Apple silicon, select ``--accelerator mps``. With the tested PyTorch 2.9.1
build, SVD requires ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so that unsupported
spectral operations run on the CPU. This is a mixed MPS/CPU execution path;
it is not a native MPS SVD implementation.

Pass ``--save-backbones`` to export ``initial-backbone.pt`` before training
and ``final-backbone.pt`` after training, including BatchNorm buffers. Use a
dedicated, fresh ``--cache-dir`` for each experiment. Snapshot export rejects
existing files to preserve the comparison; when resuming an interrupted run,
retain its original initial snapshot and do not reuse a completed run's cache.
A frozen probe fitted separately on each snapshot with the same held-out split
can distinguish representation improvement from online-probe optimization.

Pass ``--resume PATH_TO_CHECKPOINT --epochs TOTAL_EPOCHS`` with the same
model and experiment arguments to resume. ``--precision bf16-mixed`` keeps
the encoder in mixed precision while the loss's spectral operations use FP32.
Windows defaults to ``--workers 0``, with a guarded main entry point. The
example uses a single device and does not require SLURM, WSL, or Docker.
