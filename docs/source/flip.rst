FLIP: masked image-text pretraining
===================================

FLIP :cite:`li2023scaling` removes random image patches before transformer
attention and trains paired image/text embeddings with a contrastive loss.
It reuses ``MaskedEncoder``, ``PatchMasking``, and ``CLIPLoss``.

Model interface
---------------

Supply a text encoder returning pooled features of shape ``[batch, text_dim]``.
Its ``forward`` must accept ``input_ids`` and, when present in the batch,
``attention_mask`` as keyword arguments. Tokenization belongs to the data pipeline.

.. code-block:: python

    import stable_pretraining as spt

    model = spt.FLIP(
        text_encoder=text_encoder,
        text_dim=512,
        encoder_name="vit_base_patch16_224",
        mask_ratio=0.5,
        optim={"optimizer": {"type": "AdamW", "lr": 3e-4}},
    )
    outputs = model({"image": images, "input_ids": tokens}, stage="fit")

Training requires paired rows. Outputs include ``loss``, normalized projected
``image_embedding`` and ``text_embedding``, and pre-projection image ``embedding``
for probing callbacks. Image pooling averages all encoded tokens, including CLS.
The learned similarity scale is bounded between 1 and 100.

Use ``spt.Manager`` with a ``spt.data.DataModule`` for training. The composable
``spt.forward.flip`` exposes the same logic for ``spt.Module`` and YAML configs;
its API documentation lists the required components.

Call ``model.eval()`` for full-image inference. Evaluation accepts either modality
independently; paired evaluation also returns loss. ``stage`` controls logging,
while module mode controls masking. ``encode_image`` and ``encode_text`` preserve
gradient mode, so use ``torch.inference_mode()`` for retrieval. To train on full
images, set ``mask_ratio=0`` and retain training mode.

Checkpoint loading requires supplying ``text_encoder`` again. If the image
encoder was passed as a module, also supply it as ``encoder_name``; named image
encoders are recorded in the saved hyperparameters.

Use standard timm ViTs with absolute positional embeddings. Masking retains
``N - int(N * mask_ratio)`` patches, rounding fractional counts upward. This
matches the reference at 50% and 75% of 196 patches, but differs from its downward
rounding otherwise. Distributed training requires equal local batch sizes
(``drop_last=True``) and correctly paired rows. Avoid duplicate images within a
global batch: off-diagonal pairs are treated as negatives.

COCO 2017 example
-----------------

Download the official COCO 2017 images and captions from
https://cocodataset.org/#download and arrange::

    coco2017/
        train2017/*.jpg
        val2017/*.jpg
        annotations/captions_train2017.json
        annotations/captions_val2017.json

``examples/flip.py`` trains one model per invocation on 117,287 ``train2017``
images, holds out 1,000 development images, and evaluates the final model on all
5,000 ``val2017`` images and their 25,014 captions. Each training row samples one
caption. Caption choices and augmentations are keyed by seed, epoch, and image ID,
independently of masking, rank, and worker count. With seed 0, the development IDs
are positions 20,000--20,999 in the shuffled, initially sorted training IDs,
preserving the original pilot's split. Training uses the sorted remaining IDs.
Retrieval uses every associated caption and the entire evaluation candidate pool.
Recall@K is the percentage of queries with any correct match in the top K results;
recall values from different candidate-pool sizes are not directly comparable.

Both encoders train from scratch: ViT-Small/16 and a six-layer, 256-wide,
eight-head bidirectional text transformer with first-token pooling and
256-dimensional projections. The tokenizer is
``bert-base-uncased`` with length 32 and no added CLS/SEP tokens. Padding participates
in attention. The smaller models, tokenizer vocabulary, and PyTorch initialization
differ from the
`reference recipe <https://github.com/facebookresearch/flip>`_.

Defaults are 100 epochs and one device with 256 pairs per device. AdamW uses
betas ``(0.9, 0.95)`` and weight decay 0.2, excluding biases, normalization,
positions, and special tokens. The learning rate warms up from ``1e-6`` to
``1e-4`` over 5% of updates, then follows cosine decay to ``1e-6`` on the final
update. Gradients are clipped to norm 1. Incomplete training batches are dropped
without padding the distributed sampler with duplicate images.

Run the reported four-GPU protocol (global batch 1,024, 114 updates per epoch)
with 50% masking::

    python examples/flip.py --data /path/to/coco2017 \
        --output /path/to/runs/flip-0.5 --mask-ratio 0.5 --devices 4

Use ``--mask-ratio 0`` for an unmasked CLIP baseline or ``0.75`` for 75% masking.
Omitting ``--devices 4`` uses one device and reduces the global batch to 256.
For a short training check, use ``--train-size 16 --dev-size 8 --batch-size 4
--epochs 1 --workers 0``; final evaluation still uses all of ``val2017``.
Development loss is checked every 10 epochs. After training, rank zero evaluates
the unmasked final model in FP32 and writes bidirectional retrieval to
``metrics.json`` under ``--output``. Manager provides the run registry and
checkpoints. Choose settings using development data before running final
``val2017`` evaluation.

The example uses pinned memory and persistent workers; BF16 is enabled
automatically on supported GPUs, with contrastive loss computed in FP32.
Workers return resized
``uint8`` images; the DataModule batch-transform hook converts and normalizes
them after transfer. Adjust ``--workers`` for the available CPU allocation.

The four-GPU command follows the training and final evaluation recipe below.
The reported measurements came from the archived experiment; the example uses
the library's Manager/Lightning lifecycle. Hardware, software versions, and
floating-point execution can affect results, so identical scores are not guaranteed.

Tests cover masking, numerical loss computation, gradients, training integration,
checkpoint restoration, and distributed correctness. The public example was
compared against the archived protocol and validated with four-process training
and a separate BF16 Blackwell GPU smoke test, including full val2017 evaluation.
The final example was not rerun for the full 100-epoch experiments.

Validation
----------

Matched seed-0 COCO experiments trained CLIP, FLIP 50%, and FLIP 75% from scratch
for 100 epochs (11,673,600 pairs each), using ViT-Small/16 and a six-layer
width-256 text encoder. Training used 117,287 images, global batch 1,024, BF16,
four NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs, and identical initialization,
data streams, and optimizer settings. The image encoder processed 197, 99, and
50 tokens respectively, including CLS. All three fixed final checkpoints were
evaluated unmasked on all 5,000 val2017 images and 25,014 captions.

.. csv-table:: Final COCO retrieval (percent)
   :header: "Model", "Image-to-text R@1 / R@5 / R@10", "Text-to-image R@1 / R@5 / R@10"

   CLIP, 14.28 / 34.84 / 46.92, 10.25 / 28.42 / 39.84
   FLIP 50%, 14.72 / 34.96 / 46.54, 10.22 / 28.28 / 39.68
   FLIP 75%, 12.84 / 33.30 / 44.94, 9.43 / 26.78 / 37.74

FLIP 50% achieved similar retrieval scores to CLIP. At 75% masking,
image-to-text R@10 decreased by 1.98 percentage points and text-to-image R@10
by 2.10 points relative to CLIP.

Mean bidirectional R@1 was 12.27%, 12.47%, and 11.14% for CLIP, FLIP 50%, and
FLIP 75%, respectively. Peak allocated memory per GPU was 9.21, 5.36, and
3.40 GiB. Training processes took 40.97, 40.09, and 41.06 minutes, including
development checks and checkpointing but excluding final retrieval evaluation.
These are single-seed, reduced-scale validation results, not a reproduction of
the paper's large-scale LAION experiments. Validation scores had been inspected
in earlier experiments; the fixed final checkpoints were not selected using
these scores.

Single-GPU update benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A separate benchmark used the same model sizes on one Blackwell Max-Q GPU with
batch size 256, BF16, and fused AdamW. Synthetic images and cached COCO captions
were resident on the GPU. Timings include forward, contrastive loss, backward,
gradient clipping, and optimizer updates; they exclude decoding, loading, and
transfer. Each result averages two runs in reversed order, each with 20 warmup
and 100 measured updates, synchronizing only at timing boundaries.

.. csv-table:: GPU-resident update performance
   :header: "Model", "Time/update (ms)", "Pairs/s", "Speedup", "Peak allocated memory (GiB)"

   CLIP, 83.6, 3061, 1.00x, 9.08
   FLIP 50%, 42.8, 5988, 1.96x, 5.23
   FLIP 75%, 26.3, 9744, 3.18x, 3.27

Peak allocated memory decreased by approximately 42% at 50% masking and 64% at
75% masking relative to CLIP.

Single-GPU input-pipeline profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Short diagnostic runs used the FLIP implementation with the archived data
pipeline on one Blackwell Max-Q GPU: batch size 256, BF16, standard AdamW,
eight workers, and node-local COCO images. Each result averages two runs
in reversed order, each with 20 warmup and 80 measured updates. These measurements
cover the update loop, including data loading and transfer, rather than a full
example run.

.. csv-table:: CLIP versus FLIP 75% update times
   :header: "Input preparation", "CLIP (ms/update)", "FLIP 75% (ms/update)", "Speedup"

   Live COCO with CPU normalization, 116.8, 114.6, 1.02x
   Live COCO with GPU normalization, 99.9, 101.6, 0.98x
   Repeated prepared COCO batch, 87.6, 29.5, 2.97x

The prepared-input control repeatedly reused one real batch from pinned
CPU memory, still transferring it on every update. It measures capacity with
image preparation removed, not full-dataset training throughput with fresh
augmentations. All three cases omit per-update diagnostic synchronization and
run without distributed communication.

Live COCO updates took nearly the same time for both models, while prepared
input enabled a 2.97x speedup. These results suggest that input preparation is a
major bottleneck. Masking happens after image loading, decoding, and augmentation,
so those costs remain. GPU normalization improves both models' absolute times
but does not restore a relative speedup. Decoding, augmentation, and worker
scheduling were not isolated individually; further input-pipeline optimization
was outside this validation.
