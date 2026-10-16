# Flickr30k SigLIP ViT-B/16 — Image–Text Retrieval

Results for `benchmarks/flickr30k/siglip-vitb.py`, which trains and
evaluates `stable_pretraining.methods.SigLIP` on Flickr30k
(`AnyModal/flickr30k`, Karpathy splits: 29,000 train / 1,014 val /
1,000 test images, 5 captions per image).

All runs use the benchmark script as it is in this PR (commit `f0f3cbee`
on the PR branch), seed 0, 2× NVIDIA L40S, DDP, bf16-mixed, and global
batch 512 (256 per GPU), so `SigLIPLoss` gathers text features across both
GPUs. Training goes through the library's `siglip` forward function.

## Retrieval on the 1k test split (1,000 images, 5,000 captions)

Image→text (i2t) counts a hit if any of the image's 5 captions is in the
top K. Text→image (t2i) ranks the 1,000 images for each of the 5,000
captions. rsum is the sum of all six recalls (max 600).

| Mode | Init | Checkpoint | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | rsum |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| zeroshot | `google/siglip-base-patch16-224` | released | 88.6 | 98.0 | 99.2 | 74.6 | 92.4 | 95.5 | 548.2 |
| finetune | `google/siglip-base-patch16-224` | epoch 5 of 5 | **93.7** | **99.4** | **99.7** | **82.4** | **96.3** | **98.2** | **569.7** |
| scratch | random | epoch 100 of 100 | 5.3 | 16.3 | 24.9 | 5.2 | 14.3 | 21.3 | 87.2 |
| scratch | random | epoch 26 (lowest val loss) | 4.6 | 16.4 | 23.0 | 4.1 | 14.2 | 21.3 | 83.6 |

The script evaluates each training run twice: at the last epoch and at the
epoch with the lowest validation loss (all 1,014 val images, first caption
per image). For fine-tuning these are the same epoch. For the scratch run
both are shown. Its validation loss bottoms out at epoch 26 and then rises,
while test retrieval still improves slightly, so validation loss is only a
rough guide to retrieval once the model starts to overfit.

Wall time: fine-tune 3.8 min, scratch 57 min.

## What each mode shows

- **zeroshot** loads the released checkpoint through `SigLIP(pretrained=True)`,
  including its learned logit scale and bias (117.3 and −12.9). This checks
  that the method class wires the Hugging Face towers correctly.
- **finetune** continues training that checkpoint with `SigLIPLoss` under
  DDP. It improves every recall over zero-shot, which shows that the loss,
  the cross-GPU gather, and the training loop produce useful gradients.
- **scratch** trains both towers from random init with the paper's logit
  init (log 10, −10). It is a learning sanity check, not a pretraining
  result: SigLIP was trained on billions of image–text pairs, Flickr30k has
  29k images. R@1 of about 5% is roughly 50× chance (0.1%).

## Hyperparameters

| Mode | lr | Weight decay | Warmup | Crop scale | Grad clip |
|---|---:|---:|---:|---|---:|
| finetune | 1e-5 | 1e-4 | 10% | 0.9–1.0 | 1.0 |
| scratch | 1e-4 | 0.2 | 10% | 0.5–1.0 | 1.0 |

Shared settings:

- AdamW with betas (0.9, 0.95), linear warmup then cosine decay.
- Weight decay applies to weight matrices only. Biases, norm parameters,
  and the logit scale and bias are excluded (`exclude_bias_norm=True`).
- Training images get random resized crops. There are no flips, because
  captions mention left and right. Each step uses one random caption of the
  five per image.
- Images are resized to 224×224 and normalized with the checkpoint's image
  mean and std.
- Text is tokenized with `padding="max_length"` and no attention mask,
  which matches how the SigLIP text tower was trained.

## How the scratch recipe was chosen

Earlier scratch runs used 30 epochs, no augmentation, and weight decay 1e-4:

| lr | rsum | Outcome |
|---:|---:|---|
| 1e-3 | 7.2 | Collapsed to near-identical embeddings (loss stuck at 7.24) |
| 3e-4 | 36.8 | Learned |
| 1e-4 | 46.1 | Learned; validation loss rose after epoch ~12 |

The paper's lr of 1e-3 assumes batches of ~32k. At batch 512 it collapses.
With lr 1e-4 the model overfit early, so the final recipe adds random
resized crops, raises weight decay to 0.2 (with AdamW the decay is
multiplied by the learning rate, so 1e-4 had almost no effect), and trains
for 100 epochs. This moved the lowest validation loss from epoch ~12 to
epoch 26 and nearly doubled rsum (46.1 → 83.6–87.2). These are single-seed
runs, so differences of a few rsum points are within noise.

## Why SigLIP is not in the ImageNet-10 table

`benchmarks/imagenet10/RESULTS.md` trains each method on unlabeled
Imagenette images and scores it with a linear probe and k-NN on 10 classes.
SigLIP needs paired captions, which Imagenette does not have. Building
captions from class names would turn the labels into training data, so the
comparison would not be fair. Retrieval on Flickr30k is the standard
evaluation for image–text methods.

## Reproducing

Single node, 2 GPUs (adjust the Slurm flags to your cluster):

```bash
python benchmarks/flickr30k/siglip-vitb.py --mode zeroshot --num_devices 1
srun --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=6 --mem=128G \
  python benchmarks/flickr30k/siglip-vitb.py --mode finetune --num_devices 2
srun --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=6 --mem=128G \
  python benchmarks/flickr30k/siglip-vitb.py --mode scratch --num_devices 2
```

Each run writes `<output_dir>/<mode>/results.json` with the git SHA, the
hyperparameters, the last-epoch logit scale and bias, and retrieval metrics
for the last epoch (`test`) and the lowest-val-loss epoch (`test_best_val`).
The first run downloads the dataset (~4.3 GB) and the checkpoint (~0.8 GB)
to the Hugging Face cache. Tokenization needs `sentencepiece`.
