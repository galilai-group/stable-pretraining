# SimCo on CIFAR-10 and CIFAR-100

The table reports the online linear probe's top-1 accuracy on held-out
validation images for one training run per dataset (seed 0).

| Dataset | Epochs | Optimizer steps | Best online validation top-1 | Final online validation top-1 | Training wall time |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 500 | 87,500 | 88.58% | 88.18% | 9,682.42 s (2.69 h) |
| CIFAR-100 | 500 | 87,500 | 60.76% | 60.34% | 10,357.26 s (2.88 h) |

Each run starts from a fresh initialization and partitions the official training
split into 45,000 images for SSL training and 5,000 for validation. The probe
trains on detached backbone features using labels from the training partition.
Best accuracy is the maximum across validation epochs. Final accuracy is
measured at epoch 500. The official test images are held out throughout.

## Configuration

- Encoder: ResNet-18 with the low-resolution stem, initialized without pretrained
  weights; projector dimensions 512 → 128 → 128 with ReLU between linear layers.
- Loss: symmetric SimCo, intra-anchor temperature 0.1 and inter-anchor temperature
  1.0, with global cross-view negatives gathered across ranks.
- Hardware: one node with two NVIDIA RTX A5000 GPUs; DDP and SyncBatchNorm, FP32.
- Global batch size 256, local batch size 128; training drops the final incomplete
  batch. There are 175 optimizer steps per epoch and 87,500 per run.
- Backbone/projector optimizer: SGD, learning rate 0.03, momentum 0.9, weight decay
  0.0005; 10-epoch linear warmup (1,750 steps), followed by cosine decay to zero.
- Online probe: SGD, constant learning rate 0.1, momentum 0.9.
- Python 3.11.17, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, Lightning 2.6.6.

[results.json](results.json) contains full-precision metrics and the SHA256
fingerprints for each run's protocol and split. To repeat either run, launch
[simco_cifar.py](simco_cifar.py) with two processes. The
[SimCo guide](../../../docs/source/guides/simco_cifar.md) gives the exact
commands for both datasets.
