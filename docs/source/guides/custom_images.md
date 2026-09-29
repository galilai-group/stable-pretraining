# Train SimCLR on your images

Start from the source installation in [the quickstart](quickstart.md).
The runnable example accepts ImageFolder directories with matching class names:

```text
images/
  train/
    class_a/001.jpg
    class_b/002.jpg
  val/
    class_a/101.jpg
    class_b/102.jpg
```

Keep validation images separate. Provide at least eight training images and two
classes for this example's classification probe. Labels train the probe only;
SimCLR uses the paired views, not the class labels.

```bash
python -m stable_pretraining.quickstart --data ./images --epochs 2 --cache-dir ./image-runs
spt web ./image-runs/runs
```

The example intentionally uses 16x16 crops, batch size 8, and a small CPU encoder
for a fast first run. Before a research experiment, copy the starter and choose
the crop size, encoder, augmentation, batch size, and training duration for your
data. The quickstart's hyperparameters are not an accuracy baseline.

Data wiring uses the library's existing adapters:

```python
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import stable_pretraining as spt

root = Path("images")
t = spt.data.transforms
augment = t.Compose(
    t.RandomResizedCrop((16, 16)),
    t.RandomHorizontalFlip(),
    t.ToImage(scale=True),
)
dataset = spt.data.FromTorchDataset(
    ImageFolder(root / "train"),
    names=["image", "label"],
    transform=t.MultiViewTransform([augment, augment]),
)
loader = DataLoader(dataset, batch_size=8, drop_last=True)
batch = next(iter(loader))
assert len(batch["views"]) == 2
assert batch["views"][0]["image"].shape == (8, 3, 16, 16)
```

`Compose` takes transforms as positional arguments; `MultiViewTransform` takes
a list (or a dict of named views). Default PyTorch collation preserves the
`{"views": [{"image": ..., "label": ...}, ...]}` structure used by
`spt.forward.simclr`. For your own objective, inspect an actual collated batch
before writing the forward function.
