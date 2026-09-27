# Add an online probe

An `OnlineProbe` trains a classifier on detached features. The encoder continues
to optimize its SSL objective. The forward function should return `embedding`
and, when views duplicate samples, a correspondingly repeated `label` tensor.

This complete construction can be adapted to the training script from the
[quickstart](quickstart.md):

```python
import torch
from torchmetrics.classification import MulticlassAccuracy
import stable_pretraining as spt

module = spt.Module(
    forward=spt.forward.simclr,
    backbone=torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 16)),
    projector=torch.nn.Linear(16, 8),
    simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
    optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
)
probe = spt.OnlineProbe(
    module,
    name="probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(16, 2),
    loss=torch.nn.CrossEntropyLoss(),
    metrics={"accuracy": MulticlassAccuracy(2)},
)
```

Pass `callbacks=[probe]` to `lightning.Trainer` and run through `spt.Manager`.
The quickstart already does this; its `eval/probe_accuracy` metric comes from
held-out validation images. Feature width must match the probe's input width,
and prediction and label batch lengths must match. An online probe is a training
diagnostic; report a separate, controlled frozen-encoder evaluation when your
research protocol requires it.

The regression suite compares backbone updates with and without evaluation
callbacks. A probe should not change the number of backbone optimizer steps.
