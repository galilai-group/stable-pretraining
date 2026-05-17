# Methods

Full catalog of all SSL methods in `stable-pretraining`.

Methods come in two forms:

- **Forward functions** (`stable_pretraining/forward.py`) — stateless functions used with `spt.Module`. You compose them with a backbone, loss, and callbacks yourself. Best for custom experiments.
- **Method classes** (`stable_pretraining/methods/`) — full `LightningModule` implementations with the backbone, loss, and optimizer pre-wired. Best for reproducing published results quickly.

---

## Complete Method Table

`TeacherStudent†` in the callbacks column means the method requires `TeacherStudentCallback` when using the forward-function approach (EMA teacher updates). Method classes handle this internally.

| Method | Forward fn | Method class | Loss class(es) | Key callbacks | Paper |
|--------|-----------|--------------|----------------|---------------|-------|
| Supervised | `supervised_forward` | — | any | — | — |
| SimCLR | `simclr_forward` | `SimCLR` | `NTXEntLoss` | — | [Chen et al., 2020](https://arxiv.org/abs/2002.05709) |
| BYOL | `byol_forward` | `BYOL` | `BYOLLoss` | `TeacherStudent†` | [Grill et al., 2020](https://arxiv.org/abs/2006.07733) |
| VICReg | `vicreg_forward` | `VICReg` | `VICRegLoss` | — | [Bardes et al., 2022](https://arxiv.org/abs/2105.04906) |
| Barlow Twins | `barlow_twins_forward` | `BarlowTwins` | `BarlowTwinsLoss` | — | [Zbontar et al., 2021](https://arxiv.org/abs/2103.03230) |
| SwAV | `swav_forward` | `SwAV` | `SwAVLoss` | `OnlineQueue` | [Caron et al., 2020](https://arxiv.org/abs/2006.09882) |
| NNCLR | `nnclr_forward` | `NNCLR` | `NTXEntLoss` | `OnlineQueue` | [Dwibedi et al., 2021](https://arxiv.org/abs/2104.14548) |
| DINO | `dino_forward` | `DINO` | `DINOv1Loss` | `TeacherStudent†` | [Caron et al., 2021](https://arxiv.org/abs/2104.14294) |
| DINOv2 | `dinov2_forward` | `DINOv2` | `DINOv2Loss`, `iBOTPatchLoss` | `TeacherStudent†` | [Oquab et al., 2024](https://arxiv.org/abs/2304.07193) |
| BEiT | — | `BEiT` | — | — | [Bao et al., 2022](https://arxiv.org/abs/2106.08254) |
| CMAE | — | `CMAE` | — | — | [Huang et al., 2023](https://arxiv.org/abs/2207.13532) |
| Data2Vec | — | `Data2Vec` | — | `TeacherStudent†` | [Baevski et al., 2022](https://arxiv.org/abs/2202.03555) |
| DINOv3 | — | `DINOv3` | `DINOv2Loss` | `TeacherStudent†` | [Siméoni et al., 2025](https://arxiv.org/abs/2309.16588) |
| iBOT | — | `iBOT` | `DINOv1Loss`, `iBOTPatchLoss` | `TeacherStudent†` | [Zhou et al., 2022](https://arxiv.org/abs/2111.07832) |
| iGPT | — | `iGPT` | — | — | [El-Nouby et al., 2024](https://arxiv.org/abs/2401.08541) |
| IJEPA | — | `IJEPA` | — | — | [Assran et al., 2023](https://arxiv.org/abs/2301.08243) |
| LeJEPA | — | `LeJEPA` | — | — | [Balestriero & LeCun, 2025](https://arxiv.org/abs/2511.08544) |
| MAE | — | `MAE` | `MAELoss` | — | [He et al., 2022](https://arxiv.org/abs/2111.06377) |
| MaskFeat | — | `MaskFeat` | — | — | [Wei et al., 2022](https://arxiv.org/abs/2112.09133) |
| MIMRefiner | — | `MIMRefiner` | `DINOv1Loss`, `iBOTPatchLoss` | — | [Lehner et al., 2024](https://arxiv.org/abs/2402.10093) |
| MoCov2 | — | `MoCov2` | `NTXEntLoss` | — | [Chen et al., 2020](https://arxiv.org/abs/2003.04297) |
| MoCov3 | — | `MoCov3` | `NTXEntLoss` | — | [Chen et al., 2021](https://arxiv.org/abs/2104.02057) |
| MSN | — | `MSN` | — | — | [Assran et al., 2022](https://arxiv.org/abs/2204.07141) |
| NEPA | — | `NEPA` | — | — | — |
| PIRL | — | `PIRL` | — | — | [Misra & van der Maaten, 2020](https://arxiv.org/abs/1912.01991) |
| SALT | — | `SALT` | — | — | [Li et al., 2025](https://arxiv.org/pdf/2509.24317) |
| SimMIM | — | `SimMIM` | — | — | [Xie et al., 2022](https://arxiv.org/abs/2111.09886) |
| SimSiam | — | `SimSiam` | — | — | [Chen & He, 2021](https://arxiv.org/abs/2011.10566) |
| TiCO | — | `TiCO` | — | — | [Zhu et al., 2022](https://arxiv.org/abs/2206.10698) |
| VICRegL | — | `VICRegL` | `VICRegLoss` | — | [Bardes et al., 2022](https://arxiv.org/abs/2210.01571) |
| WMSE | — | `WMSE` | — | — | [Ermolov et al., 2021](https://arxiv.org/abs/2007.06346) |

---

## Using Forward Functions

Forward functions are stateless and work with `spt.Module`. You own the backbone, loss, and callbacks.

```python
from stable_pretraining import Module
from stable_pretraining.forward import simclr_forward
from stable_pretraining.losses import NTXEntLoss
from stable_pretraining.backbone import from_torchvision

module = Module(
    forward=simclr_forward,
    backbone=from_torchvision("resnet50"),
    projector=...,
    simclr_loss=NTXEntLoss(temperature=0.1),
)
```

All forward functions are importable from `stable_pretraining.forward`. They can also be
referenced by dotted path in YAML configs:

```yaml
module:
  _target_: stable_pretraining.Module
  forward: stable_pretraining.forward.simclr_forward
```

---

## Using Method Classes

Method classes are full `LightningModule` subclasses. All are importable from
`stable_pretraining.methods`:

```python
from stable_pretraining.methods import SimCLR, BYOL, DINO, MAE

model = SimCLR(backbone=backbone, projector=projector, temperature=0.1)
```

Each method file (`stable_pretraining/methods/{method}.py`) contains a runnable example
config in its module docstring.

---

## Loss Classes

All loss classes are importable from `stable_pretraining.losses`:

| Loss | Methods that use it |
|------|-------------------|
| `NTXEntLoss` | SimCLR, NNCLR, MoCov2, MoCov3 |
| `BYOLLoss` | BYOL |
| `VICRegLoss` | VICReg, VICRegL |
| `BarlowTwinsLoss` | Barlow Twins |
| `SwAVLoss` | SwAV |
| `DINOv1Loss` | DINO, iBOT, MIMRefiner |
| `DINOv2Loss` | DINOv2, DINOv3 |
| `iBOTPatchLoss` | DINOv2, iBOT, MIMRefiner |
| `MAELoss` | MAE |
| `CLIPLoss` | Multimodal CLIP-style objectives |
| `NegativeCosineSimilarity` | SimSiam, general use |
