stable_pretraining.module
=========================
.. module:: stable_pretraining.module
.. currentmodule:: stable_pretraining.module

The module provides the main PyTorch Lightning module class for self-supervised learning. This is the core component that handles all training orchestration - you only need to implement the **forward** method!

Core Module
-----------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   Module

User Implementation
-------------------

The key insight of `stable-pretraining` is **simplicity**: you only need to implement the `forward` method. Everything else (optimizers, schedulers, training loops, logging) is handled automatically.

**Required Implementation:**

.. code-block:: python

    def forward(self, batch, stage):
        # Your custom logic here
        batch["embedding"] = self.backbone(batch["image"])

        if self.training:
            # Training-specific logic
            proj = self.projector(batch["embedding"])
            views = spt.data.fold_views(proj, batch["sample_idx"])
            batch["loss"] = self.simclr_loss(views[0], views[1])

        return batch

**Module Creation:**

.. code-block:: python

    module = spt.Module(
        backbone=backbone,           # Your model components
        projector=projector,         # Any kwargs become self.attributes
        forward=forward,            # Your forward function
        simclr_loss=spt.losses.NTXEntLoss(temperature=0.1),
    )

**What's Handled Automatically:**

- ✅ **Optimizer Configuration**: Default AdamW with CosineAnnealingLR scheduler
- ✅ **Training Loop**: Automatic gradient accumulation, clipping, and stepping
- ✅ **Stage Management**: Training/validation/test/predict stages
- ✅ **Metrics**: Automatic metric logging and computation
- ✅ **Callbacks**: Integration with all stable-pretraining callbacks
- ✅ **Logging**: Rich logging and monitoring

**Key Features:**

- **Dictionary-based**: Input and output are dictionaries for maximum flexibility
- **Stage-aware**: The `stage` parameter tells you if you're in training/validation/test/predict
- **Loss-driven**: Include `"loss"` key for training, omit for evaluation-only
- **Automatic optimization**: Set `optim=False` if you don't need training
- **Flexible components**: Any kwargs become module attributes accessible via `self`

**Example Use Cases:**

- **Self-supervised learning**: Implement contrastive losses
- **Supervised learning**: Standard classification/regression
- **Multi-task learning**: Multiple losses and outputs
- **Evaluation-only**: No loss key for inference

The `Module` class is designed to be the **only** component you need to understand for implementing any SSL algorithm! 🎯
