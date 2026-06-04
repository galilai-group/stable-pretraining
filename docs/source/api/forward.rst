stable_pretraining.forward
===========================

.. module:: stable_pretraining.forward
.. currentmodule:: stable_pretraining.forward

Forward functions define the core training logic for different self-supervised learning methods. They are called during the forward pass of the :class:`~stable_pretraining.Module` and handle how data flows through the model, compute losses, and return outputs.

Overview
--------

Forward functions are the heart of each SSL method implementation. They:

- Take a batch of data and training stage as input
- Process data through backbone and projection heads
- Compute method-specific losses during training
- Return a dictionary containing loss and embeddings

The forward function is bound to the Module instance at initialization, giving it access to all module attributes (backbone, projector, loss functions, etc.).

Usage in Config
---------------

Forward functions can be specified in YAML configs as string references:

.. code-block:: yaml

    module:
      _target_: stable_pretraining.Module
      forward: stable_pretraining.forward.simclr
      backbone: ...
      projector: ...
      simclr_loss: ...

Or in Python code:

.. code-block:: python

    from stable_pretraining import Module
    from stable_pretraining.forward import simclr

    module = Module(
        forward=simclr,
        backbone=backbone,
        projector=projector,
        simclr_loss=loss_fn
    )

Available Forward Functions
---------------------------

SimCLR
~~~~~~

.. autofunction:: simclr

**Required Module Attributes:**

- ``backbone``: Feature extraction network
- ``projector``: Projection head for embedding transformation
- ``simclr_loss``: NTXent contrastive loss function

**Example Config:**

.. code-block:: yaml

    module:
      forward: stable_pretraining.forward.simclr
      backbone:
        _target_: stable_pretraining.backbone.from_torchvision
        model_name: resnet50
      projector:
        _target_: torch.nn.Sequential
        _args_:
          - _target_: torch.nn.Linear
            in_features: 2048
            out_features: 2048
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Linear
            in_features: 2048
            out_features: 128
      simclr_loss:
        _target_: stable_pretraining.losses.NTXEntLoss
        temperature: 0.5

NNCLR
~~~~~

.. autofunction:: nnclr

**Required Module Attributes:**

- ``backbone``: Feature extraction network
- ``projector``: Projection head for embedding transformation
- ``predictor``: Prediction head for the online network
- ``nnclr_loss``: NTXent contrastive loss function

**Key Features:**

- Uses a support set of past embeddings to find nearest-neighbor positives.
- Encourages semantic similarity, going beyond instance-level discrimination.
- Requires an ``OnlineQueue`` callback with a matching ``key``.

**Example Config:**

.. code-block:: yaml

    module:
      forward: stable_pretraining.forward.nnclr
      backbone:
        _target_: stable_pretraining.backbone.from_torchvision
        model_name: resnet18
      projector:
        _target_: torch.nn.Sequential
        _args_:
          - _target_: torch.nn.Linear
            in_features: 512
            out_features: 2048
          - _target_: torch.nn.BatchNorm1d
            num_features: 2048
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Linear
            in_features: 2048
            out_features: 256
      predictor:
        _target_: torch.nn.Sequential
        _args_:
          - _target_: torch.nn.Linear
            in_features: 256
            out_features: 4096
          - _target_: torch.nn.BatchNorm1d
            num_features: 4096
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Linear
            in_features: 4096
            out_features: 256
      nnclr_loss:
        _target_: stable_pretraining.losses.NTXEntLoss
        temperature: 0.5
      hparams:
        support_set_size: 16384
        projection_dim: 256

    callbacks:
      - _target_: stable_pretraining.callbacks.OnlineQueue
        key: nnclr_support_set
        queue_length: ${module.hparams.support_set_size}
        dim: ${module.hparams.projection_dim}

BYOL
~~~~

.. autofunction:: byol

**Required Module Attributes:**

- ``backbone``: Online network backbone
- ``projector``: Online network projector
- ``predictor``: Online network predictor
- ``target_backbone``: Target network backbone (momentum encoder)
- ``target_projector``: Target network projector

**Key Features:**

- Uses momentum encoder for target network
- No negative pairs required
- MSE loss between predictions and targets

**Example Config:**

.. code-block:: yaml

    module:
      forward: stable_pretraining.forward.byol
      backbone: ...
      projector: ...
      predictor:
        _target_: torch.nn.Sequential
        _args_:
          - _target_: torch.nn.Linear
            in_features: 256
            out_features: 4096
          - _target_: torch.nn.BatchNorm1d
            num_features: 4096
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Linear
            in_features: 4096
            out_features: 256

VICReg
~~~~~~

.. autofunction:: vicreg

**Required Module Attributes:**

- ``backbone``: Feature extraction network
- ``projector``: Projection head
- ``vicreg_loss``: VICReg loss (variance + invariance + covariance)

**Key Features:**

- Variance regularization to maintain information
- Invariance to augmentations
- Covariance regularization to decorrelate features

**Example Config:**

.. code-block:: yaml

    module:
      forward: stable_pretraining.forward.vicreg
      backbone: ...
      projector: ...
      vicreg_loss:
        _target_: stable_pretraining.losses.VICRegLoss
        sim_weight: 25.0
        var_weight: 25.0
        cov_weight: 1.0

Barlow Twins
~~~~~~~~~~~~

.. autofunction:: barlow_twins

**Required Module Attributes:**

- ``backbone``: Feature extraction network
- ``projector``: Projection head
- ``barlow_loss``: Barlow Twins loss function

**Key Features:**

- Reduces redundancy between embedding components
- Makes cross-correlation matrix close to identity
- No negative pairs or momentum encoder needed

**Example Config:**

.. code-block:: yaml

    module:
      forward: stable_pretraining.forward.barlow_twins
      backbone: ...
      projector: ...
      barlow_loss:
        _target_: stable_pretraining.losses.BarlowTwinsLoss
        lambda_: 0.005

Supervised
~~~~~~~~~~

.. autofunction:: supervised

**Required Module Attributes:**

- ``backbone``: Feature extraction network
- ``classifier``: Classification head

**Key Features:**

- Standard supervised learning
- Cross-entropy loss for classification
- Useful for baseline comparisons

**Example Config:**

.. code-block:: yaml

    module:
      forward: stable_pretraining.forward.supervised
      backbone: ...
      classifier:
        _target_: torch.nn.Linear
        in_features: 2048
        out_features: 1000

Custom Forward Functions
------------------------

You can create custom forward functions for new SSL methods:

.. code-block:: python

    def custom_ssl_forward(self, batch, stage):
        """Custom SSL method forward function.

        Args:
            self: Module instance with access to all attributes
            batch: Dict containing 'image' and other data
            stage: One of 'train', 'val', or 'test'

        Returns:
            Dict with at least 'loss' key during training
        """
        out = {}

        # Extract features
        out["embedding"] = self.backbone(batch["image"])

        if self.training:
            # Your custom SSL logic here
            proj = self.projector(out["embedding"])

            # Compute custom loss
            out["loss"] = self.custom_loss(proj)

        return out

**Requirements for Custom Functions:**

1. **Signature**: Must accept ``(self, batch, stage)``
2. **Return**: Dictionary with ``"loss"`` key during training (this is the only hardcoded requirement)
3. **Training Mode**: Check ``self.training`` or ``stage == "train"``
4. **Outputs**: You can return any other keys you want (embeddings, projections, logits, etc.) with any names you choose

**Important Note on Output Keys:**

The keys you use in the output dictionary (like ``"embedding"``, ``"logits"``, etc.) are not hardcoded requirements, but they serve as references for callbacks. For example:

- If you return ``out["embedding"]``, callbacks can access it via ``outputs["embedding"]``
- If you return ``out["features"]``, callbacks would access ``outputs["features"]``
- The OnlineProbe callback expects its ``input`` parameter to match one of your output keys

This allows flexible integration between your forward function and various callbacks.

**Example: Connecting Forward Outputs to Callbacks**

.. code-block:: yaml

    module:
      forward: my_custom_forward
      # Forward function returns: {"loss": ..., "my_features": ..., "my_projection": ...}

    callbacks:
      - _target_: stable_pretraining.callbacks.OnlineProbe
        name: probe1
        input: my_features  # References the "my_features" key from forward output
        target: label

      - _target_: stable_pretraining.callbacks.OnlineKNN
        name: knn1
        input: my_projection  # References the "my_projection" key from forward output
        target: label

The callback's ``input`` parameter must match the key name you chose in your forward function

Integration with Module
-----------------------

The forward function becomes the ``forward`` method of the Module:

.. code-block:: python

    class Module(pl.LightningModule):
        def __init__(self, forward, **kwargs):
            # Bind the forward function to this instance
            self.forward = forward.__get__(self, self.__class__)

        def training_step(self, batch, batch_idx):
            # Forward function is called here
            state = self(batch, stage="train")
            return state["loss"]

This design allows maximum flexibility while keeping the implementation clean and modular.

See Also
--------

- :doc:`losses` - Loss functions used by forward functions
- :doc:`module` - The Module class that uses forward functions
- :doc:`data` - Data utilities including ``fold_views``
