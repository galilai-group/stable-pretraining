:orphan:

Examples
========

Configuration examples for stable-pretraining.


Bayesian Hyperparameter Search with Optuna
------------------------------------------

Sweeping over a search space is very easy with Hydra and Optuna.
``hp_search.yaml`` provides an example configuration for performing
hyperparameter optimization using Optuna's TPE sampler (bayesian
optimization).

First, make sure to install Optuna if you haven't already:

.. code-block:: bash

    pip install optuna

Then, register the ``HPMetricLogger`` callback so the metric you want
Optuna to optimise gets logged correctly. More complex logic can also be
implemented in this callback.

.. code-block:: python

    from spt.callbacks import HPMetricLogger

    callbacks = [HPMetricLogger(metric_name="eval/some_metric")]

Finally, make sure your train script returns the ``hp_metric`` to Optuna:

.. code-block:: python

    ...
    manager = spt.Manager(...)
    manager()

    if hasattr(module, "hp_metric"):
        result = module.hp_metric.item()
        if np.isnan(result):
            logger.warning("HP Metric is NaN, returning inf for optimization.")
            result = float("inf")
        logger.info(f"HP Metric: {result}")
        return result

Now you can run the hyperparameter search and it will automatically run
multiple trials:

.. code-block:: bash

    python train.py --config-name=hydra_hp_search

It is recommended to use the ``EarlyStopping`` callback in combination
with hyperparameter optimization to avoid wasting resources on bad
trials.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Retrieve run data from wandb using the stable_pretraining library.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_wandb_reader_thumb.png
    :alt:

  :doc:`/auto_examples/wandb_reader`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reading WandB Runs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This mirrors the torch examples/ scripts but uses the parallel JAX backend (``stable_pretraining.jax``). The forward-function shape, the dict-flow, the callbacks, and the Lightning-style trainer hooks are all the same as on the torch path — only the engine underneath is JAX-native (functional grads + optax). Run with::">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_jax_simclr_minimal_thumb.png
    :alt:

  :doc:`/auto_examples/jax_simclr_minimal`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Minimal SimCLR run on the JAX / Flax-NNX backend.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use stable-SSL to train a supervised model on CIFAR10 with class imbalance.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_imbalance_supervised_learning_thumb.png
    :alt:

  :doc:`/auto_examples/imbalance_supervised_learning`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Imbalanced Supervised Learning</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Retrieve run data from wandb using the stable_pretraining library and create plots from it.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_wandb_figures_thumb.png
    :alt:

  :doc:`/auto_examples/wandb_figures`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting from WandB Runs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to train models using supervised learning with stable-pretraining, including support for various datasets like ImageNet-10, ImageNet-100, and ImageNet-1k.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_supervised_learning_thumb.png
    :alt:

  :doc:`/auto_examples/supervised_learning`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Supervised Learning Example</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduce the two iconic DINO pictures — the patch-token feature PCA and the CLS self-attention maps — on a pretrained DINO ViT, using the PCATokenVisualizer and AttentionVisualizer callbacks.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_dino_visualization_thumb.png
    :alt:

  :doc:`/auto_examples/dino_visualization`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DINO feature PCA & CLS-attention visualisation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Design choice (see the JAX backend docs): augmentation stays on CPU using the existing torchvision pipeline and we feed NHWC numpy arrays into the JAX trainer. Nothing in data/transforms.py is reimplemented — the array boundary is the clean seam between the two backends.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_jax_simclr_imagenette_thumb.png
    :alt:

  :doc:`/auto_examples/jax_simclr_imagenette`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SimCLR on ImageNette with the JAX / Flax-NNX backend, data-parallel over GPUs.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Train probes attached to multiple layers of a frozen backbone to monitor representation quality across depth.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_multi_layer_probe_thumb.png
    :alt:

  :doc:`/auto_examples/multi_layer_probe`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-layer Probe for Vision Models</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A reference recipe for from-scratch supervised ViT classification that targets the &quot;typical&quot; ~82%+ top-1 (ViT-L; ViT-e is a scale stress-test, see note below), built to run efficiently and shard cleanly:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_imagenet1k_supervised_vit_fsdp2_thumb.png
    :alt:

  :doc:`/auto_examples/imagenet1k_supervised_vit_fsdp2`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Supervised ImageNet-1k ViT training — SOTA (DeiT/AugReg) recipe, FSDP2, GPU-fast.</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/wandb_reader
   /auto_examples/jax_simclr_minimal
   /auto_examples/imbalance_supervised_learning
   /auto_examples/wandb_figures
   /auto_examples/supervised_learning
   /auto_examples/dino_visualization
   /auto_examples/jax_simclr_imagenette
   /auto_examples/multi_layer_probe
   /auto_examples/imagenet1k_supervised_vit_fsdp2

# An editable SSL experiment

Install the current stable-pretraining source checkout first; the packaged
quickstart is unreleased. Copy this directory into your own project, then extract
the runnable example from that installed package:

```bash
python -c "from importlib.resources import files; from pathlib import Path; Path('train.py').write_text(files('stable_pretraining').joinpath('quickstart.py').read_text())"
python train.py --cache-dir ./runs-cache
```

This gives you a standalone, editable script rather than a new training engine.
It delegates training, online probing, checkpoints, and registry logging to
stable-pretraining. Modify its backbone or forward callable for your research.

```bash
python train.py --data ./images --epochs 2 --cache-dir ./runs-cache
spt web ./runs-cache/runs
```

The image directory needs `train/<class>/*` and `val/<class>/*`. The script's
16x16 crops and tiny encoder are for onboarding. Select an appropriate research
configuration before treating results as a benchmark. Save the installed
package version/source revision with your experiment.

The accompanying `AGENTS.md` is opt-in project guidance. Remove or adapt it if
this project does not use stable-pretraining.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
