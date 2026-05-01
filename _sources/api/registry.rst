stable_pretraining.registry
===========================
.. module:: stable_pretraining.registry
.. currentmodule:: stable_pretraining.registry

Filesystem-backed run registry. Every training run writes a small
``sidecar.json`` next to its CSV logs; a separate scanner indexes those
sidecars into a SQLite cache for fast queries. No SQLite server, no
network — just files under ``cache_dir``.

Logger
------

The Lightning logger that writes the sidecar (auto-injected by
:class:`~stable_pretraining.manager.Manager`).

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   RegistryLogger

In addition to the standard ``log_metrics`` / ``log_hyperparams`` Lightning
hooks, ``RegistryLogger`` exposes ``log_image(key, images, step=…, caption=…)``
and ``log_video(key, videos, step=…, caption=…, fps=…)``. These match
:class:`~lightning.pytorch.loggers.WandbLogger`'s signatures, so existing
callbacks that call ``trainer.logger.log_image(...)`` start writing to disk
without any code change. Files land under ``{run_dir}/media/<safe_tag>/``
and each event is appended to ``{run_dir}/media.jsonl`` for indexing.

Query API
---------

Read-only interface to the indexed runs — used by
:doc:`spt registry <../cli>` and any custom analysis scripts.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   Registry
   RunRecord

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst

   open_registry
