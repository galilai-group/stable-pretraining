stable_pretraining.web
======================
.. module:: stable_pretraining.web
.. currentmodule:: stable_pretraining.web

Local, dependency-free web viewer for spt runs. Reads ``sidecar.json`` +
``metrics.csv`` (and optional ``media.jsonl``) produced by
:class:`~stable_pretraining.registry.RegistryLogger` and serves a
wandb-like UI backed by Python's stdlib ``http.server`` + Server-Sent
Events. NFS-safe by design (mtime polling, no inotify), no external
dependencies (uPlot is loaded from a CDN).

Launch with ``spt web`` — see :doc:`../cli` for command-line flags.
The Python entry point :func:`serve` is exposed for embedding in custom
scripts (e.g. launching alongside training).

Entry point
-----------

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst

   serve

Internals
---------

These are useful when extending the viewer (custom endpoints, alternate
scanners) but normal users only need :func:`serve`.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   scan.RunScanner
