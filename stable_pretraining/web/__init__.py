"""Local, dependency-free web viewer for stable-pretraining runs.

Reads ``sidecar.json`` + ``metrics.csv`` produced by
:class:`stable_pretraining.registry.logger.RegistryLogger` and serves a
wandb-like UI backed by ``http.server`` + Server-Sent Events.  No
FastAPI / uvicorn / inotify: safe over NFS.

Entry point: ``spt web <dir>``.
"""

from .server import serve

__all__ = ["serve"]
