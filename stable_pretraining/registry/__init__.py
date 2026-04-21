# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Filesystem-backed run registry.

Every training run writes a ``sidecar.json`` + ``heartbeat`` into its
run directory; there is no server and no network I/O.  A scanner
(``spt registry scan`` or the implicit lazy scan in
:func:`open_registry`) turns the sidecars into a SQLite cache for fast
querying.
"""

from .logger import RegistryLogger
from .query import Registry, RunRecord, open_registry

__all__ = ["RegistryLogger", "Registry", "RunRecord", "open_registry"]
