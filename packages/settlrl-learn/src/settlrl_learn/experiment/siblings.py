"""Importing another framework's same-dir helper module.

A framework directory (``experiments/NNNN_slug/``) is a script dir, not a
package, so a helper it owns is only importable once its directory is on
``sys.path``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def sibling_module(framework_dir: Path, name: str) -> ModuleType:
    """Import ``<framework_dir>/<name>.py``, putting the directory on
    ``sys.path`` first (idempotent)."""
    if str(framework_dir) not in sys.path:
        sys.path.insert(0, str(framework_dir))
    return importlib.import_module(name)
