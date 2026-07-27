"""Make the experiment frameworks importable from the smoke tests.

Each ``NNNN_slug/`` is a script directory, not a package. The shared harness
(``Run`` / ``start_run`` / ``Config``) is a normal import from
``settlrl_learn.experiment``; what needs help is each framework's *same-dir*
helpers (``value_fitting`` / ``data`` / ``models`` / ...). ``load_run`` loads a
``run.py`` by path, so this conftest puts every framework directory on
``sys.path`` for those intra-framework imports to resolve.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import jax
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]

# Persist XLA compilations across runs (~/.cache/jax-settlrl unless overridden):
# the smokes are dominated by JIT compilation, so a warm cache (CI restores it)
# turns the repeated `evaluate` traces into cache hits. Under `-n auto`, force
# the xdist workers onto CPU — one JAX process per core each initialising the
# GPU breaks CUDA init for the rest (mirrors the engine/agents conftests).
if "JAX_PLATFORMS" not in os.environ and "PYTEST_XDIST_WORKER" in os.environ:
    jax.config.update("jax_platforms", "cpu")  # type: ignore[no-untyped-call]
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    jax.config.update(  # type: ignore[no-untyped-call]
        "jax_compilation_cache_dir", str(Path.home() / ".cache/jax-settlrl")
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: end-to-end smoke whose JAX recompiles make it CI-only "
        "(pre-commit runs `-m 'not slow'`)",
    )


for _framework in sorted(EXPERIMENTS.glob("[0-9][0-9][0-9][0-9]_*")):
    sys.path.insert(0, str(_framework))


def load_run(framework: str, *, module: str = "run") -> ModuleType:
    """Import ``experiments/<framework>/<module>.py`` under a unique name."""
    path = EXPERIMENTS / framework / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"{framework}.{module}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
