"""jaxtyping runtime checks for settlrl_learn + the shared XLA cache settings
(same pattern as the other packages' conftests)."""

import os
import pathlib

import jax
from jaxtyping import install_import_hook

# nn/mlp.py / train.py are excluded: MLPParams' per-layer shapes are
# heterogeneous, and jaxtyping unifies dim names across a variadic tuple, so
# the annotation is per-layer documentation that cannot hold globally.
install_import_hook(
    ["settlrl_learn.features", "settlrl_learn.nn.graph"], "beartype.beartype"
)

if "JAX_PLATFORMS" not in os.environ and "PYTEST_XDIST_WORKER" in os.environ:
    jax.config.update("jax_platforms", "cpu")  # type: ignore[no-untyped-call]

# Persist XLA compilations across test runs (~/.cache/jax-settlrl unless
# JAX_COMPILATION_CACHE_DIR overrides; shared with the engine/experiments
# suites on purpose -- same cache dir, same tiny programs). The resume/carry
# tests each run a miniature training loop, so a warm cache turns their
# repeated `learn`/`self_play` traces into cache hits.
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    jax.config.update(  # type: ignore[no-untyped-call]
        "jax_compilation_cache_dir", str(pathlib.Path.home() / ".cache/jax-settlrl")
    )
