"""Turn the rule modules' jaxtyping annotations into enforced runtime checks.

The single-game rule modules annotate their array params with the un-batched
aliases defined in state / resources / layout / dev_cards (``EdgeRoadVec``,
``IntScalar``, ...). Installing jaxtyping's import hook (backed by beartype) here
makes those annotations *checked* during the test run -- every call, and every
``jax.jit`` trace, verifies shapes and dtypes -- at zero cost to the shipped
package (the hook only exists in the test session).

The hook must be installed before the target modules are first imported, so this
lives in the top-level ``tests`` conftest, which pytest loads before any test
module (and before ``tests/mechanics/actions/conftest.py``).

``action`` / ``env`` are intentionally excluded: their cores are annotated with
*batched* aliases but execute unbatched under ``vmap``, so enforcing them needs a
separate batched -> single-game re-annotation pass first.
"""

import os
import pathlib

import jax
from jaxtyping import install_import_hook

install_import_hook(
    [
        "settlrl_engine.mechanics.placement",
        "settlrl_engine.mechanics.awards",
        "settlrl_engine.mechanics.longest_road",
        "settlrl_engine.mechanics.dice",
        "settlrl_engine.mechanics.robber",
        "settlrl_engine.mechanics.setup",
        "settlrl_engine.mechanics.trade",
        "settlrl_engine.mechanics.development",
        "settlrl_engine.mechanics.turn",
        "settlrl_engine.belief",
    ],
    "beartype.beartype",
)

# Persist XLA compilations across test runs (~/.cache/jax-settlrl unless
# JAX_COMPILATION_CACHE_DIR overrides). Cache hits skip XLA compilation only;
# jit traces still execute, so the jaxtyping hook's checks are unaffected.
# Parallel (xdist) workers run on CPU: one JAX process per core each
# initialising the GPU (and preallocating its memory) breaks CUDA init
# for the rest. Single-process runs (the benchmarks) keep the GPU.
if "JAX_PLATFORMS" not in os.environ and "PYTEST_XDIST_WORKER" in os.environ:
    jax.config.update("jax_platforms", "cpu")  # type: ignore[no-untyped-call]

if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    jax.config.update(  # type: ignore[no-untyped-call]
        "jax_compilation_cache_dir", str(pathlib.Path.home() / ".cache/jax-settlrl")
    )
