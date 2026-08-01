"""Turn the search modules' jaxtyping annotations into enforced runtime checks
(same pattern as settlrl-engine's top-level conftest: the hook must be installed
before the target modules are first imported)."""

import os
import pathlib

import jax
from jaxtyping import install_import_hook

install_import_hook(
    [
        "settlrl_search.rows",
        "settlrl_search.policy",
        "settlrl_search.value",
        "settlrl_search.priors",
        "settlrl_search.sample",
        "settlrl_search._common",
        "settlrl_search.ismcts",
        "settlrl_search.expectimax",
        "settlrl_search",
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
