"""Search stand-ins for the self-play tests: a uniform policy over the legal set
instead of a real tree, so the tests that only exercise the *loop* around the
search stay seconds-fast."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array
from settlrl_engine.belief import belief_view
from settlrl_engine.board.layout import BoardLayout
from settlrl_learn.training.backend import Backend


def uniform_weights(
    key: Array, layout: BoardLayout, view: Any, player: Array, mask: Array
) -> Array:
    """A stand-in for the search: uniform over the legal set (no net, no tree)."""
    return mask.astype(jnp.float32)


def uniform_legal_dist(
    key: Array, layout: BoardLayout, view: Any, player: Array, mask: Array
) -> Array:
    """A *normalised* uniform-over-legal stand-in -- a proper distribution, like
    the real search's visit-count target (the bare mask is unnormalised)."""
    m = mask.astype(jnp.float32)
    return m / jnp.sum(m)


def jitted(weights_fn: Any, backend: Backend) -> dict[str, Any]:
    """Build the pre-jitted+vmapped callables `self_play` now expects from a bare
    `weights_fn` stand-in and a backend (no setup search)."""
    return {
        "search": jax.jit(jax.vmap(weights_fn, in_axes=(0, 0, 0, 0, 0))),
        "observe_of": jax.jit(jax.vmap(backend.observe, in_axes=(0, 0, 0))),
        "view_of": jax.jit(jax.vmap(belief_view, in_axes=(0, 0, 0))),
    }


def uniform_weights_value(
    key: Array, layout: BoardLayout, view: Any, player: Array, mask: Array
) -> tuple[Array, Array]:
    """Uniform policy + a constant root value (a PolicyWeightsValue stand-in)."""
    return mask.astype(jnp.float32), jnp.float32(0.3)
