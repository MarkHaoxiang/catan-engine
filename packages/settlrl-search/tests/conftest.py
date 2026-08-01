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

import jax.numpy as jnp  # noqa: E402  (after the import hook installs)
from settlrl_engine.board.layout import EDGE_V, TILE_V, BoardLayout  # noqa: E402
from settlrl_engine.board.resources import CITY_COST, SETTLEMENT_COST  # noqa: E402
from settlrl_engine.board.state import BoardState, Player  # noqa: E402
from settlrl_engine.mechanics.common import player_total_vp  # noqa: E402
from settlrl_search.value import Value  # noqa: E402


def heuristic_value(layout: BoardLayout, state: BoardState, player: Player) -> Value:
    """A self-contained leaf for the search contract tests.

    Engine-only (no settlrl-agents dependency), and deliberately not the shipped
    heuristic: it just needs to *discriminate* moves and reward expansion enough
    to drive a game to a win, since the contracts pinned on it (legality,
    reproducibility, distribution support, q-range, visit concentration, game
    completion, setup bit-identity) require those two properties of the leaf and
    nothing more. Per-player strength is total VP (buildings + awards + VP
    cards), owned-vertex production pips, progress toward the cheapest next
    build, the best open spot reachable from the player's road network (the
    expansion driver), roads, and cards; value is own strength minus the best
    opponent's.
    """
    players = jnp.arange(state.n_players)
    n_vertices = state.vertex_owner.shape[-1]
    # Pips per tile from its dice number (0 for the desert / unnumbered), spread
    # to its corner vertices: a vertex's production is the pips it touches.
    num = layout.tile_number.astype(jnp.int32)
    tile_pips = jnp.where(num == 0, 0, 6 - jnp.abs(7 - num)).astype(jnp.float32)
    vertex_pips = (
        jnp.zeros(n_vertices, jnp.float32)
        .at[TILE_V.reshape(-1)]
        .add(jnp.repeat(tile_pips, TILE_V.shape[-1]))
    )
    settlement_cost = jnp.asarray(SETTLEMENT_COST, jnp.float32)
    city_cost = jnp.asarray(CITY_COST, jnp.float32)
    edge_a, edge_b = EDGE_V[:, 0], EDGE_V[:, 1]
    open_v = state.vertex_owner == 0

    def strength(p: Player) -> Value:
        mine_v = state.vertex_owner.astype(jnp.int32) == p + 1
        mine_e = state.edge_road.astype(jnp.int32) == p + 1
        held = state.player_resources[p].astype(jnp.float32)
        # Progress toward the cheapest next build: rewards accumulating the
        # *right* cards (so the search expands rather than idles).
        progress = jnp.maximum(
            jnp.minimum(held, settlement_cost).sum(),
            jnp.minimum(held, city_cost).sum(),
        )
        # Best open spot the player's network already touches (its settlements
        # plus the endpoints of its roads): the road-to-settlement reach signal.
        touched = (
            jnp.zeros(n_vertices, bool)
            .at[jnp.where(mine_e, edge_a, 0)]
            .set(mine_e)
            .at[jnp.where(mine_e, edge_b, 0)]
            .set(mine_e)
        ) | mine_v
        reach = jnp.max(jnp.where(touched & open_v, vertex_pips, 0.0))
        out: Value = (
            10.0 * player_total_vp(state, p).astype(jnp.float32)
            + 3.0 * mine_v.sum().astype(jnp.float32)
            + 0.6 * (mine_v.astype(jnp.float32) * vertex_pips).sum()
            + 3.0 * progress
            + 1.5 * reach
            + 0.15 * mine_e.sum().astype(jnp.float32)
            + 0.05 * held.sum()
        )
        return out

    strengths = jax.vmap(strength)(players)
    mine = strengths[player]
    best_other = jnp.max(jnp.where(players == player, -jnp.inf, strengths))
    out: Value = mine - best_other
    return out
