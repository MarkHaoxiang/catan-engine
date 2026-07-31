"""Initial-placement (setup phase) rules and action cores.

Holds the snake turn order, the 2nd-settlement resource grant, and the
``SetupSettlement`` / ``SetupRoad`` action cores that drive the setup snake.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from settlrl_engine.board import Board
from settlrl_engine.board.layout import N_EDGES, N_VERTICES, TILE_V, BoardLayout
from settlrl_engine.board.resources import BANK_INITIAL, N_RESOURCES
from settlrl_engine.board.state import (
    SETTLEMENT,
    BoardState,
    BoolScalar,
    GamePhase,
    IntScalar,
    tree_select,
)
from settlrl_engine.board.tile import Tile
from settlrl_engine.mechanics import placement
from settlrl_engine.mechanics.common import (
    INVALID,
    SUCCESS,
    IndexParam,
    Mask,
    ResultCode,
)

# Setup placement order over the 2 * n_players starting settlements: a snake /
# boustrophedon (0..n-1 then back n-1..0).


def setup_order(n_players: int) -> list[int]:
    """The snake setup order for ``n_players`` (host-side, not traceable)."""
    return list(range(n_players)) + list(range(n_players - 1, -1, -1))


def _setup_player(setup_index: IntScalar, n_players: int) -> IntScalar:
    """Who places the ``setup_index``-th starting settlement (traceable snake)."""
    return jnp.where(
        setup_index < n_players, setup_index, 2 * n_players - 1 - setup_index
    )


def grant_setup_resources(
    layout: BoardLayout, state: BoardState, vertex: IntScalar, player: IntScalar
) -> BoardState:
    """Grant one resource per (non-desert) tile adjacent to a 2nd settlement."""
    res = state.player_resources.astype(jnp.int32)
    bank = BANK_INITIAL - res.sum(axis=0)  # (R,)
    # Tiles touching ``vertex`` (a vertex appears in at most 3 tiles' corners).
    incident = (vertex == TILE_V).any(axis=1)  # (N_TILES,)
    resource = layout.tile_resource.astype(jnp.int32)  # (N_TILES,)
    produce = (incident & (layout.tile_resource != Tile.DESERT)).astype(jnp.int32)
    # Scatter demand per resource, then grant what the bank can cover (granting
    # min(demand, bank) matches a per-tile sequential payout).
    demand = jnp.zeros((N_RESOURCES,), jnp.int32).at[resource].add(produce)
    res = res.at[player].add(jnp.minimum(demand, bank))
    return state._replace(player_resources=res.astype(jnp.uint8))


# ===========================================================================
# SetupSettlement
# ===========================================================================


def _setup_settlement_avail(
    layout: BoardLayout, state: BoardState, vertex: IntScalar
) -> BoolScalar:
    in_range = (vertex >= 0) & (vertex < N_VERTICES)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    phase_ok = state.phase == GamePhase.SETUP_SETTLEMENT
    dist = placement.distance_rule_ok(state.vertex_owner, v)
    return in_range & phase_ok & dist


def _setup_settlement_apply(
    layout: BoardLayout, state: BoardState, vertex: IntScalar, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    placed = state._replace(
        vertex_owner=state.vertex_owner.at[v].set((player + 1).astype(jnp.uint8)),
        vertex_type=state.vertex_type.at[v].set(SETTLEMENT),
        victory_points=state.victory_points.at[player].add(1),
    )
    # The second settlement (placed in the reverse pass) grants resources.
    granted = grant_setup_resources(layout, placed, v, player)
    placed = tree_select(
        state.setup_index.astype(jnp.int32) >= state.n_players,
        granted,
        placed,
    )
    cand = placed._replace(phase=jnp.uint8(GamePhase.SETUP_ROAD))
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


_setup_settlement_avail_b = jax.jit(jax.vmap(_setup_settlement_avail))
_setup_settlement_apply_b = jax.jit(jax.vmap(_setup_settlement_apply))


def setup_settlement_available(board: Board, vertex: IndexParam) -> Mask:
    """``(batch,)`` legality of placing a free setup settlement on ``vertex``."""
    return cast(Mask, _setup_settlement_avail_b(board[0], board[1], vertex))


def setup_settlement_step(
    board: Board, vertex: IndexParam
) -> tuple[BoardState, ResultCode]:
    """Place a free starting settlement on ``vertex`` per game.

    The second settlement (placed in the reverse setup pass, when
    ``setup_index >= n_players``) grants one resource per adjacent tile. Always
    advances to SETUP_ROAD.
    """
    available = _setup_settlement_avail_b(board[0], board[1], vertex)
    return cast(
        "tuple[BoardState, ResultCode]",
        _setup_settlement_apply_b(board[0], board[1], vertex, available),
    )


# ===========================================================================
# SetupRoad
# ===========================================================================


def _setup_road_avail(
    layout: BoardLayout, state: BoardState, edge: IntScalar
) -> BoolScalar:
    in_range = (edge >= 0) & (edge < N_EDGES)
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    phase_ok = state.phase == GamePhase.SETUP_ROAD
    empty = state.edge_road[e] == 0
    touches = placement.setup_road_placeable(
        state.edge_road, state.vertex_owner, player, e
    )
    return in_range & phase_ok & empty & touches


def _setup_road_apply(
    layout: BoardLayout, state: BoardState, edge: IntScalar, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    n = state.n_players
    new_index = state.setup_index.astype(jnp.int32) + 1
    setup_continues = new_index < 2 * n
    next_player = jnp.where(
        setup_continues,
        _setup_player(new_index, n),
        0,
    )
    next_phase = jnp.where(setup_continues, GamePhase.SETUP_SETTLEMENT, GamePhase.ROLL)
    cand = state._replace(
        edge_road=state.edge_road.at[e].set((player + 1).astype(jnp.uint8)),
        setup_index=new_index.astype(state.setup_index.dtype),
        phase=next_phase.astype(state.phase.dtype),
        current_player=next_player.astype(state.current_player.dtype),
    )
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


_setup_road_avail_b = jax.jit(jax.vmap(_setup_road_avail))
_setup_road_apply_b = jax.jit(jax.vmap(_setup_road_apply))


def setup_road_available(board: Board, edge: IndexParam) -> Mask:
    """``(batch,)`` legality of placing the setup road on ``edge``."""
    return cast(Mask, _setup_road_avail_b(board[0], board[1], edge))


def setup_road_step(board: Board, edge: IndexParam) -> tuple[BoardState, ResultCode]:
    """Place the road next to the just-placed setup settlement, on ``edge``.

    Advances the snake setup order: the next settlement placement, or ROLL with
    player 0 once setup is complete.
    """
    available = _setup_road_avail_b(board[0], board[1], edge)
    return cast(
        "tuple[BoardState, ResultCode]",
        _setup_road_apply_b(board[0], board[1], edge, available),
    )
