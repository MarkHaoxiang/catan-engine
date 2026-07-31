"""The flat action space: one enumerated index per concrete move.

The static table (``N_FLAT`` rows; Discard is one row per resource), its
decode ``flat_to_action``, the reverse-lookup ``flat_legality``, the
switch-free legality sweeps, and the per-index enumerations ``INDEX_MASKS``.
Everything calls the per-action legality cores directly over static slices of
the table -- never through ``action.py``'s ``lax.switch``, which under
``vmap`` would evaluate every branch per entry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Int

from settlrl_engine.board.layout import (
    N_EDGES,
    N_TILES,
    N_VERTICES,
    BoardLayout,
)
from settlrl_engine.board.resources import N_PLAYERS, N_RESOURCES
from settlrl_engine.board.state import BoardState, IntScalar, KeyScalar
from settlrl_engine.mechanics.action import (
    N_ACTION_TYPES,
    ActionParams,
    ActionType,
)
from settlrl_engine.mechanics.common import (
    ActionTypeArray,
    IndexAvail,
    IndexParam,
    Mask,
    NoneAvail,
    PairAvail,
)
from settlrl_engine.mechanics.development import (
    _buy_dev_avail,
    _knight_avail,
    _monopoly_avail,
    _road_building_avail,
    _yop_avail,
)
from settlrl_engine.mechanics.dice import _roll_avail
from settlrl_engine.mechanics.placement import (
    _build_city_avail,
    _build_road_avail,
    _build_settlement_avail,
)
from settlrl_engine.mechanics.robber import (
    _discard_avail,
    _move_robber_avail,
)
from settlrl_engine.mechanics.setup import (
    _setup_road_avail,
    _setup_settlement_avail,
)
from settlrl_engine.mechanics.trade import (
    _accept_trade_avail,
    _maritime_avail,
    _propose_trade_avail,
    _reject_trade_avail,
    pack_trade_single,
)
from settlrl_engine.mechanics.turn import _end_turn_avail

__all__ = [
    "FLAT_ATYPE",
    "FLAT_IDX",
    "FLAT_TARGET",
    "INDEX_MASKS",
    "N_FLAT",
    "FlatMaskArray",
    "FlatMaskVec",
    "TypeMaskArray",
    "flat_available_b",
    "flat_available_for",
    "flat_legality",
    "flat_to_action",
    "random_flat",
    "type_mask_from_flat",
]


def _build_action_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat action index -> (ActionType, primary index, secondary target)."""
    atype: list[int] = []
    idx: list[int] = []
    target: list[int] = []

    def add(t: ActionType, i: int = 0, tg: int = 0) -> None:
        atype.append(int(t))
        idx.append(i)
        target.append(tg)

    for v in range(N_VERTICES):
        add(ActionType.SETUP_SETTLEMENT, v)
    for e in range(N_EDGES):
        add(ActionType.SETUP_ROAD, e)
    add(ActionType.ROLL_DICE)
    for r in range(N_RESOURCES):
        add(ActionType.DISCARD, r)
    for t in range(N_TILES):
        for victim in range(-1, N_PLAYERS):
            add(ActionType.MOVE_ROBBER, t, victim)
    for e in range(N_EDGES):
        add(ActionType.BUILD_ROAD, e)
    for v in range(N_VERTICES):
        add(ActionType.BUILD_SETTLEMENT, v)
    for v in range(N_VERTICES):
        add(ActionType.BUILD_CITY, v)
    add(ActionType.BUY_DEVELOPMENT_CARD)
    for t in range(N_TILES):
        for victim in range(-1, N_PLAYERS):
            add(ActionType.PLAY_KNIGHT, t, victim)
    add(ActionType.PLAY_ROAD_BUILDING)
    for a in range(N_RESOURCES):
        for b in range(N_RESOURCES):
            add(ActionType.PLAY_YEAR_OF_PLENTY, a, b)
    for r in range(N_RESOURCES):
        add(ActionType.PLAY_MONOPOLY, r)
    for g in range(N_RESOURCES):
        for r in range(N_RESOURCES):
            add(ActionType.MARITIME_TRADE, g, r)
    # Domestic trade supports arbitrary bundles via the packed params (see
    # trade.pack_trade); the table enumerates only the 1:1 subset.
    for g in range(N_RESOURCES):
        for r in range(N_RESOURCES):
            for partner in range(N_PLAYERS):
                add(ActionType.PROPOSE_TRADE, *pack_trade_single(g, r, partner))
    add(ActionType.ACCEPT_TRADE)
    add(ActionType.REJECT_TRADE)
    add(ActionType.END_TURN)

    return (
        np.asarray(atype, dtype=np.int32),
        np.asarray(idx, dtype=np.int32),
        np.asarray(target, dtype=np.int32),
    )


_ATYPE, _IDX, _TARGET = _build_action_table()

N_FLAT = int(_ATYPE.shape[0])
"""Size of the flat action space (one index per concrete move)."""

# Device-side columns of the table (row -> action type / primary / secondary).
FLAT_ATYPE = jnp.asarray(_ATYPE)
FLAT_IDX = jnp.asarray(_IDX)
FLAT_TARGET = jnp.asarray(_TARGET)

# Legality-mask aliases, shaped to the flat table / the action-type axis.
FlatMaskArray = Bool[Array, f"batch flat={N_FLAT}"]
FlatMaskVec = Bool[Array, f"flat={N_FLAT}"]
TypeMaskArray = Bool[Array, f"batch action_types={N_ACTION_TYPES}"]


def flat_to_action(
    flat: Int[Array, "*shape"],
) -> tuple[Int[Array, "*shape"], ActionParams]:
    """Decode flat action indices (any shape) into ``(action_type, ActionParams)``."""
    return FLAT_ATYPE[flat], ActionParams(idx=FLAT_IDX[flat], target=FLAT_TARGET[flat])


def random_flat(key: KeyScalar, mask: FlatMaskVec) -> IntScalar:
    """A random legal flat action: a uniform legal action *type*, then a
    uniform legal row of that type (one game; batch by vmapping).

    Two-stage rather than flat-uniform so bulk-enumerated types don't crowd
    out single-row ones (the ~100 trade-proposal rows would make almost every
    MAIN move an offer, stretching random games several-fold). With no legal
    move the returned index is arbitrary (the engine rejects it as INVALID).
    The one sampler behind ``BatchedSettlrlEnv.random_actions``,
    ``record_game``'s default chooser, and settlrl-agents' ``random_policy``.
    """
    k_type, k_row = jax.random.split(key)
    type_legal = jnp.zeros((N_ACTION_TYPES,), jnp.bool_).at[FLAT_ATYPE].max(mask)
    t = jnp.argmax(
        jnp.where(type_legal, jax.random.uniform(k_type, (N_ACTION_TYPES,)), -1.0)
    )
    noise = jax.random.uniform(k_row, (N_FLAT,))
    return jnp.argmax(jnp.where(mask & (t == FLAT_ATYPE), noise, -1.0))


# ===========================================================================
# Reverse lookup: (action_type, idx, target) -> flat row.
#
# Lets a caller read one ``(action_type, params)`` action's legality straight out
# of a cached ``(B, N_FLAT)`` flat-legality sweep (:func:`flat_legality`) instead
# of recomputing avail for the chosen action. The flat table holds one row per
# concrete move, so ``(action_type, idx, target)`` is unique per row; we pack the
# triple into a dense int32 index and store its row. PROPOSE_TRADE is excluded:
# its bit-packed params span ~2^27 values (a dense table can't), and the
# enumerable rows are only the 1:1 subset of the bundle domain anyway — callers
# evaluate proposals with the trade core directly (see ``env.batched``).
# ===========================================================================
_DENSE = int(ActionType.PROPOSE_TRADE) != _ATYPE
_IDX_RANGE = int(_IDX[_DENSE].max()) + 1
_TGT_MIN = int(_TARGET[_DENSE].min())
_TGT_RANGE = int(_TARGET[_DENSE].max()) - _TGT_MIN + 1
_REVERSE = np.zeros(N_ACTION_TYPES * _IDX_RANGE * _TGT_RANGE, dtype=np.int32)
_REVERSE[
    (_ATYPE[_DENSE] * _IDX_RANGE + _IDX[_DENSE]) * _TGT_RANGE
    + (_TARGET[_DENSE] - _TGT_MIN)
] = np.flatnonzero(_DENSE).astype(np.int32)
_REVERSE_J = jnp.asarray(_REVERSE)


def flat_legality(
    avail_flat: FlatMaskArray,
    action_type: ActionTypeArray,
    idx: IndexParam,
    target: IndexParam,
) -> Mask:
    """``(B,)`` legality of each lane's ``(action_type, idx, target)`` move.

    Gathers each lane's bit from the ``(B, N_FLAT)`` mask ``avail_flat``. An
    action that is not a row of the flat table (out-of-range params, and every
    PROPOSE_TRADE — see the reverse-lookup note) reads ``False``.
    """
    pack = (action_type * _IDX_RANGE + idx) * _TGT_RANGE + (target - _TGT_MIN)
    pack = jnp.clip(pack, 0, _REVERSE_J.shape[0] - 1)
    row = _REVERSE_J[pack]  # (B,)
    matches = (
        (FLAT_ATYPE[row] == action_type)
        & (FLAT_IDX[row] == idx)
        & (FLAT_TARGET[row] == target)
        & (action_type != ActionType.PROPOSE_TRADE)
    )
    rows = jnp.arange(avail_flat.shape[0])
    return avail_flat[rows, row] & matches


# ===========================================================================
# Switch-free flat legality sweep.
#
# ``action_available`` dispatches the N_ACTION_TYPES action types with
# ``jax.lax.switch``. ``vmap``-ing it over the whole flat table (whose action
# type *varies* per entry) forces XLA to evaluate *every* branch for *every*
# entry -- an ~18x blow-up, worse once the expensive cores (Longest Road,
# placement scatters) are counted. Because the flat table is *static* we instead call each action's
# legality core *directly* over its own slice of the table, board closed over,
# and stitch the results back in table order: same mask, no switch.
# ===========================================================================
def _flat_positions(t: ActionType) -> np.ndarray:
    """Flat-table row indices belonging to action type ``t`` (static)."""
    return np.where(int(t) == _ATYPE)[0]


# (core, flat rows, primary index per row) for the single-index actions.
_INDEX_AVAIL: tuple[tuple[IndexAvail, jax.Array, jax.Array], ...] = tuple(
    (core, jnp.asarray(p), jnp.asarray(_IDX[p]))
    for core, p in (
        (_setup_settlement_avail, _flat_positions(ActionType.SETUP_SETTLEMENT)),
        (_setup_road_avail, _flat_positions(ActionType.SETUP_ROAD)),
        (_discard_avail, _flat_positions(ActionType.DISCARD)),
        (_build_road_avail, _flat_positions(ActionType.BUILD_ROAD)),
        (_build_settlement_avail, _flat_positions(ActionType.BUILD_SETTLEMENT)),
        (_build_city_avail, _flat_positions(ActionType.BUILD_CITY)),
        (_monopoly_avail, _flat_positions(ActionType.PLAY_MONOPOLY)),
    )
)

# (core, flat rows, primary, secondary) for the (idx, target) pair actions.
_PAIR_AVAIL: tuple[tuple[PairAvail, jax.Array, jax.Array, jax.Array], ...] = tuple(
    (core, jnp.asarray(p), jnp.asarray(_IDX[p]), jnp.asarray(_TARGET[p]))
    for core, p in (
        (_move_robber_avail, _flat_positions(ActionType.MOVE_ROBBER)),
        (_knight_avail, _flat_positions(ActionType.PLAY_KNIGHT)),
        (_yop_avail, _flat_positions(ActionType.PLAY_YEAR_OF_PLENTY)),
        (_maritime_avail, _flat_positions(ActionType.MARITIME_TRADE)),
        (_propose_trade_avail, _flat_positions(ActionType.PROPOSE_TRADE)),
    )
)

# (core, single flat row) for the parameterless actions.
_NONE_AVAIL: tuple[tuple[NoneAvail, int], ...] = tuple(
    (core, int(_flat_positions(t)[0]))
    for core, t in (
        (_roll_avail, ActionType.ROLL_DICE),
        (_buy_dev_avail, ActionType.BUY_DEVELOPMENT_CARD),
        (_road_building_avail, ActionType.PLAY_ROAD_BUILDING),
        (_accept_trade_avail, ActionType.ACCEPT_TRADE),
        (_reject_trade_avail, ActionType.REJECT_TRADE),
        (_end_turn_avail, ActionType.END_TURN),
    )
)


def _sweep_index(
    core: IndexAvail, layout: BoardLayout, state: BoardState, idxs: jax.Array
) -> jax.Array:
    """Map a single-index legality core over ``idxs`` (board closed over)."""
    return jax.vmap(lambda i: core(layout, state, i))(idxs)


def _sweep_pair(
    core: PairAvail,
    layout: BoardLayout,
    state: BoardState,
    idxs: jax.Array,
    tgts: jax.Array,
) -> jax.Array:
    """Map a ``(primary, secondary)`` legality core over ``idxs``/``tgts``."""
    return jax.vmap(lambda i, tg: core(layout, state, (i, tg)))(idxs, tgts)


def flat_available_for(layout: BoardLayout, state: BoardState) -> FlatMaskVec:
    """``(N_FLAT,)`` legality of every flat action for one game.

    Equivalent to :func:`action_available` over every row of the flat table.
    """
    out = jnp.zeros(N_FLAT, dtype=bool)
    for core, pos, idxs in _INDEX_AVAIL:
        out = out.at[pos].set(_sweep_index(core, layout, state, idxs))
    for pair_core, pos, idxs, tgts in _PAIR_AVAIL:
        out = out.at[pos].set(_sweep_pair(pair_core, layout, state, idxs, tgts))
    for none_core, p in _NONE_AVAIL:
        out = out.at[p].set(none_core(layout, state, None))
    return out


flat_available_b: Callable[[BoardLayout, BoardState], FlatMaskArray] = jax.jit(
    jax.vmap(flat_available_for)
)
"""``(B, N_FLAT)`` flat legality per lane for its acting player (switch-free)."""


def _type_mask_core(avail_flat: FlatMaskArray) -> TypeMaskArray:
    """``(B, N_ACTION_TYPES)`` per-action-type legality, reduced from the
    ``(B, N_FLAT)`` flat mask: an action type is legal iff some concrete move
    of that type is."""
    b = avail_flat.shape[0]
    return jnp.zeros((b, N_ACTION_TYPES), jnp.bool_).at[:, FLAT_ATYPE].max(avail_flat)


type_mask_from_flat: Callable[[FlatMaskArray], TypeMaskArray] = jax.jit(_type_mask_core)


# ===========================================================================
# Per-index legality enumerations.
#
# A finer view over the same cores than the flat sweep: ``INDEX_MASKS`` sweeps an
# index-parameterized action's whole primary domain (the env's
# ``available_indices``). The per-action-type "is any move of this type legal"
# mask (the env's ``action_mask``) is not a separate sweep: the env reduces it
# from the cached flat-legality mask.
# ===========================================================================

# Static parameter domains for the robber-tile legality sweep.
_TILE_DOM = jnp.arange(N_TILES, dtype=jnp.int32)
_VICTIM_DOM = jnp.arange(-1, N_PLAYERS, dtype=jnp.int32)  # -1 = steal from no one


_BatchedMask = Callable[[BoardLayout, BoardState], Bool[Array, "batch domain"]]


def _index_mask_factory(avail: IndexAvail, n: int) -> _BatchedMask:
    """Batched ``(B, n)`` legality sweep over a single index parameter."""
    dom = jnp.arange(n, dtype=jnp.int32)

    def single(layout: BoardLayout, state: BoardState) -> jax.Array:
        return jax.vmap(lambda i: avail(layout, state, i))(dom)

    return cast(_BatchedMask, jax.jit(jax.vmap(single)))


def _robber_tile_mask_factory(avail: PairAvail) -> _BatchedMask:
    """Batched ``(B, N_TILES)`` mask: a tile is legal if some victim choice works."""

    def single(layout: BoardLayout, state: BoardState) -> jax.Array:
        return jax.vmap(
            lambda t: jnp.any(
                jax.vmap(lambda v: avail(layout, state, (t, v)))(_VICTIM_DOM)
            )
        )(_TILE_DOM)

    return cast(_BatchedMask, jax.jit(jax.vmap(single)))


# ActionType -> batched legality sweep over that action's primary index domain.
# (Multi-parameter / parameterless actions are absent; use ``action_mask`` /
# ``available`` for those.)
INDEX_MASKS = {
    ActionType.SETUP_SETTLEMENT: _index_mask_factory(
        _setup_settlement_avail, N_VERTICES
    ),
    ActionType.SETUP_ROAD: _index_mask_factory(_setup_road_avail, N_EDGES),
    ActionType.DISCARD: _index_mask_factory(_discard_avail, N_RESOURCES),
    ActionType.BUILD_ROAD: _index_mask_factory(_build_road_avail, N_EDGES),
    ActionType.BUILD_SETTLEMENT: _index_mask_factory(
        _build_settlement_avail, N_VERTICES
    ),
    ActionType.BUILD_CITY: _index_mask_factory(_build_city_avail, N_VERTICES),
    ActionType.PLAY_MONOPOLY: _index_mask_factory(_monopoly_avail, N_RESOURCES),
    ActionType.MOVE_ROBBER: _robber_tile_mask_factory(_move_robber_avail),
    ActionType.PLAY_KNIGHT: _robber_tile_mask_factory(_knight_avail),
}
