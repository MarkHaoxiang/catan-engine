"""Board -> graph featurization for the learned architectures.

The board graph has fixed topology (54 vertices, 72 edges; the senders/receivers
never change shape), so a sample carries only per-node, per-edge and global
*features* -- the static incidence lives here as module constants. Perspective is
one player: node/edge ownership is relative (empty / mine / other -- a board
element has one owner, so "other" is complete there), and the global player
summary pairs the perspective player's own quantities with a *symmetric* summary
of the opponent multiset (sum / max / spread of their VP, hand, dev, knights,
roads) -- invariant to relabeling the opponents, so the same featurization
serves any seat. Computed on the true board (this measures representation
capacity, not belief handling).

Three readouts of the same position are produced per sample so architectures can
be compared on a level field:

- ``nodes`` / ``edges`` / ``glob`` -- the graph (GNN, DeepSet, flat-MLP all read
  this);
- ``extra`` -- an optional, caller-configured per-sample vector (a "form of
  feature engineering", e.g. the hand-tuned :mod:`settlrl_learn.features`
  baseline); ``None`` by default, so the graph-only consumers pay nothing.

The featurization is **versioned**: ``board_sample(..., version=...)`` selects a
frozen feature set (see ``FEATURE_VERSIONS``), so an existing net/checkpoint keeps
reading exactly the bytes it was trained on while a new one opts into a wider set.
``dims(version)`` resolves the widths per version; the module-level ``*_DIM``
constants are version 1's.

A training-side module (equinox/jraph consumers build on it): not imported by
the package root.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from settlrl_agents.internal.feature_engineering import tile_pips
from settlrl_engine.board.layout import (
    EDGE_V,
    N_EDGES,
    N_TILES,
    N_VERTICES,
    PORT_V,
    TILE_V,
    BoardLayout,
)
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.board.state import (
    CITY,
    MAX_ROADS,
    NO_INDEX,
    SETTLEMENT,
    BoardState,
    GamePhase,
    IntScalar,
)
from settlrl_engine.mechanics.common import player_total_vp

# A form of feature engineering: board -> an extra per-sample vector.
FeatureFn = Callable[[BoardLayout, BoardState, IntScalar], Float[Array, "feat"]]

FEATURE_VERSIONS = (1, 2)
"""Selectable feature sets. 1 is the original; 2 adds the own hand / dev-card
composition, the Road Building counter, the Longest Road lengths and the pending
discard to the globals, and puts the node production term on the same pips scale
as the per-hex features."""

# --- static topology (shared by every sample) ---

# Board edges are undirected; the message-passing graph uses both directions.
_edge_np = np.asarray(EDGE_V)
SENDERS = jnp.asarray(np.concatenate([_edge_np[:, 0], _edge_np[:, 1]]))
RECEIVERS = jnp.asarray(np.concatenate([_edge_np[:, 1], _edge_np[:, 0]]))
N_DIR_EDGES = 2 * N_EDGES

# vertex-tile incidence (V, T): vertex v touches tile t.
_inc = np.zeros((N_VERTICES, N_TILES), dtype=np.float32)
for _t, _vs in enumerate(np.asarray(TILE_V)):
    _inc[_vs, _t] = 1.0
VERTEX_TILE = jnp.asarray(_inc)

# vertex<->tile incidence as message-passing edges (heterogeneous graph): one
# (vertex, tile) pair per tile corner. ``VT_V`` are the vertex ends, ``VT_T`` the
# tile ends; vertex->tile aggregates over tiles, tile->vertex over vertices.
_tilev = np.asarray(TILE_V)  # (T, corners)
VT_V = jnp.asarray(_tilev.reshape(-1))
VT_T = jnp.asarray(np.repeat(np.arange(N_TILES), _tilev.shape[1]))

# vertex-port incidence (V, P): vertex v sits on port p.
_pinc = np.zeros((N_VERTICES, PORT_V.shape[0]), dtype=np.float32)
for _p, _vs in enumerate(np.asarray(PORT_V)):
    _pinc[_vs, _p] = 1.0
VERTEX_PORT = jnp.asarray(_pinc)


class Sample(NamedTuple):
    """One featurized position. ``nodes``/``edges``/``tiles``/``glob`` are the
    graph (``tiles`` = the per-hex node features, consumed only by the
    heterogeneous trunk); ``extra`` is an optional caller-configured vector
    (``None`` is graph-only)."""

    nodes: Float[Array, f"v={N_VERTICES} node_f"]
    edges: Float[Array, f"e={N_DIR_EDGES} edge_f"]
    glob: Float[Array, "global_f"]
    tiles: Float[Array, f"t={N_TILES} tile_f"]
    extra: Float[Array, "feat"] | None = None


_PIPS_SCALE = 5.0
"""Pips (``6 - |7 - n|``, 0 at the desert) top out at 5, so dividing by 5 lands a
hex's frequency in [0, 1]. Both the per-hex and (from version 2) the per-vertex
production terms use it, so the two readouts of the same quantity share a scale."""


def _node_features(
    layout: BoardLayout, state: BoardState, p: IntScalar, version: int = 1
) -> Float[Array, f"v={N_VERTICES} node_f"]:
    owner = state.vertex_owner.astype(jnp.int32)
    mine = owner == p + 1
    occupied = owner > 0
    owner_oh = jnp.stack([~occupied, mine, occupied & ~mine], axis=1).astype(
        jnp.float32
    )
    building = jnp.stack(
        [state.vertex_type == SETTLEMENT, state.vertex_type == CITY], axis=1
    ).astype(jnp.float32)

    pips = tile_pips(layout.tile_number) * (jnp.arange(N_TILES) != state.robber)
    if version >= 2:
        pips = pips / _PIPS_SCALE  # same scale as `_tile_features` (v1 left them raw)
    res_oh = jax.nn.one_hot(layout.tile_resource.astype(jnp.int32) % N_RESOURCES, 5)
    node_prod = VERTEX_TILE @ (res_oh * pips[:, None])  # (V, 5)
    robber_adj = (VERTEX_TILE[:, state.robber] > 0).astype(jnp.float32)[:, None]

    alloc = layout.port_allocation.astype(jnp.int32)
    port_res_oh = jax.nn.one_hot(alloc % N_RESOURCES, 5) * (alloc < 5)[:, None]
    v_2to1 = jnp.clip(VERTEX_PORT @ port_res_oh, 0.0, 1.0)  # (V, 5)
    v_3to1 = jnp.clip(VERTEX_PORT @ (alloc == 5).astype(jnp.float32), 0.0, 1.0)[:, None]

    return jnp.concatenate(
        [owner_oh, building, node_prod, robber_adj, v_2to1, v_3to1], axis=1
    )


def _edge_features(
    state: BoardState, p: IntScalar
) -> Float[Array, f"e={N_DIR_EDGES} edge_f"]:
    owner = state.edge_road.astype(jnp.int32)
    mine = owner == p + 1
    built = owner > 0
    one_dir = jnp.stack([~built, mine, built & ~mine], axis=1).astype(jnp.float32)
    return jnp.concatenate([one_dir, one_dir], axis=0)  # mirror both directions


def _tile_features(
    layout: BoardLayout, state: BoardState, p: IntScalar
) -> Float[Array, f"t={N_TILES} tile_f"]:
    """Per-hex node features (relative to ``p``): resource, pips, robber, and the
    own/other building weight summed over the hex's corners (settlement 1, city 2)."""
    tres = layout.tile_resource.astype(jnp.int32)
    # DESERT (== N_RESOURCES) has no resource -- zero its one-hot row rather
    # than let `% N_RESOURCES` alias it onto SHEEP's.
    res_oh = jax.nn.one_hot(tres % N_RESOURCES, 5) * (tres < N_RESOURCES)[:, None]
    pips = (tile_pips(layout.tile_number) / _PIPS_SCALE)[:, None]  # (T, 1)
    robber = (jnp.arange(N_TILES) == state.robber).astype(jnp.float32)[:, None]
    owner = state.vertex_owner.astype(jnp.int32)
    bval = jnp.where(
        state.vertex_type == CITY, 2.0, jnp.where(state.vertex_type == SETTLEMENT, 1.0, 0.0)
    )  # fmt: skip
    mine = ((owner == p + 1).astype(jnp.float32) * bval) @ VERTEX_TILE  # (T,)
    other = (((owner > 0) & (owner != p + 1)).astype(jnp.float32) * bval) @ VERTEX_TILE
    return jnp.concatenate(
        [res_oh, pips, robber, mine[:, None], other[:, None]], axis=1
    )


# Version-2 scaling divisors: each count feature is divided by a "large but
# ordinary" value for that quantity, so every added feature lands in roughly
# [0, 1] and the single ``glob`` encoder Linear sees no feature dominating the
# others by magnitude alone (nothing normalises `glob`).

# cards of ONE resource: 5 is already a big pile (a hand over 7 risks the discard)
_HAND_SCALE = 5.0
# dev cards of one type: 3 of a kind is a hoard (only 2 of most types exist)
_DEV_SCALE = 3.0
# Road Building grants exactly 2
_FREE_ROADS_SCALE = 2.0
# a trail cannot exceed the player's road pieces
_ROAD_LEN_SCALE = float(MAX_ROADS)
# half of a 10-card hand; owing more than that is rare
_DISCARD_SCALE = 5.0


def _own_features_v2(state: BoardState, p: IntScalar) -> Float[Array, "own_f"]:
    """Version-2 globals: the perspective player's *composition* (which resources,
    which dev cards) and the turn-state counters v1 dropped entirely.

    All player-relative, hence relabel-invariant. Longest Road: the engine stores
    only the award holder's length (``longest_road_len``), so own/opponent lengths
    are read off the holder rather than re-running the per-player DFS -- which the
    engine gates hard for cost and which this function pays on *every* search leaf.
    ``free_roads`` is game-level (it belongs to whoever is on turn), so it is
    masked by "it is my turn" to stay own-relative.
    """
    mine = state.current_player.astype(jnp.int32) == p
    resources = state.player_resources[p].astype(jnp.float32) / _HAND_SCALE
    dev = state.dev_hand[p].astype(jnp.float32) / _DEV_SCALE
    free_roads = state.free_roads.astype(jnp.float32) / _FREE_ROADS_SCALE
    lr_owner = state.longest_road_owner.astype(jnp.int32)
    lr_len = state.longest_road_len.astype(jnp.float32) / _ROAD_LEN_SCALE
    held = lr_owner != jnp.int32(NO_INDEX)
    return jnp.concatenate(
        [
            resources,  # (5,)
            dev,  # (5,)
            jnp.stack(
                [
                    jnp.where(mine, free_roads, 0.0),
                    jnp.where(held & (lr_owner == p), lr_len, 0.0),
                    jnp.where(held & (lr_owner != p), lr_len, 0.0),
                    state.pending_discard[p].astype(jnp.float32) / _DISCARD_SCALE,
                ]
            ),
        ]
    )


def _global_features(
    layout: BoardLayout, state: BoardState, p: IntScalar, version: int = 1
) -> Float[Array, "global_f"]:
    held = state.player_resources.astype(jnp.float32).sum(axis=0)
    bank = 19.0 - held
    phase = jax.nn.one_hot(state.phase.astype(jnp.int32), len(GamePhase))

    players = jnp.arange(state.n_players)
    vps = jax.vmap(lambda q: player_total_vp(state, q))(players).astype(jnp.float32)
    hands = state.player_resources.astype(jnp.float32).sum(axis=1)
    devs = state.dev_hand.astype(jnp.float32).sum(axis=1)
    knights = state.knights_played.astype(jnp.float32)
    roads = jnp.zeros((state.n_players + 1,), jnp.float32).at[state.edge_road].add(1.0)
    roads = roads[1:]  # drop the "empty" bucket -> per-player road count
    per_player = jnp.stack([vps, hands, devs, knights, roads])  # (5, P)

    # Own values, then a symmetric summary of the opponent *multiset* (sum, max,
    # spread) per quantity: invariant to relabeling the opponents, and -- unlike
    # the old (max-VP, total-hand) pair -- it distinguishes "one strong + one
    # weak" from "two medium" opponents, and adds their dev/knight/road context.
    opp = players != p
    own = per_player[:, p]
    opp_sum = jnp.sum(jnp.where(opp, per_player, 0.0), axis=1)
    opp_max = jnp.max(jnp.where(opp, per_player, -jnp.inf), axis=1)
    opp_min = jnp.min(jnp.where(opp, per_player, jnp.inf), axis=1)
    opp_summary = jnp.concatenate([opp_sum, opp_max, opp_max - opp_min])  # (15,)

    robber_pips = tile_pips(layout.tile_number)[state.robber]
    if version >= 2:
        robber_pips = robber_pips / _PIPS_SCALE  # one pips scale everywhere in v2
    scalars = jnp.concatenate(
        [
            jnp.asarray(
                [
                    state.dev_deck.astype(jnp.float32).sum(),
                    state.has_rolled.astype(jnp.float32),
                    robber_pips,
                    (state.longest_road_owner == p).astype(jnp.float32),
                    (state.largest_army_owner == p).astype(jnp.float32),
                    (state.current_player.astype(jnp.int32) == p).astype(jnp.float32),
                    jnp.float32(state.n_players),
                ],
                dtype=jnp.float32,
            ),
            own,  # (5,) own vp/hand/dev/knights/roads
            opp_summary,  # (15,) symmetric opponent-multiset summary
        ]
    )
    parts = [bank, scalars, phase]
    if version >= 2:
        # appended last, so v1's block keeps its layout (a v2 vector's prefix is
        # v1's, up to the rescaled robber pips)
        parts.append(_own_features_v2(state, p))
    return jnp.concatenate(parts)


def board_sample(
    layout: BoardLayout,
    state: BoardState,
    p: IntScalar,
    features: FeatureFn | None = None,
    *,
    with_tiles: bool = True,
    version: int = 1,
) -> Sample:
    """Featurize one position from ``p``'s perspective into a :class:`Sample`.
    ``features`` (when given) computes the optional ``extra`` vector; by default
    no extra is computed (graph-only). ``with_tiles=False`` skips the per-hex
    features (a constant-zero ``tiles``) -- the non-heterogeneous trunk ignores
    them, so this keeps its graph free of tile ops (byte-identical to the
    pre-hetero forward). ``version`` selects the feature set (``FEATURE_VERSIONS``);
    a net must be built for the same version (``dims``)."""
    if version not in FEATURE_VERSIONS:
        raise ValueError(f"unknown feature version {version} (have {FEATURE_VERSIONS})")
    return Sample(
        nodes=_node_features(layout, state, p, version),
        edges=_edge_features(state, p),
        glob=_global_features(layout, state, p, version),
        tiles=(
            _tile_features(layout, state, p)
            if with_tiles
            else jnp.zeros((N_TILES, TILE_DIM), jnp.float32)
        ),
        extra=None if features is None else features(layout, state, p),
    )


@cache
def dims(version: int = 1) -> tuple[int, int, int, int]:
    """``(node, edge, global, tile)`` feature widths at ``version`` (the topology
    dims ``N_VERTICES`` / ``N_DIR_EDGES`` / ``N_TILES`` are fixed)."""
    from settlrl_engine.board import make_board

    layout, state = make_board(batch_size=1, n_players=2)
    one = jax.tree.map(lambda x: x[0], (layout, state))
    s = jax.eval_shape(
        lambda lo, st: board_sample(lo, st, jnp.int32(0), version=version), *one
    )
    return (
        int(s.nodes.shape[1]),
        int(s.edges.shape[1]),
        int(s.glob.shape[0]),
        int(s.tiles.shape[1]),
    )


NODE_DIM, EDGE_DIM, GLOBAL_DIM, TILE_DIM = dims(1)
"""Version-1 feature widths (``dims(version)`` for any other version)."""
