"""The fixed-capacity tree storage and the selection/expand/backup helpers.

``children`` / ``n`` / ``mover`` / ``kind`` hold only small exact
integers, so they are stored in narrow dtypes (see :mod:`config`) and cast to
float at the read sites where arithmetic needs it. The stored values are exact
integers, so results are bit-identical to a float32/int32 store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
from settlrl_engine.board.state import IntScalar, Player

from settlrl_search.value import Value

from ._types import (
    _Action,
    _EdgeF,
    _EdgeI,
    _FlatVec,
    _LegalMask,
    _Node,
    _NodeF,
    _NodeI,
    _PriorLogits,
)

if TYPE_CHECKING:
    from .config import _Cfg
    from .descent import _Descent

# completed-Q scale: (_MAXVISIT_INIT + max_visits) * _MIX_SCALE, added to the
# prior logits (absolute, not min-max normalized).
_MIX_SCALE = 0.1
_MAXVISIT_INIT = 50.0


class _Tree(NamedTuple):
    """The fixed-capacity tree a :data:`TreeSearch` builds and consumes: ``node``
    rows index up to ``num_simulations + 1`` nodes, ``act`` columns the flat
    action space.

    ``children`` / ``n`` / ``mover`` / ``kind`` are narrow integer stores (small
    exact values); ``w`` / ``prior`` / ``raw`` stay float32.
    """

    # TODO(perf): the 126 setup rows (a contiguous [0:126] prefix of N_FLAT) are
    # always illegal in the main loop, where ISMCTS runs, so the `act` axis could
    # be the suffix slice. Profile first — the cost is the per-descent engine
    # steps + roll_ev, not the action-axis width, so this likely won't move it.

    mover: _NodeI  # int8 player id
    children: _EdgeI  # child node id per edge, -1 = unexpanded (narrow node dtype)
    n: _EdgeI  # edge visit counts (narrow count dtype; cast to float on read)
    w: _EdgeF  # edge value sums (searcher frame)
    prior: _EdgeF  # raw prior logits (root logits at node 0, interior prior elsewhere)
    raw: _NodeF  # searcher-frame node value
    kind: _NodeI  # _DECISION or _CHANCE (chance nodes index children by outcome), int8
    size: IntScalar  # nodes in use


# --- select: the Gumbel-MuZero policy over the determinized legal set ---


def _completed_q(
    tree: _Tree, node: _Node, legal: _LegalMask, player: Player
) -> tuple[_FlatVec, _PriorLogits]:
    """The scaled completed Q-values and legal-masked prior logits at one node,
    in the mover's frame over this determinization's legal set. Unvisited actions
    take a prior-weighted blend of the node value and its visited children's Q."""
    n = tree.n[node].astype(jnp.float32)
    w = tree.w[node]
    sign = jnp.where(tree.mover[node] == player, 1.0, -1.0)
    visited = n > 0
    q = sign * jnp.where(visited, w / jnp.maximum(n, 1.0), 0.0)
    raw = sign * tree.raw[node]
    logits = jnp.where(legal > 0, tree.prior[node], -jnp.inf)
    probs = jax.nn.softmax(logits)
    # mixed value: blend the node's raw value with its prior-weighted visited-Q.
    sum_n = n.sum()
    floored_probs = jnp.maximum(1e-37, probs)  # floor a zero-prior visited action
    sum_probs = jnp.where(visited, floored_probs, 0.0).sum()
    weighted_q = jnp.where(
        visited, floored_probs * q / jnp.where(sum_probs > 0, sum_probs, 1.0), 0.0
    ).sum()
    mixed = (raw + sum_n * weighted_q) / (sum_n + 1.0)
    completed = jnp.where(visited, q, mixed)
    scaled = (_MAXVISIT_INIT + n.max()) * _MIX_SCALE * completed
    return scaled, logits


def _interior_select(
    tree: _Tree, node: _Node, legal: _LegalMask, player: Player
) -> _Action:
    """The interior action whose visit share most lags ``softmax(prior +
    completed_Q)``, so visits track the improved policy."""
    cq, logits = _completed_q(tree, node, legal, player)
    improved = jax.nn.softmax(jnp.where(legal > 0, logits + cq, -jnp.inf))
    n = tree.n[node].astype(jnp.float32)
    sum_n = n.sum()
    to_argmax = jnp.where(legal > 0, improved - n / (1.0 + sum_n), -jnp.inf)
    return jnp.argmax(to_argmax).astype(jnp.int32)


def _root_select(
    tree: _Tree,
    cfg: "_Cfg",
    gumbel: _FlatVec,
    num_considered: IntScalar,
    candidates: _LegalMask,
    world_legal: _LegalMask,
    player: Player,
) -> _Action:
    """The root action for this simulation: the Sequential-Halving pick over the
    fixed ``candidates`` set (the highest ``gumbel + prior + completed_Q`` among
    candidates at the schedule's current visit count), guarded to this
    determinization's ``world_legal``. A candidate can be illegal in some worlds
    (e.g. steal targets), so fall back to interior selection when the pick is."""
    visits = tree.n[0].astype(jnp.float32)
    sim_index = jnp.minimum(visits.sum().astype(jnp.int32), cfg.num_simulations - 1)
    cq, logits = _completed_q(tree, jnp.int32(0), candidates, player)
    considered_visit = cfg.table[num_considered, sim_index].astype(jnp.float32)
    norm_logits = logits - jnp.max(logits)
    penalty = jnp.where(visits == considered_visit, 0.0, -jnp.inf)
    score = jnp.maximum(-1e9, gumbel + norm_logits + cq) + penalty
    a_root = jnp.argmax(jnp.where(candidates > 0, score, -jnp.inf)).astype(jnp.int32)
    return jnp.where(
        world_legal[a_root] > 0,
        a_root,
        _interior_select(tree, jnp.int32(0), world_legal, player),
    )


# --- expand + backup: grow the tree, then propagate the leaf value ---


def _expand(
    tree: _Tree, walk: "_Descent", value: Value, mover: Player, leaf_prior: _PriorLogits
) -> _Tree:
    """Attach the descent's new leaf node — its mover, prior, and value — at the
    next free slot (a no-op when the simulation grew no node)."""
    grew = walk.exp_parent >= 0
    new_id = tree.size  # always <= n_nodes - 1 (<=1 node added per sim)
    safe_parent = jnp.maximum(walk.exp_parent, 0)
    node_dtype = tree.children.dtype
    return tree._replace(
        mover=tree.mover.at[new_id].set(
            jnp.where(grew, mover.astype(jnp.int8), tree.mover[new_id])
        ),
        prior=tree.prior.at[new_id].set(jnp.where(grew, leaf_prior, tree.prior[new_id])),
        raw=tree.raw.at[new_id].set(jnp.where(grew, value, tree.raw[new_id])),
        kind=tree.kind.at[new_id].set(
            jnp.where(grew, walk.exp_kind.astype(jnp.int8), tree.kind[new_id])
        ),
        children=tree.children.at[safe_parent, walk.exp_act].set(
            jnp.where(grew, new_id.astype(node_dtype), tree.children[safe_parent, walk.exp_act])
        ),
        size=tree.size + grew.astype(jnp.int32),
    )  # fmt: skip


def _backup(tree: _Tree, walk: "_Descent", value: Value, max_depth: int) -> _Tree:
    """Add the descent's leaf value and a visit to every edge on its path."""
    count_dtype = tree.n.dtype

    def body(j: IntScalar, tree: _Tree) -> _Tree:
        node, act = walk.path_node[j], walk.path_act[j]
        in_path = j < walk.depth  # past the real depth -> no-op
        use_i = in_path.astype(count_dtype)  # integer visit increment
        use_f = in_path.astype(jnp.float32)  # float value increment
        return tree._replace(
            n=tree.n.at[node, act].add(use_i),
            w=tree.w.at[node, act].add(use_f * value),
        )

    return cast(_Tree, jax.lax.fori_loop(0, max_depth, body, tree))
