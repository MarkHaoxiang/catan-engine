"""The specialized setup-phase policies (the initial settlement/road
placements): a one-ply value sweep restricted to the setup rows
(``make_setup_lookahead``) and a depth-limited, beam-pruned *probabilistic
expectimax* (``make_setup_search``).

Setup is a short, fully-observable, high-leverage sub-game: there are no cards
yet, so nothing is hidden, and the two opening settlements decide the whole
economy. It is also structurally distinct from the main loop, so the learned net
does not handle it -- this fixed policy does.

The search backs up a **per-player value vector** (the heuristic value from each
seat) so every node knows the mover's own value. At the searcher's nodes it
maximizes (we assume we play the opener optimally); at an opponent's nodes it
takes a **Boltzmann expectation** over that opponent's self-interested moves --
``softmax(opponent_value / temperature)``. ``temperature`` is the chance the
opponent plays a non-optimal placement: ``-> 0`` is an optimal (minimax)
opponent, large is uniform-random. Modeling opponents as imperfect matters most
with **more than two players**, where assuming three optimal openers is far too
pessimistic.

It is built as a **full ``beam``-ary tree expanded level by level** (node ``i``'s
children are ``beam*i .. beam*i+beam-1`` one level down), one
``vmap(apply_action)`` over the frontier per level. A modest ``beam`` keeps
``beam**depth`` states tractable, so depth 6-8 -- enough to reach the
*complementary second settlement* in snake order -- is reachable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import N_VERTICES, BoardLayout
from settlrl_engine.board.state import BoardState, KeyScalar, Player
from settlrl_engine.env import N_FLAT
from settlrl_engine.mechanics.action import ActionType, action_available, apply_action
from settlrl_engine.mechanics.awards import resolve_step
from settlrl_engine.mechanics.common import agent_selection_single
from settlrl_engine.mechanics.setup import _setup_road_apply, _setup_settlement_apply

from settlrl_search._common import _ILLEGAL
from settlrl_search.policy import BeliefPolicy, FlatAction, FlatMask
from settlrl_search.rows import ROW_PARAMS, ROW_TYPE
from settlrl_search.sample import sample_world
from settlrl_search.value import ValueFunction

__all__ = ["make_setup_lookahead", "make_setup_search"]

# The flat rows that are setup placements (initial settlement or road).
_SETUP_IDX = jnp.asarray(
    np.flatnonzero(
        (np.asarray(ROW_TYPE) == int(ActionType.SETUP_SETTLEMENT))
        | (np.asarray(ROW_TYPE) == int(ActionType.SETUP_ROAD))
    )
)
_SETUP_TYPE = ROW_TYPE[_SETUP_IDX]
_SETUP_PARAMS = jax.tree.map(lambda x: x[_SETUP_IDX], ROW_PARAMS)
_N_SETUP = int(_SETUP_IDX.shape[0])

# ``make_setup_lookahead`` relies on the setup rows being the flat table's
# leading block -- all settlements, then all roads -- so a sweep index is
# already its flat row and the mask/noise prefixes line up.
assert np.array_equal(np.asarray(_SETUP_IDX), np.arange(_N_SETUP)), (
    "setup rows must be the flat table's leading block "
    "(make_setup_lookahead slices prefixes)"
)
assert np.all(
    np.asarray(_SETUP_TYPE[:N_VERTICES]) == int(ActionType.SETUP_SETTLEMENT)
), "setup block must lead with the settlement rows (make_setup_lookahead)"
assert np.all(np.asarray(_SETUP_TYPE[N_VERTICES:]) == int(ActionType.SETUP_ROAD)), (
    "setup block must end with the road rows (make_setup_lookahead)"
)


def make_setup_lookahead(value: ValueFunction) -> BeliefPolicy:
    """The setup-phase one-ply opener: at a state whose legal actions are all
    setup placements, returns exactly the move
    ``make_search(value, num_simulations=0)`` (at factory defaults) would --
    same successor values, same tie-break noise. At any other state the
    returned index is meaningless (the caller gates on the phase)."""

    def policy(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: Player,
        mask: FlatMask,
    ) -> FlatAction:
        # Mirror ``make_search``'s draws exactly: the tie-break noise is a
        # full-width draw (a narrower shape would change the bits), and the
        # world key is split off through the same num_trees=1 chain.
        noise = jax.random.uniform(key, (N_FLAT,))[:_N_SETUP] * 1e-4
        (tree_key,) = jax.random.split(key, 1)
        world_key, _ = jax.random.split(tree_key)  # unused half: the propose gate
        world = sample_world(world_key, view, player)
        setup_mask = mask[:_N_SETUP]
        # The two setup cores called directly over their own slice of the table
        # (vmapping apply_action's lax.switch would evaluate every branch per
        # row), then stage 2 exactly as apply_action runs it on a setup row:
        # the longest-road recompute gated off (a setup placement is never a
        # BuildRoad / BuildSettlement).
        idx = _SETUP_PARAMS.idx
        settlement_successors, settlement_result = jax.vmap(
            lambda v, legal: _setup_settlement_apply(layout, world, v, legal)
        )(idx[:N_VERTICES], setup_mask[:N_VERTICES])
        road_successors, road_result = jax.vmap(
            lambda e, legal: _setup_road_apply(layout, world, e, legal)
        )(idx[N_VERTICES:], setup_mask[N_VERTICES:])
        successors = jax.tree.map(
            lambda a, b: jnp.concatenate([a, b]),
            settlement_successors,
            road_successors,
        )
        successors, _ = jax.vmap(lambda s, r: resolve_step(s, r, False))(
            successors, jnp.concatenate([settlement_result, road_result])
        )
        values = jax.vmap(value, in_axes=(None, 0, None))(layout, successors, player)
        weights = jnp.where(setup_mask, values, _ILLEGAL)
        return jnp.argmax(jnp.where(setup_mask, weights + noise, -jnp.inf))

    return policy


_Vec = Float[Array, "n players"]


def make_setup_search(
    value: ValueFunction,
    *,
    n_players: int,
    depth: int = 6,
    temperature: float = 2.0,
    beam: int = 4,
) -> BeliefPolicy:
    """A setup-phase :class:`BeliefPolicy`: ``depth``-ply probabilistic expectimax
    over the opening placements, opponents Boltzmann-rational at ``temperature``,
    a full ``beam``-ary tree expanded level by level. Meaningful only at a setup
    state (the caller gates on the phase)."""
    k = min(beam, _N_SETUP)

    def value_vecs(layout: BoardLayout, states: BoardState) -> _Vec:
        """``(n, n_players)`` heuristic value of each state from every seat."""
        return jax.vmap(
            lambda s: jax.vmap(lambda p: value(layout, s, p))(
                jnp.arange(n_players, dtype=jnp.int32)
            )
        )(states)

    def expand(
        layout: BoardLayout, states: BoardState
    ) -> tuple[BoardState, Array, Array, Array]:
        """One level: each frontier state -> its top-``k`` placements. Returns the
        ``(n*k,)`` child states, their flat actions, their legality, and the
        ``(n,)`` movers of the frontier (parents)."""
        movers = jax.vmap(lambda s: agent_selection_single(s).astype(jnp.int32))(states)

        def node(state: BoardState, mover: Player) -> tuple[BoardState, Array, Array]:
            avail = jax.vmap(
                lambda t, prm: action_available(layout, state, t, prm)
            )(_SETUP_TYPE, _SETUP_PARAMS)  # fmt: skip
            succ, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
                layout, state, _SETUP_TYPE, _SETUP_PARAMS, avail
            )
            oneply = jax.vmap(lambda s: value(layout, s, mover))(succ)
            top = jax.lax.top_k(jnp.where(avail, oneply, -jnp.inf), k)[1]
            child = jax.tree.map(lambda x: x[top], succ)
            return child, _SETUP_IDX[top], avail[top]

        child, actions, legal = jax.vmap(node)(states, movers)

        def flat(x: Array) -> Array:
            return x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))

        return jax.tree.map(flat, child), actions.reshape(-1), legal.reshape(-1), movers

    def policy(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: Player,
        mask: FlatMask,
    ) -> FlatAction:
        me = player.astype(jnp.int32)
        state = sample_world(key, view, me)  # exact at setup (nothing hidden yet)

        # Forward: expand the full k-ary tree one level at a time.
        states = jax.tree.map(lambda x: x[None], state)  # the root frontier (n=1)
        movers_at: list[Array] = []
        legal_at: list[Array] = []
        root_actions: Array = jnp.zeros((k,), jnp.int32)
        for d in range(depth):
            states, actions, legal, movers = expand(layout, states)
            movers_at.append(movers)  # movers of the level-d parents
            legal_at.append(legal)  # legality of the level-(d+1) children
            if d == 0:
                root_actions = actions
        vec = value_vecs(layout, states)  # (k**depth, n_players)

        # Backward: aggregate children into parents (max for us, Boltzmann for the
        # opponents); at the root, the searcher takes the argmax placement.
        for d in reversed(range(depth)):
            n_d = k**d
            child_vec = vec.reshape((n_d, k, n_players))
            score = jnp.take_along_axis(child_vec, movers_at[d][:, None, None], axis=2)
            score = jnp.where(legal_at[d].reshape((n_d, k)), score[:, :, 0], -jnp.inf)
            if d == 0:
                my = jnp.where(legal_at[0], child_vec[0, :, me], -jnp.inf)
                return root_actions[jnp.argmax(my)].astype(jnp.int32)
            has = score.max(axis=1, keepdims=True) > -jnp.inf
            dist = jnp.where(
                (movers_at[d] == me)[:, None],
                jax.nn.one_hot(jnp.argmax(score, axis=1), k),
                jax.nn.softmax(jnp.where(has, score, 0.0) / temperature, axis=1),
            )
            dist = jnp.where(has, dist, 1.0 / k)  # all-illegal: children identical
            vec = (dist[:, :, None] * child_vec).sum(1)
        raise AssertionError(
            "unreachable: the d == 0 branch returns"
        )  # pragma: no cover

    return policy
