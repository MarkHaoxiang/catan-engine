"""Setup-phase opener contracts.

``make_setup_lookahead`` is the setup-restricted fast path of the one-ply
lookahead ``make_search(value, num_simulations=0)``; its contract is
*bit-identity* -- at every setup-phase position both must pick the same move
under the same key (same sampled world, same successor values, same tie-break
noise bits).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from settlrl_engine.belief import belief_view
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, GamePhase, Player
from settlrl_engine.env import BatchedSettlrlEnv, flat_to_action
from settlrl_search import make_search
from settlrl_search.expectimax import make_setup_lookahead
from settlrl_search.value import Value
from test_ismcts import heuristic_value


def test_setup_lookahead_matches_the_full_sweep_bitwise() -> None:
    # Drive a batch through its whole setup phase (8 plies per lane at 2p) and
    # a few main-loop steps beyond. On every setup-phase lane the restricted
    # sweep must return the full sweep's move exactly; on main-loop lanes its
    # output is unspecified (the caller discards it) but must still evaluate.
    batch = 4
    env = BatchedSettlrlEnv(batch_size=batch, seed=5, n_players=2, track_beliefs=True)
    full = jax.jit(
        jax.vmap(
            make_search(heuristic_value, num_simulations=0), in_axes=(0, 0, 0, 0, 0)
        )
    )
    fast = jax.jit(
        jax.vmap(make_setup_lookahead(heuristic_value), in_axes=(0, 0, 0, 0, 0))
    )

    # A constant value forces every legal row into an exact tie, so the argmax
    # is decided purely by the tie-break noise bits -- pinning the RNG mirroring
    # (draw shape and split order) that value differences would otherwise mask.
    def constant_value(layout: BoardLayout, state: BoardState, player: Player) -> Value:
        return jnp.zeros((), jnp.float32)

    full_tied = jax.jit(
        jax.vmap(
            make_search(constant_value, num_simulations=0), in_axes=(0, 0, 0, 0, 0)
        )
    )
    fast_tied = jax.jit(
        jax.vmap(make_setup_lookahead(constant_value), in_axes=(0, 0, 0, 0, 0))
    )
    view_of = jax.jit(jax.vmap(belief_view, in_axes=(0, 0, 0)))
    key = jax.random.key(0)
    compared = 0
    for _ in range(12):
        layout, state = env.board
        selection = jnp.asarray(env.agent_selection)
        mask = env.flat_mask()
        view = view_of(state, env.beliefs, selection)
        key, step_key = jax.random.split(key)
        keys = jax.random.split(step_key, batch)
        is_setup = np.asarray(state.phase <= int(GamePhase.SETUP_ROAD))
        full_move = np.asarray(full(keys, layout, view, selection, mask))
        fast_move = np.asarray(fast(keys, layout, view, selection, mask))
        assert np.array_equal(fast_move[is_setup], full_move[is_setup])
        tied_full = np.asarray(full_tied(keys, layout, view, selection, mask))
        tied_fast = np.asarray(fast_tied(keys, layout, view, selection, mask))
        assert np.array_equal(tied_fast[is_setup], tied_full[is_setup])
        compared += int(is_setup.sum())
        env.step(*flat_to_action(jnp.asarray(full_move)))
    assert compared == 8 * batch  # every lane's full setup phase was compared
