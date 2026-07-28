"""The optional action-ordering lock-out (``settlrl_engine.ordering``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from settlrl_engine.board import make_board
from settlrl_engine.board.state import BoardState, GamePhase
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_engine.mechanics.action import ActionType
from settlrl_engine.mechanics.flat import FLAT_ATYPE, flat_available_b
from settlrl_engine.ordering import ORDER_CATEGORY, next_category, ordering_mask

_ATYPE = np.asarray(FLAT_ATYPE)
_FLAT_CAT = np.asarray(ORDER_CATEGORY)[_ATYPE]


def _main_state() -> BoardState:
    _, state = make_board(batch_size=1, seed=0, n_players=2)
    single: BoardState = jax.tree.map(lambda x: x[0], state)
    return single._replace(phase=jnp.uint8(GamePhase.MAIN))


def test_category_zero_imposes_no_constraint() -> None:
    m = np.asarray(ordering_mask(_main_state(), jnp.int32(0)))
    assert bool(m.all())  # category 0 -> everything allowed


def test_lockout_blocks_earlier_categories() -> None:
    # At category 5 (a road was built), only road (5), settlement (6), and
    # uncategorised actions (0, incl. END_TURN) stay legal; dev/trade/buy/city go.
    m = np.asarray(ordering_mask(_main_state(), jnp.int32(5)))
    for t in (
        ActionType.PLAY_KNIGHT,
        ActionType.MARITIME_TRADE,
        ActionType.BUY_DEVELOPMENT_CARD,
        ActionType.BUILD_CITY,
    ):
        assert not m[int(t) == _ATYPE].any(), t
    for t in (ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT, ActionType.END_TURN):
        assert m[int(t) == _ATYPE].all(), t


def test_lockout_inactive_outside_main() -> None:
    _, state = make_board(batch_size=1, seed=0, n_players=2)
    roll = jax.tree.map(lambda x: x[0], state)._replace(phase=jnp.uint8(GamePhase.ROLL))
    assert bool(np.asarray(ordering_mask(roll, jnp.int32(6))).all())  # no MAIN -> free


def test_next_category_resets_on_turn_change_else_runs_max() -> None:
    settle = jnp.int32(ActionType.BUILD_SETTLEMENT)
    road = jnp.int32(ActionType.BUILD_ROAD)
    # running max within a turn (road=5 then settlement=6 -> 6; not back down)
    c = next_category(jnp.int32(0), road, jnp.bool_(False))
    assert int(c) == 5
    assert int(next_category(c, settle, jnp.bool_(False))) == 6
    assert int(next_category(jnp.int32(6), road, jnp.bool_(False))) == 6  # no decrease
    # turn change wipes it
    assert int(next_category(jnp.int32(6), road, jnp.bool_(True))) == 0
    # a non-main action (roll) never raises the category
    assert (
        int(
            next_category(
                jnp.int32(0), jnp.int32(ActionType.ROLL_DICE), jnp.bool_(False)
            )
        )
        == 0
    )


def test_settlement_then_road_is_a_blocked_transposition() -> None:
    # The canonical order is road-before-settlement: once a settlement (6) is
    # built, building a road (5) is locked out -- the transposition is cut.
    after_settlement = next_category(
        jnp.int32(0), jnp.int32(ActionType.BUILD_SETTLEMENT), jnp.bool_(False)
    )
    m = np.asarray(ordering_mask(_main_state(), after_settlement))
    assert not m[int(ActionType.BUILD_ROAD) == _ATYPE].any()


def test_env_track_ordering_overlay_only_removes_and_fires() -> None:
    # Over random play, the constrained mask is always a subset of the engine's
    # true legality (the overlay never adds a move), and it genuinely fires in
    # MAIN at least once (some real-legal action gets locked out).
    env = BatchedSettlrlEnv(batch_size=8, n_players=2, seed=0, track_ordering=True)
    key = jax.random.key(0)
    fired = False
    for _ in range(150):
        real = np.asarray(flat_available_b(*env.board))
        masked = np.asarray(env.flat_mask())
        assert not (masked & ~real).any()  # overlay only removes
        in_main = np.asarray(env.board[1].phase) == int(GamePhase.MAIN)
        if (in_main[:, None] & (real & ~masked)).any():
            fired = True
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
    assert fired  # the lock-out actually constrained some MAIN position


def test_env_track_ordering_off_is_unconstrained() -> None:
    off = BatchedSettlrlEnv(batch_size=8, n_players=2, seed=0)
    assert bool(
        (np.asarray(off.flat_mask()) == np.asarray(flat_available_b(*off.board))).all()
    )
