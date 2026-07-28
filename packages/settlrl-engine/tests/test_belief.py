"""Soundness of belief tracking (``settlrl_engine.belief``) under random play.

The contract: everything in a ``BeliefState`` is derivable from public
information, the bounds always bracket the truth, and a ``BeliefView`` carries
nothing the observer couldn't know — structurally, since its type has no field
for anything hidden.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from settlrl_engine.belief import BeliefState, PublicState, make_belief, update_belief
from settlrl_engine.board import Board, give, make_board, to_main
from settlrl_engine.board.dev_cards import DEV_CARD_COUNTS, DevCard
from settlrl_engine.board.state import BoardState
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_engine.mechanics.action import (
    ActionParams,
    ActionType,
    action_available,
    apply_action,
)

_DEV_COUNTS = jnp.asarray(DEV_CARD_COUNTS, jnp.int32)


def _single(board: Board) -> BoardState:
    """Lane-0 single-game ``BoardState`` (belief functions are unbatched)."""
    return cast(BoardState, jax.tree.map(lambda x: x[0], board[1]))


def _belief_pinned_to(n_players: int, hands: list[list[int]]) -> BeliefState:
    """A belief where every observer's bounds are pinned exactly to ``hands``
    (the truth). Lets a single transition's bound *deltas* be read off cleanly."""
    P = n_players
    truth = jnp.asarray(hands, jnp.uint8)  # (P, R)
    bounds = jnp.broadcast_to(truth[None], (P, P, truth.shape[1]))
    return BeliefState(
        res_lo=bounds,
        res_hi=bounds,
        dev_played=jnp.zeros_like(make_belief(1, P).dev_played[0]),
    )


def _rollout(env: BatchedSettlrlEnv, n_steps: int, seed: int = 42) -> None:
    key = jax.random.key(seed)
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))


# 2p (exact-belief regime) and 4p (third-party-steal slack) are the boundaries;
# 3p is interior — the diff logic branches on "third party / not", not the count.
@pytest.mark.parametrize("n_players", [2, 4])
def test_bounds_bracket_the_truth(n_players: int) -> None:
    env = BatchedSettlrlEnv(
        batch_size=4, seed=0, n_players=n_players, track_beliefs=True
    )
    key = jax.random.key(42)
    for _ in range(100):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
        true = env.board[1].player_resources.astype(jnp.int32)  # (B, P, R)
        lo = env.beliefs.res_lo.astype(jnp.int32)  # (B, O, P, R)
        hi = env.beliefs.res_hi.astype(jnp.int32)
        assert bool(jnp.all(lo <= true[:, None]))
        assert bool(jnp.all(true[:, None] <= hi))
        # Each observer knows its own hand exactly.
        o = jnp.arange(n_players)
        assert bool(jnp.all(lo[:, o, o] == true))
        assert bool(jnp.all(hi[:, o, o] == true))


@pytest.mark.parametrize("n_players", [2, 4])
def test_dev_pool_conservation(n_players: int) -> None:
    env = BatchedSettlrlEnv(
        batch_size=4, seed=1, n_players=n_players, track_beliefs=True
    )
    _rollout(env, 100)
    state = env.board[1]
    held = state.dev_hand.astype(jnp.int32).sum(axis=1)
    played = env.beliefs.dev_played.astype(jnp.int32)
    assert bool(
        jnp.all(state.dev_deck.astype(jnp.int32) + held + played == _DEV_COUNTS)
    )


def test_two_player_beliefs_are_exact() -> None:
    # With two players every flow is mutually visible (a steal involves both
    # seats) and public totals pin the rest: the bounds must stay tight.
    env = BatchedSettlrlEnv(batch_size=4, seed=2, n_players=2, track_beliefs=True)
    key = jax.random.key(7)
    for _ in range(100):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
        assert bool(jnp.all(env.beliefs.res_lo == env.beliefs.res_hi))


def test_third_party_steals_create_uncertainty() -> None:
    # The converse: with 4 players, random play eventually produces a steal
    # some observer didn't take part in, and their bounds must open up. The
    # slack is checked per step (not at the end): bounds legitimately
    # re-tighten as public totals pin emptied hands back down.
    env = BatchedSettlrlEnv(batch_size=4, seed=3, n_players=4, track_beliefs=True)
    key = jax.random.key(42)
    max_slack = 0
    for _ in range(120):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
        slack = env.beliefs.res_hi.astype(jnp.int32) - env.beliefs.res_lo.astype(
            jnp.int32
        )
        max_slack = max(max_slack, int(slack.sum()))
    assert max_slack > 0


_HIDDEN_FIELDS = {"player_resources", "dev_hand", "dev_deck", "dev_bought", "key"}


def test_every_board_field_is_classified() -> None:
    # PublicState plus hidden must cover BoardState exactly: a new BoardState
    # field can't slip into (or out of) the view layer unclassified.
    assert set(PublicState._fields) | _HIDDEN_FIELDS == set(BoardState._fields)
    assert set(PublicState._fields) & _HIDDEN_FIELDS == set()


def test_belief_view_carries_nothing_hidden() -> None:
    env = BatchedSettlrlEnv(batch_size=4, seed=4, n_players=4, track_beliefs=True)
    _rollout(env, 100)
    state = env.board[1]
    for me in range(4):
        view = env.belief_view(me)
        # Structural: the view's type has no field for anything hidden.
        for name in _HIDDEN_FIELDS:
            assert not hasattr(view, name) and not hasattr(view.public, name)
        # The public fields are the truth's, field for field.
        for name in PublicState._fields:
            assert bool(jnp.all(getattr(view.public, name) == getattr(state, name)))
        # Own knowledge is exact; bounds are the observer's slice.
        assert bool(jnp.all(view.own_dev == state.dev_hand[:, me]))
        assert bool(jnp.all(view.belief.res_lo == env.beliefs.res_lo[:, me]))
        assert bool(jnp.all(view.belief.res_hi == env.beliefs.res_hi[:, me]))
        # Own purchases survive only on the observer's turn.
        own_turn = state.current_player.astype(jnp.int32) == me
        assert bool(
            jnp.all(
                view.own_bought == jnp.where(own_turn[:, None], state.dev_bought, 0)
            )
        )
        # The unseen dev pool by conservation.
        pool = (
            _DEV_COUNTS
            - env.beliefs.dev_played.astype(jnp.int32)
            - state.dev_hand[:, me].astype(jnp.int32)
        )
        assert bool(jnp.all(view.unseen_dev.astype(jnp.int32) == pool))
        # The public counts are the truth's.
        pb = view.belief
        assert bool(
            jnp.all(
                pb.hand_size == state.player_resources.astype(jnp.int32).sum(axis=2)
            )
        )
        assert bool(
            jnp.all(pb.dev_count == state.dev_hand.astype(jnp.int32).sum(axis=2))
        )
        assert bool(
            jnp.all(
                pb.res_total == state.player_resources.astype(jnp.int32).sum(axis=1)
            )
        )


def test_belief_resets_with_auto_reset() -> None:
    # Run long enough for lanes to finish (2p games end quickly): beliefs of
    # replaced lanes must restart from the empty-board belief, staying sound.
    env = BatchedSettlrlEnv(batch_size=4, seed=5, n_players=2, track_beliefs=True)
    _rollout(env, 1_700)
    true = env.board[1].player_resources.astype(jnp.int32)
    assert bool(jnp.all(env.beliefs.res_lo.astype(jnp.int32) <= true[:, None]))
    assert bool(jnp.all(true[:, None] <= env.beliefs.res_hi.astype(jnp.int32)))


def test_tracking_off_raises() -> None:
    env = BatchedSettlrlEnv(batch_size=1, seed=0, n_players=2)
    with pytest.raises(RuntimeError, match="track_beliefs"):
        _ = env.beliefs


def test_belief_unchanged_under_invalid_action() -> None:
    # An INVALID action leaves the board untouched (before == after), so the
    # belief diff is empty: update_belief over that transition must be a no-op.
    # Use a real env-tracked (board, belief) pair so the belief is a fixpoint for
    # the board (an inconsistent belief would legitimately move under _tighten).
    env = BatchedSettlrlEnv(batch_size=1, seed=0, n_players=4, track_beliefs=True)
    _rollout(env, 80)
    layout = jax.tree.map(lambda x: x[0], env.board[0])
    state = jax.tree.map(lambda x: x[0], env.board[1])
    belief = jax.tree.map(lambda x: x[0], env.beliefs)

    # An action illegal in any phase: an unowned-tile MOVE_ROBBER outside MAIN /
    # robber phases. Confirm illegality, then that the board is a true no-op.
    params = ActionParams(idx=jnp.int32(0), target=jnp.int32(-1))
    at = jnp.int32(ActionType.BUILD_CITY)
    avail = action_available(layout, state, at, params)
    assert not bool(avail)  # the action really is illegal
    after, _ = apply_action(layout, state, at, params, avail)
    for name in BoardState._fields:
        if name == "key":
            continue
        assert bool(jnp.all(getattr(after, name) == getattr(state, name)))

    out = update_belief(belief, state, after, at, params)
    assert bool(jnp.all(out.res_lo == belief.res_lo))
    assert bool(jnp.all(out.res_hi == belief.res_hi))
    assert bool(jnp.all(out.dev_played == belief.dev_played))


def test_steal_bound_semantics() -> None:
    # Isolate a 4p robber steal so seats 2 and 3 are third parties. Build the
    # (before, after) by hand: thief (0) takes one card off victim (1); seats 2,3
    # only learn a card moved, not its type. Start every observer pinned to the
    # truth so the post-steal bound deltas read off cleanly.
    P = 4
    victim_before = [0, 2, 1, 0, 0]  # 3 cards: 2 wheat, 1 wood
    thief_before = [1, 0, 0, 0, 0]
    hands_before = [thief_before, victim_before, [0] * 5, [0] * 5]
    # After: one wheat moved 1 -> 0 (the engine picks a type; choose wheat).
    victim_after = [0, 1, 1, 0, 0]
    thief_after = [1, 1, 0, 0, 0]
    hands_after = [thief_after, victim_after, [0] * 5, [0] * 5]

    base = to_main(make_board(seed=0, n_players=P), player=0)
    before = _single(give(give(base, 0, thief_before), 1, victim_before))
    after = _single(give(give(base, 0, thief_after), 1, victim_after))
    at = jnp.int32(ActionType.MOVE_ROBBER)
    params = ActionParams(idx=jnp.int32(0), target=jnp.int32(1))  # steal from seat 1

    belief = _belief_pinned_to(P, hands_before)
    out = update_belief(belief, before, after, at, params)
    lo = out.res_lo.astype(jnp.int32)
    hi = out.res_hi.astype(jnp.int32)
    bef_lo = belief.res_lo.astype(jnp.int32)
    bef_hi = belief.res_hi.astype(jnp.int32)

    truth_after = jnp.asarray(hands_after, jnp.int32)
    # Bounds still bracket the truth for every observer.
    assert bool(jnp.all(lo <= truth_after[None]))
    assert bool(jnp.all(truth_after[None] <= hi))

    # Thief (0) and victim (1) saw the type: their rows are exact post-steal.
    for o in (0, 1):
        assert bool(jnp.all(lo[o] == truth_after))
        assert bool(jnp.all(hi[o] == truth_after))

    # Third parties (2, 3): the steal type stays hidden -- they do NOT pin it.
    for o in (2, 3):
        # Victim could have lost any held type: lower bound drops by exactly 1
        # for each type the victim provably held (here wheat: 2->... and wood),
        # never by more than 1, and only for held types.
        d_lo_victim = lo[o, 1] - bef_lo[o, 1]
        assert bool(jnp.all(d_lo_victim >= -1))
        assert bool(jnp.all((bef_lo[o, 1] == 0) | (d_lo_victim <= 0)))
        # The thief may now hold one more of any type the victim could have had:
        # upper bound rises by exactly 1 for those types, by 0 otherwise.
        d_hi_thief = hi[o, 0] - bef_hi[o, 0]
        could = bef_hi[o, 1] > 0
        assert bool(jnp.all(d_hi_thief <= 1))
        assert bool(jnp.all(jnp.where(could, d_hi_thief >= 0, d_hi_thief == 0)))
        # The exact stolen type is NOT pinned: the victim's wheat slot keeps slack
        # (lo < hi), so seat o cannot tell wheat from wood was taken.
        assert bool(lo[o, 1, 1] < hi[o, 1, 1])


def test_monopoly_reveals_pin_the_type() -> None:
    # PLAY_MONOPOLY on a resource is a public surrender: afterwards every
    # observer knows every player's exact count of that type (lo == hi).
    P = 4
    # Before: wheat scattered across seats with hidden hands.
    hands_before = [[0, 2, 0, 0, 0], [0, 1, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, 0, 0, 0]]
    # Player 0 monopolizes wheat (type 1): all wheat flows to seat 0.
    hands_after = [[0, 6, 0, 0, 0], [0] * 5, [0] * 5, [0] * 5]

    base = to_main(make_board(seed=1, n_players=P), player=0)
    before = base
    after = base
    for p in range(P):
        before = give(before, p, hands_before[p])
        after = give(after, p, hands_after[p])
    before_s, after_s = _single(before), _single(after)
    # The played MONOPOLY card must show up as a hand decrease for the diff to
    # register the reveal, so seat 0 holds (and plays) one.
    before_s = before_s._replace(
        dev_hand=before_s.dev_hand.at[0, DevCard.MONOPOLY].set(1)
    )
    at = jnp.int32(ActionType.PLAY_MONOPOLY)
    params = ActionParams(idx=jnp.int32(1), target=jnp.int32(-1))  # monopolize wheat

    # Hidden hands: third parties don't know who holds the wheat going in.
    belief = make_belief(1, P)
    belief = jax.tree.map(lambda x: x[0], belief)
    # Open every non-own row wide for wheat so the test exercises a real collapse.
    res_hi = belief.res_hi.astype(jnp.int32).at[:, :, 1].set(6)
    belief = belief._replace(res_hi=res_hi.astype(jnp.uint8))

    out = update_belief(belief, before_s, after_s, at, params)
    lo = out.res_lo.astype(jnp.int32)[:, :, 1]  # (observers, players) wheat
    hi = out.res_hi.astype(jnp.int32)[:, :, 1]
    truth = jnp.asarray(hands_after, jnp.int32)[:, 1]  # (players,) wheat after
    # Every observer's wheat bounds collapse to exact, for every player.
    assert bool(jnp.all(lo == hi))
    assert bool(jnp.all(lo == truth[None]))
