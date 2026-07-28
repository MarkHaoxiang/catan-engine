"""Protocol-level tests, run against every shipped agent.

An agent in the ``POLICIES`` registry must pick a legal flat action whenever
one exists, be able to drive whole games in self-play, and be a pure function
of its inputs (same seed -> same trajectory). Each agent is exercised at a
player count it supports, through whichever protocol (observation / belief)
it declares.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from settlrl_agents import POLICIES, BeliefSpec, ObservationSpec, StatefulSpec, evaluate
from settlrl_engine.belief import BeliefState, BeliefView, belief_view
from settlrl_engine.board import Board, give, make_board, to_main
from settlrl_engine.env import (
    ActionType,
    BatchedSettlrlEnv,
    Observation,
    flat_to_action,
    observe_for,
)
from settlrl_engine.mechanics.flat import FLAT_ATYPE, flat_available_for
from settlrl_engine.mechanics.trade import pack_trade_single, propose_trade_step

BATCH = 4

# Each family at its for_testing parameters -- the protocol properties under
# test are parameter-independent, so the cheap member suffices.
SPECS = {name: spec.for_tests for name, spec in POLICIES.items()}


def _acting_obs(env: BatchedSettlrlEnv) -> Observation:
    """Per-lane observation of that lane's acting player."""
    per_seat = [env.observe(i) for i in range(env.n_players)]
    lanes = jnp.arange(env.batch_size)
    return cast(
        Observation,
        jax.tree.map(lambda *xs: jnp.stack(xs)[env.agent_selection, lanes], *per_seat),
    )


def _acting_view(env: BatchedSettlrlEnv) -> BeliefView:
    """Per-lane ``BeliefView`` of that lane's acting player."""
    per_seat = [env.belief_view(i) for i in range(env.n_players)]
    lanes = jnp.arange(env.batch_size)
    return cast(
        BeliefView,
        jax.tree.map(lambda *xs: jnp.stack(xs)[env.agent_selection, lanes], *per_seat),
    )


Spec = ObservationSpec | BeliefSpec | StatefulSpec


def _self_play(spec: Spec, seed: int, n_steps: int) -> tuple[jax.Array, jax.Array]:
    """Drive ``n_steps`` of self-play; return the per-step ``(masks, actions)``."""
    n_players = max(spec.n_players)
    env = BatchedSettlrlEnv(
        batch_size=BATCH,
        seed=seed,
        n_players=n_players,
        track_beliefs=isinstance(spec, BeliefSpec),
    )
    act: Callable[[jax.Array], jax.Array]
    if isinstance(spec, ObservationSpec):
        obs_act = jax.jit(jax.vmap(spec.policy))
        act = lambda keys: obs_act(keys, _acting_obs(env), env.flat_mask())  # noqa: E731
    elif isinstance(spec, StatefulSpec):
        # One stateful agent per (lane, seat), driven lane by lane in Python.
        seats = [
            {s: spec.policy(seed + lane * n_players + s) for s in range(n_players)}
            for lane in range(BATCH)
        ]

        def stateful_act(keys: jax.Array) -> jax.Array:
            mask = np.asarray(env.flat_mask())
            sel = np.asarray(env.agent_selection)
            picks = []
            for lane in range(BATCH):
                obs = cast(
                    "dict[str, np.ndarray]",
                    jax.device_get(env.observe(int(sel[lane]))),
                )
                obs_l = {k: v[lane] for k, v in obs.items()}
                picks.append(seats[lane][int(sel[lane])].act(obs_l, mask[lane]))
            return jnp.asarray(picks, jnp.int32)

        act = stateful_act
    else:
        belief_act = jax.jit(jax.vmap(spec.policy))
        act = lambda keys: belief_act(  # noqa: E731
            keys, env.board[0], _acting_view(env), env.agent_selection, env.flat_mask()
        )
    key = jax.random.key(seed)
    masks, actions = [], []
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        mask = env.flat_mask()
        flat = act(jax.random.split(k, BATCH))
        masks.append(mask)
        actions.append(flat)
        env.step(*flat_to_action(flat))
    return jnp.stack(masks), jnp.stack(actions)


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_self_play_is_legal_and_reproducible(spec: Spec) -> None:
    # One rollout carries both properties: legality is checked on it directly,
    # and a second same-seed rollout must replay it exactly (so its legality
    # follows without re-checking).
    masks, actions = _self_play(spec, seed=3, n_steps=60)
    legal = jnp.take_along_axis(masks, actions[..., None], axis=2)[..., 0]
    assert bool(jnp.all(~masks.any(axis=2) | legal))

    _, replayed = _self_play(spec, seed=3, n_steps=60)
    assert bool(jnp.all(actions == replayed))


SHEEP, WOOD = 0, 2
_ACCEPT_ROW = int(np.flatnonzero(np.asarray(FLAT_ATYPE) == ActionType.ACCEPT_TRADE)[0])
_REJECT_ROW = int(np.flatnonzero(np.asarray(FLAT_ATYPE) == ActionType.REJECT_TRADE)[0])


def _pending_trade_board(responder_hand: list[int]) -> Board:
    """A 3p TRADE_RESPONSE board: player 0 has offered player 1 wood for sheep.

    Player 2 is far ahead on victory points, so the best-opponent term of the
    two-sided heuristic is theirs (unchanged by the trade) and the responder's
    decision reduces to its own side of the swap.
    """
    board = to_main(make_board(1, seed=0, n_players=3))
    board = give(board, 0, [0, 0, 4, 0, 0])
    board = give(board, 1, responder_hand)
    layout, st = board
    st = st._replace(victory_points=st.victory_points.at[0, 2].set(5))
    idx, target = pack_trade_single(WOOD, SHEEP, partner=1)
    st, _ = propose_trade_step((layout, st), (jnp.array([idx]), jnp.array([target])))
    return layout, st


# Jitted once per agent name and reused across the accept/reject cases below --
# eager dispatch of these policies' full op graph is orders of magnitude
# slower than a single compile (root sweep, ~40s eager vs <1s jitted+cached).
_JITTED_POLICY = {name: jax.jit(SPECS[name].policy) for name in ("greedy", "lookahead")}


@pytest.mark.parametrize("name", ["greedy", "lookahead"])
@pytest.mark.parametrize("favorable", [True, False], ids=["accepts", "rejects"])
def test_responds_to_trades_by_benefit(name: str, favorable: bool) -> None:
    # Favorable: pay 1 of 6 sheep for a first wood. Unfavorable: pay the only
    # sheep for a seventh wood.
    hand = [6, 0, 0, 0, 0] if favorable else [1, 0, 6, 0, 0]
    layout, state = _pending_trade_board(hand)
    layout0, state0 = jax.tree.map(lambda x: x[0], (layout, state))
    mask = flat_available_for(layout0, state0)
    spec = SPECS[name]
    policy = _JITTED_POLICY[name]
    key = jax.random.key(0)
    if isinstance(spec, ObservationSpec):
        obs = cast(
            Observation,
            jax.tree.map(
                lambda x: x[0], observe_for(layout, state, jnp.array([1], jnp.int32))
            ),
        )
        flat = policy(key, obs, mask)
    else:
        assert isinstance(spec, BeliefSpec)  # the parametrized names
        res = state0.player_resources  # exact public knowledge: lo == hi == truth
        belief = BeliefState(
            res_lo=jnp.broadcast_to(res, (3, *res.shape)),
            res_hi=jnp.broadcast_to(res, (3, *res.shape)),
            dev_played=jnp.zeros_like(state0.dev_deck),
        )
        view = belief_view(state0, belief, 1)
        flat = policy(key, layout0, view, jnp.int32(1), mask)
    assert int(flat) == (_ACCEPT_ROW if favorable else _REJECT_ROW)


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_self_play_rollouts_complete_games(spec: Spec) -> None:
    # The episode budget stops as soon as a game finishes (instead of a fixed
    # step count), which is what bounds the expensive search agents' runtime.
    # Three seats so domestic trade is live: a proposer stuck re-offering a
    # trade its partner keeps rejecting would stall the game and fail here (a
    # stall never reaches one completion, so the test hangs/fails as intended).
    # One completion at batch_size=2 is the cheapest form of the guard: cost is
    # ~game_length x lanes-until-first-finish, far below running games to a
    # two-finish budget at the full batch.
    result = evaluate([spec, spec, spec], n_episodes=1, batch_size=2, seed=0)
    assert result.episodes >= 1
