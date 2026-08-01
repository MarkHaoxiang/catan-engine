"""Head-to-head evaluation: seat agents in a batch of games and count wins."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from settlrl_engine.belief import BeliefState, belief_view
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, KeyScalar
from settlrl_engine.env import (
    ActionParams,
    BatchedSettlrlEnv,
    flat_to_action,
    observe_for,
)

# `_rollout_core` (the engine's fused rollout body, private): compile_evaluate
# rebuilds the actor from *traced* agent parameters inside its own jit, which
# the public `BatchedSettlrlEnv.rollout` (actor identity static, closures
# baked) cannot express. Kept beside `evaluate` so the two paths cannot drift.
from settlrl_engine.env.batched import Actor, AgentSelectionArray, _rollout_core
from settlrl_engine.mechanics.flat import FlatMaskArray
from settlrl_search.policy import (
    AgentSpec,
    BeliefSpec,
    GameAgent,
    ObservationSpec,
    OpponentSpec,
    Policy,
    StatefulSpec,
)


class EvalResult(NamedTuple):
    """Outcome of an :func:`evaluate` run."""

    wins: Float[Array, "seats"]
    """Games won by each seat."""

    episodes: int
    """Total games completed within the step budget."""


_Picker = Callable[
    [KeyScalar, BoardLayout, BoardState, BeliefState | None, FlatMaskArray],
    Int[Array, "batch"],
]
"""Internal: one seat's per-lane flat-action picks, traceable inside a scan."""


def _picker(agent: ObservationSpec | BeliefSpec | Policy, n: int, i: int) -> _Picker:
    """Wrap one agent as seat ``i`` of ``n``."""
    spec = (
        agent
        if isinstance(agent, AgentSpec)
        else ObservationSpec(lambda: agent, frozenset((2, 3, 4)))
    )
    if n not in spec.n_players:
        raise ValueError(f"seat {i} does not support {n}-player games")
    if isinstance(spec, ObservationSpec):
        obs_act = jax.vmap(spec.policy)

        def pick_obs(
            key: KeyScalar,
            layout: BoardLayout,
            state: BoardState,
            belief: BeliefState | None,
            avail: FlatMaskArray,
        ) -> Int[Array, "batch"]:
            b = avail.shape[0]
            obs = observe_for(layout, state, jnp.full((b,), i, jnp.int32))
            return obs_act(jax.random.split(key, b), obs, avail)

        return pick_obs
    belief_act = jax.vmap(spec.policy, in_axes=(0, 0, 0, None, 0))

    def pick_belief(
        key: KeyScalar,
        layout: BoardLayout,
        state: BoardState,
        belief: BeliefState | None,
        avail: FlatMaskArray,
    ) -> Int[Array, "batch"]:
        assert belief is not None  # evaluate() tracks beliefs for BeliefSpec seats
        b = avail.shape[0]
        view = jax.vmap(belief_view, in_axes=(0, 0, None))(state, belief, i)
        return belief_act(jax.random.split(key, b), layout, view, jnp.int32(i), avail)

    return pick_belief


def _actor(pickers: Sequence[_Picker]) -> Actor:
    """Every seat picks a move in every lane; the acting seat's is kept."""

    def actor(
        key: KeyScalar,
        layout: BoardLayout,
        state: BoardState,
        belief: BeliefState | None,
        avail: FlatMaskArray,
        agent_sel: AgentSelectionArray,
    ) -> tuple[jax.Array, ActionParams]:
        keys = jax.random.split(key, len(pickers))
        picks = jnp.stack(
            [
                pick(k, layout, state, belief, avail)
                for pick, k in zip(pickers, keys, strict=True)
            ]
        )
        flat = picks[agent_sel, jnp.arange(avail.shape[0])]
        return flat_to_action(flat)

    return actor


def _evaluate_stepwise(
    agents: Sequence[OpponentSpec | Policy],
    *,
    n_steps: int | None,
    n_episodes: int | None,
    batch_size: int,
    seed: int,
    number_placement: Literal["random", "spiral"],
) -> EvalResult:
    """The per-step Python driver behind :func:`evaluate` when a stateful
    seat is present: same seating and budget semantics, but each step calls
    the acting stateful agents lane by lane (their state lives across calls),
    so nothing fuses into a scan. Auto-reset lanes get fresh agents."""
    n = len(agents)
    env = BatchedSettlrlEnv(
        batch_size=batch_size,
        seed=seed,
        reward="sparse",
        n_players=n,
        number_placement=number_placement,
        track_beliefs=any(isinstance(a, BeliefSpec) for a in agents),
    )
    factories: dict[int, Callable[[int], GameAgent]] = {}
    pickers: dict[int, _Picker] = {}
    for i, a in enumerate(agents):
        if isinstance(a, StatefulSpec):
            if n not in a.n_players:
                raise ValueError(f"seat {i} does not support {n}-player games")
            factories[i] = a.policy
        else:
            # Jitted here: the fused path traces pickers inside its scan, but
            # this loop calls them step by step (eager greedy is ~450x).
            pickers[i] = cast(_Picker, jax.jit(_picker(a, n, i)))

    next_episode = 0

    def fresh_seats() -> dict[int, GameAgent]:
        nonlocal next_episode
        next_episode += 1
        return {i: f(seed + (next_episode - 1) * n + i) for i, f in factories.items()}

    lanes_agents = [fresh_seats() for _ in range(batch_size)]
    key = jax.random.key(seed)
    wins = np.zeros((n,), np.float64)
    total = (
        n_steps
        if n_steps is not None
        else _MAX_STEPS_PER_EPISODE * ((n_episodes or 0) // batch_size + 1)
    )
    for _ in range(total):
        # Re-read the board every step: an auto-reset lane regenerates it.
        layout, state = env.board
        mask = env.flat_mask()
        sel = np.asarray(env.agent_selection)
        flat = np.zeros((batch_size,), np.int32)
        key, k = jax.random.split(key)
        seat_keys = jax.random.split(k, n)
        mask_host: np.ndarray | None = None
        for i in range(n):
            lanes = np.flatnonzero(sel == i)
            if lanes.size == 0:
                continue
            if i in pickers:
                belief = env.beliefs if env.track_beliefs else None
                picks = pickers[i](seat_keys[i], layout, state, belief, mask)
                flat[lanes] = np.asarray(picks)[lanes]
            else:
                # One host fetch per seat-step; agents see numpy lane slices.
                obs = cast("dict[str, np.ndarray]", jax.device_get(env.observe(i)))
                if mask_host is None:
                    mask_host = np.asarray(mask)
                for lane in lanes:
                    obs_l = {k: v[lane] for k, v in obs.items()}
                    flat[lane] = lanes_agents[int(lane)][i].act(
                        obs_l, mask_host[int(lane)]
                    )
        env.step(*flat_to_action(jnp.asarray(flat)))
        wins += np.asarray(env.rewards).sum(axis=0)
        for lane in np.flatnonzero(np.asarray(env.terminations).any(axis=1)):
            lanes_agents[int(lane)] = fresh_seats()
        if n_episodes is not None and int(wins.sum()) >= n_episodes:
            break
    return EvalResult(
        wins=jnp.asarray(wins, jnp.float32), episodes=round(float(wins.sum()))
    )


# Step cap per requested episode in n_episodes mode, guarding against agents
# that never finish a game (a full game is well under this many steps).
_MAX_STEPS_PER_EPISODE = 5_000

# Steps fused into one rollout scan between win-count syncs: long enough to
# amortise the dispatch, short enough to keep the n_episodes overshoot small.
_SYNC_WINDOW = 64


def evaluate(
    agents: Sequence[OpponentSpec | Policy],
    *,
    n_steps: int | None = None,
    n_episodes: int | None = None,
    batch_size: int = 64,
    seed: int = 0,
    number_placement: Literal["random", "spiral"] = "random",
) -> EvalResult:
    """Play ``len(agents)`` seats (2..4 players) under exactly one budget:
    ``n_steps`` env steps, or until at least ``n_episodes`` games finish.

    Each agent (an :class:`AgentSpec`, or a bare :class:`Policy` valid at any
    count) occupies one seat in every game of the batch; finished games
    auto-reset, so ``episodes`` counts every game completed within the budget
    (games still running at the end are discarded; lanes finishing within the
    same sync window can overshoot ``n_episodes``). Deterministic for a given
    configuration and ``seed``. A :class:`StatefulSpec` seat switches the run
    to the per-step Python driver (same semantics, no fused scan, win counts
    sync every step so the overshoot is at most a batch).
    """
    if (n_steps is None) == (n_episodes is None):
        raise ValueError("provide exactly one of n_steps / n_episodes")
    if any(isinstance(a, StatefulSpec) for a in agents):
        return _evaluate_stepwise(
            agents,
            n_steps=n_steps,
            n_episodes=n_episodes,
            batch_size=batch_size,
            seed=seed,
            number_placement=number_placement,
        )
    n = len(agents)
    pure = cast("Sequence[ObservationSpec | BeliefSpec | Policy]", agents)
    env = BatchedSettlrlEnv(
        batch_size=batch_size,
        seed=seed,
        reward="sparse",
        n_players=n,
        number_placement=number_placement,
        track_beliefs=any(isinstance(a, BeliefSpec) for a in agents),
    )
    actor = _actor([_picker(agent, n, i) for i, agent in enumerate(pure)])
    key = jax.random.key(seed)
    wins = jnp.zeros((n,), jnp.float32)
    total = (
        n_steps
        if n_steps is not None
        else _MAX_STEPS_PER_EPISODE * ((n_episodes or 0) // batch_size + 1)
    )
    done = 0
    while done < total:
        window = min(_SYNC_WINDOW, total - done)
        key, k = jax.random.split(key)
        # One fused scan per window; the win count only syncs between windows.
        wins = wins + env.rollout(k, window, actor=actor).sum(axis=0)
        done += window
        if n_episodes is not None and int(wins.sum()) >= n_episodes:
            break
    return EvalResult(wins=wins, episodes=int(wins.sum()))


def compile_evaluate(
    make_agents: Callable[[Any], Sequence[ObservationSpec | BeliefSpec | Policy]],
    *,
    batch_size: int = 64,
    number_placement: Literal["random", "spiral"] = "random",
) -> Callable[..., EvalResult]:
    """:func:`evaluate`'s fused path, compiled once over traced agent parameters.

    Returns ``run(params, *, n_steps=None, n_episodes=None, seed=0) ->
    EvalResult``, bit-identical to ``evaluate(make_agents(params), ...)`` at
    the same budget/seed. ``make_agents`` is called *inside* the compiled
    rollout with ``params`` (any pytree) as traced arrays, so repeated ``run``
    calls with new arrays of the same structure reuse the compiled scan --
    a plain ``evaluate`` call rebuilds its actor closure and retraces instead.
    ``make_agents`` must return the same seating structure for any ``params``
    and pure (non-stateful) agents only."""

    @functools.partial(
        jax.jit,
        static_argnames=(
            "n",
            "n_steps",
            "reward_mode",
            "auto_reset",
            "victory_points_to_win",
        ),
    )
    def _window(
        params: Any,
        carry: Any,
        sample_key: KeyScalar,
        *,
        n: int,
        n_steps: int,
        reward_mode: str,
        auto_reset: bool,
        victory_points_to_win: int,
    ) -> tuple[Any, Float[Array, "batch seats"]]:
        layout, state, avail, vps, belief, extras, env_key = carry
        agents = make_agents(params)
        actor = _actor([_picker(a, n, i) for i, a in enumerate(agents)])
        out = _rollout_core(
            layout, state, avail, vps, belief, extras, env_key, sample_key,
            n_steps, actor, batch_size, reward_mode, auto_reset,
            number_placement, n, victory_points_to_win,
        )  # fmt: skip
        return out[:7], out[7]

    def run(
        params: Any,
        *,
        n_steps: int | None = None,
        n_episodes: int | None = None,
        seed: int = 0,
    ) -> EvalResult:
        if (n_steps is None) == (n_episodes is None):
            raise ValueError("provide exactly one of n_steps / n_episodes")
        agents = make_agents(params)
        n = len(agents)
        for i, a in enumerate(agents):
            if isinstance(a, StatefulSpec):
                raise ValueError(f"seat {i} is stateful; compile_evaluate fuses")
            if isinstance(a, AgentSpec) and n not in a.n_players:
                raise ValueError(f"seat {i} does not support {n}-player games")
        env = BatchedSettlrlEnv(
            batch_size=batch_size,
            seed=seed,
            reward="sparse",
            n_players=n,
            number_placement=number_placement,
            track_beliefs=any(isinstance(a, BeliefSpec) for a in agents),
        )
        carry = (
            env._layout, env._state, env._avail, env._vps, env._belief,
            (env._reward, env._terminations, env._result, env._agent_sel),
            env._key,
        )  # fmt: skip
        key = jax.random.key(seed)
        wins = jnp.zeros((n,), jnp.float32)
        total = (
            n_steps
            if n_steps is not None
            else _MAX_STEPS_PER_EPISODE * ((n_episodes or 0) // batch_size + 1)
        )
        done = 0
        while done < total:
            window = min(_SYNC_WINDOW, total - done)
            key, k = jax.random.split(key)
            carry, cum = _window(
                params, carry, k,
                n=n, n_steps=window, reward_mode=env.reward_mode,
                auto_reset=env.auto_reset,
                victory_points_to_win=env.victory_points_to_win,
            )  # fmt: skip
            wins = wins + cum.sum(axis=0)
            done += window
            if n_episodes is not None and int(wins.sum()) >= n_episodes:
                break
        return EvalResult(wins=wins, episodes=int(wins.sum()))

    return run
