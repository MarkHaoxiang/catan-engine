"""The Stage-1 gate: the net's win rate vs. a fixed opponent -- a ``POLICIES``
entry, any pre-built spec, or a :class:`NetOpponent` (a frozen checkpoint) --
seat-swapped at 2p.

A learned value worth shipping beats ``lookahead(heuristic)``; ``random`` is the
lower-bound sanity check. The agent (search, plus any setup delegation) comes from
the backend, so this is net-agnostic.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
from settlrl_agents import POLICIES, BeliefSpec, evaluate
from settlrl_agents.evaluate import EvalResult, compile_evaluate
from settlrl_search.policy import OpponentSpec as OpponentSpec
from settlrl_search.policy import StatefulSpec

if TYPE_CHECKING:
    from settlrl_learn.training.backends.base import Backend


class ArenaResult(NamedTuple):
    """Seat-swapped match outcome: net wins over actual completed episodes."""

    wins: float
    episodes: int

    @property
    def winrate(self) -> float:
        return self.wins / max(self.episodes, 1)


class NetOpponent(NamedTuple):
    """A frozen net seated as an arena opponent, played by its own
    ``backend.play_agent`` at the calling arena's search budget.

    Unlike a pre-built spec (whose arrays a policy closure bakes in), the net's
    arrays enter the cached evaluate callable as traced arguments, so every
    round it plays reuses one compiled program."""

    backend: Backend
    net: Any


class _CachedEval(NamedTuple):
    backend: Any
    opponent: Any
    net_static: Any
    opp_static: Any
    run: Callable[..., EvalResult]


_EVAL_CACHE: dict[tuple[Any, ...], _CachedEval] = {}
"""``(id(backend), opponent id, net seats first, sims, considered, batch) ->``
the compiled evaluate callable built under it. Mirrors
``loop._CALLABLES_CACHE``: unbounded and id-keyed, with the referents held
alive in the value so their ids stay reserved; both nets' arrays are traced
arguments, never closure-baked, so a new round's net is not a retrace."""

_EVAL_BUILDS = 0
"""Compiled-callable builds (cache misses) -- test introspection."""


def _eval_callable(
    backend: Backend,
    net: Any,
    opponent: OpponentSpec | NetOpponent,
    *,
    net_first: bool,
    num_simulations: int,
    max_num_considered_actions: int,
    batch_size: int,
) -> tuple[Callable[..., EvalResult], Any]:
    """The memoised compiled evaluate for one seating, plus the traced
    ``(net arrays, opponent arrays)`` params to call it with."""
    global _EVAL_BUILDS
    net_opp = opponent if isinstance(opponent, NetOpponent) else None
    net_arrays, net_static = eqx.partition(net, eqx.is_array)
    opp_arrays, opp_static = eqx.partition(
        net_opp.net if net_opp is not None else None, eqx.is_array
    )
    params = (net_arrays, opp_arrays)
    opp_key = ("net", id(net_opp.backend)) if net_opp is not None else id(opponent)
    key = (
        id(backend), opp_key, net_first,
        num_simulations, max_num_considered_actions, batch_size,
    )  # fmt: skip
    hit = _EVAL_CACHE.get(key)
    if (
        hit is not None
        and eqx.tree_equal(hit.net_static, net_static)
        and eqx.tree_equal(hit.opp_static, opp_static)
    ):
        return hit.run, params

    def make_agents(params: Any) -> list[Any]:
        arrays, opp_arrays = params
        agent = backend.play_agent(
            eqx.combine(arrays, net_static),
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
        )
        net_spec = BeliefSpec(lambda: agent, frozenset((2,)))
        if net_opp is not None:
            opp_agent = net_opp.backend.play_agent(
                eqx.combine(opp_arrays, opp_static),
                num_simulations=num_simulations,
                max_num_considered_actions=max_num_considered_actions,
            )
            opp_spec: Any = BeliefSpec(lambda: opp_agent, frozenset((2,)))
        else:
            opp_spec = opponent
        return [net_spec, opp_spec] if net_first else [opp_spec, net_spec]

    run = compile_evaluate(make_agents, batch_size=batch_size)
    _EVAL_BUILDS += 1
    _EVAL_CACHE[key] = _CachedEval(backend, opponent, net_static, opp_static, run)
    return run, params


def arena_spec(
    backend: Backend,
    net: Any,
    *,
    opponent: OpponentSpec | NetOpponent,
    n_games: int = 40,
    num_simulations: int = 64,
    max_num_considered_actions: int = 16,
    batch_size: int = 16,
    seed: int = 0,
) -> ArenaResult:
    """The net's wins/episodes vs. a pre-built opponent, seat-swapped at 2p.

    ``episodes`` is the actual completed-game count and may differ from the
    requested ``n_games``. The evaluate callable is compiled once per
    (backend, opponent, seat order, search budget, batch) and memoised with
    the nets' arrays as traced arguments, so repeated rounds never retrace; a
    stateful opponent takes the uncached stepwise :func:`evaluate` instead."""
    half = max(1, n_games // 2)
    if isinstance(opponent, StatefulSpec):
        # The stepwise driver fuses no scan, so there is nothing to memoise.
        def make_agent() -> Any:
            return backend.play_agent(
                net,
                num_simulations=num_simulations,
                max_num_considered_actions=max_num_considered_actions,
            )

        net_spec = BeliefSpec(make_agent, frozenset((2,)))
        r1 = evaluate(
            [net_spec, opponent], n_episodes=half, batch_size=batch_size, seed=seed
        )
        r2 = evaluate(
            [opponent, net_spec], n_episodes=half, batch_size=batch_size, seed=seed + 1
        )
    else:
        budget = {
            "num_simulations": num_simulations,
            "max_num_considered_actions": max_num_considered_actions,
            "batch_size": batch_size,
        }
        run1, params = _eval_callable(backend, net, opponent, net_first=True, **budget)
        run2, _ = _eval_callable(backend, net, opponent, net_first=False, **budget)
        r1 = run1(params, n_episodes=half, seed=seed)
        r2 = run2(params, n_episodes=half, seed=seed + 1)
    return ArenaResult(float(r1.wins[0] + r2.wins[1]), int(r1.episodes + r2.episodes))


def arena(
    backend: Backend,
    net: Any,
    *,
    opponent: str = "lookahead",
    n_games: int = 40,
    num_simulations: int = 64,
    max_num_considered_actions: int = 16,
    batch_size: int = 16,
    seed: int = 0,
) -> ArenaResult:
    """:func:`arena_spec` against the registry entry ``POLICIES[opponent]``."""
    return arena_spec(
        backend, net, opponent=POLICIES[opponent], n_games=n_games,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        batch_size=batch_size, seed=seed,
    )  # fmt: skip
