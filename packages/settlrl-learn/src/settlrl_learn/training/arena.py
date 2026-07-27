"""The Stage-1 gate: the net's win rate vs. a fixed ``POLICIES`` opponent,
seat-swapped at 2p.

A learned value worth shipping beats ``lookahead(heuristic)``; ``random`` is the
lower-bound sanity check. The agent (search, plus any setup delegation) comes from
the backend, so this is net-agnostic.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from settlrl_agents import POLICIES, BeliefSpec, evaluate

from settlrl_learn.training.backend import Backend


class ArenaResult(NamedTuple):
    """Seat-swapped match outcome: net wins over actual completed episodes."""

    wins: float
    episodes: int

    @property
    def winrate(self) -> float:
        return self.wins / max(self.episodes, 1)


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
    """The net's wins/episodes vs. ``POLICIES[opponent]``, seat-swapped at 2p.

    ``episodes`` is the actual completed-game count and may differ from the
    requested ``n_games``."""

    def make_agent() -> Any:
        return backend.play_agent(
            net,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
        )

    net_spec = BeliefSpec(make_agent, frozenset((2,)))
    base = POLICIES[opponent]
    half = max(1, n_games // 2)
    r1 = evaluate([net_spec, base], n_episodes=half, batch_size=batch_size, seed=seed)
    r2 = evaluate(
        [base, net_spec], n_episodes=half, batch_size=batch_size, seed=seed + 1
    )
    return ArenaResult(float(r1.wins[0] + r2.wins[1]), int(r1.episodes + r2.episodes))
