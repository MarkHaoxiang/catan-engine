"""Command-line tools over the agent registry (``compare``, ``bench``)."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Annotated, NamedTuple

import typer
from settlrl_search.policy import OpponentSpec

from settlrl_agents import POLICIES
from settlrl_agents.evaluate import evaluate
from settlrl_agents.value import make_heuristic, tuned_value


class CompareResult(NamedTuple):
    """Outcome of a :func:`compare` run (totals are per agent, in name order)."""

    names: tuple[str, str]
    wins: tuple[int, int]
    first_seat_wins: tuple[int, int]
    """Each agent's wins from the games where it was seated first."""

    first_seat_episodes: tuple[int, int]
    """How many games each agent was seated first in."""

    episodes: int


def compare(
    a: str,
    b: str,
    *,
    n_games: int = 100,
    batch_size: int = 32,
    seed: int = 0,
) -> CompareResult:
    """Head-to-head between two registered agents over ``n_games`` two-player
    games, seats swapped halfway to cancel the first-mover advantage.

    ``a`` / ``b`` are ``POLICIES`` names. Lanes finishing on the same step can
    overshoot ``n_games`` slightly; every finished game is counted.
    """
    for name in (a, b):
        if name not in POLICIES:
            raise ValueError(f"unknown agent {name!r} (choose from {sorted(POLICIES)})")
    half = n_games // 2
    r_ab = evaluate(
        [POLICIES[a], POLICIES[b]],
        n_episodes=half,
        batch_size=batch_size,
        seed=seed,
    )
    r_ba = evaluate(
        [POLICIES[b], POLICIES[a]],
        n_episodes=n_games - half,
        batch_size=batch_size,
        seed=seed + 1,
    )
    return CompareResult(
        names=(a, b),
        wins=(
            int(r_ab.wins[0]) + int(r_ba.wins[1]),
            int(r_ab.wins[1]) + int(r_ba.wins[0]),
        ),
        first_seat_wins=(int(r_ab.wins[0]), int(r_ba.wins[0])),
        first_seat_episodes=(r_ab.episodes, r_ba.episodes),
        episodes=r_ab.episodes + r_ba.episodes,
    )


def build_spec(text: str) -> OpponentSpec:
    """An agent spec from a registry name or a JSON configuration.

    JSON shape: ``{"kind": <name>, "params": {<make kwargs>},
    "value": {<make_heuristic weights>}}`` — ``params`` are knob overrides for
    the family builder; ``value`` builds a reweighted heuristic for the
    families that take one, or the string ``"tuned"`` for the count-tuned
    weights.
    """
    if text in POLICIES:
        return POLICIES[text]
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"unknown agent {text!r} (a registry name or JSON spec)"
        ) from exc
    kind = doc.get("kind")
    if kind not in POLICIES:
        raise ValueError(
            f"unknown agent kind {kind!r} (choose from {sorted(POLICIES)})"
        )
    base = POLICIES[kind]
    overrides = dict(doc.get("params") or {})
    if doc.get("value") == "tuned":
        overrides["value"] = tuned_value
    elif doc.get("value") is not None:
        overrides["value"] = make_heuristic(**doc["value"])
    if not overrides:
        return base
    return dataclasses.replace(base, defaults={**base.defaults, **overrides})


class BenchResult(NamedTuple):
    """Outcome of a :func:`bench` run, from agent ``a``'s side."""

    wins_a: int
    wins_b: int
    episodes: int
    rate: float
    """``a``'s win share of all finished games (chance: 1 / players)."""

    se: float
    """Binomial standard error of ``rate``."""

    by_position: tuple[tuple[int, int], ...]
    """Per seating of ``a`` (seat 0 first): (a's wins, episodes)."""


def bench(
    spec_a: str,
    spec_b: str,
    *,
    n_games: int = 200,
    players: int = 2,
    batch_size: int = 32,
    seed: int = 0,
) -> BenchResult:
    """Head-to-head between two (possibly configured) agents.

    At 2 players the seats swap halfway; at 3-4, ``a`` rotates through every
    seat with ``b`` filling the rest. Specs are :func:`build_spec` strings.
    """
    a, b = build_spec(spec_a), build_spec(spec_b)
    per = n_games // players
    wins_a = wins_b = episodes = 0
    by_position: list[tuple[int, int]] = []
    for pos in range(players):
        agents: list[OpponentSpec] = [b] * players
        agents[pos] = a
        r = evaluate(
            agents,
            n_episodes=per if pos < players - 1 else n_games - per * (players - 1),
            batch_size=batch_size,
            seed=seed + pos,
        )
        wa = int(r.wins[pos])
        by_position.append((wa, r.episodes))
        wins_a += wa
        wins_b += r.episodes - wa
        episodes += r.episodes
    rate = wins_a / episodes if episodes else 0.0
    se = math.sqrt(rate * (1.0 - rate) / episodes) if episodes else 0.0
    return BenchResult(wins_a, wins_b, episodes, rate, se, tuple(by_position))


def _format_bench(result: BenchResult, players: int) -> str:
    pos = ", ".join(f"seat {i}: {w}/{n}" for i, (w, n) in enumerate(result.by_position))
    return (
        f"a={result.wins_a} b={result.wins_b} n={result.episodes} "
        f"rate={100 * result.rate:.1f}% se={100 * result.se:.1f}% "
        f"(chance {100 / players:.1f}%)\n  a seated at {pos}"
    )


def _format(result: CompareResult) -> str:
    (a, b), (wa, wb) = result.names, result.wins
    (fa, fb), (ea, eb) = result.first_seat_wins, result.first_seat_episodes
    rate = 100.0 * wa / result.episodes if result.episodes else 0.0
    return (
        f"{a} vs {b}: {wa}-{wb} over {result.episodes} games ({a} {rate:.1f}%)\n"
        f"  seated first: {a} {fa}/{ea}, {b} {fb}/{eb}"
    )


app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Tools over the shipped agent registry.",
)


@app.command(name="compare", help="head-to-head between two agents (seat-swapped)")
def _compare_cmd(
    agent_a: str,
    agent_b: str,
    games: int = 100,
    batch_size: int = 32,
    seed: int = 0,
) -> None:
    try:
        result = compare(
            agent_a, agent_b, n_games=games, batch_size=batch_size, seed=seed
        )
    except ValueError as exc:  # unknown agent -> a clean CLI error, not a traceback
        raise typer.BadParameter(str(exc)) from exc
    print(_format(result))


@app.command(
    name="bench",
    help="head-to-head between configured agents (names or JSON specs; "
    "seat-swapped at 2 players, seat-rotated at 3-4)",
)
def _bench_cmd(
    spec_a: str,
    spec_b: str,
    games: int = 200,
    players: int = 2,
    batch_size: int = 32,
    seed: int = 0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="emit one JSON object (for experiment gates)"),
    ] = False,
) -> None:
    try:
        result = bench(
            spec_a,
            spec_b,
            n_games=games,
            players=players,
            batch_size=batch_size,
            seed=seed,
        )
    except ValueError as exc:  # unknown spec -> a clean CLI error
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        print(json.dumps({**result._asdict(), "players": players}))
    else:
        print(_format_bench(result, players))


def main() -> None:
    app()
