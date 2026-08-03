"""Paired search duels over one frozen net: the guard's measurement layer.

An :class:`Arm` is one search configuration. :func:`duel` plays two of them
against each other with the focus arm rotated through every seat on paired
seeds, so the only difference between the sides is search behavior; the
anchors' pinned setup opener plays the setup phase on both sides.
:func:`seconds_per_move` prices an arm's search in wall-clock, which an
equal-simulation duel does not.

The net and the frozen setup semantics come from ``0004_alphazero/anchors.py``
(cross-framework sibling import).
"""

from __future__ import annotations

import json
import time
from math import ceil
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, NamedTuple

from settlrl_learn.experiment import sibling_module

if TYPE_CHECKING:
    from jaxtyping import Array
    from settlrl_engine.board.layout import BoardLayout
    from settlrl_engine.board.state import IntScalar
    from settlrl_learn.training import SearchSettings
    from settlrl_search.policy import BeliefPolicy, OpponentSpec

_ALPHAZERO_DIR = Path(__file__).resolve().parents[1] / "0004_alphazero"

GATE_ELO = 35.0
"""The production ship threshold (``../CLAUDE.md``) this screen sizes against:
the arena-Elo gate a full training A/B is judged by."""

_FLAGS = ("expected_rolls", "chance_nodes", "dev_chance", "ordered")
"""The search flags an arm sets — the whole difference between two arms."""


class Arm(NamedTuple):
    """One side of a duel: the search behavior under test and its budget."""

    name: str
    expected_rolls: bool
    chance_nodes: bool
    dev_chance: bool
    ordered: bool
    num_simulations: int


class MatchResult(NamedTuple):
    """One arm's wins over the games a match decided."""

    wins: int
    episodes: int

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.episodes, 1)

    @property
    def standard_error(self) -> float:
        """Binomial standard error of :attr:`win_rate`."""
        p = self.win_rate
        return float((p * (1.0 - p) / max(self.episodes, 1)) ** 0.5)


def anchor_semantics(anchor: str) -> dict[str, bool]:
    """The search flags the anchor's recorded strength was measured under, from
    its sidecar's ``search_semantics``. ``expected_rolls`` is not among them —
    the sidecar pins the arena's block, and the arena inherits whatever the play
    path defaults to."""
    anchors = sibling_module(_ALPHAZERO_DIR, "anchors")
    meta = json.loads((anchors.ANCHOR_DIR / f"{anchor}.json").read_text())
    semantics = meta["search_semantics"]
    return {k: bool(semantics[k]) for k in ("chance_nodes", "dev_chance", "ordered")}


def load_anchor(name: str) -> Any:
    """The frozen net from ``0004``'s committed anchors (its architecture rides
    on the net, so the search seams need nothing else)."""
    return sibling_module(_ALPHAZERO_DIR, "anchors").load_anchor(name)[0]


def search_settings(arm: Arm, *, considered: int) -> SearchSettings:
    """The search configuration ``arm`` runs under, built through the production
    settings model — which silently forces ``expected_rolls`` off under
    ``chance_nodes``. Raises when that rewrites a flag the arm set: an arm whose
    flags don't survive validation is not the arm a verdict would name."""
    from settlrl_learn.training import SearchSettings

    settings = SearchSettings(
        num_simulations=arm.num_simulations, max_considered=considered,
        expected_rolls=arm.expected_rolls, chance_nodes=arm.chance_nodes,
        dev_chance=arm.dev_chance, ordered=arm.ordered,
    )  # fmt: skip
    asked = {flag: getattr(arm, flag) for flag in _FLAGS}
    effective = {flag: getattr(settings, flag) for flag in _FLAGS}
    if asked != effective:
        raise ValueError(
            f"arm {arm.name!r} asked for {asked} and the search would run "
            f"{effective} -- the duel would vary flags the variant never named"
        )
    return settings


def arm_spec(arm: Arm, net: Any, *, n_players: int, considered: int) -> OpponentSpec:
    """``arm``'s seatable agent playing ``net``, under the anchors' frozen setup
    opener — so two arms differ only in their search."""
    from settlrl_agents import BeliefSpec

    agent = _net_agent(
        net, search_settings(arm, considered=considered), n_players=n_players
    )
    return BeliefSpec(lambda: agent, frozenset((n_players,)))


def _net_agent(net: Any, settings: SearchSettings, *, n_players: int) -> BeliefPolicy:
    """``training.make_net_agent`` (the setup opener up to the first non-setup
    legal action, the net's search after it) with every flag of ``settings``
    reaching the search.

    Composed here rather than through ``GNNBackend.play_agent``: an arm is a
    whole ``SearchSettings`` (``max_depth``, ``value_scale``, ``fused_leaf``),
    wider than the backend's keyword surface, and a duel has to move exactly
    the flags its variant names.
    """
    import jax.numpy as jnp
    from settlrl_engine.mechanics.action import ActionType
    from settlrl_learn.nn.board_gnn import gnn_seams
    from settlrl_learn.training import setup_policy
    from settlrl_search import make_search
    from settlrl_search.rows import ROW_TYPE

    setup_rows = (int(ActionType.SETUP_SETTLEMENT) == ROW_TYPE) | (
        int(ActionType.SETUP_ROAD) == ROW_TYPE
    )
    anchors = sibling_module(_ALPHAZERO_DIR, "anchors")
    value_fn, prior_fn = gnn_seams(net)
    search = make_search(
        value_fn, prior=prior_fn,
        num_simulations=settings.num_simulations,
        max_depth=settings.max_depth,
        max_num_considered_actions=settings.max_considered,
        value_scale=settings.value_scale,
        expected_rolls=settings.expected_rolls,
        chance_nodes=settings.chance_nodes,
        dev_chance=settings.dev_chance,
        ordered=settings.ordered,
        fused_leaf=settings.fused_leaf,
    )  # fmt: skip
    setup = setup_policy(
        n_players,
        setup_depth=anchors.NET_OPPONENT_SETUP_DEPTH,
        setup_temperature=anchors.NET_OPPONENT_SETUP_TEMPERATURE,
        setup_beam=anchors.NET_OPPONENT_SETUP_BEAM,
    )

    def policy(
        key: Array, layout: BoardLayout, view: Any, player: IntScalar, mask: Array
    ) -> Array:
        main_legal = (mask & ~setup_rows).any()
        return jnp.where(
            main_legal,
            search(key, layout, view, player, mask),
            setup(key, layout, view, player, mask),
        )

    return policy


def duel(
    focus: OpponentSpec,
    opponent: OpponentSpec,
    *,
    n_players: int,
    games: int,
    batch: int,
    seed: int,
) -> MatchResult:
    """``focus``'s wins with it rotated through every seat of an otherwise
    all-``opponent`` table, ``max(1, games // n_players)`` games per seating on
    paired seeds (``seed + position``).

    At 2 players this is the arena's seat swap
    (``settlrl_learn.evaluation.arena_spec``) with the same seeds; the no-edge
    reference is ``1 / n_players``. ``episodes`` is the decided-game count,
    which overshoots the request by up to a batch per seating."""
    from settlrl_agents import evaluate

    wins = episodes = 0
    per_seating = max(1, games // n_players)
    for position in range(n_players):
        agents: list[OpponentSpec] = [opponent] * n_players
        agents[position] = focus
        result = evaluate(
            agents, n_episodes=per_seating, batch_size=batch, seed=seed + position
        )
        wins += round(float(result.wins[position]))
        episodes += result.episodes
    return MatchResult(wins, episodes)


def seconds_per_move(
    spec: OpponentSpec,
    *,
    n_players: int,
    batch: int,
    steps: int,
    seed: int,
    repeats: int,
) -> float:
    """Median wall-clock a table of ``spec`` spends per searched move, over
    ``repeats`` timed windows of ``steps`` env steps; one untimed call first
    pays the compilation. Every repeat replays the same window, so the spread
    across them is measurement noise."""
    from settlrl_agents.evaluate import compile_evaluate

    def make_agents(_params: Any) -> Any:
        return [spec] * n_players

    def play_window() -> float:
        # float() of the win count is the device sync -- without it the window
        # is still in flight when the clock stops.
        start = time.perf_counter()
        float(compiled(None, n_steps=steps, seed=seed).wins.sum())
        return time.perf_counter() - start

    compiled = compile_evaluate(make_agents, batch_size=batch)
    play_window()  # compilation, untimed
    elapsed = median(play_window() for _ in range(repeats))
    return elapsed / (steps * batch * n_players)


def elo_delta(result: MatchResult) -> tuple[float, float]:
    """The focus arm's Elo and its standard error on the scale that pins its
    opponent at 0 — the arena's own anchored-Elo MLE
    (``settlrl_learn.evaluation.elo``). The model is pairwise, so this reads
    only at 2 players."""
    from settlrl_learn.evaluation.elo import anchored_elo, anchored_elo_se

    anchors = [(0.0, float(result.wins), result.episodes)]
    return anchored_elo(anchors), anchored_elo_se(anchors)


def games_for_elo(elo: float = GATE_ELO, *, win_rate: float = 0.5) -> int:
    """Decided games at which a win rate's 2-sigma half-width shrinks to
    ``elo``'s win-rate offset — what a duel costs to resolve an edge that
    size."""
    from settlrl_learn.evaluation.elo import expected_score

    offset = expected_score(elo, 0.0) - 0.5
    return ceil(4.0 * win_rate * (1.0 - win_rate) / offset**2)
