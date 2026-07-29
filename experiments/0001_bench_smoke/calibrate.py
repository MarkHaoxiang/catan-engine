"""Anchor calibration: a joint Elo fit over a round-robin among the shipped
policy rungs (``random``/``greedy``/``lookahead``/``mcts``) plus the frozen
``az0_gnn96x4`` checkpoint (0004_alphazero's arena mid-rung).

The fit holds ``lookahead`` pinned at 0 -- settlrl-learn's arena scale
(``anchored_elo`` in ``settlrl_learn.training.elo``) -- and coordinate-ascends
the logistic MLE over the rest (Zermelo's algorithm / the Bradley-Terry MM
update: each player's rating is re-solved against its current opponents via
``anchored_elo`` itself, which is already that per-player MLE against
fixed-Elo opponents; iterating it to a fixed point is the joint MLE for a
connected round-robin).

The result is only valid for arena runs whose search semantics match
:func:`search_semantics` exactly -- see ``experiments/JOURNAL.md``'s
scale-reset entry.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from omegaconf import OmegaConf
from settlrl_agents import BeliefSpec
from settlrl_agents.evaluate import evaluate
from settlrl_learn.training import OpponentSpec
from settlrl_learn.training.elo import anchored_elo, anchored_elo_se

RUNGS: tuple[str, ...] = ("random", "greedy", "lookahead", "mcts")
"""The ``POLICIES`` rungs in the round-robin (registry names)."""

AZ0_NAME = "az0_gnn96x4"
"""The frozen checkpoint's anchor artifact name (``0004_alphazero/anchors/``)."""

FIXED_ELOS: dict[str, float] = {"lookahead": 0.0}
"""The scale's origin: ``anchored_elo``'s convention (heuristic lookahead = 0)."""

# Budgets (task brief): heuristic-vs-heuristic pairs are cheap (n>=600); a pair
# touching mcts or az0 pays search cost (n=300-400).
PAIR_PLAN: list[tuple[str, str, int]] = [
    ("random", "greedy", 600),
    ("random", "lookahead", 600),
    ("greedy", "lookahead", 600),
    ("random", "mcts", 350),
    ("greedy", "mcts", 350),
    ("lookahead", "mcts", 350),
    ("random", AZ0_NAME, 350),
    ("greedy", AZ0_NAME, 350),
    ("lookahead", AZ0_NAME, 350),
    ("mcts", AZ0_NAME, 350),
]
"""Every unordered pair among ``RUNGS`` + ``AZ0_NAME``, once."""

_ALPHAZERO_DIR = Path(__file__).resolve().parents[1] / "0004_alphazero"


class PairResult(NamedTuple):
    """One unordered pair's seat-swapped 2p outcome: ``a``'s wins over
    ``episodes`` completed games (``b`` took the rest -- Catan games never
    draw)."""

    a: str
    b: str
    wins_a: int
    episodes: int


def search_semantics() -> dict[str, object]:
    """The frozen arena + search settings az0 plays the calibration at, read
    straight from ``0004_alphazero``'s conf so this can't drift from what the
    config change pins: ``conf/arena/scale.yaml``'s ``sims``/``considered`` and
    ``conf/search/scale.yaml``'s ``chance_nodes``/``dev_chance``/``ordered``."""
    arena = OmegaConf.to_container(
        OmegaConf.load(_ALPHAZERO_DIR / "conf" / "arena" / "scale.yaml")
    )
    search = OmegaConf.to_container(
        OmegaConf.load(_ALPHAZERO_DIR / "conf" / "search" / "scale.yaml")
    )
    assert isinstance(arena, dict) and isinstance(search, dict)
    return {
        "sims": int(arena["sims"]),
        "considered": int(arena["considered"]),
        "chance_nodes": bool(search["chance_nodes"]),
        "dev_chance": bool(search["dev_chance"]),
        "ordered": bool(search["ordered"]),
    }


def az0_spec(
    *,
    sims: int,
    considered: int,
    chance_nodes: bool,
    dev_chance: bool,
    ordered: bool,
) -> BeliefSpec:
    """The frozen az0 checkpoint as a seatable 2p spec, played by its own GNN
    search at the given (arena-scale) budget -- mirrors 0004_alphazero's
    ``arena_helpers.py::build_net_opponents``."""
    if str(_ALPHAZERO_DIR) not in sys.path:
        sys.path.insert(0, str(_ALPHAZERO_DIR))
    # cross-framework sibling import (0004_alphazero/anchors.py); no stub
    # there since script dirs aren't packages.
    from anchors import (  # type: ignore[import-not-found]
        NET_OPPONENT_SETUP_BEAM,
        NET_OPPONENT_SETUP_DEPTH,
        NET_OPPONENT_SETUP_TEMPERATURE,
        load_anchor,
    )
    from settlrl_learn.training import GNNBackend

    net, netcfg = load_anchor(AZ0_NAME)
    # explicit, not default-coincidence: this calibration IS what pinned
    # 0004_alphazero's anchors.NET_OPPONENT_SETUP_* (see that module's comment).
    backend = GNNBackend(
        netcfg, setup_depth=NET_OPPONENT_SETUP_DEPTH,
        setup_temperature=NET_OPPONENT_SETUP_TEMPERATURE,
        setup_beam=NET_OPPONENT_SETUP_BEAM,
        chance_nodes=chance_nodes, dev_chance=dev_chance, ordered=ordered,
    )  # fmt: skip
    agent = backend.play_agent(
        net, num_simulations=sims, max_num_considered_actions=considered
    )
    return BeliefSpec(lambda agent=agent: agent, frozenset((2,)))


def play_pair(
    a_name: str,
    a_spec: OpponentSpec,
    b_name: str,
    b_spec: OpponentSpec,
    *,
    n_games: int,
    seed: int,
    batch_size: int = 64,
) -> PairResult:
    """``a``'s wins over ``b``, seat-swapped at 2p (mirrors ``cli.bench``'s
    per-seat loop, generalized to pre-built specs rather than registry
    names -- the az0 spec is not a ``build_spec`` string)."""
    half = n_games // 2
    r0 = evaluate([a_spec, b_spec], n_episodes=half, batch_size=batch_size, seed=seed)
    r1 = evaluate(
        [b_spec, a_spec],
        n_episodes=n_games - half,
        batch_size=batch_size,
        seed=seed + 1,
    )
    wins_a = int(r0.wins[0]) + int(r1.wins[1])
    episodes = r0.episodes + r1.episodes
    return PairResult(a_name, b_name, wins_a, episodes)


def _index(results: list[PairResult]) -> dict[str, list[tuple[str, int, int]]]:
    """Each player's ``(opponent, wins, games)`` anchors, both directions of
    every recorded pair."""
    idx: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for r in results:
        idx[r.a].append((r.b, r.wins_a, r.episodes))
        idx[r.b].append((r.a, r.episodes - r.wins_a, r.episodes))
    return idx


def joint_fit(
    results: list[PairResult],
    fixed: dict[str, float],
    *,
    iters: int = 200,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Coordinate-ascent joint Elo MLE over a round-robin: hold ``fixed``
    ratings pinned, solve each free player's rating against its current
    opponents via :func:`~settlrl_learn.training.elo.anchored_elo`, repeat to
    convergence."""
    idx = _index(results)
    free = sorted(set(idx) - set(fixed))
    ratings: dict[str, float] = {**fixed, **dict.fromkeys(free, 0.0)}
    for _ in range(iters):
        delta = 0.0
        for p in free:
            anchors = [(ratings[opp], float(w), g) for opp, w, g in idx[p]]
            new_r = anchored_elo(anchors)
            delta = max(delta, abs(new_r - ratings[p]))
            ratings[p] = new_r
        if delta < tol:
            break
    return ratings


def joint_fit_se(
    results: list[PairResult], ratings: dict[str, float]
) -> dict[str, float]:
    """Fisher SE per rung (Fisher information at the fitted point, the same
    formula ``anchored_elo_se`` uses for a single MLE rating)."""
    idx = _index(results)
    return {
        p: anchored_elo_se(
            [(ratings[o], float(w), g) for o, w, g in opps], rating=ratings[p]
        )
        for p, opps in idx.items()
    }


def sanity_gates(
    ratings: dict[str, float], az0_name: str = AZ0_NAME
) -> dict[str, bool]:
    """The task's sanity checks: ``lookahead`` pinned at 0 by construction;
    ``greedy`` within a wide band around the -330 expectation; ``random`` well
    below every heuristic rung; ``az0`` strictly between ``random``-tier and
    ``lookahead``."""
    return {
        "lookahead_zero": ratings["lookahead"] == 0.0,
        "greedy_in_range": -480.0 <= ratings["greedy"] <= -180.0,
        "random_below": ratings["random"] < -600.0,
        "az0_in_range": -400.0 < ratings[az0_name] < 0.0,
    }
