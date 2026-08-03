"""Anchored Elo: a single comparable strength number per checkpoint.

The arena scores the net against a *fixed* set of anchors whose Elo never moves
(``lookahead(heuristic)`` pinned at 0, optionally ``random`` and frozen
self-play checkpoints). :func:`anchored_elo` then places the net on that fixed
scale by maximum likelihood -- the AlphaZero/MuZero anchored-baseline scheme,
not a within-pool round-robin (which drifts when the pool changes).

The number is comparable wherever the anchor set and its pinned Elos are the
same: within a run, across its checkpoints, always; across runs only when they
scored against the same anchors under the same search semantics (the end-of-run
gauntlet, where nothing is skipped).

A training-side module: not imported by the package root.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def expected_score(rating: float, opponent: float) -> float:
    """Logistic win probability of a player at ``rating`` vs ``opponent`` (the
    standard Elo curve, 400-point scale)."""
    return float(1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0)))


def anchored_elo(
    anchors: Iterable[tuple[float, float, int]],
    *,
    lo: float = -4000.0,
    hi: float = 4000.0,
    iters: int = 64,
) -> float:
    """Maximum-likelihood Elo of a player from results vs fixed-Elo anchors.

    ``anchors`` is ``(anchor_elo, wins, games)`` per anchor. The expected total
    score ``sum_a games_a * expected_score(R, elo_a)`` is monotone in ``R``, so
    the MLE solves ``= sum_a wins_a`` by bisection. Wins are continuity-corrected
    to ``[0.5, games-0.5]`` so a saturated anchor (0% / 100%) can't drive ``R`` to
    ``+-inf``. Returns ``nan`` if no anchor has games."""
    data = [(elo, w, g) for elo, w, g in anchors if g > 0]
    if not data:
        return float("nan")
    target = sum(min(max(w, 0.5), g - 0.5) for _, w, g in data)

    def predicted(r: float) -> float:
        return sum(g * expected_score(r, elo) for elo, _, g in data)

    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if predicted(m) < target:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def anchored_elo_se(
    anchors: Iterable[tuple[float, float, int]], *, rating: float | None = None
) -> float:
    """Standard error of :func:`anchored_elo` from the Fisher information at the
    MLE: ``(400/ln 10) / sqrt(sum_a games_a * p_a * (1 - p_a))``, ``p_a`` the
    fitted win probability vs anchor ``a``. ``rating`` overrides the MLE fit.
    Returns ``nan`` if no anchor has games."""
    data = [(elo, w, g) for elo, w, g in anchors if g > 0]
    if not data:
        return float("nan")
    r = anchored_elo(data) if rating is None else rating
    info = sum(
        g * expected_score(r, elo) * (1.0 - expected_score(r, elo))
        for elo, _, g in data
    )
    return (400.0 / math.log(10.0)) / math.sqrt(info) if info > 0 else float("inf")
