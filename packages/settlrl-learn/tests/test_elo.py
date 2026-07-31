"""Anchored-Elo math contracts (pure, no JAX): the per-checkpoint strength
number the arena reports."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from settlrl_learn.training.elo import anchored_elo, expected_score


def test_expected_score_endpoints() -> None:
    assert expected_score(0.0, 0.0) == 0.5  # equal ratings -> coin flip
    # the canonical 400-point step is ~10:1 odds (~0.909).
    assert math.isclose(expected_score(400.0, 0.0), 10 / 11, rel_tol=1e-9)
    assert math.isclose(
        expected_score(0.0, 400.0) + expected_score(400.0, 0.0), 1.0, abs_tol=1e-12
    )


def test_parity_vs_anchor_is_the_anchor_rating() -> None:
    # 50% vs an anchor at R puts the player exactly at R, wherever R sits.
    for anchor in (-300.0, 0.0, 250.0):
        r = anchored_elo([(anchor, 100, 200)])
        assert abs(r - anchor) < 1e-3


def test_55_percent_winrate_maps_to_35_elo() -> None:
    # vs the heuristic pinned at 0, 55% winrate is +~35 Elo (this derives the
    # gate_elo threshold); 45% is the mirror.
    up = anchored_elo([(0.0, 110, 200)])
    down = anchored_elo([(0.0, 90, 200)])
    assert math.isclose(up, 400 * math.log10(0.55 / 0.45), abs_tol=0.5)
    assert math.isclose(up, -down, abs_tol=1e-3)  # symmetric about parity


def test_monotone_in_winrate() -> None:
    elos = [anchored_elo([(0.0, w, 200)]) for w in (40, 80, 100, 140, 180)]
    assert all(b > a for a, b in pairwise(elos))


def test_saturated_anchor_is_finite() -> None:
    # 100% / 0% vs an anchor must not send R to +-inf (continuity correction).
    assert math.isfinite(anchored_elo([(0.0, 200, 200)]))
    assert math.isfinite(anchored_elo([(0.0, 0, 200)]))
    # a saturated `random` anchor far below shouldn't swamp a discriminating one:
    # heuristic@0 at 55% dominates random@-800 at 100%.
    r = anchored_elo([(0.0, 110, 200), (-800.0, 200, 200)])
    assert 0.0 < r < 120.0


def test_no_games_is_nan() -> None:
    assert math.isnan(anchored_elo([]))
    assert math.isnan(anchored_elo([(0.0, 0, 0)]))


def test_anchored_elo_se_single_even_anchor() -> None:
    from settlrl_learn.training.elo import anchored_elo_se

    # 20/40 vs a 0-Elo anchor: MLE rating 0, SE = (400/ln10)/sqrt(40*0.25) ~= 54.93
    assert anchored_elo_se([(0.0, 20.0, 40)]) == pytest.approx(54.93, abs=0.05)


def test_anchored_elo_se_saturated_anchor_adds_little() -> None:
    from settlrl_learn.training.elo import anchored_elo_se

    base = anchored_elo_se([(0.0, 20.0, 40)])
    with_saturated = anchored_elo_se([(0.0, 20.0, 40), (-800.0, 39.5, 40)])
    with_even = anchored_elo_se([(0.0, 20.0, 40), (0.0, 20.0, 40)])
    # another near-parity anchor cuts the SE ~sqrt(2)x; a saturated one barely moves it
    assert with_even < with_saturated < base


def test_anchored_elo_se_empty_is_nan() -> None:
    from settlrl_learn.training.elo import anchored_elo_se

    assert math.isnan(anchored_elo_se([]))
