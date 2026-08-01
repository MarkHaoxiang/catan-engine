"""The anchor-calibration joint Elo fit.

Pure-Python: no JAX, no GPU. Synthetic win matrices are built at the *exact*
expected score for known Elos -- what an infinite-sample round-robin converges
to -- so the joint MLE must recover them (minus the fixed anchor) to high
precision. Composition-only checks (``PAIR_PLAN``, ``search_semantics``) run
here too since they're free (no JAX import, cheap file reads).
"""

from __future__ import annotations

import math
from types import ModuleType

import pytest
from conftest import load_run
from settlrl_learn.training.elo import anchored_elo, expected_score


@pytest.fixture
def calibrate() -> ModuleType:
    return load_run("0004_alphazero", module="calibrate")


def _synthetic_results(
    calibrate: ModuleType, true_elos: dict[str, float], games: int
) -> list[object]:
    names = sorted(true_elos)
    results = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            wins_a = round(games * expected_score(true_elos[a], true_elos[b]))
            results.append(calibrate.PairResult(a, b, wins_a, games))
    return results


def test_joint_fit_recovers_known_elos(calibrate: ModuleType) -> None:
    true_elos = {
        "lookahead": 0.0,
        "random": -900.0,
        "greedy": -300.0,
        "mcts": 150.0,
        "az0": -60.0,
    }
    results = _synthetic_results(calibrate, true_elos, games=200_000)
    fitted = calibrate.joint_fit(results, {"lookahead": 0.0})
    for name, elo in true_elos.items():
        assert fitted[name] == pytest.approx(elo, abs=2.0)


def test_joint_fit_degenerates_to_anchored_elo(calibrate: ModuleType) -> None:
    # One free player vs one fixed anchor: the joint fit's coordinate ascent
    # should take exactly its first update and match anchored_elo directly.
    results = [calibrate.PairResult("random", "lookahead", 40, 200)]
    fitted = calibrate.joint_fit(results, {"lookahead": 0.0})
    assert fitted["random"] == pytest.approx(anchored_elo([(0.0, 40, 200)]), abs=1e-6)


def test_joint_fit_se_is_finite_and_positive(calibrate: ModuleType) -> None:
    true_elos = {"lookahead": 0.0, "random": -900.0, "greedy": -300.0}
    results = _synthetic_results(calibrate, true_elos, games=1000)
    ratings = calibrate.joint_fit(results, {"lookahead": 0.0})
    ses = calibrate.joint_fit_se(results, ratings)
    for name in true_elos:
        assert math.isfinite(ses[name]) and ses[name] > 0.0


def test_index_is_symmetric_no_draws(calibrate: ModuleType) -> None:
    results = [calibrate.PairResult("a", "b", 30, 100)]
    idx = calibrate._index(results)
    assert idx["a"] == [("b", 30, 100)]
    assert idx["b"] == [("a", 70, 100)]


def test_pair_plan_covers_every_unordered_pair_once(calibrate: ModuleType) -> None:
    names = {*calibrate.RUNGS, calibrate.AZ0_NAME}
    seen: set[frozenset[str]] = set()
    for a, b, n in calibrate.PAIR_PLAN:
        assert a in names and b in names and a != b
        pair = frozenset((a, b))
        assert pair not in seen, f"duplicate pair {pair}"
        seen.add(pair)
        involves_search = "mcts" in pair or calibrate.AZ0_NAME in pair
        assert n >= (300 if involves_search else 600)
    assert len(seen) == math.comb(len(names), 2)


def test_search_semantics_reads_0004_scale_conf(calibrate: ModuleType) -> None:
    s = calibrate.search_semantics()
    assert s["sims"] == 24
    assert s["considered"] == 16
    assert s["chance_nodes"] is False
    assert s["dev_chance"] is True
    assert s["ordered"] is False


def test_sanity_gates_thresholds(calibrate: ModuleType) -> None:
    ok = {
        "lookahead": 0.0,
        "greedy": -330.0,
        "random": -900.0,
        calibrate.AZ0_NAME: -150.0,
    }
    assert all(calibrate.sanity_gates(ok).values())

    bad = {**ok, "random": -400.0}  # not < -600
    gates = calibrate.sanity_gates(bad)
    assert not gates["random_below"]
    assert (
        gates["lookahead_zero"] and gates["greedy_in_range"] and gates["az0_in_range"]
    )
