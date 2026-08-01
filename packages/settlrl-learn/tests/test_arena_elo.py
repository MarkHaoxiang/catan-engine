"""The arena's wiring into the Elo report: `run_arena`'s real-counts feed, the
per-opponent schedules, the net-opponent seeds, and the name-based `arena`
wrapper. The arena itself is stubbed out -- no games are played."""

from __future__ import annotations

import sys
from typing import Any, cast

from settlrl_agents import POLICIES, BeliefSpec
from settlrl_learn.training import MLPBackend
from settlrl_learn.training.arena import ArenaResult, arena
from settlrl_learn.training.config import ArenaConfig
from settlrl_learn.training.elo import anchored_elo, anchored_elo_se
from settlrl_learn.training.steps import run_arena


def test_run_arena_uses_real_counts_for_elo_and_reports_se(monkeypatch: Any) -> None:
    # run_arena must feed the *actual* (wins, episodes) arena returns into the
    # Elo MLE -- not wr * cfg.games. Two anchors with different overshoot ratios
    # (50/40 vs 20/40 episodes-per-nominal-game) make the real-counts and
    # nominal-counts Elo provably different -- a single anchor can't discriminate
    # the two paths, since anchored_elo there depends only on the win ratio.
    results = {
        "lookahead": ArenaResult(wins=30.0, episodes=50),
        "random": ArenaResult(wins=10.0, episodes=20),
    }
    monkeypatch.setattr(
        "settlrl_learn.training.steps.arena",
        lambda *a, opponent, **k: results[opponent],
    )
    cfg = ArenaConfig(
        games=40,
        opponents=["lookahead", "random"],
        anchor_elos={"lookahead": 0.0, "random": -1115.0},
    )
    metrics = run_arena(MLPBackend((16,)), object(), cfg, seed=0, round_index=1)
    real_inputs = [(0.0, 30.0, 50), (-1115.0, 10.0, 20)]
    nominal_inputs = [
        (0.0, 0.6 * 40, 40),
        (-1115.0, 0.5 * 40, 40),
    ]  # the nominal (games-requested x winrate) feed -- run_arena must not use it
    assert metrics["arena_winrate"] == results["lookahead"].winrate
    assert metrics["arena_elo"] == anchored_elo(real_inputs)
    assert metrics["arena_elo"] != anchored_elo(nominal_inputs)
    assert metrics["arena_elo_se"] == anchored_elo_se(real_inputs)


def test_run_arena_opponent_every_skips_off_rounds(monkeypatch: Any) -> None:
    # opponent_every={"random": 5} plays random only on round_index multiples of
    # 5; lookahead (absent from the map) plays every round. A skipped opponent
    # contributes no arena_vs_<opp> metric and no Elo input that round.
    calls: list[str] = []
    results = {
        "lookahead": ArenaResult(wins=30.0, episodes=50),
        "random": ArenaResult(wins=10.0, episodes=20),
    }

    def _fake_arena(*a: Any, opponent: str, **k: Any) -> ArenaResult:
        calls.append(opponent)
        return results[opponent]

    monkeypatch.setattr("settlrl_learn.training.steps.arena", _fake_arena)
    cfg = ArenaConfig(
        games=40,
        opponents=["lookahead", "random"],
        anchor_elos={"lookahead": 0.0, "random": -1115.0},
        opponent_every={"random": 5},
    )
    backend = MLPBackend((16,))

    for round_index in range(1, 5):
        calls.clear()
        metrics = run_arena(backend, object(), cfg, seed=0, round_index=round_index)
        assert calls == ["lookahead"]
        assert "arena_vs_random" not in metrics
        assert metrics["arena_elo"] == anchored_elo([(0.0, 30.0, 50)])
        assert metrics["arena_elo_se"] == anchored_elo_se([(0.0, 30.0, 50)])

    calls.clear()
    metrics = run_arena(backend, object(), cfg, seed=0, round_index=5)
    assert calls == ["lookahead", "random"]
    assert metrics["arena_vs_random"] == results["random"].winrate


def _dummy_spec() -> BeliefSpec:
    # The agent is never built: these tests stub the arena out.
    return BeliefSpec(lambda: cast("Any", None), frozenset((2,)))


def test_arena_name_path_delegates_to_the_spec_core(monkeypatch: Any) -> None:
    # The name-based `arena` only resolves POLICIES and hands the spec to the
    # shared core -- the seat-swap/seed/episode logic exists once.
    seen: dict[str, Any] = {}

    def _fake_spec_arena(backend: Any, net: Any, **kwargs: Any) -> ArenaResult:
        seen.update(kwargs)
        return ArenaResult(wins=1.0, episodes=2)

    # by module object: the training package rebinds `arena` to the function, so
    # the dotted path does not reach the submodule.
    arena_module = sys.modules["settlrl_learn.training.arena"]
    monkeypatch.setattr(arena_module, "arena_spec", _fake_spec_arena)
    res = arena(
        MLPBackend((16,)), object(), opponent="random", n_games=8,
        num_simulations=17, max_num_considered_actions=5, batch_size=9, seed=3,
    )  # fmt: skip
    assert res == ArenaResult(1.0, 2)
    assert seen == {
        "opponent": POLICIES["random"], "n_games": 8, "num_simulations": 17,
        "max_num_considered_actions": 5, "batch_size": 9, "seed": 3,
    }  # fmt: skip


def test_run_arena_net_opponent_joins_metrics_and_elo(monkeypatch: Any) -> None:
    # A pre-built spec opponent (a frozen checkpoint) plays alongside the registry
    # anchors: it reports arena_vs_<name> and its (elo, wins, episodes) joins the
    # same MLE. Its seed comes off a base disjoint from the registry opponents'.
    seeds: dict[str, int] = {}

    def _fake_arena(*a: Any, opponent: str, seed: int, **k: Any) -> ArenaResult:
        seeds[opponent] = seed
        return ArenaResult(wins=30.0, episodes=50)

    def _fake_spec_arena(*a: Any, opponent: Any, seed: int, **k: Any) -> ArenaResult:
        seeds["az0"] = seed
        return ArenaResult(wins=24.0, episodes=40)

    monkeypatch.setattr("settlrl_learn.training.steps.arena", _fake_arena)
    monkeypatch.setattr("settlrl_learn.training.steps.arena_spec", _fake_spec_arena)
    cfg = ArenaConfig(games=40, opponents=["lookahead"], anchor_elos={"lookahead": 0.0})
    metrics = run_arena(
        MLPBackend((16,)), object(), cfg, seed=7, round_index=1,
        net_opponents={"az0": (_dummy_spec(), -100.0, 1, 0)},
    )  # fmt: skip
    inputs = [(0.0, 30.0, 50), (-100.0, 24.0, 40)]
    assert metrics["arena_winrate"] == 0.6
    assert metrics["arena_vs_az0"] == 0.6
    assert metrics["arena_elo"] == anchored_elo(inputs)
    assert metrics["arena_elo_se"] == anchored_elo_se(inputs)
    assert seeds["lookahead"] == 7  # registry base, untouched
    assert seeds["az0"] == 7 + 50_000  # disjoint net-opponent base


def test_run_arena_net_opponent_every_and_registry_seeds(monkeypatch: Any) -> None:
    # `every` schedules a net opponent exactly like opponent_every does a registry
    # one (skipped rounds contribute no metric and no Elo input), and adding net
    # opponents never shifts the registry opponents' seeds.
    reg_seeds: list[int] = []
    net_calls: list[int] = []

    def _fake_arena(*a: Any, opponent: str, seed: int, **k: Any) -> ArenaResult:
        reg_seeds.append(seed)
        return ArenaResult(wins=30.0, episodes=50)

    def _fake_spec_arena(*a: Any, opponent: Any, seed: int, **k: Any) -> ArenaResult:
        net_calls.append(seed)
        return ArenaResult(wins=24.0, episodes=40)

    monkeypatch.setattr("settlrl_learn.training.steps.arena", _fake_arena)
    monkeypatch.setattr("settlrl_learn.training.steps.arena_spec", _fake_spec_arena)
    cfg = ArenaConfig(
        games=40,
        opponents=["lookahead", "random"],
        anchor_elos={"lookahead": 0.0, "random": -1115.0},
    )
    backend = MLPBackend((16,))
    net_opponents = {
        "az0": (_dummy_spec(), -100.0, 3, 0),
        "az1": (_dummy_spec(), 50.0, 1, 0),
    }

    metrics = run_arena(
        backend, object(), cfg, seed=0, round_index=1, net_opponents=net_opponents
    )
    assert reg_seeds == [0, 10_000]
    assert net_calls == [50_000 + 10_000]  # az0 skipped (round 1 % 3), az1 played
    assert "arena_vs_az0" not in metrics
    assert metrics["arena_elo"] == anchored_elo(
        [(0.0, 30.0, 50), (-1115.0, 30.0, 50), (50.0, 24.0, 40)]
    )

    reg_seeds.clear()
    net_calls.clear()
    metrics = run_arena(
        backend, object(), cfg, seed=0, round_index=3, net_opponents=net_opponents
    )
    assert reg_seeds == [0, 10_000]  # unchanged by the extra opponents
    assert net_calls == [50_000, 50_000 + 10_000]
    assert metrics["arena_vs_az0"] == 0.6

    # ... and identical to the no-net-opponents path.
    reg_seeds.clear()
    net_calls.clear()
    base = run_arena(backend, object(), cfg, seed=0, round_index=3)
    assert reg_seeds == [0, 10_000] and net_calls == []
    assert base["arena_elo"] == anchored_elo([(0.0, 30.0, 50), (-1115.0, 30.0, 50)])


def test_run_arena_net_opponent_phase_rotates_rungs(monkeypatch: Any) -> None:
    # Three rungs at every=3 with phases 0/1/2 rotate: exactly one plays per
    # round, each keeping its own enumeration-position seed regardless of which
    # rounds its phase selects (phase schedules, never re-seeds).
    played: list[tuple[str, int]] = []

    def _fake_spec_arena(*a: Any, opponent: Any, seed: int, **k: Any) -> ArenaResult:
        played.append((cast("str", opponent.policy), seed))
        return ArenaResult(wins=24.0, episodes=40)

    monkeypatch.setattr(
        "settlrl_learn.training.steps.arena",
        lambda *a, **k: ArenaResult(wins=30.0, episodes=50),
    )
    monkeypatch.setattr("settlrl_learn.training.steps.arena_spec", _fake_spec_arena)
    cfg = ArenaConfig(games=40, opponents=["lookahead"], anchor_elos={"lookahead": 0.0})
    net_opponents = {
        name: (
            BeliefSpec(lambda name=name: cast("Any", name), frozenset((2,))),
            e,
            3,
            p,
        )
        for name, e, p in [("az0", -58.0, 0), ("az1", 110.0, 1), ("az2", 187.0, 2)]
    }

    schedule: dict[int, list[tuple[str, int]]] = {}
    for round_index in range(1, 7):
        played.clear()
        metrics = run_arena(
            MLPBackend((16,)), object(), cfg, seed=0,
            round_index=round_index, net_opponents=net_opponents,
        )  # fmt: skip
        schedule[round_index] = list(played)
        assert sum(f"arena_vs_az{i}" in metrics for i in range(3)) == 1

    assert schedule == {
        1: [("az2", 70_000)],  # (1 + 2) % 3 == 0
        2: [("az1", 60_000)],  # (2 + 1) % 3 == 0
        3: [("az0", 50_000)],  # (3 + 0) % 3 == 0
        4: [("az2", 70_000)],
        5: [("az1", 60_000)],
        6: [("az0", 50_000)],
    }
