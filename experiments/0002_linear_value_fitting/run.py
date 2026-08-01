"""Linear value fitting over the hand-engineered features.

A framework for a class of experiments: optimize linear weights over
``BoardFeatures``, deploy them in one-step lookahead, and gate against the
hand-tuned weights. Pick a variant (and optional ``key=value`` overrides)::

    uv run python experiments/0002_linear_value_fitting/run.py [variant] [k=v ...]

- ``predict``   — fit weights to predict game outcomes vs a known opponent
- ``maximise``  — CEM search over weights for match win rate vs a known
  opponent
- ``self_play`` — iterated maximise: each round's opponent is the current
  champion (round 0: the hand weights), challengers accepted only by
  beating it; rounds are configurable
- ``self_play_wide`` — the 2p ladder over *all* engineered features (new
  terms start at weight 0; the search decides what earns weight)
- ``self_play_4p`` — the same ladder in a four-player arena (one challenger
  seat rotating against a champion table; acceptance and the gate clear the
  1/players chance line)
- ``smoke``     — trivial budgets for the end-to-end plumbing test (not a
  real result; the gate is meaningless here)

The schema below carries the defaults; ``VARIANTS`` are deltas onto them.
"""

import sys
from pathlib import Path

from pydantic import Field, model_validator
from settlrl_agents.internal.feature_engineering import BoardFeatures
from settlrl_learn.experiment import Config, start_run
from value_fitting import HAND_WEIGHTS, run_experiment


class Collect(Config):
    steps: int = 12_000
    batch_size: int = 64
    snapshot_every: int = 4


class Maximise(Config):
    iterations: int = 3
    population: int = 6
    elites: int = 3
    eval_games: int = 60
    sigma: float = 0.3


class ValueFittingConfig(Config):
    seed: int = 0
    players: int = 2  # the optimization arena's player count
    # BoardFeatures names to optimize (or "ALL", expanded by _expand_all).
    features: list[str] = Field(default_factory=lambda: list(HAND_WEIGHTS))
    collect: Collect = Field(default_factory=Collect)
    maximise: Maximise = Field(default_factory=Maximise)
    probe_games: int = 120
    # deployment numbers per count (the gate is the 2p arena)
    eval_players: list[int] = Field(default_factory=lambda: [2, 4])
    games_multi: int = 240  # 3-4p matches need n>=240 (seed-batch variance)
    bench_opponent: str = "greedy"
    bench_games: int = 200
    gate_games: int = 300
    # Seat-swap every 2p match for seat-fairness. Off only for the smoke, where a
    # single seating halves the (compile-bound) `evaluate` retrace count.
    seat_swap: bool = True
    # Parallel games per `evaluate`. 32 saturates a GPU; on CPU it is ~linear
    # extra work, so the smoke shrinks it to ~what `n_episodes` needs.
    eval_batch: int = 32
    target: str = "predict"  # predict | maximise
    opponent: str = "greedy"  # a POLICIES name, or "self" for the ladder
    rounds: int = 1
    variant: str = "predict"

    @model_validator(mode="before")
    @classmethod
    def _expand_all(cls, data: object) -> object:
        if isinstance(data, dict) and data.get("features") == "ALL":
            data = {**data, "features": list(BoardFeatures._fields)}
        return data


VARIANTS: dict[str, dict[str, object]] = {
    "predict": {"target": "predict", "opponent": "greedy", "rounds": 1},
    "maximise": {"target": "maximise", "opponent": "greedy", "rounds": 1},
    "self_play": {"target": "maximise", "opponent": "self", "rounds": 3},
    "self_play_wide": {
        "target": "maximise",
        "opponent": "self",
        "rounds": 4,
        "features": "ALL",
        "maximise": {
            "iterations": 3,
            "population": 8,
            "elites": 3,
            "eval_games": 60,
            "sigma": 0.3,
        },
    },
    "self_play_4p": {
        "target": "maximise",
        "opponent": "self",
        "rounds": 2,
        "players": 4,
        "maximise": {
            "iterations": 2,
            "population": 5,
            "elites": 2,
            "eval_games": 80,
            "sigma": 0.3,
        },
        "probe_games": 160,
        "eval_players": [4],
        "games_multi": 200,
    },
    # The smoke only needs to exercise the value-fitting path unique to this
    # framework (CEM search -> a deployed `make_linear` lookahead -> verdict).
    # Each `evaluate` retraces a fresh lookahead scan (~40s, cache or not), so
    # cost is the *count* of matches, not their game budgets. `eval_players: []`
    # skips the deployment gate/bench block (3 more matches) -- its seat-swapped
    # `evaluate`/bench machinery is settlrl-agents library code with its own
    # test coverage -- so the verdict
    # defaults to "fail" (the smoke asserts only that a verdict was recorded).
    "smoke": {
        "target": "maximise",
        "opponent": "greedy",
        "rounds": 1,
        "maximise": {
            "iterations": 1,
            "population": 2,
            "elites": 1,
            "eval_games": 2,
            "sigma": 0.3,
        },
        "probe_games": 2,
        "eval_players": [],
        "seat_swap": False,
        "eval_batch": 2,
    },
}


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if variant not in VARIANTS:
        raise SystemExit(f"usage: run.py [{'|'.join(VARIANTS)}] [key=value ...]")
    cfg = ValueFittingConfig.resolve(
        {**VARIANTS[variant], "variant": variant}, overrides=sys.argv[2:]
    )
    run_experiment(start_run(Path(__file__).parent, cfg.dump()), cfg.dump())


if __name__ == "__main__":
    main()
