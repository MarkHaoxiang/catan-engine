"""Bench smoke: greedy must beat random.

Hypothesis: the scripted greedy agent beats uniform-random play decisively
(2-player, seat-swapped). Known true from the strength ladder — this
experiment exists as the worked example of the contract: a manifest-pinned
run, streamed metrics, a saved bench verdict, and a gate asserted in code.

    uv run python experiments/0001_bench_smoke/run.py [key=value ...]

This framework also holds the one-off anchor-calibration round-robin (measurement
wave 2, task 4): a joint Elo fit over {random, greedy, lookahead, mcts} + the
frozen az0 checkpoint, holding lookahead = 0 (``calibrate.py``, `joint_fit`).
Chunked into per-pair CLI invocations (crash-safe: each pair's outcome is
appended to the run dir before the next one starts)::

    uv run python experiments/0001_bench_smoke/run.py calibrate init
    uv run python experiments/0001_bench_smoke/run.py calibrate pair <run_dir> <a> <b> <n>
    uv run python experiments/0001_bench_smoke/run.py calibrate fit <run_dir>
"""

import json
import sys
from pathlib import Path

from calibrate import (
    AZ0_NAME,
    FIXED_ELOS,
    PairResult,
    az0_spec,
    joint_fit,
    joint_fit_se,
    play_pair,
    sanity_gates,
    search_semantics,
)
from settlrl_agents import POLICIES
from settlrl_agents.cli import bench
from settlrl_learn.experiment import Config, Run, start_run
from settlrl_learn.training import OpponentSpec


class BenchSmokeConfig(Config):
    a: str = "greedy"
    b: str = "random"
    players: int = 2
    games: int = 60
    batch_size: int = 32
    seed: int = 0
    gate: float = 0.70  # pass iff rate - 2*se >= gate


def run_bench(run: Run, cfg: BenchSmokeConfig) -> str:
    """Bench ``a`` vs ``b``, log the per-seat split, gate on the lower 2-sigma
    bound. Returns the verdict."""
    result = bench(
        cfg.a,
        cfg.b,
        n_games=cfg.games,
        players=cfg.players,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    for seat, (wins, episodes) in enumerate(result.by_position):
        run.log(seat=seat, wins=wins, episodes=episodes)
    run.save_json("bench.json", result._asdict())
    lower = result.rate - 2 * result.se
    verdict = "pass" if lower >= cfg.gate else "fail"
    run.finish(verdict, rate=result.rate, se=result.se, lower_2se=lower)
    return verdict


class CalibrateConfig(Config):
    seed: int = 0
    batch_size: int = 64


def _spec(name: str, semantics: dict[str, object]) -> OpponentSpec:
    if name == AZ0_NAME:
        return az0_spec(
            sims=int(semantics["sims"]),  # type: ignore[call-overload]
            considered=int(semantics["considered"]),  # type: ignore[call-overload]
            chance_nodes=bool(semantics["chance_nodes"]),
            dev_chance=bool(semantics["dev_chance"]),
            ordered=bool(semantics["ordered"]),
        )
    return POLICIES[name]


def run_calibrate_pair(
    run: Run, cfg: CalibrateConfig, a: str, b: str, n_games: int
) -> PairResult:
    """Play one round-robin pair and append it to this run's matches
    (crash-safe: recorded before the next pair starts, so a chunked sequence
    of CLI invocations can be interrupted and resumed by re-reading the run
    dir)."""
    semantics = search_semantics()
    result = play_pair(
        a, _spec(a, semantics), b, _spec(b, semantics),
        n_games=n_games, seed=cfg.seed, batch_size=cfg.batch_size,
    )  # fmt: skip
    with (run.dir / "matches.jsonl").open("a") as f:
        f.write(json.dumps(result._asdict()) + "\n")
    run.log(pair=f"{a}-{b}", **result._asdict())
    return result


def run_calibrate_fit(run: Run) -> str:
    """Fit the joint Elo MLE over every recorded pair, check the sanity gates,
    and record the matrix + fit + search semantics as this run's verdict
    (``pass`` iff every gate holds, else ``blocked``)."""
    lines = (run.dir / "matches.jsonl").read_text().splitlines()
    results = [PairResult(**json.loads(line)) for line in lines]
    ratings = joint_fit(results, FIXED_ELOS)
    ses = joint_fit_se(results, ratings)
    gates = sanity_gates(ratings)
    verdict = "pass" if all(gates.values()) else "blocked"
    run.save_json("matches.json", [r._asdict() for r in results])
    run.finish(
        verdict,
        ratings=ratings,
        se=ses,
        gates=gates,
        search_semantics=search_semantics(),
        n_pairs=len(results),
        n_games=sum(r.episodes for r in results),
    )
    return verdict


def _calibrate_main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(
            "usage: run.py calibrate {init|pair <dir> <a> <b> <n>|fit <dir>} "
            "[key=value ...]"
        )
    sub = argv[0]
    if sub == "init":
        cfg = CalibrateConfig.resolve({}, overrides=argv[1:])
        run = start_run(Path(__file__).parent, cfg.dump())
        print(run.dir)
        return
    run_dir = Path(argv[1])
    run = Run(run_dir)
    if sub == "pair":
        a, b, n = argv[2], argv[3], int(argv[4])
        cfg = CalibrateConfig.resolve({}, overrides=argv[5:])
        print(run_calibrate_pair(run, cfg, a, b, n))
        return
    if sub == "fit":
        print(run_calibrate_fit(run))
        return
    raise SystemExit(f"unknown calibrate subcommand {sub!r}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        _calibrate_main(sys.argv[2:])
        return
    cfg = BenchSmokeConfig.resolve({}, overrides=sys.argv[1:])
    run_bench(start_run(Path(__file__).parent, cfg.dump()), cfg)


if __name__ == "__main__":
    main()
