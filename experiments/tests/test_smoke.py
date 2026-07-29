"""Smoke tests: composition checks plus one tiny end-to-end run per framework.

Most of these are config-composition checks — every named variant/preset
resolves and validates (catches a typo'd knob or a renamed seam in seconds,
without paying for a JAX compile). The actual training loop, backends, and
search are covered end-to-end at tiny budgets by the owning packages
(``settlrl-learn``'s ``test_training.py``, ``settlrl-search``'s
``test_ismcts.py``); a framework's unique surface is only the composition
layer (hydra/`resolve` config groups, `run.py` wiring, the bench gate, a
recorded verdict), so each framework keeps at most one genuinely tiny
end-to-end run to prove that layer, plus config-only checks for the rest of
its variants. They write into a ``tmp_path`` ``Run`` rather than ``runs/``
and never assert a strength claim, only that the framework completes and
records a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import EXPERIMENTS, load_run
from settlrl_learn.experiment import Run


def _verdict(run_dir: Path) -> str:
    result = json.loads((run_dir / "result.json").read_text())
    assert result["verdict"] in {"pass", "fail"}
    return str(result["verdict"])


def test_0001_bench_smoke(tmp_path: Path) -> None:
    run = load_run("0001_bench_smoke")
    cfg = run.BenchSmokeConfig.resolve({}, overrides=["games=4", "batch_size=4"])
    run.run_bench(Run(tmp_path), cfg)
    _verdict(tmp_path)


def test_0002_value_fitting_smoke(tmp_path: Path) -> None:
    run = load_run("0002_linear_value_fitting")
    cfg = run.ValueFittingConfig.resolve({**run.VARIANTS["smoke"], "variant": "smoke"})
    run.run_experiment(Run(tmp_path), cfg.dump())
    _verdict(tmp_path)


def test_0002_variants_resolve() -> None:
    run = load_run("0002_linear_value_fitting")
    for name, variant in run.VARIANTS.items():
        run.ValueFittingConfig.resolve({**variant, "variant": name})


def test_0003_neural_board_architectures_smoke(tmp_path: Path) -> None:
    run = load_run("0003_neural_board_architectures")
    cfg = run.NeuralBoardArchitecturesConfig.resolve(run.VARIANTS["smoke"])
    run.run_experiment(Run(tmp_path), cfg)
    _verdict(tmp_path)


def test_0003_variants_resolve() -> None:
    run = load_run("0003_neural_board_architectures")
    for variant in run.VARIANTS.values():
        run.NeuralBoardArchitecturesConfig.resolve(variant)


# The one genuinely tiny end-to-end 0004 run: the mlp path (not gnn -- gnn
# training is covered end-to-end by settlrl-learn's
# test_learn_resume_bit_exact_gnn, and chance/ordered search by
# settlrl-search's test_ismcts.py, so a second full run here would only
# re-prove the arena/hydra wiring the mlp path already proves).
@pytest.mark.slow
def test_0004_alphazero_smoke(tmp_path: Path) -> None:
    run = load_run("0004_alphazero")
    cfg = run.compose_config(["+experiment=smoke"])
    run.run_experiment(Run(tmp_path), cfg)
    _verdict(tmp_path)


def test_0004_bench_throughput_smoke(tmp_path: Path) -> None:
    run = load_run("0004_alphazero")
    cfg = run.compose_config(
        [
            "+experiment=bench_throughput",
            "selfplay.samples=4",
            "selfplay.batch=2",
            "search.num_simulations=2",
            "bench.repeats=1",
            "bench.warmup=0",
        ]
    )
    run.run_experiment(Run(tmp_path), cfg)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["verdict"] == "recorded"
    assert "samples_per_s" in result


def test_0004_experiment_presets_compose() -> None:
    # Every conf/experiment/*.yaml (gnn_smoke included -- its full training run
    # is not re-proven here, see test_0004_alphazero_smoke's comment) resolves
    # and validates into AlphaZeroConfig. Cheap (hydra compose + pydantic, no
    # JAX), and broader than running any one variant: catches a typo'd knob in
    # any preset, not just the ones a smoke happens to execute.
    run = load_run("0004_alphazero")
    conf_dir = EXPERIMENTS / "0004_alphazero" / "conf" / "experiment"
    presets = sorted(p.stem for p in conf_dir.glob("*.yaml"))
    assert presets  # guard against a silently-empty glob
    for name in presets:
        run.compose_config([f"+experiment={name}"])


def test_0004_scale_presets_compose() -> None:
    # The nano/small/medium budget tiers share one recipe (gnn + warm-up + Canopy
    # q-blend, no chance/EV, B256, sims64) and differ only in budget. Fast guard
    # (compose + validate only, no run) against drift in the shared scale groups.
    run = load_run("0004_alphazero")
    for name, n_iters in {"nano": 36, "small": 300, "medium": 3000}.items():
        cfg = run.compose_config([f"+experiment={name}"])
        assert cfg.n_iterations == n_iters
        assert cfg.net.kind == "gnn" and cfg.net.width == 96 and cfg.net.layers == 4
        assert cfg.teacher.enabled and cfg.teacher.iters == 8
        assert cfg.search.num_simulations == 64
        assert not cfg.search.chance_nodes and not cfg.search.expected_rolls
        assert cfg.selfplay.samples == 16384 and cfg.optim.batch_size == 1024
        assert cfg.value_blend.max == 0.85 and cfg.optim.grad_clip == 1.0
        assert cfg.arena.every == 10 and cfg.arena.sims == 24


def test_0004_anchor_loads_and_forwards() -> None:
    import jax
    from settlrl_engine.env import BatchedSettlrlEnv
    from settlrl_learn.nn.board_gnn import gnn_seams

    anchors = load_run("0004_alphazero", module="anchors")
    net, netcfg = anchors.load_anchor("az0_gnn96x4")
    assert (netcfg.width, netcfg.layers, netcfg.head_depth) == (96, 4, 2)

    env = BatchedSettlrlEnv(batch_size=1, n_players=2)
    layout = jax.tree.map(lambda x: x[0], env.board[0])
    state = jax.tree.map(lambda x: x[0], env.board[1])
    value_fn, _ = gnn_seams(net)
    v = value_fn(layout, state, jax.numpy.int32(0))
    assert bool(jax.numpy.isfinite(v))


def test_0004_scale_arena_names_the_az0_rung() -> None:
    # The frozen-checkpoint arena rung is config-named (the spec itself is built
    # in run.py and handed to `learn`), and it must not leak into the library's
    # ArenaConfig.
    run = load_run("0004_alphazero")
    cfg = run.compose_config(["+experiment=small"])
    az0 = cfg.arena.net_opponents["az0_gnn96x4"]
    assert (az0.elo, az0.every) == (-100.0, 1)
    assert not hasattr(cfg.to_learn_config().arena, "net_opponents")


def test_0004_builds_net_opponent_specs() -> None:
    # Composition only: the named anchor loads and becomes a seatable spec at the
    # arena's budget (no game is played -- that is a GPU-scale cost).
    run = load_run("0004_alphazero")
    cfg = run.compose_config(["+experiment=small"])
    opponents = run.build_net_opponents(cfg)
    spec, elo, every = opponents["az0_gnn96x4"]
    assert (elo, every) == (-100.0, 1)
    assert 2 in spec.n_players and callable(spec.policy)
