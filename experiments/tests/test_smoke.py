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
from typing import Any

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


def test_0004_v2_four_arm_study_presets_compose() -> None:
    # The featurization-v2 four-arm study (docs/superpowers/plans/
    # 2026-07-30-featurization-v2.md, Task 5): each arm changes exactly one
    # net knob on top of v2_base (net.incidence / net.layers / net.preset).
    # Two things are checked: each arm's own delta (below), and -- the
    # isolation guarantee -- that popping just those three known-varying
    # `net` keys leaves *everything else* identical across all four dumped
    # configs: at minimum selfplay.{batch,persistent,pcr_full_prob,
    # pcr_fast_sims,temperature_moves}, search.{num_simulations,max_considered,
    # expected_rolls,chance_nodes}, optim.*, replay.buffer_max, n_iterations,
    # checkpoint_every, final_games, and net.width (net.layers too, across the
    # three non-deep arms) -- but the dump-and-pop form checks the *whole*
    # config, not just this list, so a future edit to any field (including one
    # not named here) that silently drifted one arm from the shared recipe
    # fails this test too.
    run = load_run("0004_alphazero")
    names = ["v2_base", "v2_incidence", "v2_deep", "v2_hetero"]
    cfgs = {name: run.compose_config([f"+experiment={name}"]) for name in names}

    base, incidence, deep, hetero = (cfgs[n] for n in names)
    assert base.net.kind == "gnn" and base.net.feature_version == 2
    assert (
        not base.net.incidence
        and base.net.layers == 4
        and base.net.preset == "gn_global"
    )
    assert incidence.net.feature_version == 2 and incidence.net.incidence
    assert (
        incidence.net.layers == base.net.layers
        and incidence.net.preset == base.net.preset
    )
    assert deep.net.feature_version == 2 and deep.net.layers == base.net.layers + 2
    assert not deep.net.incidence and deep.net.preset == base.net.preset
    assert hetero.net.feature_version == 2 and hetero.net.preset == "gn_hetero"
    assert not hetero.net.incidence and hetero.net.layers == base.net.layers

    varying_net_keys = {"incidence", "layers", "preset"}
    stripped = {}
    for name, cfg in cfgs.items():
        dump = cfg.dump()
        dump["net"] = {
            k: v for k, v in dump["net"].items() if k not in varying_net_keys
        }
        stripped[name] = dump
    reference = stripped[names[0]]
    for name in names[1:]:
        assert stripped[name] == reference, (
            f"{name} drifted from v2_base outside its declared delta "
            f"({varying_net_keys} within `net`) -- arm isolation broken"
        )


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
    assert (az0.elo, az0.every) == (-58.0, 1)
    assert not hasattr(cfg.to_learn_config().arena, "net_opponents")


def test_0004_builds_net_opponent_specs() -> None:
    # Composition only: the named anchor loads and becomes a seatable spec at the
    # arena's budget (no game is played -- that is a GPU-scale cost).
    run = load_run("0004_alphazero")
    cfg = run.compose_config(["+experiment=small"])
    opponents = run.build_net_opponents(cfg)
    spec, elo, every = opponents["az0_gnn96x4"]
    assert (elo, every) == (-58.0, 1)
    assert 2 in spec.n_players and callable(spec.policy)


def test_0004_final_gauntlet_neutralizes_schedules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The final gauntlet ignores the in-loop `arena.games` and every schedule
    # (`opponent_every`, a net opponent's `every`) -- every rung plays, at
    # `final_games` games, in the one end-of-run call. `run_arena` is stubbed
    # so this checks the composed `ArenaConfig`/`net_opponents`, not real play.
    # `run_final_gauntlet` lives in arena_helpers (run.py re-exports it), and
    # its `run_arena` lookup resolves against *that* module's globals -- so the
    # patch target must be the real `arena_helpers` module (a plain import,
    # not `load_run`'s by-path copy, which would be a distinct module object).
    import arena_helpers  # type: ignore[import-not-found]

    run = load_run("0004_alphazero")
    cfg = run.compose_config(["+experiment=smoke", "final_games=7"])
    cfg = cfg.model_copy(
        update={
            "arena": cfg.arena.model_copy(
                update={"games": 999, "opponent_every": {"random": 5}}
            )
        }
    )
    captured: dict[str, Any] = {}

    def fake_run_arena(
        backend: object,
        net: object,
        arena_cfg: object,
        *,
        seed: int,
        round_index: int,
        net_opponents: object = None,
    ) -> dict[str, float]:
        captured["arena_cfg"] = arena_cfg
        captured["seed"] = seed
        captured["net_opponents"] = net_opponents
        return {"arena_elo": 10.0, "arena_elo_se": 1.0, "arena_winrate": 0.5}

    monkeypatch.setattr(arena_helpers, "run_arena", fake_run_arena)
    net_opponents = {"az0": (object(), -100.0, 5)}  # every=5, to be neutralized

    metrics = run.run_final_gauntlet(object(), object(), cfg, net_opponents)

    arena_cfg = captured["arena_cfg"]
    assert arena_cfg.games == 7
    assert arena_cfg.opponent_every == {}
    assert captured["seed"] == cfg.seed + 99
    got_opponents = captured["net_opponents"]
    assert got_opponents["az0"][2] == 1
    assert metrics == {"arena_elo": 10.0, "arena_elo_se": 1.0, "arena_winrate": 0.5}


def test_0004_gauntlet_verdict_elo_boundary() -> None:
    # pass iff arena_elo - 2*arena_elo_se >= gate_elo -- both sides of the
    # 2-sigma boundary.
    run = load_run("0004_alphazero")
    gate = 35.0
    assert (
        run.gauntlet_verdict({"arena_elo": 45.0, "arena_elo_se": 5.0}, gate) == "pass"
    )
    assert (
        run.gauntlet_verdict({"arena_elo": 44.9, "arena_elo_se": 5.0}, gate) == "fail"
    )


def test_0004_resume_state_reads_checkpoint_and_wandb_id(tmp_path: Path) -> None:
    # Shared by both run paths (mlp/gnn) so a resumed gnn run also continues
    # its wandb curve, not just the mlp path.
    run = load_run("0004_alphazero")
    assert run._resume_state("") == (None, None)
    assert run._resume_state(str(tmp_path / "missing")) == (None, None)

    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "runstate.eqx").write_bytes(b"x")
    (prior / "wandb_id.txt").write_text("abc123\n")
    checkpoint, wandb_id = run._resume_state(str(prior))
    assert checkpoint == prior / "runstate.eqx"
    assert wandb_id == "abc123"
