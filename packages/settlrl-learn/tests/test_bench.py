"""The self-play throughput probe's contract: the reported keys, the persistent
carry threading, and the configurations it refuses."""

from __future__ import annotations

import jax
import pytest
from settlrl_learn.training import LearnConfig, MLPBackend
from settlrl_learn.training.bench import bench_selfplay
from settlrl_learn.training.config import (
    ArenaConfig,
    SearchSettings,
    SelfPlayConfig,
)


def _bench_cfg() -> LearnConfig:
    """The smallest config that still runs the real net search (bench wiring)."""
    return LearnConfig(
        n_iterations=1, seed=0,
        search=SearchSettings(num_simulations=1, max_considered=4),
        selfplay=SelfPlayConfig(samples=4, batch=2),
        arena=ArenaConfig(games=0),
    )  # fmt: skip


def test_bench_selfplay_reports_throughput() -> None:
    backend = MLPBackend((16,))
    net = backend.init(jax.random.key(0))
    out = bench_selfplay(backend, net, _bench_cfg(), warmup=1, repeats=2, seed=0)
    assert set(out) == {
        "samples_per_s", "moves_per_s", "sims_per_s", "samples",
        "env_steps", "discarded", "t_median_s", "t_0", "t_1",
    }  # fmt: skip
    assert out["samples_per_s"] > 0.0 and out["samples"] > 0.0
    assert out["sims_per_s"] == out["moves_per_s"] * 1  # num_simulations


def test_bench_selfplay_persistent_threads_carry() -> None:
    # Persistent bench: the warmup call creates the carry, timed repeats thread
    # it -- so every repeat's `discarded` is honestly trims-only (the pending
    # games survive in the carry instead of being silently dropped by a fresh
    # env each call), unlike the guard this replaces (which fired precisely
    # because an un-threaded persistent call would under-report `discarded`).
    backend = MLPBackend((16,))
    cfg = _bench_cfg()
    cfg = cfg.model_copy(
        update={"selfplay": cfg.selfplay.model_copy(update={"persistent": True})}
    )
    out = bench_selfplay(
        backend, backend.init(jax.random.key(0)), cfg, warmup=1, repeats=2, seed=0
    )
    assert set(out) == {
        "samples_per_s", "moves_per_s", "sims_per_s", "samples",
        "env_steps", "discarded", "t_median_s", "t_0", "t_1",
    }  # fmt: skip
    assert out["samples_per_s"] >= 0.0
    # No `max_game_len` trim can fire at this budget (a handful of env steps
    # against the 800-row default cap), so discarded is exactly 0 -- honest
    # trims-only accounting because the pending games rode the threaded carry
    # instead of vanishing with a discarded fresh env.
    assert out["discarded"] == 0.0


def test_bench_selfplay_rejects_playout_cap() -> None:
    # sims_per_s assumes every step ran the full search; PCR would make it a lie.
    backend = MLPBackend((16,))
    cfg = _bench_cfg()
    cfg = cfg.model_copy(
        update={"selfplay": cfg.selfplay.model_copy(update={"pcr_full_prob": 0.5})}
    )
    with pytest.raises(ValueError, match="pcr_full_prob"):
        bench_selfplay(backend, backend.init(jax.random.key(0)), cfg)


def test_bench_selfplay_rejects_persistent_without_warmup() -> None:
    # warmup=0 under persistent would put pool creation + the XLA compile
    # inside the first timed repeat, silently corrupting the headline number.
    backend = MLPBackend((16,))
    cfg = _bench_cfg()
    cfg = cfg.model_copy(
        update={"selfplay": cfg.selfplay.model_copy(update={"persistent": True})}
    )
    with pytest.raises(ValueError, match="warmup"):
        bench_selfplay(backend, backend.init(jax.random.key(0)), cfg, warmup=0)
