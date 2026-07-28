"""pytest-benchmark micro-timings for the units the training loop spends its
time in: the net forward, one vmapped search step, one self-play window, and
one optimiser step.

Four measures, all on a **random-init** ``BoardGNN`` (``gn_global`` preset,
width 96, layers 4 -- the pinned ``az0_gnn96x4`` config) -- the committed
training anchor lives in ``experiments/``, out of reach of package tests, and
kernel timing does not depend on trained weights:

- ``test_net_forward`` -- one jitted+vmapped forward at B=256.
- ``test_search_step`` -- one warmed dispatch of the vmapped net search
  (``make_net_search(64)``) on a mid-game batch, B in {64, 256}. This is
  ``search_step_ms``, the unit the parallel-descent work will move.
- ``test_selfplay_window`` -- one warmed self-play window
  (:func:`~settlrl_learn.training.self_play` over
  :func:`~settlrl_learn.training.selfplay_callables`) at a reduced budget.
  The XLA-compiling call runs untimed (``benchmark.pedantic``'s ``setup``),
  so the headline stat is steady-state throughput, on the same footing as
  the other three benchmarks here.
- ``test_optimizer_step`` -- one warmed ``backend.make_step`` dispatch on a
  broadcast zero batch at ``batch_size=1024``.

All swept over devices (CPU always; CUDA when a GPU-enabled jaxlib sees a
device, otherwise the CUDA variants skip), pinned via ``jax.default_device``.
JIT is warmed up before each timed region.
"""

from __future__ import annotations

import functools
import time
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from settlrl_engine.belief import BeliefView
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_learn.nn.graph import Sample
from settlrl_learn.nn.graphnet import PRESETS
from settlrl_learn.training import (
    GNNBackend,
    LearnConfig,
    SelfPlayConfig,
    make_optimizer,
    self_play,
    selfplay_callables,
)
from settlrl_learn.training.config import OptimConfig


def _cuda_available() -> bool:
    try:
        return bool(jax.devices("cuda"))
    except RuntimeError:  # no CUDA plugin installed, or plugin sees no GPU
        return False


# Device sweep: CPU always; CUDA only when available (install the `cuda` extra).
_DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(
            not _cuda_available(),
            reason="no CUDA device (needs the `cuda` extra and an NVIDIA GPU)",
        ),
    ),
]

# gn_global at the pinned az0_gnn96x4 width/depth (experiments/0004's anchor).
_NET_CFG = PRESETS["gn_global"]._replace(width=96, layers=4)

_PLAYERS = 2
_MIDGAME_STEPS = 150


@functools.cache
def _midgame_env(batch_size: int, device: str) -> BatchedSettlrlEnv:
    """A batch of mid-game positions (random play, beliefs tracked), reused
    across benchmarks that only read it."""
    with jax.default_device(jax.devices(device)[0]):
        env = BatchedSettlrlEnv(
            batch_size=batch_size, seed=0, n_players=_PLAYERS, track_beliefs=True
        )
        env.rollout(jax.random.key(0), _MIDGAME_STEPS)
    return env


def _acting_view(env: BatchedSettlrlEnv) -> BeliefView:
    """Per-lane ``BeliefView`` of that lane's acting player."""
    per_seat = [env.belief_view(i) for i in range(env.n_players)]
    lanes = jnp.arange(env.batch_size)
    return cast(
        BeliefView,
        jax.tree.map(lambda *xs: jnp.stack(xs)[env.agent_selection, lanes], *per_seat),
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("device", _DEVICES)
def test_net_forward(benchmark: Any, device: str) -> None:
    """Latency of one jitted+vmapped ``BoardGNN`` forward at B=256."""
    benchmark.group = f"net_forward[{device}]"
    with jax.default_device(jax.devices(device)[0]):
        backend = GNNBackend(_NET_CFG)
        net = backend.init(jax.random.key(0))
        env = _midgame_env(256, device)
        obs = jax.jit(jax.vmap(backend.observe, in_axes=(0, 0, 0)))(
            env.board[0], env.board[1], env.agent_selection
        )
        sample = Sample(obs["nodes"], obs["edges"], obs["glob"], obs["tiles"], None)
        fwd = eqx.filter_jit(lambda s: jax.vmap(net)(s))
        jax.block_until_ready(fwd(sample))  # type: ignore[no-untyped-call]
        benchmark(lambda: jax.block_until_ready(fwd(sample)))  # type: ignore[no-untyped-call]


_SEARCH_SIMS = 64


@pytest.mark.benchmark
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [64, 256], ids=lambda b: f"B{b}")
def test_search_step(benchmark: Any, batch_size: int, device: str) -> None:
    """Latency of one warmed dispatch of the vmapped net search
    (``make_net_search(64)``) on a mid-game batch -- ``search_step_ms``, the
    unit the parallel-descent work will move."""
    benchmark.group = f"search_step[B{batch_size}-{device}]"
    with jax.default_device(jax.devices(device)[0]):
        backend = GNNBackend(_NET_CFG)
        net = backend.init(jax.random.key(0))
        cfg = LearnConfig(n_iterations=1)
        calls = selfplay_callables(backend, cfg, net)
        net_search = calls.make_net_search(_SEARCH_SIMS)
        search = functools.partial(net_search, eqx.partition(net, eqx.is_array)[0])

        env = _midgame_env(batch_size, device)
        keys = jax.random.split(jax.random.key(1), batch_size)
        layout, view, player = env.board[0], _acting_view(env), env.agent_selection
        mask = env.flat_mask()

        np.asarray(search(keys, layout, view, player, mask))  # warm up JIT
        benchmark(lambda: np.asarray(search(keys, layout, view, player, mask)))


_SELFPLAY_SAMPLES = 256
_SELFPLAY_BATCH = 64


@pytest.mark.benchmark
@pytest.mark.parametrize("device", _DEVICES)
def test_selfplay_window(benchmark: Any, device: str) -> None:
    """Throughput of one warmed self-play window (:func:`self_play` over
    :func:`selfplay_callables`) at a reduced budget -- the loop's dominant
    per-iteration cost. ``setup`` (untimed) pays the XLA compile with one
    self-play call; ``target`` (timed) runs exactly one more, already-warmed
    collection, so the pytest-benchmark headline is steady-state throughput,
    not compile-inclusive wall time. The rates ``bench_selfplay`` would report
    (samples/moves/sims per second, computed from the same timed call) surface
    via ``benchmark.extra_info``."""
    benchmark.group = f"selfplay_window[{device}]"
    with jax.default_device(jax.devices(device)[0]):
        backend = GNNBackend(_NET_CFG)
        net = backend.init(jax.random.key(0))
        cfg = LearnConfig(
            n_iterations=1,
            selfplay=SelfPlayConfig(samples=_SELFPLAY_SAMPLES, batch=_SELFPLAY_BATCH),
        )
        calls = selfplay_callables(backend, cfg, net)
        search = functools.partial(
            calls.make_net_search(cfg.search.num_simulations),
            eqx.partition(net, eqx.is_array)[0],
        )

        def play(seed: int) -> tuple[int, int, int]:
            samples, stats = self_play(
                search, n_samples=cfg.selfplay.samples,
                observe_of=calls.observe_of, view_of=calls.view_of,
                setup_search=calls.setup_search,
                batch_size=cfg.selfplay.batch, temperature=cfg.selfplay.temperature,
                seed=seed, record_value=cfg.value_blend.max > 0,
                track_ordering=cfg.search.ordered,
                max_steps=cfg.selfplay.max_steps, max_game_len=cfg.selfplay.max_game_len,
            )  # fmt: skip
            return samples["value"].shape[0], stats.env_steps, stats.discarded

        def warmup() -> None:
            # pedantic's `setup` runs untimed, before instrumentation starts --
            # this call pays the XLA compile. Its return value must be falsy (a
            # truthy return is unpacked as `(args, kwargs)` for `timed`).
            play(seed=1000)

        result: dict[str, float] = {}

        def timed() -> None:
            t0 = time.perf_counter()
            n_samples, env_steps, discarded = play(seed=0)
            dt = time.perf_counter() - t0
            moves = env_steps * cfg.selfplay.batch
            result.update(
                samples_per_s=n_samples / dt, moves_per_s=moves / dt,
                sims_per_s=(moves / dt) * cfg.search.num_simulations,
                samples=float(n_samples), env_steps=float(env_steps),
                discarded=float(discarded),
            )  # fmt: skip

        benchmark.pedantic(timed, setup=warmup, rounds=1, iterations=1)
        benchmark.extra_info.update(result)


@pytest.mark.benchmark
@pytest.mark.parametrize("device", _DEVICES)
def test_optimizer_step(benchmark: Any, device: str) -> None:
    """Latency of one warmed ``backend.make_step`` dispatch on a broadcast
    zero batch at ``batch_size=1024``."""
    benchmark.group = f"optimizer_step[{device}]"
    with jax.default_device(jax.devices(device)[0]):
        backend = GNNBackend(_NET_CFG)
        net = backend.init(jax.random.key(0))
        optimizer = make_optimizer(OptimConfig())
        opt_state = backend.init_opt(optimizer, net)
        step = backend.make_step(optimizer)
        item = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1024, *x.shape)), backend.empty_item()
        )
        jax.block_until_ready(step(net, opt_state, item))  # type: ignore[no-untyped-call]
        benchmark(lambda: jax.block_until_ready(step(net, opt_state, item)))  # type: ignore[no-untyped-call]
