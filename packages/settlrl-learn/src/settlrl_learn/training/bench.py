"""Self-play throughput probe: warmed, repeated timings at a fixed net.

The training loop's cost is dominated by self-play, so this times exactly what
an iteration times (:func:`~settlrl_learn.training.loop.run_selfplay` over the
loop's own callables, :func:`~settlrl_learn.training.loop.selfplay_callables`)
with the net held fixed -- no optimiser, no replay, no arena. The first call is
untimed (it pays the XLA compile) and the timed repeats all run the identical
workload, so the spread across repeats is measurement noise, not workload drift.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

import functools
import statistics
import time
from typing import Any

import equinox as eqx

from settlrl_learn.training.backend import Backend
from settlrl_learn.training.config import LearnConfig
from settlrl_learn.training.loop import run_selfplay, selfplay_callables


def bench_selfplay(
    backend: Backend,
    net: Any,
    cfg: LearnConfig,
    *,
    warmup: int = 1,
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    """Time ``repeats`` self-play batches of ``cfg.selfplay.samples`` at a fixed
    ``net``, after ``warmup`` untimed calls (at ``seed + 1000``; the timed ones
    all run at ``seed``, so they share one workload).

    Reports ``samples_per_s`` / ``moves_per_s`` / ``sims_per_s`` (medians over
    the repeats), the per-repeat wall times ``t_0..t_{repeats-1}`` with their
    median ``t_median_s``, and the workload's ``samples`` / ``env_steps`` /
    ``discarded``. A *move* is one lane-step (``env_steps * cfg.selfplay.batch``)
    and each move runs ``cfg.search.num_simulations`` simulations.

    Raises ``ValueError`` under playout-cap randomization
    (``cfg.selfplay.pcr_full_prob`` < 1): the sims-per-move accounting assumes
    every step ran the full search. Raises ``ValueError`` under
    ``cfg.selfplay.persistent``: the repeats do not thread a carry, so they
    would keep the fresh-env workload while ``discarded`` reported a persistent
    run's. Raises ``ValueError`` if ``repeats`` < 1 (the median over zero timed
    repeats is undefined)."""
    if cfg.selfplay.pcr_full_prob < 1.0:
        raise ValueError(
            "bench_selfplay needs pcr_full_prob == 1.0 (playout-cap randomization "
            f"makes sims_per_s meaningless); got {cfg.selfplay.pcr_full_prob}"
        )
    if cfg.selfplay.persistent:
        raise ValueError(
            "bench_selfplay measures the fresh-env workload and does not thread a "
            "carry across repeats; persistent benching lands with the loop "
            "integration"
        )
    if repeats < 1:
        raise ValueError(f"bench_selfplay needs repeats >= 1; got {repeats}")
    calls = selfplay_callables(backend, cfg, net)
    net_search = calls.make_net_search(cfg.search.num_simulations)
    search = functools.partial(net_search, eqx.partition(net, eqx.is_array)[0])

    def play(sd: int) -> tuple[int, int, int]:
        samples, stats, _ = run_selfplay(calls, search, cfg, cfg.selfplay.samples, sd)
        return samples["value"].shape[0], stats.env_steps, stats.discarded

    for _ in range(warmup):
        play(seed + 1000)
    times: list[float] = []
    n_samples = env_steps = discarded = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        n_samples, env_steps, discarded = play(seed)
        times.append(time.perf_counter() - t0)

    t_median = statistics.median(times)
    moves = env_steps * cfg.selfplay.batch
    moves_per_s = moves / t_median
    out = {
        "samples_per_s": n_samples / t_median,
        "moves_per_s": moves_per_s,
        "sims_per_s": moves_per_s * cfg.search.num_simulations,
        "samples": float(n_samples),
        "env_steps": float(env_steps),
        "discarded": float(discarded),
        "t_median_s": t_median,
    }
    out.update({f"t_{i}": t for i, t in enumerate(times)})
    return out
