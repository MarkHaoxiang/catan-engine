"""Self-play throughput probe: warmed, repeated timings at a fixed net.

The training loop's cost is dominated by self-play, so this times exactly what
an iteration times (:func:`~settlrl_learn.training.loop.run_selfplay` over the
loop's own callables, :func:`~settlrl_learn.training.loop.selfplay_callables`)
with the net held fixed -- no optimiser, no replay, no arena. The first call is
untimed (it pays the XLA compile); under ``selfplay.persistent`` off (the
default) the timed repeats all run the identical workload, so the spread
across them is measurement noise, not workload drift -- under ``persistent``
they are sequential continuations of one self-play pool instead (see
``bench_selfplay``'s docstring).

A training-side module: not imported by the package root.
"""

from __future__ import annotations

import functools
import statistics
import time
from typing import Any

import equinox as eqx

from settlrl_learn.training.backends.base import Backend
from settlrl_learn.training.config import LearnConfig
from settlrl_learn.training.loop import run_selfplay, selfplay_callables
from settlrl_learn.training.selfplay.carry import SelfPlayCarry


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
    ``net``, after ``warmup`` untimed calls (the timed ones all start at
    ``seed``).

    Under ``cfg.selfplay.persistent`` the warmup call(s) *create* the carry
    (paying the XLA compile and ramping the pool up from a cold env) and every
    timed repeat *threads* it, continuing the games in flight rather than
    rebuilding a fresh env per call -- so ``discarded`` stays honest (trims
    only, per :func:`~settlrl_learn.training.selfplay.self_play`'s contract)
    instead of silently losing whatever was pending in a discarded env. Repeats
    are then sequential continuations of one pool, not repetitions of an
    identical workload -- their ``samples``/``env_steps``/``discarded`` can
    differ (the surplus carried into each call varies), so the reported
    ``samples``/``env_steps``/``discarded`` are the *last* repeat's, same as
    the non-persistent path, and the timing headline stays the across-repeat
    median (steady-state flush rate, not a single sample).

    Reports ``samples_per_s`` / ``moves_per_s`` / ``sims_per_s`` as the
    **median of each repeat's own ratio** (repeat *i*'s samples over its own
    ``t_i``, not the last repeat's count over the cross-repeat ``t_median_s``
    -- pairing every repeat's numerator with a different repeat's denominator
    would mismatch workload and timing whenever repeats differ, as persistent
    ones do). When repeats share one workload (non-persistent) this is
    numerically identical to the old aggregate formula: with ``n`` constant,
    ``median(n / t_i) == n / median(t_i)``, since a ratio by a positive
    constant is a monotonic transform of ``t_i`` and so preserves the median's
    rank position. Also reports the per-repeat wall times ``t_0..t_{repeats-1}``
    with their own median ``t_median_s``, and the *last* repeat's ``samples``
    / ``env_steps`` / ``discarded`` (identical to every other repeat's when
    the workload is shared; under ``persistent`` just that one repeat's, since
    repeats are sequential continuations rather than a workload total). A
    *move* is one lane-step (``env_steps * cfg.selfplay.batch``) and each move
    runs ``cfg.search.num_simulations`` simulations.

    Raises ``ValueError`` under playout-cap randomization
    (``cfg.selfplay.pcr_full_prob`` < 1): the sims-per-move accounting assumes
    every step ran the full search. Raises ``ValueError`` if ``repeats`` < 1
    (the median over zero timed repeats is undefined), or if
    ``cfg.selfplay.persistent`` and ``warmup`` < 1 (persistent bench needs the
    warmup call to create the carry; without it, the first *timed* repeat pays
    pool creation and the XLA compile alongside the flush it's meant to
    measure)."""
    if cfg.selfplay.pcr_full_prob < 1.0:
        raise ValueError(
            "bench_selfplay needs pcr_full_prob == 1.0 (playout-cap randomization "
            f"makes sims_per_s meaningless); got {cfg.selfplay.pcr_full_prob}"
        )
    if repeats < 1:
        raise ValueError(f"bench_selfplay needs repeats >= 1; got {repeats}")
    if cfg.selfplay.persistent and warmup < 1:
        raise ValueError(
            "bench_selfplay needs warmup >= 1 under selfplay.persistent (the "
            "warmup call creates the carry -- pool ramp-up and the XLA compile "
            f"-- so a timed repeat measures steady-state flush only); got warmup={warmup}"
        )
    calls = selfplay_callables(backend, cfg, net)
    net_search = calls.make_net_search(cfg.search.num_simulations)
    search = functools.partial(net_search, eqx.partition(net, eqx.is_array)[0])

    def play(
        sd: int, carry: SelfPlayCarry | None
    ) -> tuple[int, int, int, SelfPlayCarry | None]:
        samples, stats, new_carry = run_selfplay(
            calls, search, cfg, cfg.selfplay.samples, sd, carry=carry
        )
        return samples["value"].shape[0], stats.env_steps, stats.discarded, new_carry

    # Non-persistent: `run_selfplay` always hands back `None`, so `carry` stays
    # `None` across every call below and each one rebuilds a fresh env at `sd`
    # -- unchanged from the pre-persistent behaviour. Persistent: the warmup
    # call(s) build the carry (at `seed + 1000`) and every later call threads
    # it, so `sd` stops mattering once the pool exists (`self_play`'s contract:
    # `seed` seeds only the first call of a persistent chain).
    carry: SelfPlayCarry | None = None
    for _ in range(warmup):
        *_, carry = play(seed + 1000, carry)
    times: list[float] = []
    samples_ratios: list[float] = []
    moves_ratios: list[float] = []
    n_samples = env_steps = discarded = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        n_samples, env_steps, discarded, carry = play(seed, carry)
        dt = time.perf_counter() - t0
        times.append(dt)
        samples_ratios.append(n_samples / dt)
        moves_ratios.append((env_steps * cfg.selfplay.batch) / dt)

    moves_per_s = statistics.median(moves_ratios)
    out = {
        "samples_per_s": statistics.median(samples_ratios),
        "moves_per_s": moves_per_s,
        "sims_per_s": moves_per_s * cfg.search.num_simulations,
        "samples": float(n_samples),
        "env_steps": float(env_steps),
        "discarded": float(discarded),
        "t_median_s": statistics.median(times),
    }
    out.update({f"t_{i}": t for i, t in enumerate(times)})
    return out
