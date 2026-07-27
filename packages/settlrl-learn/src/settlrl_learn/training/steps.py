"""The per-iteration training steps, extracted from ``learn`` as pure units.

Each is a function of its inputs (no loop state, no hidden RNG): the loop derives
every key from ``seed`` + iteration index and threads it in, so the steps stay
testable in isolation and bit-exact resume is preserved. ``learn`` orchestrates
them; this module holds the bodies.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

from typing import Any

import jax
import optax
from jaxtyping import Array

from settlrl_learn.training.arena import arena
from settlrl_learn.training.backend import Backend
from settlrl_learn.training.config import ArenaConfig, OptimConfig
from settlrl_learn.training.elo import anchored_elo, anchored_elo_se
from settlrl_learn.training.selfplay import Samples


def make_optimizer(cfg: OptimConfig) -> optax.GradientTransformation:
    """adamw, optionally preceded by global-norm gradient clipping
    (``cfg.grad_clip`` > 0). The clip is stateless, so an unclipped checkpoint
    must be resumed with ``grad_clip=0`` (its opt-state has no clip layer)."""
    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.grad_clip > 0:
        opt = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), opt)
    return opt


def prepare_targets(
    fresh: Samples,
    *,
    blend: bool,
    blend_max: float,
    blend_ramp: int,
    iteration: int,
) -> tuple[Samples, float]:
    """Apply the value-target blend to a fresh self-play batch (all of it trains;
    the eval slice is a separate fresh generation, not held out here).

    ``blend`` mixes the value to ``(1-alpha)*z + alpha*(q in [0,1])`` with
    ``alpha = blend_max * min(1, iteration/max(blend_ramp,1))``; returns
    ``(train_samples, alpha)``."""
    alpha = blend_max * min(1.0, iteration / max(blend_ramp, 1)) if blend else 0.0
    if blend:
        q_prob = (fresh["q"] + 1.0) / 2.0  # searcher frame [-1,1] -> P(win) [0,1]
        fresh = {**fresh, "value": (1.0 - alpha) * fresh["value"] + alpha * q_prob}
    return fresh, alpha


def train_epochs(
    net: Any,
    opt_state: Any,
    buffer: Any,
    buf_state: Any,
    step: Any,
    steps: int,
    key: Array,
) -> tuple[Any, Any, dict[str, float]]:
    """Run ``steps`` minibatch updates, sampling from ``buffer``; return the
    updated ``(net, opt_state)`` and the per-step-averaged metrics."""
    sums: dict[str, float] = {}
    for _ in range(steps):
        key, k = jax.random.split(key)
        item = buffer.sample(buf_state, k).experience
        net, opt_state, m = step(net, opt_state, item)
        for kk, vv in m.items():
            sums[kk] = sums.get(kk, 0.0) + float(vv)
    return net, opt_state, {kk: vv / steps for kk, vv in sums.items()}


def evaluate(backend: Backend, net: Any, ev: Samples) -> dict[str, float]:
    """Held-out diagnostics over the eval accumulator (``val_*`` metrics)."""
    vm = backend.eval_metrics(net, backend.to_item(ev))
    return {kk: float(vv) for kk, vv in vm.items()}


def run_arena(
    backend: Backend, net: Any, cfg: ArenaConfig, *, seed: int
) -> dict[str, float]:
    """Play the net against each configured opponent; ``lookahead`` -> the gate
    metric ``arena_winrate``, others -> ``arena_vs_<opponent>``, plus ``arena_elo``
    -- the MLE Elo on the fixed ``anchor_elos`` scale -- and ``arena_elo_se``, its
    standard error. Opponents get well-separated seeds (``seed + j*10_000``); the
    loop holds ``seed`` fixed across iterations so every checkpoint faces the same
    games (a paired strength curve)."""
    metrics: dict[str, float] = {}
    elo_inputs: list[tuple[float, float, int]] = []
    for j, opp in enumerate(cfg.opponents):
        res = arena(
            backend, net, opponent=opp, n_games=cfg.games,
            num_simulations=cfg.sims, batch_size=cfg.batch,
            max_num_considered_actions=cfg.considered, seed=seed + j * 10_000,
        )  # fmt: skip
        metrics["arena_winrate" if opp == "lookahead" else f"arena_vs_{opp}"] = (
            res.winrate
        )
        if opp in cfg.anchor_elos:
            elo_inputs.append((cfg.anchor_elos[opp], res.wins, res.episodes))
    if elo_inputs:
        metrics["arena_elo"] = anchored_elo(elo_inputs)
        metrics["arena_elo_se"] = anchored_elo_se(elo_inputs)
    return metrics
