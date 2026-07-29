"""The net-agnostic training loop: self-play -> replay -> train -> periodic arena.

Each iteration self-plays under the current net (or a fixed teacher during the
warm-up), buffers the positions into a flashbax on-device replay, trains, and --
once past the warm-up -- scores the net vs. ``lookahead(heuristic)``. The
:class:`~settlrl_learn.training.backend.Backend` supplies everything net-specific;
this loop is shared by the flat-MLP and board-GNN paths.

:func:`learn` takes a single :class:`~settlrl_learn.training.config.LearnConfig`
(the grouped, validated knob surface) and orchestrates the per-iteration steps
(:mod:`settlrl_learn.training.steps`). Per-iteration RNG derives from
``cfg.seed`` and the iteration index, so ``resume_from`` (a prior ``runstate.eqx``)
continues a run bit-exactly.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import flashbax as fbx
import jax
import jax.numpy as jnp
from settlrl_agents.value import ValueFunction
from settlrl_engine.belief import belief_view
from settlrl_search import (
    PolicyWeights,
    PolicyWeightsValue,
    make_search_weights,
    make_search_weights_value,
)

from settlrl_learn.training.arena import OpponentSpec
from settlrl_learn.training.backend import (
    Backend,
    RunState,
    load_run_state,
    save_run_state,
)
from settlrl_learn.training.carry import (
    PaddedCarry,
    SelfPlayCarry,
    carry_template,
    from_padded,
    to_padded,
)
from settlrl_learn.training.config import LearnConfig
from settlrl_learn.training.selfplay import Samples, SelfPlayStats, self_play
from settlrl_learn.training.steps import (
    evaluate,
    make_optimizer,
    prepare_targets,
    run_arena,
    train_epochs,
)


class SelfPlayCallables(NamedTuple):
    """The pre-built jitted+vmapped callables :func:`self_play` takes.

    ``make_net_search(num_simulations)`` builds the net's search: a jitted+vmapped
    callable whose *first* argument is the net's array params (bind them with
    ``functools.partial`` per weight update).
    """

    view_of: Callable[..., Any]
    observe_of: Callable[..., Any]
    setup_search: Callable[..., Any] | None
    make_net_search: Callable[[int], Any]


def _weights_factory(cfg: LearnConfig) -> tuple[Any, dict[str, Any]]:
    """The search-weights factory (root-value-returning under value blending) and
    the shared search knobs, less ``num_simulations``."""
    s = cfg.search
    mk = make_search_weights_value if cfg.value_blend.max > 0 else make_search_weights
    return mk, {
        "max_depth": s.max_depth, "max_num_considered_actions": s.max_considered,
        "expected_rolls": s.expected_rolls, "chance_nodes": s.chance_nodes,
        "dev_chance": s.dev_chance, "ordered": s.ordered,
    }  # fmt: skip


def _callables_key(cfg: LearnConfig) -> tuple[str, bool]:
    """Everything :func:`selfplay_callables` bakes into its traced program: the
    whole search config (all of it reaches the search factory) and whether value
    blending selects the root-value-returning one. Batch and seat counts are not
    in it -- they ride the traced arguments' shapes, which jax keys its own
    compilation cache on -- and neither are the setup knobs, which live on the
    backend (keyed by identity)."""
    return cfg.search.model_dump_json(), cfg.value_blend.max > 0


_CALLABLES_CACHE: dict[
    tuple[int, tuple[str, bool]], tuple[Any, Any, SelfPlayCallables]
] = {}
"""``(id(backend), config key) -> (backend, net static, callables)``. The backend
rides in the *value* so the cache holds it alive -- its ``id`` cannot then be
recycled by a later object while the entry exists. The static rides along so a
hit *checks* the same-architecture assumption instead of assuming it (two nets of
one architecture have equal statics; a differently-shaped net rebuilds). Unbounded:
a process runs a handful of configurations."""


def selfplay_callables(
    backend: Backend, cfg: LearnConfig, net: Any
) -> SelfPlayCallables:
    """The self-play callables for ``net``'s architecture -- the search closes
    over the net's *static* (non-array) part and takes its arrays as a traced
    arg. Shared by :func:`learn` and
    :func:`~settlrl_learn.training.bench.bench_selfplay`.

    Memoised on ``(backend, the search-affecting config, the net's static)``, so
    a second call -- a second :func:`learn` in one process -- reuses the same
    jitted objects instead of re-tracing them. The net's arrays are a traced
    argument and never enter the key."""
    key = (id(backend), _callables_key(cfg))
    net_static = eqx.partition(net, eqx.is_array)[1]
    hit = _CALLABLES_CACHE.get(key)
    if hit is not None and eqx.tree_equal(hit[1], net_static):
        return hit[2]
    calls = _build_selfplay_callables(backend, cfg, net_static)
    _CALLABLES_CACHE[key] = (backend, net_static, calls)
    return calls


def _build_selfplay_callables(
    backend: Backend, cfg: LearnConfig, net_static: Any
) -> SelfPlayCallables:
    s = cfg.search
    setup_fn = backend.setup_policy()
    mk, search_kwargs = _weights_factory(cfg)
    view_of = jax.jit(jax.vmap(belief_view, in_axes=(0, 0, 0)))
    observe_of = jax.jit(jax.vmap(backend.observe, in_axes=(0, 0, 0)))
    setup_search = (
        jax.jit(jax.vmap(setup_fn, in_axes=(0, 0, 0, 0, 0)))
        if setup_fn is not None
        else None
    )

    # Memoised too: `make_net_search` builds a *fresh* closure per call, and a
    # fresh closure is a jit cache miss -- so without this the caching above
    # would save the encoders but not the search itself.
    @functools.cache
    def make_net_search(num_simulations: int) -> Any:
        def _net_weights(
            arrays: Any, key: Any, layout: Any, view: Any, player: Any, mask: Any
        ) -> Any:
            model = eqx.combine(arrays, net_static)
            v_fn, p_fn = backend.seams(model)
            wfn = mk(
                v_fn, prior=p_fn, value_scale=s.value_scale,
                num_simulations=num_simulations, **search_kwargs,
            )  # fmt: skip
            return wfn(key, layout, view, player, mask)

        return jax.jit(jax.vmap(_net_weights, in_axes=(None, 0, 0, 0, 0, 0)))

    return SelfPlayCallables(view_of, observe_of, setup_search, make_net_search)


def run_selfplay(
    calls: SelfPlayCallables,
    search: Any,
    cfg: LearnConfig,
    n: int,
    seed: int,
    *,
    fast_search: Any = None,
    full_prob: float = 1.0,
    carry: SelfPlayCarry | None = None,
) -> tuple[Samples, SelfPlayStats, SelfPlayCarry | None]:
    """:func:`~settlrl_learn.training.selfplay.self_play` over ``calls`` and
    ``cfg``'s selfplay/search knobs -- shared by :func:`learn`,
    :func:`~settlrl_learn.training.bench.bench_selfplay`, and the
    ``test_selfplay_window`` benchmark so the kwarg wiring cannot drift.

    Under ``cfg.selfplay.persistent`` the returned carry resumes the games in
    flight on the next call (and ``seed`` then only seeds the first one)."""
    return self_play(
        search, fast_search=fast_search, full_prob=full_prob, n_samples=n,
        observe_of=calls.observe_of, view_of=calls.view_of,
        setup_search=calls.setup_search,
        batch_size=cfg.selfplay.batch, temperature=cfg.selfplay.temperature,
        temperature_moves=cfg.selfplay.temperature_moves,
        seed=seed, record_value=cfg.value_blend.max > 0,
        track_ordering=cfg.search.ordered,
        max_steps=cfg.selfplay.max_steps, max_game_len=cfg.selfplay.max_game_len,
        persistent=cfg.selfplay.persistent, carry=carry,
    )  # fmt: skip


def learn(
    backend: Backend,
    cfg: LearnConfig,
    *,
    teacher_value: ValueFunction | None = None,
    net_opponents: Mapping[str, tuple[OpponentSpec, float, int]] | None = None,
    checkpoint_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    on_iter: Callable[[int, dict[str, float], Any], None] | None = None,
    progress: bool = False,
) -> Any:
    """One training loop over ``backend`` under ``cfg``; returns the final net.

    ``teacher_value`` (with ``cfg.teacher.iters`` > 0) warm-starts the loop: the
    first ``cfg.teacher.iters`` iterations draw their moves and policy targets from
    a fixed strong search (``cfg.teacher.sims`` simulations) over ``teacher_value``
    instead of the cold net.

    ``net_opponents`` (name -> ``(spec, anchor_elo, every)``) adds pre-built arena
    opponents alongside ``cfg.arena.opponents`` -- the loop stays agnostic about
    where a spec comes from, so a frozen checkpoint is composed by the caller
    (see :func:`~settlrl_learn.training.steps.run_arena`).

    The full :class:`RunState` is checkpointed to ``checkpoint_dir/runstate.eqx``
    every ``cfg.checkpoint_every`` iterations; ``resume_from`` continues it
    bit-exactly. ``on_iter(i, metrics, net)`` runs after each iteration.
    ``progress`` shows a tqdm bar over the iterations."""
    s = cfg.search
    optimizer = make_optimizer(cfg.optim)
    buffer = fbx.make_item_buffer(
        max_length=cfg.replay.buffer_max, min_length=cfg.replay.buffer_min,
        sample_batch_size=cfg.optim.batch_size, add_batches=True,
    )  # fmt: skip
    net0 = backend.init(jax.random.key(cfg.seed))
    zero_carry: PaddedCarry | None = carry_template(backend, cfg)
    fresh_state = RunState(
        net0, backend.init_opt(optimizer, net0),
        buffer.init(backend.empty_item()), jnp.int32(0), jnp.float32(-1.0), zero_carry,
    )  # fmt: skip
    # A non-persistent run never reads the pool: a `None` template carry skips
    # that section, which a persistent checkpoint's pad would otherwise fail on.
    template = (
        fresh_state
        if cfg.selfplay.persistent
        else fresh_state._replace(selfplay_carry=None)
    )
    state = load_run_state(resume_from, template) if resume_from else fresh_state
    net, opt_state, buf_state = state.net, state.opt_state, state.buffer_state
    best, start = float(state.best), int(state.iteration)
    # The pool the checkpoint left in flight (a pre-carry or fresh state has
    # none, so `persistent` then starts a new one seeded as usual).
    carry: SelfPlayCarry | None = (
        from_padded(state.selfplay_carry, track_ordering=cfg.search.ordered)
        if cfg.selfplay.persistent and bool(state.selfplay_carry.present)
        else None
    )
    del state, fresh_state, template  # the padded carry is large: keep one copy
    if cfg.selfplay.persistent:
        # Every persistent checkpoint pads the live carry, so the full-size zero
        # template has no reader left.
        zero_carry = None
    ckpt = Path(checkpoint_dir) / "runstate.eqx" if checkpoint_dir else None

    step = backend.make_step(optimizer)
    # Warm up the jitted step (one-off XLA compile) on a zero batch so the recorded
    # per-iteration `t_train` is the optimiser step, not the compile. The returned
    # update is discarded -- net/opt_state are untouched.
    _warm = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (cfg.optim.batch_size, *x.shape)),
        backend.empty_item(),
    )
    jax.block_until_ready(step(net, opt_state, _warm))  # type: ignore[no-untyped-call]
    blend = cfg.value_blend.max > 0
    mk, search_kwargs = _weights_factory(cfg)
    # The teacher search uses the heuristic value at its own (factory) value_scale,
    # not the net's `s.value_scale`; the net's leaf is a win-probability logit.
    teacher_weights: PolicyWeights | PolicyWeightsValue | None = (
        mk(teacher_value, num_simulations=cfg.teacher.sims, **search_kwargs)
        if teacher_value is not None
        else None
    )

    calls = selfplay_callables(backend, cfg, net)
    teacher_search = (
        jax.jit(jax.vmap(teacher_weights, in_axes=(0, 0, 0, 0, 0)))
        if teacher_weights is not None
        else None
    )
    net_search = calls.make_net_search(s.num_simulations)
    # Playout-cap randomization: a cheaper search for the value-only (fast) steps.
    pcr = cfg.selfplay.pcr_full_prob < 1.0 and cfg.selfplay.pcr_fast_sims > 0
    net_search_fast: Any = (
        calls.make_net_search(cfg.selfplay.pcr_fast_sims) if pcr else None
    )

    def _play(
        search: Any,
        n: int,
        seed: int,
        *,
        fast_search: Any = None,
        full_prob: float = 1.0,
    ) -> tuple[Samples, SelfPlayStats, SelfPlayCarry | None]:
        return run_selfplay(
            calls, search, cfg, n, seed,
            fast_search=fast_search, full_prob=full_prob, carry=carry,
        )  # fmt: skip

    iters: Iterable[int] = range(start, cfg.n_iterations)
    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(iters, initial=start, total=cfg.n_iterations, unit="iter")
        iters = bar

    for i in iters:
        t0 = time.perf_counter()
        teaching = teacher_weights is not None and i < cfg.teacher.iters
        net_arrays = eqx.partition(net, eqx.is_array)[0]
        search: Any
        fast: Any = None
        full_prob = 1.0
        if teaching:
            search = teacher_search  # warm-up: always full, no PCR
        else:
            search = functools.partial(net_search, net_arrays)
            if pcr:
                fast = functools.partial(net_search_fast, net_arrays)
                full_prob = cfg.selfplay.pcr_full_prob
        fresh, sp_stats, carry = _play(
            search, cfg.selfplay.samples, cfg.seed + 1 + i,
            fast_search=fast, full_prob=full_prob,
        )  # fmt: skip
        t_selfplay = time.perf_counter() - t0
        # `selfplay_discarded` is the pending positions thrown away at the
        # iteration boundary (games still unfinished) -- the self-play waste.
        sp = {
            "selfplay_steps": float(sp_stats.env_steps),
            "selfplay_discarded": float(sp_stats.discarded),
        }
        nf = fresh["value"].shape[0]
        metrics: dict[str, float] = {
            "samples": float(nf), "lr": cfg.optim.lr, "t_selfplay": t_selfplay, **sp,
        }  # fmt: skip

        # A zero-sample iteration is normal under `persistent` (the carried
        # surplus already covered the request, so the call took no env step) and
        # means a degenerate net elsewhere. Either way it skips the *data* steps
        # -- eval, replay add, and the optimiser (no fresh data, no update) --
        # but still counts, checkpoints and proceeds.
        eval_d: dict[str, float] = {}
        steps = 0
        if nf:
            # Periodic generalization check: score val_* on this iter's fresh
            # batch under the *pre-train* net -- the net generated these positions
            # but has not trained on them yet, so it is a valid held-out-in-time
            # signal; the batch then trains as normal (no data wasted). Gated past
            # the warm-up.
            if (
                cfg.eval.every
                and (i + 1) % cfg.eval.every == 0
                and (i + 1) >= cfg.teacher.iters
            ):
                te = time.perf_counter()
                sl = {k: v[: cfg.eval.samples] for k, v in fresh.items()}
                eval_d = evaluate(backend, net, sl)
                eval_d["t_eval"] = time.perf_counter() - te

            fr, alpha = prepare_targets(
                fresh, blend=blend,
                blend_max=cfg.value_blend.max, blend_ramp=cfg.value_blend.ramp,
                iteration=i,
            )  # fmt: skip
            buf_state = buffer.add(buf_state, backend.to_item(fr))
            steps = (
                cfg.optim.train_steps
                if cfg.optim.reuse <= 0
                else max(
                    1,
                    int(cfg.optim.reuse * fr["value"].shape[0] / cfg.optim.batch_size),
                )
            )
            # entropy of the search policy *targets* (degenerate targets -> the
            # net learns a degenerate policy).
            tp = jnp.asarray(fr["policy"])
            metrics["target_entropy"] = float(
                -jnp.mean(jnp.sum(tp * jnp.log(jnp.clip(tp, 1e-9, 1.0)), axis=-1))
            )
            metrics["value_blend_alpha"] = alpha
            metrics["train_steps"] = float(steps)

        t1 = time.perf_counter()
        if steps and bool(buffer.can_sample(buf_state)):
            net, opt_state, tm = train_epochs(
                net, opt_state, buffer, buf_state, step, steps,
                jax.random.key(cfg.seed + 10_000 + i),
            )  # fmt: skip
            metrics.update(tm)
        metrics["t_train"] = time.perf_counter() - t1
        metrics.update(eval_d)  # val_* from the pre-train eval above (eval iters)

        # Arena only once the net is past the warm-up: a half-trained net drags
        # games out, and the search arena pays full cost per step.
        if (
            cfg.arena.games
            and (i + 1) % cfg.arena.every == 0
            and (i + 1) >= cfg.teacher.iters
        ):
            t2 = time.perf_counter()
            # Fixed seed (no +i): every checkpoint faces the *same* games, so the
            # arena curve is paired across iterations -- only the net varies and
            # the dice/board luck differences out (the big variance cut).
            round_index = (i + 1) // cfg.arena.every
            am = run_arena(
                backend, net, cfg.arena, seed=cfg.seed + 20_000,
                round_index=round_index, net_opponents=net_opponents,
            )  # fmt: skip
            metrics.update(am)
            metrics["t_arena"] = time.perf_counter() - t2
            if "arena_winrate" in am:
                best = max(best, am["arena_winrate"])

        if ckpt is not None and (i + 1) % cfg.checkpoint_every == 0:
            if carry is not None:
                pool = to_padded(carry, cfg.selfplay.max_game_len)
            else:
                assert zero_carry is not None  # only persistent runs drop it
                pool = zero_carry
            save_run_state(
                ckpt,
                RunState(
                    net, opt_state, buf_state, jnp.int32(i + 1), jnp.float32(best), pool
                ),
            )
            # Persistent: `pool` is a fresh `to_padded` copy built just above,
            # so this frees it (the next checkpoint pads the live carry again).
            # Non-persistent: `pool` merely aliases the reused `zero_carry`
            # template, which the name `zero_carry` keeps alive for the next
            # checkpoint -- `del` here only drops this local binding.
            del pool
        if bar is not None:
            bar.set_postfix(
                {
                    k: round(metrics[k], 3)
                    for k in ("loss", "arena_winrate")
                    if k in metrics
                }
            )
        if on_iter is not None:
            on_iter(i, metrics, net)
    return net
