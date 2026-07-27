"""Fast structural contracts for the unified training package -- the backend
seams and the net-agnostic self-play, exercised without the search (a uniform
policy stands in for `weights_fn`, so these stay seconds-fast).

Expect tests: the inline snapshot is the contract; regenerate with
``EXPECTTEST_ACCEPT=1 pytest``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline
from jaxtyping import Array
from settlrl_engine.belief import belief_view
from settlrl_engine.board import Board, make_board
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.env import N_FLAT
from settlrl_learn.nn.graphnet import PRESETS
from settlrl_learn.training import (
    GNNBackend,
    LearnConfig,
    MLPBackend,
    OptimConfig,
    ReplayConfig,
    RunState,
    SearchSettings,
    SelfPlayConfig,
    ValueBlendConfig,
    make_optimizer,
    prepare_targets,
    train_epochs,
)
from settlrl_learn.training.backend import Backend, load_run_state, save_run_state
from settlrl_learn.training.config import ArenaConfig, EvalConfig
from settlrl_learn.training.gnn_backend import _SETUP_ROWS
from settlrl_learn.training.loop import learn
from settlrl_learn.training.selfplay import Samples, self_play


def _shapes(tree: object) -> str:
    """Trailing shapes of a pytree's array leaves, one per line (the leading
    sample/batch axis is run-dependent, so it is dropped)."""
    leaves = jax.tree.leaves(tree)
    return "\n".join(str(tuple(np.asarray(x).shape)) for x in leaves)


def _single(n_players: int = 2, seed: int = 0) -> Board:
    layout, state = make_board(batch_size=1, seed=seed, n_players=n_players)
    return jax.tree.map(lambda x: x[0], layout), jax.tree.map(lambda x: x[0], state)


def test_mlp_backend_item_and_observe_shapes() -> None:
    backend = MLPBackend((16,))
    layout, state = _single()
    obs = backend.observe(layout, state, jnp.int32(0))
    assert_expected_inline(
        f"keys={sorted(obs)}\nempty_item:\n{_shapes(backend.empty_item())}",
        """\
keys=['features']
empty_item:
(118,)
(662,)
()
()""",
    )


def test_gnn_backend_item_and_observe_shapes() -> None:
    backend = GNNBackend(
        PRESETS["gn_global"]._replace(width=16, layers=2, head_depth=1)
    )
    layout, state = _single()
    obs = backend.observe(layout, state, jnp.int32(0))
    assert_expected_inline(
        f"keys={sorted(obs)}\nempty_item:\n{_shapes(backend.empty_item())}",
        """\
keys=['edges', 'glob', 'nodes', 'tiles']
empty_item:
(54, 17)
(144, 3)
(40,)
(19, 9)
(662,)
(662,)
()
()""",
    )


def _uniform_weights(
    key: Array, layout: BoardLayout, view: Any, player: Array, mask: Array
) -> Array:
    """A stand-in for the search: uniform over the legal set (no net, no tree)."""
    return mask.astype(jnp.float32)


def _uniform_legal_dist(
    key: Array, layout: BoardLayout, view: Any, player: Array, mask: Array
) -> Array:
    """A *normalised* uniform-over-legal stand-in -- a proper distribution, like
    the real search's visit-count target (the bare mask is unnormalised)."""
    m = mask.astype(jnp.float32)
    return m / jnp.sum(m)


def _jitted(weights_fn: Any, backend: Backend) -> dict[str, Any]:
    """Build the pre-jitted+vmapped callables `self_play` now expects from a bare
    `weights_fn` stand-in and a backend (no setup search)."""
    return {
        "search": jax.jit(jax.vmap(weights_fn, in_axes=(0, 0, 0, 0, 0))),
        "observe_of": jax.jit(jax.vmap(backend.observe, in_axes=(0, 0, 0))),
        "view_of": jax.jit(jax.vmap(belief_view, in_axes=(0, 0, 0))),
    }


def test_self_play_samples_shape_under_uniform_policy() -> None:
    # Drives the real generic self-play (env stepping, pending flush, outcome
    # credit) with the MLP observation but a trivial policy -- fast, no search.
    backend = MLPBackend((16,))
    samples = self_play(
        n_samples=8, batch_size=4, seed=0,
        **_jitted(_uniform_weights, backend),
    )  # fmt: skip
    n = samples["value"].shape[0]
    assert n >= 8 and all(v.shape[0] == n for v in samples.values())
    trailing = {k: tuple(v.shape[1:]) for k, v in sorted(samples.items())}
    assert_expected_inline(
        str(trailing),
        "{'features': (118,), 'mask': (662,), 'policy': (662,), "
        "'train_policy': (), 'value': ()}",
    )
    # the env mask is binary; the policy target is recorded over the legal set.
    assert set(np.unique(samples["mask"])).issubset({0.0, 1.0})
    assert samples["policy"].shape[1] == N_FLAT


def _uniform_weights_value(
    key: Array, layout: BoardLayout, view: Any, player: Array, mask: Array
) -> tuple[Array, Array]:
    """Uniform policy + a constant root value (a PolicyWeightsValue stand-in)."""
    return mask.astype(jnp.float32), jnp.float32(0.3)


def test_self_play_records_root_value_when_asked() -> None:
    backend = MLPBackend((16,))
    samples = self_play(
        n_samples=8, batch_size=4, seed=0, record_value=True,
        **_jitted(_uniform_weights_value, backend),
    )  # fmt: skip
    assert "q" in samples and samples["q"].shape == samples["value"].shape
    assert bool(np.all(np.abs(samples["q"] - 0.3) < 1e-5))  # the stand-in's q


def test_runstate_serialise_roundtrip_is_bit_exact(tmp_path: Path) -> None:
    # The resume invariant at the serialization layer (no training): a fresh
    # RunState round-trips bit-exactly through eqx for both backends.
    import optax

    backends: list[tuple[str, Backend]] = [
        ("mlp", MLPBackend((16,))),
        ("gnn", GNNBackend(PRESETS["gn_global"]._replace(width=16, layers=2))),
    ]
    for name, backend in backends:
        net = backend.init(jax.random.key(0))
        opt = optax.adamw(1e-3)
        state = RunState(
            net, backend.init_opt(opt, net), {}, jnp.int32(3), jnp.float32(0.4)
        )
        path = tmp_path / f"{name}.eqx"
        save_run_state(path, state)
        back = load_run_state(path, state)
        a, b = jax.tree.leaves(state.net), jax.tree.leaves(back.net)
        assert all(
            np.array_equal(np.asarray(x), np.asarray(y))
            for x, y in zip(a, b, strict=True)
        )
        assert int(back.iteration) == 3 and float(back.best) == float(jnp.float32(0.4))


# --------------------------------------------------------------------------- #
# Bit-exact resume, end-to-end (both backends)                                #
# --------------------------------------------------------------------------- #


def _net_arrays(net: Any) -> list[np.ndarray]:
    """The numeric array leaves of a net (an AZParams pytree or an eqx module)."""
    arrays = eqx.filter(net, eqx.is_array)
    return [np.asarray(x) for x in jax.tree.leaves(arrays)]


def _assert_nets_bit_exact(a: Any, b: Any) -> None:
    la, lb = _net_arrays(a), _net_arrays(b)
    assert len(la) == len(lb) and la, "expected matching, non-empty leaf sets"
    for x, y in zip(la, lb, strict=True):
        assert np.array_equal(x, y)


def _learn_cfg(
    n_iterations: int,
    *,
    seed: int = 7,
    train_steps: int = 3,
    value_blend: ValueBlendConfig | None = None,
) -> LearnConfig:
    """Tiny, arena-free LearnConfig -- the resume property holds regardless of
    arena, so we skip it (games=0) to keep the run seconds-fast."""
    return LearnConfig(
        n_iterations=n_iterations, seed=seed,
        search=SearchSettings(num_simulations=2, max_considered=4),
        selfplay=SelfPlayConfig(samples=8, batch=4),
        optim=OptimConfig(batch_size=4, train_steps=train_steps),
        replay=ReplayConfig(buffer_min=4),
        eval=EvalConfig(),
        arena=ArenaConfig(games=0),
        value_blend=value_blend or ValueBlendConfig(),
    )  # fmt: skip


def test_learn_resume_bit_exact_mlp(tmp_path: Path) -> None:
    # Headline durability: a straight 3-iteration run must equal a 1-iter
    # checkpoint + resume to 3, leaf-for-leaf. Resume RNG is seed+iter, so the
    # split run must reproduce the contiguous one bit-for-bit.
    straight = learn(MLPBackend((16,)), _learn_cfg(3))
    learn(MLPBackend((16,)), _learn_cfg(1), checkpoint_dir=tmp_path)
    resumed = learn(
        MLPBackend((16,)), _learn_cfg(3), resume_from=tmp_path / "runstate.eqx"
    )
    _assert_nets_bit_exact(straight, resumed)


def test_learn_resume_bit_exact_gnn(tmp_path: Path) -> None:
    cfg = PRESETS["gn_global"]._replace(width=16, layers=2, head_depth=1)
    straight = learn(GNNBackend(cfg), _learn_cfg(3))
    learn(GNNBackend(cfg), _learn_cfg(1), checkpoint_dir=tmp_path)
    resumed = learn(
        GNNBackend(cfg), _learn_cfg(3), resume_from=tmp_path / "runstate.eqx"
    )
    _assert_nets_bit_exact(straight, resumed)


# --------------------------------------------------------------------------- #
# Self-play data semantics                                                     #
# --------------------------------------------------------------------------- #


def test_self_play_value_is_acting_seat_win_loss() -> None:
    # Credit assignment: the recorded value is the *acting seat's* eventual
    # win (1) / loss (0), not a constant and not the raw reward. The labels
    # must therefore be exactly {0, 1}, and -- the nontrivial part -- a finished
    # 2p game produces positions for *both* seats (they alternate), so the
    # winner's positions are labelled 1 and the loser's 0: both classes must
    # appear. A bug that always credited seat 0, or that stored the seat index
    # / raw VP reward, would break one of these. (We use the same batch_size=4
    # config as the existing shape test, which is known to finish games under
    # the uniform stand-in; the flat output hides the lane partition, so the
    # both-classes-present check is the strongest lane-agnostic form of the
    # complementary-per-game property.)
    backend = MLPBackend((16,))
    samples = self_play(
        n_samples=16, batch_size=4, seed=0, temperature=0.0,
        **_jitted(_uniform_weights, backend),
    )  # fmt: skip
    sv = samples["value"]
    assert set(np.unique(sv)).issubset({0.0, 1.0})  # win/loss only, never a VP/seat
    assert sv.sum() > 0 and sv.sum() < len(sv)  # both a winner's and a loser's slice


def test_self_play_policy_target_is_legal() -> None:
    # The recorded policy target is exactly the weights_fn output, verbatim
    # (the real search returns a normalised visit distribution; here a
    # normalised uniform-over-legal stand-in). Property: non-negative, sums to
    # ~1, and -- the load-bearing part -- ZERO mass on illegal actions, since
    # the search may only propose legal moves.
    backend = MLPBackend((16,))
    samples = self_play(
        n_samples=16, batch_size=4, seed=3, temperature=0.0,
        **_jitted(_uniform_legal_dist, backend),
    )  # fmt: skip
    pol, mask = samples["policy"], samples["mask"]
    assert np.all(pol >= 0.0)
    sums = pol.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-5), f"policy rows not normalised: {sums}"
    illegal_mass = np.where(mask == 0, pol, 0.0).sum()
    assert illegal_mass == 0.0, f"policy put {illegal_mass} mass on illegal actions"


def test_self_play_excludes_setup_gnn() -> None:
    # With the GNN backend's fixed setup policy playing the opening, no setup
    # position leaks into training data. The observation carries no phase field,
    # so we assert it via the mask: a recorded position is in the main loop iff a
    # non-setup action is legal there. Every recorded mask must satisfy that.
    backend = GNNBackend(
        PRESETS["gn_global"]._replace(width=16, layers=2, head_depth=1)
    )
    setup_search = jax.jit(jax.vmap(backend.setup_policy(), in_axes=(0, 0, 0, 0, 0)))
    samples = self_play(
        n_samples=8, batch_size=4, seed=4, temperature=0.0,
        setup_search=setup_search,
        **_jitted(_uniform_weights, backend),
    )  # fmt: skip
    mask = samples["mask"].astype(bool)
    setup_rows = np.asarray(_SETUP_ROWS)
    main_legal = (mask & ~setup_rows).any(axis=-1)
    assert main_legal.all(), "a recorded position had only setup actions legal"
    # stronger: no recorded position is purely a setup placement (some lane is in
    # SETUP only when every legal action is a setup row).
    pure_setup = mask.any(axis=-1) & ~main_legal
    assert not pure_setup.any()


# --------------------------------------------------------------------------- #
# Value-blend formula                                                          #
# --------------------------------------------------------------------------- #


def test_value_blend_alpha_ramp() -> None:
    # The loop ramps alpha linearly 0 -> value_blend_max over value_blend_ramp
    # iterations. We read the live per-iteration alpha off the on_iter metrics
    # and check it against the documented schedule (loop.py:181-183). This is
    # the side the loop owns; iteration 0 must be a pure-z no-op (alpha 0).
    alphas: dict[int, float] = {}

    def on_iter(i: int, metrics: dict[str, float], net: Any) -> None:
        # a degenerate (no-game) iteration emits no alpha; only record real ones.
        if "value_blend_alpha" in metrics:
            alphas[i] = metrics["value_blend_alpha"]

    learn(
        MLPBackend((16,)),
        _learn_cfg(
            4, seed=11, train_steps=2, value_blend=ValueBlendConfig(max=0.5, ramp=4)
        ),
        on_iter=on_iter,
    )
    # alpha[i] = value_blend_max * min(1, i / max(ramp, 1)); ramp=4, max=0.5.
    schedule = {0: 0.0, 1: 0.5 * (1 / 4), 2: 0.5 * (2 / 4), 3: 0.5 * (3 / 4)}
    assert alphas, "no iteration produced samples"
    assert alphas == {i: schedule[i] for i in alphas}  # every real iter on-schedule
    assert alphas[0] == 0.0  # iteration 0 is always a pure-z no-op


def test_prepare_targets_value_blend() -> None:
    # Direct test of the extracted step (no full learn run): all data trains
    # (the eval slice is a separate fresh generation), so this pins the
    # value-blend formula against the real function.
    rng = np.random.default_rng(0)
    n = 20
    fresh: Samples = {
        "value": (rng.random(n) < 0.5).astype(np.float32),  # z in {0, 1}
        "q": np.full(n, 0.3, np.float32),  # searcher frame -> q_prob 0.65
        "policy": rng.random((n, 5)).astype(np.float32),
    }

    # blend off: value untouched, alpha 0.
    fr, alpha = prepare_targets(
        fresh, blend=False, blend_max=0.0, blend_ramp=1, iteration=3
    )
    assert alpha == 0.0
    assert np.array_equal(fr["value"], fresh["value"])

    # blend on at the ramp midpoint: alpha = 0.5 * min(1, 2/4) = 0.25.
    fr, alpha = prepare_targets(
        fresh, blend=True, blend_max=0.5, blend_ramp=4, iteration=2
    )
    assert abs(alpha - 0.25) < 1e-12
    # value -> affine mix (1-a)z + a*0.65, i.e. one of two values, valid P(win).
    lo, hi = 0.25 * 0.65, 0.75 + 0.25 * 0.65  # blend of z=0 and z=1
    assert np.all(np.isclose(fr["value"], lo) | np.isclose(fr["value"], hi))
    assert np.all(fr["value"] >= 0.0) and np.all(fr["value"] <= 1.0)


def test_train_epochs_is_deterministic_in_key() -> None:
    # The inner update loop is a pure function of (net, opt_state, key): the same
    # key replays the same minibatch draws and yields a bit-identical net -- the
    # property bit-exact resume rests on, isolated from the rest of the loop.
    import flashbax as fbx
    import optax

    backend = MLPBackend((16,))
    samples = self_play(
        n_samples=16, batch_size=4, seed=0, **_jitted(_uniform_legal_dist, backend)
    )
    optimizer = optax.adamw(1e-3)
    net = backend.init(jax.random.key(0))
    # a finished game flushes all its positions at once, so the batch can be large.
    buffer = fbx.make_item_buffer(
        max_length=max(64, samples["value"].shape[0]),
        min_length=4, sample_batch_size=4, add_batches=True,
    )  # fmt: skip
    buf = buffer.add(buffer.init(backend.empty_item()), backend.to_item(samples))
    step = backend.make_step(optimizer)
    key = jax.random.key(123)
    n1, _, m1 = train_epochs(
        net, backend.init_opt(optimizer, net), buffer, buf, step, 3, key
    )
    n2, _, m2 = train_epochs(
        net, backend.init_opt(optimizer, net), buffer, buf, step, 3, key
    )
    _assert_nets_bit_exact(n1, n2)
    assert m1.keys() == m2.keys()
    assert all(abs(m1[k] - m2[k]) < 1e-9 for k in m1)


def test_periodic_eval_emits_val_metrics() -> None:
    # The held-out slice is gone: eval is a separate fresh generation every
    # `cfg.eval.every` iters. Assert it fires and produces the val_* metrics.
    seen: dict[str, float] = {}

    def on_iter(i: int, metrics: dict[str, float], net: Any) -> None:
        seen.update({k: v for k, v in metrics.items() if k.startswith("val_")})

    cfg = LearnConfig(
        n_iterations=2, seed=5,
        search=SearchSettings(num_simulations=2, max_considered=4),
        selfplay=SelfPlayConfig(samples=8, batch=4),
        optim=OptimConfig(batch_size=4, train_steps=2),
        replay=ReplayConfig(buffer_min=4),
        eval=EvalConfig(every=1, samples=8),
        arena=ArenaConfig(games=0),
    )  # fmt: skip
    learn(MLPBackend((16,)), cfg, on_iter=on_iter)
    assert "val_value_acc" in seen  # the periodic eval ran and scored a fresh batch


def test_make_optimizer_grad_clip() -> None:
    import optax
    from settlrl_learn.training.config import OptimConfig

    # grad_clip > 0 caps the raw gradient's global norm before adamw -- verify the
    # clip layer's semantics directly (adamw then rescales per-coordinate).
    g = {"w": jnp.array([3.0, 4.0])}  # global norm 5
    clip = optax.clip_by_global_norm(2.0)
    out, _ = clip.update(g, clip.init(g))
    assert abs(float(optax.global_norm(out)) - 2.0) < 1e-5
    # the clip is stateless, so it adds no opt-state leaves: a clipped and an
    # unclipped optimiser carry the same adamw moments (only the nesting differs).
    p = {"w": jnp.zeros(2)}
    n_clip = len(jax.tree.leaves(make_optimizer(OptimConfig(grad_clip=1.0)).init(p)))
    n_plain = len(jax.tree.leaves(make_optimizer(OptimConfig(grad_clip=0.0)).init(p)))
    assert n_clip == n_plain


# --------------------------------------------------------------------------- #
# Playout-cap randomization (PCR)                                              #
# --------------------------------------------------------------------------- #


def test_self_play_pcr_marks_fast_positions() -> None:
    # With a fast_search + full_prob < 1, each step is full (train_policy 1) or
    # fast (0); the data side of PCR. value is recorded for both (fast positions
    # still train the value head).
    backend = MLPBackend((16,))
    j = _jitted(_uniform_legal_dist, backend)
    samples = self_play(
        n_samples=64, batch_size=8, seed=1,
        fast_search=j["search"], full_prob=0.5, **j,
    )  # fmt: skip
    tp = samples["train_policy"]
    assert set(np.unique(tp)).issubset({0.0, 1.0})
    assert tp.min() == 0.0 and tp.max() == 1.0  # both full and fast steps occurred
    assert tp.shape == samples["value"].shape  # a flag per recorded position


def test_run_arena_uses_real_counts_for_elo_and_reports_se(monkeypatch: Any) -> None:
    # run_arena must feed the *actual* (wins, episodes) arena returns into the
    # Elo MLE -- not wr * cfg.games -- and report the SE beside arena_elo.
    from settlrl_learn.training.arena import ArenaResult
    from settlrl_learn.training.elo import anchored_elo, anchored_elo_se
    from settlrl_learn.training.steps import run_arena

    result = ArenaResult(wins=30.0, episodes=50)
    monkeypatch.setattr("settlrl_learn.training.steps.arena", lambda *a, **k: result)
    cfg = ArenaConfig(games=40, opponents=["lookahead"], anchor_elos={"lookahead": 0.0})
    metrics = run_arena(MLPBackend((16,)), object(), cfg, seed=0)
    assert metrics["arena_winrate"] == result.winrate
    assert metrics["arena_elo"] == anchored_elo([(0.0, 30.0, 50)])
    assert metrics["arena_elo_se"] == anchored_elo_se([(0.0, 30.0, 50)])


def test_self_play_no_pcr_marks_all_full() -> None:
    # Default (no fast_search): every position is a full-search position.
    backend = MLPBackend((16,))
    samples = self_play(
        n_samples=8, batch_size=4, seed=0, **_jitted(_uniform_weights, backend)
    )
    assert np.all(samples["train_policy"] == 1.0)


def test_mlp_loss_masks_policy_by_train_policy() -> None:
    # The loss side of PCR: the policy CE averages over train_policy=1 positions
    # only (so it equals the loss on that subset), while value loss spans all.
    from settlrl_learn.features import FEATURE_DIM
    from settlrl_learn.training import mlp_loss
    from settlrl_learn.training.mlp_backend import MLPItem

    rng = np.random.default_rng(0)
    n = 6
    net = MLPBackend((8,)).init(jax.random.key(0))
    feats = jnp.asarray(rng.standard_normal((n, FEATURE_DIM)), jnp.float32)
    pol = jnp.asarray(rng.random((n, N_FLAT)), jnp.float32)
    val = jnp.asarray((rng.random(n) < 0.5).astype(np.float32))
    full = MLPItem(feats, pol, val, jnp.ones(n, jnp.float32))
    half = full._replace(train_policy=jnp.array([1, 1, 1, 0, 0, 0], jnp.float32))
    first3 = MLPItem(feats[:3], pol[:3], val[:3], jnp.ones(3, jnp.float32))

    _, a_full = mlp_loss(net, full, 1.0)
    _, a_half = mlp_loss(net, half, 1.0)
    _, a_first3 = mlp_loss(net, first3, 1.0)
    # value loss spans every position -> unchanged by the policy mask.
    assert abs(float(a_full["value_loss"]) - float(a_half["value_loss"])) < 1e-5
    # masked policy loss == the policy loss over the unmasked subset alone.
    assert abs(float(a_half["policy_loss"]) - float(a_first3["policy_loss"])) < 1e-4
