"""The training loop: the backend seams, bit-exact resume (both backends), the
per-iteration steps, and the self-play callable cache.

The self-play, carry, arena/Elo and bench areas have their own files
(`test_selfplay.py`, `test_carry.py`, `test_arena_elo.py`, `test_bench.py`).

Expect tests: the inline snapshot is the contract; regenerate with
``EXPECTTEST_ACCEPT=1 pytest``."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _selfplay_stubs import jitted, uniform_legal_dist
from expecttest import assert_expected_inline
from settlrl_engine.board import Board, make_board
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
    loop,
    make_optimizer,
    prepare_targets,
    train_epochs,
)
from settlrl_learn.training.backends.base import Backend, load_run_state, save_run_state
from settlrl_learn.training.config import ArenaConfig, EvalConfig
from settlrl_learn.training.loop import learn, selfplay_callables
from settlrl_learn.training.selfplay import Samples, self_play
from settlrl_learn.training.selfplay.carry import carry_template


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


def test_runstate_serialise_roundtrip_is_bit_exact(tmp_path: Path) -> None:
    # The resume invariant at the serialization layer (no training): a fresh
    # RunState round-trips bit-exactly through eqx for both backends.
    import optax

    backends: list[tuple[str, Backend]] = [
        ("mlp", MLPBackend((16,))),
        ("gnn", GNNBackend(PRESETS["gn_global"]._replace(width=8, layers=1))),
    ]
    for name, backend in backends:
        net = backend.init(jax.random.key(0))
        opt = optax.adamw(1e-3)
        state = RunState(
            net, backend.init_opt(opt, net), {}, jnp.int32(3), jnp.float32(0.4),
            carry_template(backend, _learn_cfg(1, selfplay=_PERSISTENT)),
        )  # fmt: skip
        path = tmp_path / f"{name}.eqx"
        save_run_state(path, state)
        back = load_run_state(path, state)
        a, b = jax.tree.leaves(state.net), jax.tree.leaves(back.net)
        assert all(
            np.array_equal(np.asarray(x), np.asarray(y))
            for x, y in zip(a, b, strict=True)
        )
        assert int(back.iteration) == 3 and float(back.best) == float(jnp.float32(0.4))


def test_save_run_state_leaves_no_tmp_and_loads_back_whole(tmp_path: Path) -> None:
    # Happy path for the atomic write: no leftover `.tmp`, and the file the
    # rename leaves behind loads back completely.
    import optax

    backend = MLPBackend((16,))
    net = backend.init(jax.random.key(0))
    state = RunState(
        net, backend.init_opt(optax.adamw(1e-3), net), {}, jnp.int32(3),
        jnp.float32(0.4), carry_template(backend, _learn_cfg(1)),
    )  # fmt: skip
    path = tmp_path / "runstate.eqx"
    save_run_state(path, state)
    assert path.exists()
    assert not (tmp_path / "runstate.eqx.tmp").exists()
    back = load_run_state(path, state)
    assert int(back.iteration) == 3 and float(back.best) == float(jnp.float32(0.4))


def test_save_run_state_overwrites_a_stale_tmp(tmp_path: Path) -> None:
    # A `.tmp` left behind by a prior kill mid-write must not be mistaken for
    # a real checkpoint, and must not block the next write.
    import optax

    backend = MLPBackend((16,))
    net = backend.init(jax.random.key(0))
    state = RunState(
        net, backend.init_opt(optax.adamw(1e-3), net), {}, jnp.int32(3),
        jnp.float32(0.4), carry_template(backend, _learn_cfg(1)),
    )  # fmt: skip
    path = tmp_path / "runstate.eqx"
    stale = tmp_path / "runstate.eqx.tmp"
    stale.write_bytes(b"truncated by a prior kill")
    save_run_state(path, state)
    assert not stale.exists()
    back = load_run_state(path, state)
    assert int(back.iteration) == 3


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
    train_steps: int = 2,
    num_simulations: int = 1,
    value_blend: ValueBlendConfig | None = None,
    selfplay: SelfPlayConfig | None = None,
) -> LearnConfig:
    """Tiny, arena-free LearnConfig -- the resume property holds regardless of
    arena, so we skip it (games=0) to keep the run seconds-fast. Defaults to a
    single simulation, exercising the real tree-search jit (not the
    ``num_simulations=0`` lookahead special case) -- resume correctness does
    not depend on search *depth*, but it does depend on running the real
    search at least once per backend (the two headline bit-exact tests).
    Callers whose own assertions don't depend on how many env steps a game
    takes to finish (checked per call site, since ``lookahead`` self-plays
    measurably differently, not just faster) may pass ``num_simulations=0``
    for the 3-4x cheaper trace."""
    return LearnConfig(
        n_iterations=n_iterations, seed=seed,
        search=SearchSettings(num_simulations=num_simulations, max_considered=4),
        selfplay=selfplay or SelfPlayConfig(samples=8, batch=4),
        optim=OptimConfig(batch_size=4, train_steps=train_steps),
        replay=ReplayConfig(buffer_min=4),
        eval=EvalConfig(),
        arena=ArenaConfig(games=0),
        value_blend=value_blend or ValueBlendConfig(),
    )  # fmt: skip


# `max_steps` cuts each iteration mid-game, so the carried env, pending buffers
# and RNG (not just the surplus) decide what the next iteration plays -- and a
# game finishes partway through the run, so real samples train the net.
_PERSISTENT = SelfPlayConfig(samples=8, batch=4, persistent=True, max_steps=60)


def test_learn_resume_bit_exact_persistent(tmp_path: Path) -> None:
    # The headline durability property with the carry threaded: a straight
    # 6-iteration persistent run must equal a 2-iteration checkpoint + resume,
    # leaf-for-leaf. Only a bit-exact carry in the checkpoint can do that.
    # n_iterations is not free to shrink here: at this seed/config a game
    # finishes (produces samples) only on iteration 6 exactly -- fewer
    # iterations make the "real samples trained" assertion below vacuous.
    seen: list[float] = []

    def cfg(n: int) -> LearnConfig:
        return _learn_cfg(n, selfplay=_PERSISTENT)

    straight = learn(
        MLPBackend((16,)), cfg(6), on_iter=lambda i, m, n: seen.append(m["samples"])
    )
    assert sum(seen) > 0, "no iteration produced samples -- the test is vacuous"
    learn(MLPBackend((16,)), cfg(2), checkpoint_dir=tmp_path)
    resumed = learn(MLPBackend((16,)), cfg(6), resume_from=tmp_path / "runstate.eqx")
    _assert_nets_bit_exact(straight, resumed)


def test_learn_persistent_zero_sample_iteration_checkpoints(tmp_path: Path) -> None:
    # Under `persistent` a zero-sample iteration is legitimate -- the surplus of
    # an earlier overshoot already covers the request, so the call takes no env
    # step at all. It must still count and checkpoint (a `continue` here would
    # wedge the run at the last data-producing iteration's checkpoint).
    # num_simulations=0: verified this seed's zero-sample-then-flush pattern
    # (samples[0] > 0, samples[-1] == 0) holds unchanged under lookahead too, at
    # 3-4x less trace cost -- unlike the persistent test above, no per-iteration
    # `max_steps` cutoff makes the *which* iteration flushes sensitive to the
    # acting policy here (self_play just runs until the sample target is met).
    samples: list[float] = []
    cfg = _learn_cfg(
        3, num_simulations=0,
        selfplay=SelfPlayConfig(samples=8, batch=4, persistent=True),
    )  # fmt: skip
    learn(
        MLPBackend((16,)), cfg, checkpoint_dir=tmp_path,
        on_iter=lambda i, m, n: samples.append(m["samples"]),
    )  # fmt: skip
    assert samples[0] > 0 and samples[-1] == 0  # the first flush overshoots by a lot
    assert len(samples) == 3  # every iteration reported, zero-sample ones included
    straight = learn(
        MLPBackend((16,)), _learn_cfg(4, num_simulations=0, selfplay=cfg.selfplay)
    )
    resumed = learn(
        MLPBackend((16,)), _learn_cfg(4, num_simulations=0, selfplay=cfg.selfplay),
        resume_from=tmp_path / "runstate.eqx",
    )  # fmt: skip
    _assert_nets_bit_exact(straight, resumed)  # the 3rd iteration did checkpoint


def test_learn_skips_the_pool_when_resuming_without_persistence(
    tmp_path: Path,
) -> None:
    # Flipping `persistent` OFF across a resume: the run never reads the pool, so
    # the checkpoint's (much larger, differently-shaped) carry section is skipped
    # rather than shape-checked. Resuming at the checkpoint's own iteration count
    # runs nothing, so the returned net must be the checkpointed one verbatim --
    # no sample-count threshold at stake, so `num_simulations=0` is safe.
    backend = MLPBackend((16,))
    trained = learn(
        backend, _learn_cfg(1, num_simulations=0, selfplay=_PERSISTENT),
        checkpoint_dir=tmp_path,
    )  # fmt: skip
    resumed = learn(
        backend, _learn_cfg(1, num_simulations=0),
        resume_from=tmp_path / "runstate.eqx",
    )  # fmt: skip
    _assert_nets_bit_exact(trained, resumed)


def test_learn_rejects_resuming_a_pool_less_checkpoint_as_persistent(
    tmp_path: Path,
) -> None:
    # Flipping `persistent` ON across a resume: the checkpoint holds no pool to
    # continue, and its zero-row pad cannot fit the padded template. That must
    # name the knob, not surface a raw eqx shape error. The raise fires before
    # any self-play, so `num_simulations=0` is safe.
    backend = MLPBackend((16,))
    learn(backend, _learn_cfg(1, num_simulations=0), checkpoint_dir=tmp_path)
    with pytest.raises(ValueError, match=r"selfplay\.persistent"):
        learn(
            backend, _learn_cfg(2, num_simulations=0, selfplay=_PERSISTENT),
            resume_from=tmp_path / "runstate.eqx",
        )  # fmt: skip


def test_learn_resumes_from_a_pre_carry_checkpoint(tmp_path: Path) -> None:
    # `RunState` grew `selfplay_carry` (last field, so the carry is the file's
    # trailing section): a checkpoint written before the change -- this one with
    # that section stripped -- must still load and resume. No sample-count
    # threshold is at stake (unlike the persistent test above), so
    # `num_simulations=0` is safe here for the cheaper trace.
    backend = MLPBackend((16,))
    cfg1 = _learn_cfg(1, num_simulations=0)
    cfg3 = _learn_cfg(3, num_simulations=0)
    learn(backend, cfg1, checkpoint_dir=tmp_path)
    ck = tmp_path / "runstate.eqx"
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, carry_template(backend, cfg1))
    ck.write_bytes(ck.read_bytes()[: -len(buf.getvalue())])
    straight = learn(backend, cfg3)
    resumed = learn(backend, cfg3, resume_from=ck)
    _assert_nets_bit_exact(straight, resumed)


def test_learn_resume_bit_exact_mlp(tmp_path: Path) -> None:
    # Headline durability: a straight 2-iteration run must equal a 1-iter
    # checkpoint + resume to 2, leaf-for-leaf. Resume RNG is seed+iter, so the
    # split run must reproduce the contiguous one bit-for-bit.
    straight = learn(MLPBackend((16,)), _learn_cfg(2))
    learn(MLPBackend((16,)), _learn_cfg(1), checkpoint_dir=tmp_path)
    resumed = learn(
        MLPBackend((16,)), _learn_cfg(2), resume_from=tmp_path / "runstate.eqx"
    )
    _assert_nets_bit_exact(straight, resumed)


def test_learn_resume_bit_exact_gnn(tmp_path: Path) -> None:
    # Resume is a loop/serialization property, not an architecture one, so the
    # smallest net that still runs the GNN backend's own code path (distinct
    # from the mlp test above) suffices.
    cfg = PRESETS["gn_global"]._replace(width=8, layers=1, head_depth=1)
    straight = learn(GNNBackend(cfg), _learn_cfg(2))
    learn(GNNBackend(cfg), _learn_cfg(1), checkpoint_dir=tmp_path)
    resumed = learn(
        GNNBackend(cfg), _learn_cfg(2), resume_from=tmp_path / "runstate.eqx"
    )
    _assert_nets_bit_exact(straight, resumed)


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
    samples, _, _ = self_play(
        n_samples=16, batch_size=4, seed=0, **jitted(uniform_legal_dist, backend)
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
        # num_simulations=0: eval-scheduling doesn't depend on search depth.
        search=SearchSettings(num_simulations=0, max_considered=4),
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


def test_mlp_loss_masks_policy_by_train_policy() -> None:
    # The loss side of PCR: the policy CE averages over train_policy=1 positions
    # only (so it equals the loss on that subset), while value loss spans all.
    from settlrl_learn.features import FEATURE_DIM
    from settlrl_learn.training import mlp_loss
    from settlrl_learn.training.backends.mlp import MLPItem

    rng = np.random.default_rng(0)
    n = 6
    net = MLPBackend((8,)).init(jax.random.key(0))
    feats = jnp.asarray(rng.standard_normal((n, FEATURE_DIM)), jnp.float32)
    pol = jnp.asarray(rng.random((n, N_FLAT)), jnp.float32)
    val = jnp.asarray((rng.random(n) < 0.5).astype(np.float32))
    full = MLPItem(feats, pol, val, jnp.ones(n, jnp.bool_))
    half = full._replace(train_policy=jnp.array([1, 1, 1, 0, 0, 0], jnp.bool_))
    first3 = MLPItem(feats[:3], pol[:3], val[:3], jnp.ones(3, jnp.bool_))

    _, a_full = mlp_loss(net, full, 1.0)
    _, a_half = mlp_loss(net, half, 1.0)
    _, a_first3 = mlp_loss(net, first3, 1.0)
    # value loss spans every position -> unchanged by the policy mask.
    assert abs(float(a_full["value_loss"]) - float(a_half["value_loss"])) < 1e-5
    # masked policy loss == the policy loss over the unmasked subset alone.
    assert abs(float(a_half["policy_loss"]) - float(a_first3["policy_loss"])) < 1e-4


def test_bool_item_dtypes_are_loss_and_grad_bit_exact_with_float32() -> None:
    # Checkpoint format rev: `to_item` stores mask/train_policy as bool (they are
    # exact 0/1 flags) and the losses cast to float32 at use, so a bool item and
    # its old float32 form must give byte-equal loss and grads -- checked on CPU
    # for both backends over a real self-play batch (PCR mixes train_policy 0/1).
    from jaxtyping import Array, Float
    from settlrl_learn.nn.graph import Sample
    from settlrl_learn.training import mlp_loss
    from settlrl_learn.training.backends.gnn import gnn_loss

    def widen(item: Any) -> Any:  # the pre-rev float32 item form
        wide = {
            k: jnp.asarray(getattr(item, k), jnp.float32)
            for k in ("mask", "train_policy")
            if hasattr(item, k)
        }
        return item._replace(**wide)

    def assert_bit_exact(pair_a: Any, pair_b: Any) -> None:
        (va, ga), (vb, gb) = pair_a, pair_b
        assert np.asarray(va).tobytes() == np.asarray(vb).tobytes()
        la, lb = jax.tree.leaves(ga), jax.tree.leaves(gb)
        assert la
        for x, y in zip(la, lb, strict=True):
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes()

    with jax.default_device(jax.devices("cpu")[0]):
        mlp = MLPBackend((16,))
        samples, _, _ = self_play(
            n_samples=8, batch_size=4, seed=0, **jitted(uniform_legal_dist, mlp)
        )
        # PCR's train_policy=0 rows must be represented, not just all-ones.
        samples["train_policy"][::2] = 0.0
        m_item = mlp.to_item(samples)
        assert m_item.train_policy.dtype == jnp.bool_

        def loss_m(params: Any, it: Any) -> Float[Array, ""]:  # noqa: F722
            return mlp_loss(params, it, 1.0)[0]

        m_net = mlp.init(jax.random.key(0))
        assert_bit_exact(
            jax.value_and_grad(loss_m)(m_net, m_item),
            jax.value_and_grad(loss_m)(m_net, widen(m_item)),
        )

        gnn = GNNBackend(PRESETS["gn_global"]._replace(width=8, layers=1))
        g_samples, _, _ = self_play(
            n_samples=8, batch_size=4, seed=0, **jitted(uniform_legal_dist, gnn)
        )
        g_samples["train_policy"][::2] = 0.0
        g_item = gnn.to_item(g_samples)
        assert g_item.mask.dtype == jnp.bool_
        assert g_item.train_policy.dtype == jnp.bool_

        def loss_g(model: Any, it: Any) -> Float[Array, ""]:  # noqa: F722
            return gnn_loss(
                model, Sample(it.nodes, it.edges, it.glob, it.tiles, None),
                it.policy, it.value, it.mask, it.train_policy,
            )[0]  # fmt: skip

        g_net = gnn.init(jax.random.key(0))
        grad_g = eqx.filter_value_and_grad(loss_g)
        assert_bit_exact(grad_g(g_net, g_item), grad_g(g_net, widen(g_item)))


# --------------------------------------------------------------------------- #
# The self-play callable cache                                                 #
# --------------------------------------------------------------------------- #


def test_selfplay_callables_warm_hit_is_bit_exact_with_the_cold_build(
    monkeypatch: Any,
) -> None:
    # The cache's soundness gate, and the only test where a *warm hit* drives a
    # run: `learn` is a pure function of (backend, cfg), so two calls over ONE
    # backend must agree leaf-for-leaf -- but the first builds the self-play
    # callables and the second reuses them, so agreement here is exactly
    # "reusing the jitted callables changes no bit". The resume tests cannot say
    # this: they build a fresh backend per call, so every one of theirs is a cold
    # miss. Reuse is counted at the builder, not the clock.
    builds: list[int] = []
    real = loop._build_selfplay_callables

    def counted(*args: Any, **kwargs: Any) -> Any:
        builds.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "_build_selfplay_callables", counted)
    backend = MLPBackend((16,))
    cfg = _learn_cfg(1, num_simulations=0)
    seen: list[float] = []
    cold = learn(backend, cfg, on_iter=lambda i, m, n: seen.append(m["samples"]))
    warm = learn(backend, cfg)
    assert len(builds) == 1  # the second run really was a hit
    assert sum(seen) > 0, "no self-play samples trained -- the comparison is vacuous"
    _assert_nets_bit_exact(cold, warm)
    # and the *same* jitted objects come back -- a fresh closure would be a jit
    # cache miss even at an unchanged cache size.
    ca = selfplay_callables(backend, cfg, cold)
    cb = selfplay_callables(backend, cfg, warm)
    assert ca is cb and ca.make_net_search(0) is cb.make_net_search(0)


def test_selfplay_callables_rebuild_for_another_net_structure() -> None:
    # Two backends never share an entry (the key holds each one's identity), and
    # a second net of the same architecture reuses its backend's entry -- the
    # static-equality check passing, which is what lets a trained net reuse the
    # callables built for the freshly-initialised one.
    cfg = _learn_cfg(1, num_simulations=0)
    small, wide = MLPBackend((16,)), MLPBackend((8, 8))
    a = selfplay_callables(small, cfg, small.init(jax.random.key(0)))
    b = selfplay_callables(wide, cfg, wide.init(jax.random.key(0)))
    assert a is not b
    assert selfplay_callables(small, cfg, small.init(jax.random.key(1))) is a
    # ... and the check fails closed: a net whose static *treedef* differs from
    # the entry's rebuilds rather than silently reusing another structure's
    # search, even at that entry's own key.
    assert selfplay_callables(small, cfg, wide.init(jax.random.key(0))) is not a


def test_selfplay_callables_rebuild_when_the_search_config_changes() -> None:
    # A search knob that changes the traced program must miss: the key carries
    # the whole search config, not just the fields the loop happens to read.
    backend = MLPBackend((16,))
    net = backend.init(jax.random.key(0))
    a = selfplay_callables(backend, _learn_cfg(1, num_simulations=0), net)
    deep = _learn_cfg(1, num_simulations=0).model_copy(
        update={"search": SearchSettings(num_simulations=0, max_considered=8)}
    )
    assert selfplay_callables(backend, deep, net) is not a
