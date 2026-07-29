"""Fast structural contracts for the unified training package -- the backend
seams and the net-agnostic self-play, exercised without the search (a uniform
policy stands in for `weights_fn`, so these stay seconds-fast).

Expect tests: the inline snapshot is the contract; regenerate with
``EXPECTTEST_ACCEPT=1 pytest``."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from expecttest import assert_expected_inline
from jaxtyping import Array
from settlrl_agents import POLICIES, BeliefSpec
from settlrl_engine.belief import belief_view
from settlrl_engine.board import Board, make_board
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.env import N_FLAT, BatchedSettlrlEnv
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
from settlrl_learn.training.arena import ArenaResult, arena
from settlrl_learn.training.backend import Backend, load_run_state, save_run_state
from settlrl_learn.training.bench import bench_selfplay
from settlrl_learn.training.config import ArenaConfig, EvalConfig
from settlrl_learn.training.elo import anchored_elo, anchored_elo_se
from settlrl_learn.training.gnn_backend import _SETUP_ROWS
from settlrl_learn.training.loop import carry_template, learn
from settlrl_learn.training.selfplay import (
    PaddedEnv,
    Samples,
    SelfPlayCarry,
    empty_padded,
    from_padded,
    self_play,
    to_padded,
)
from settlrl_learn.training.steps import run_arena


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
    samples, _, _ = self_play(
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


def test_self_play_reports_stats() -> None:
    # The stats side of the contract: env steps actually taken, recorded ==
    # the returned sample count, and a non-negative discard count (the pending
    # positions of games still unfinished when the budget ran out -- the
    # iteration-boundary waste).
    backend = MLPBackend((16,))
    samples, stats, _ = self_play(
        n_samples=4, batch_size=2, seed=0,
        **_jitted(_uniform_weights, backend),
    )  # fmt: skip
    assert stats.env_steps > 0
    assert stats.recorded == samples["value"].shape[0]
    assert stats.discarded >= 0


def _uniform_weights_value(
    key: Array, layout: BoardLayout, view: Any, player: Array, mask: Array
) -> tuple[Array, Array]:
    """Uniform policy + a constant root value (a PolicyWeightsValue stand-in)."""
    return mask.astype(jnp.float32), jnp.float32(0.3)


def test_self_play_records_root_value_when_asked() -> None:
    backend = MLPBackend((16,))
    samples, _, _ = self_play(
        n_samples=8, batch_size=4, seed=0, record_value=True,
        **_jitted(_uniform_weights_value, backend),
    )  # fmt: skip
    assert "q" in samples and samples["q"].shape == samples["value"].shape
    assert bool(np.all(np.abs(samples["q"] - 0.3) < 1e-5))  # the stand-in's q


# --------------------------------------------------------------------------- #
# Persistent self-play carry                                                   #
# --------------------------------------------------------------------------- #

# Captured from the pre-carry `self_play` (commit b6fb56b) at exactly the config
# below. The flag-off path must reproduce it bit-for-bit -- the RNG stream and
# the recording order are the contract, so these are frozen constants, NOT an
# expecttest snapshot to regenerate.
_GOLDEN_STATS = (833, 800, 866)  # env_steps, recorded, discarded
_GOLDEN_ARRAYS = {
    "features": ((800, 118), "float32", "7a7735a8c2af2582"),
    "mask": ((800, 662), "bool", "20ab5e3d5a6eaca6"),
    "policy": ((800, 662), "float32", "c5ecb941a3ea2504"),
    "q": ((800,), "float32", "ce857a5e9ccde945"),
    "train_policy": ((800,), "float32", "59e707682300eb4d"),
    "value": ((800,), "float32", "408ff942db14360c"),
}


def _fingerprint(samples: Samples) -> dict[str, tuple[tuple[int, ...], str, str]]:
    return {
        k: (
            v.shape,
            str(v.dtype),
            hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16],
        )
        for k, v in samples.items()
    }


def test_self_play_flag_off_matches_pre_carry_golden() -> None:
    backend = MLPBackend((16,))
    samples, stats, carry = self_play(
        n_samples=8, batch_size=2, seed=0, temperature=1.0, record_value=True,
        **_jitted(_uniform_weights_value, backend),
    )  # fmt: skip
    assert carry is None  # nothing survives a non-persistent call
    assert (stats.env_steps, stats.recorded, stats.discarded) == _GOLDEN_STATS
    assert _fingerprint(samples) == _GOLDEN_ARRAYS


def test_temperature_moves_zero_matches_flag_off_golden() -> None:
    # `temperature_moves=0` (its default) must reproduce the pre-anneal golden
    # fingerprint exactly -- passing it explicitly draws no extra RNG and changes
    # no behavior.
    backend = MLPBackend((16,))
    samples, stats, carry = self_play(
        n_samples=8, batch_size=2, seed=0, temperature=1.0, record_value=True,
        temperature_moves=0,
        **_jitted(_uniform_weights_value, backend),
    )  # fmt: skip
    assert carry is None
    assert (stats.env_steps, stats.recorded, stats.discarded) == _GOLDEN_STATS
    assert _fingerprint(samples) == _GOLDEN_ARRAYS


def test_temperature_moves_anneal_then_argmax_is_key_independent() -> None:
    # K=1: a lane's first recorded move samples at `temperature`; every move
    # after is argmax, independent of which key drives it. We fork an in-flight
    # carry (same board, two different keys) via the padded round trip -- the
    # env is a held *object*, so it can't be forked by sharing a reference --
    # and confirm the post-K step lands on the same board state either way.
    backend = MLPBackend((16,))
    j = _jitted(_uniform_legal_dist, backend)
    _, _, carry = self_play(
        n_samples=10_000, batch_size=1, seed=0, temperature=5.0,
        temperature_moves=1, persistent=True, max_steps=1, **j,
    )  # fmt: skip
    assert carry is not None and len(carry.pending[0]) == 1  # move 0 recorded
    padded = to_padded(carry, max_game_len=800)
    carry_a = from_padded(padded, track_ordering=False)._replace(key=jax.random.key(11))
    carry_b = from_padded(padded, track_ordering=False)._replace(key=jax.random.key(22))
    kwargs = {
        "n_samples": 10_000, "batch_size": 1, "temperature": 5.0,
        "temperature_moves": 1, "persistent": True, "max_steps": 1,
    }  # fmt: skip
    _, _, carry_a2 = self_play(carry=carry_a, **kwargs, **j)  # type: ignore[arg-type]
    _, _, carry_b2 = self_play(carry=carry_b, **kwargs, **j)  # type: ignore[arg-type]
    assert carry_a2 is not None and carry_b2 is not None

    def _plain(x: Array) -> np.ndarray:
        if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            return np.asarray(jax.random.key_data(x))
        return np.asarray(x)

    state_a = jax.tree.map(_plain, carry_a2.env.board[1])
    state_b = jax.tree.map(_plain, carry_b2.env.board[1])
    assert eqx.tree_equal(state_a, state_b) is True


def test_persistent_carry_two_calls_equal_one_long_call() -> None:
    # The carry's defining property: collection is a continuous stream cut at
    # sample counts, not restarted per call. Two carried calls of N must produce
    # exactly the concatenation the single 2N call produces -- same env stepping,
    # same RNG stream, same flush order -- including the overshoot (a finished
    # game flushes whole, so the first call returns past N and the second's
    # target accounts for that surplus).
    backend = MLPBackend((16,))
    j = _jitted(_uniform_legal_dist, backend)
    n = 250
    first, s1, carry = self_play(
        n_samples=n, batch_size=2, seed=1, temperature=1.0, persistent=True, **j
    )
    assert carry is not None
    assert s1.recorded > n  # the overshoot the surplus must account for
    second, s2, carry2 = self_play(
        n_samples=n, batch_size=2, seed=1, temperature=1.0, persistent=True,
        carry=carry, **j,
    )  # fmt: skip
    assert carry2 is not None and s2.recorded > 0  # the second call really played
    long, sl, _ = self_play(
        n_samples=2 * n, batch_size=2, seed=1, temperature=1.0, **j
    )  # fmt: skip
    assert s1.env_steps + s2.env_steps == sl.env_steps
    assert s1.recorded + s2.recorded == sl.recorded
    assert set(first) == set(second) == set(long)
    for k in long:
        joined = np.concatenate([first[k], second[k]])
        assert np.array_equal(joined, long[k]), f"{k} diverged from the long call"


def test_persistent_zero_step_call_keeps_the_stream_intact() -> None:
    # A call whose request the carried surplus already covers must take no step
    # at all (realistic at scale: one step can flush several whole buffers) --
    # and still return the full key set with the real dtypes, so it concatenates
    # into the stream as a no-op. `mask` is bool, so a float32 empty would be
    # silently promoted by np.concatenate; the carried spec keeps dtypes exact.
    backend = MLPBackend((16,))
    j = _jitted(_uniform_legal_dist, backend)
    first, s1, carry = self_play(
        n_samples=250, batch_size=2, seed=1, temperature=1.0, persistent=True, **j
    )
    assert carry is not None and s1.recorded > 100  # surplus > the next request
    mid, s_mid, carry = self_play(
        n_samples=100, batch_size=2, seed=1, temperature=1.0, persistent=True,
        carry=carry, **j,
    )  # fmt: skip
    assert (s_mid.recorded, s_mid.env_steps) == (0, 0)
    assert set(mid) == set(first)
    assert {k: mid[k].dtype for k in mid} == {k: first[k].dtype for k in first}
    assert carry is not None
    last, _, _ = self_play(
        n_samples=150, batch_size=2, seed=1, temperature=1.0, persistent=True,
        carry=carry, **j,
    )  # fmt: skip
    long, _, _ = self_play(
        n_samples=500, batch_size=2, seed=1, temperature=1.0, **j
    )  # fmt: skip
    for k in long:
        joined = np.concatenate([first[k], mid[k], last[k]])
        assert joined.dtype == long[k].dtype and np.array_equal(joined, long[k]), k


def test_persistent_discard_counts_only_trims() -> None:
    # Flag off, the unfinished games are thrown away at the call boundary (the
    # iteration waste). Flag on, they stay in the carry, so `discarded` counts
    # only `max_game_len` trims -- zero for games that never reach the cap.
    backend = MLPBackend((16,))
    j = _jitted(_uniform_legal_dist, backend)
    _, fresh, none_carry = self_play(
        n_samples=200, batch_size=2, seed=1, temperature=1.0, **j
    )
    assert none_carry is None and fresh.discarded > 0
    _, stats, carry = self_play(
        n_samples=200, batch_size=2, seed=1, temperature=1.0, persistent=True, **j
    )
    assert carry is not None
    assert stats.recorded == fresh.recorded  # the first call collects identically
    assert stats.discarded == 0
    assert any(carry.pending)  # the in-flight game survived instead


# --------------------------------------------------------------------------- #
# The carry's padded (checkpointable) form                                     #
# --------------------------------------------------------------------------- #


def _mid_game_carry(
    *, batch_size: int = 2, n_samples: int = 60, track_ordering: bool = False
) -> SelfPlayCarry:
    """A carry with games in flight (non-empty pending buffers) and a `q` key."""
    backend = MLPBackend((16,))
    _, _, carry = self_play(
        n_samples=n_samples, batch_size=batch_size, seed=1, temperature=1.0,
        persistent=True, record_value=True, track_ordering=track_ordering,
        **_jitted(_uniform_weights_value, backend),
    )  # fmt: skip
    assert carry is not None and any(carry.pending)
    return carry


def _carry_rows(carry: SelfPlayCarry) -> list[list[tuple[Any, ...]]]:
    """The pending rows in a comparable (hashable-free) form."""
    return [
        [
            (
                {k: v.tobytes() for k, v in obs.items()},
                pol.tobytes(),
                mask.tobytes(),
                seat,
                q,
                tp,
            )
            for obs, pol, mask, seat, q, tp in lane
        ]
        for lane in carry.pending
    ]


def test_carry_padded_round_trip_is_exact() -> None:
    # The checkpointable form must be lossless: the pending buffers, the RNG key,
    # the (possibly negative) surplus and the recorded-field spec all survive.
    # `pending_len` in particular carries the anneal counter (the per-lane
    # recorded-move count, `len(pending[lane])`) -- no new carry field exists for
    # it, since `pending_len` already is that count -- so its round trip is
    # asserted directly here rather than in a separate test.
    carry = _mid_game_carry()
    counts = [len(lane) for lane in carry.pending]
    padded = to_padded(carry, max_game_len=800)
    assert padded.pending_len.tolist() == counts
    back = from_padded(padded, track_ordering=False)
    assert _carry_rows(back) == _carry_rows(carry)
    assert np.array_equal(
        np.asarray(jax.random.key_data(back.key)),
        np.asarray(jax.random.key_data(carry.key)),
    )
    assert back.surplus == carry.surplus
    assert back.spec == carry.spec
    assert len(back.pending) == len(carry.pending)


def test_carry_padded_round_trip_resumes_identically() -> None:
    # The env is a held object, not a pytree: the padded form must reconstruct an
    # env that plays on identically. Continuing from the restored carry must
    # reproduce the original continuation sample-for-sample.
    backend = MLPBackend((16,))
    j = _jitted(_uniform_weights_value, backend)
    carry = _mid_game_carry()
    restored = from_padded(to_padded(carry, max_game_len=800), track_ordering=False)
    # past the carried surplus, so the continuation really steps the restored env
    kw = {
        "n_samples": 600, "batch_size": 2, "seed": 1, "temperature": 1.0,
        "persistent": True, "record_value": True,
    }  # fmt: skip
    a, sa, _ = self_play(carry=carry, **kw, **j)  # type: ignore[arg-type]
    b, sb, _ = self_play(carry=restored, **kw, **j)  # type: ignore[arg-type]
    assert sa == sb and sa.env_steps > 0
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{k} diverged after the round trip"


def test_padded_env_captures_every_env_array() -> None:
    # The padded form hand-lists the env's array state; a new engine-side array
    # attribute must break loudly here rather than silently not being carried.
    env = BatchedSettlrlEnv(
        batch_size=2, seed=0, reward="sparse", n_players=2,
        track_beliefs=True, track_ordering=True,
    )  # fmt: skip
    live = {
        name
        for name, v in vars(env).items()
        if any(hasattr(x, "dtype") for x in jax.tree.leaves(v))
    }
    assert live == {f"_{f}" for f in PaddedEnv._fields}


def test_from_padded_rejects_a_reconfigured_run() -> None:
    # Resuming a checkpoint into a run with different self-play semantics is a
    # user error, not a programming one: fail loudly. Shapes catch a changed
    # batch size; `track_ordering` is invisible to them, so it is checked.
    padded = to_padded(_mid_game_carry(track_ordering=True), max_game_len=800)
    with pytest.raises(ValueError, match="track_ordering"):
        from_padded(padded, track_ordering=False)


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


def test_runstate_carries_the_live_pool_through_eqx(tmp_path: Path) -> None:
    # The carry survives the *file*, not just the in-memory conversion pair: it
    # is deserialised into the zero template a fresh run builds.
    import optax

    backend = MLPBackend((16,))
    net = backend.init(jax.random.key(0))
    carry = _mid_game_carry()
    padded = to_padded(carry, max_game_len=800)
    fresh = RunState(
        net, backend.init_opt(optax.adamw(1e-3), net), {}, jnp.int32(0),
        jnp.float32(-1.0),
        empty_padded(
            batch_size=len(carry.pending), n_players=2, track_ordering=False,
            max_game_len=800, spec=carry.spec,
        ),
    )  # fmt: skip
    path = tmp_path / "runstate.eqx"
    save_run_state(path, fresh._replace(selfplay_carry=padded))
    back = load_run_state(path, fresh)
    assert not bool(fresh.selfplay_carry.present)  # the template stands for "none"
    assert bool(back.selfplay_carry.present)
    restored = from_padded(back.selfplay_carry, track_ordering=False)
    assert _carry_rows(restored) == _carry_rows(carry)
    assert restored.surplus == carry.surplus and restored.spec == carry.spec


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
    samples, _, _ = self_play(
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
    samples, _, _ = self_play(
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
    samples, _, _ = self_play(
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
    samples, _, _ = self_play(
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


# --------------------------------------------------------------------------- #
# Playout-cap randomization (PCR)                                              #
# --------------------------------------------------------------------------- #


def test_self_play_pcr_marks_fast_positions() -> None:
    # With a fast_search + full_prob < 1, each step is full (train_policy 1) or
    # fast (0); the data side of PCR. value is recorded for both (fast positions
    # still train the value head).
    backend = MLPBackend((16,))
    j = _jitted(_uniform_legal_dist, backend)
    samples, _, _ = self_play(
        n_samples=64, batch_size=8, seed=1,
        fast_search=j["search"], full_prob=0.5, **j,
    )  # fmt: skip
    tp = samples["train_policy"]
    assert set(np.unique(tp)).issubset({0.0, 1.0})
    assert tp.min() == 0.0 and tp.max() == 1.0  # both full and fast steps occurred
    assert tp.shape == samples["value"].shape  # a flag per recorded position


def test_run_arena_uses_real_counts_for_elo_and_reports_se(monkeypatch: Any) -> None:
    # run_arena must feed the *actual* (wins, episodes) arena returns into the
    # Elo MLE -- not wr * cfg.games. Two anchors with different overshoot ratios
    # (50/40 vs 20/40 episodes-per-nominal-game) make the real-counts and
    # nominal-counts Elo provably different -- a single anchor can't discriminate
    # the two paths, since anchored_elo there depends only on the win ratio.
    results = {
        "lookahead": ArenaResult(wins=30.0, episodes=50),
        "random": ArenaResult(wins=10.0, episodes=20),
    }
    monkeypatch.setattr(
        "settlrl_learn.training.steps.arena",
        lambda *a, opponent, **k: results[opponent],
    )
    cfg = ArenaConfig(
        games=40,
        opponents=["lookahead", "random"],
        anchor_elos={"lookahead": 0.0, "random": -1115.0},
    )
    metrics = run_arena(MLPBackend((16,)), object(), cfg, seed=0, round_index=1)
    real_inputs = [(0.0, 30.0, 50), (-1115.0, 10.0, 20)]
    nominal_inputs = [
        (0.0, 0.6 * 40, 40),
        (-1115.0, 0.5 * 40, 40),
    ]  # the old, buggy feed
    assert metrics["arena_winrate"] == results["lookahead"].winrate
    assert metrics["arena_elo"] == anchored_elo(real_inputs)
    assert metrics["arena_elo"] != anchored_elo(nominal_inputs)
    assert metrics["arena_elo_se"] == anchored_elo_se(real_inputs)


def test_run_arena_opponent_every_skips_off_rounds(monkeypatch: Any) -> None:
    # opponent_every={"random": 5} plays random only on round_index multiples of
    # 5; lookahead (absent from the map) plays every round. A skipped opponent
    # contributes no arena_vs_<opp> metric and no Elo input that round.
    calls: list[str] = []
    results = {
        "lookahead": ArenaResult(wins=30.0, episodes=50),
        "random": ArenaResult(wins=10.0, episodes=20),
    }

    def _fake_arena(*a: Any, opponent: str, **k: Any) -> ArenaResult:
        calls.append(opponent)
        return results[opponent]

    monkeypatch.setattr("settlrl_learn.training.steps.arena", _fake_arena)
    cfg = ArenaConfig(
        games=40,
        opponents=["lookahead", "random"],
        anchor_elos={"lookahead": 0.0, "random": -1115.0},
        opponent_every={"random": 5},
    )
    backend = MLPBackend((16,))

    for round_index in range(1, 5):
        calls.clear()
        metrics = run_arena(backend, object(), cfg, seed=0, round_index=round_index)
        assert calls == ["lookahead"]
        assert "arena_vs_random" not in metrics
        assert metrics["arena_elo"] == anchored_elo([(0.0, 30.0, 50)])
        assert metrics["arena_elo_se"] == anchored_elo_se([(0.0, 30.0, 50)])

    calls.clear()
    metrics = run_arena(backend, object(), cfg, seed=0, round_index=5)
    assert calls == ["lookahead", "random"]
    assert metrics["arena_vs_random"] == results["random"].winrate


def _dummy_spec() -> BeliefSpec:
    # The agent is never built: these tests stub the arena out.
    return BeliefSpec(lambda: cast("Any", None), frozenset((2,)))


def test_arena_name_path_delegates_to_the_spec_core(monkeypatch: Any) -> None:
    # The name-based `arena` only resolves POLICIES and hands the spec to the
    # shared core -- the seat-swap/seed/episode logic exists once.
    seen: dict[str, Any] = {}

    def _fake_spec_arena(backend: Any, net: Any, **kwargs: Any) -> ArenaResult:
        seen.update(kwargs)
        return ArenaResult(wins=1.0, episodes=2)

    # by module object: the training package rebinds `arena` to the function, so
    # the dotted path no longer reaches the submodule.
    arena_module = sys.modules["settlrl_learn.training.arena"]
    monkeypatch.setattr(arena_module, "arena_spec", _fake_spec_arena)
    res = arena(
        MLPBackend((16,)), object(), opponent="random", n_games=8,
        num_simulations=17, max_num_considered_actions=5, batch_size=9, seed=3,
    )  # fmt: skip
    assert res == ArenaResult(1.0, 2)
    assert seen == {
        "opponent": POLICIES["random"], "n_games": 8, "num_simulations": 17,
        "max_num_considered_actions": 5, "batch_size": 9, "seed": 3,
    }  # fmt: skip


def test_run_arena_net_opponent_joins_metrics_and_elo(monkeypatch: Any) -> None:
    # A pre-built spec opponent (a frozen checkpoint) plays alongside the registry
    # anchors: it reports arena_vs_<name> and its (elo, wins, episodes) joins the
    # same MLE. Its seed comes off a base disjoint from the registry opponents'.
    seeds: dict[str, int] = {}

    def _fake_arena(*a: Any, opponent: str, seed: int, **k: Any) -> ArenaResult:
        seeds[opponent] = seed
        return ArenaResult(wins=30.0, episodes=50)

    def _fake_spec_arena(*a: Any, opponent: Any, seed: int, **k: Any) -> ArenaResult:
        seeds["az0"] = seed
        return ArenaResult(wins=24.0, episodes=40)

    monkeypatch.setattr("settlrl_learn.training.steps.arena", _fake_arena)
    monkeypatch.setattr("settlrl_learn.training.steps.arena_spec", _fake_spec_arena)
    cfg = ArenaConfig(games=40, opponents=["lookahead"], anchor_elos={"lookahead": 0.0})
    metrics = run_arena(
        MLPBackend((16,)), object(), cfg, seed=7, round_index=1,
        net_opponents={"az0": (_dummy_spec(), -100.0, 1)},
    )  # fmt: skip
    inputs = [(0.0, 30.0, 50), (-100.0, 24.0, 40)]
    assert metrics["arena_winrate"] == 0.6
    assert metrics["arena_vs_az0"] == 0.6
    assert metrics["arena_elo"] == anchored_elo(inputs)
    assert metrics["arena_elo_se"] == anchored_elo_se(inputs)
    assert seeds["lookahead"] == 7  # registry base, untouched
    assert seeds["az0"] == 7 + 50_000  # disjoint net-opponent base


def test_run_arena_net_opponent_every_and_registry_seeds(monkeypatch: Any) -> None:
    # `every` schedules a net opponent exactly like opponent_every does a registry
    # one (skipped rounds contribute no metric and no Elo input), and adding net
    # opponents never shifts the registry opponents' seeds.
    reg_seeds: list[int] = []
    net_calls: list[int] = []

    def _fake_arena(*a: Any, opponent: str, seed: int, **k: Any) -> ArenaResult:
        reg_seeds.append(seed)
        return ArenaResult(wins=30.0, episodes=50)

    def _fake_spec_arena(*a: Any, opponent: Any, seed: int, **k: Any) -> ArenaResult:
        net_calls.append(seed)
        return ArenaResult(wins=24.0, episodes=40)

    monkeypatch.setattr("settlrl_learn.training.steps.arena", _fake_arena)
    monkeypatch.setattr("settlrl_learn.training.steps.arena_spec", _fake_spec_arena)
    cfg = ArenaConfig(
        games=40,
        opponents=["lookahead", "random"],
        anchor_elos={"lookahead": 0.0, "random": -1115.0},
    )
    backend = MLPBackend((16,))
    net_opponents = {"az0": (_dummy_spec(), -100.0, 3), "az1": (_dummy_spec(), 50.0, 1)}

    metrics = run_arena(
        backend, object(), cfg, seed=0, round_index=1, net_opponents=net_opponents
    )
    assert reg_seeds == [0, 10_000]
    assert net_calls == [50_000 + 10_000]  # az0 skipped (round 1 % 3), az1 played
    assert "arena_vs_az0" not in metrics
    assert metrics["arena_elo"] == anchored_elo(
        [(0.0, 30.0, 50), (-1115.0, 30.0, 50), (50.0, 24.0, 40)]
    )

    reg_seeds.clear()
    net_calls.clear()
    metrics = run_arena(
        backend, object(), cfg, seed=0, round_index=3, net_opponents=net_opponents
    )
    assert reg_seeds == [0, 10_000]  # unchanged by the extra opponents
    assert net_calls == [50_000, 50_000 + 10_000]
    assert metrics["arena_vs_az0"] == 0.6

    # ... and identical to the no-net-opponents path.
    reg_seeds.clear()
    net_calls.clear()
    base = run_arena(backend, object(), cfg, seed=0, round_index=3)
    assert reg_seeds == [0, 10_000] and net_calls == []
    assert base["arena_elo"] == anchored_elo([(0.0, 30.0, 50), (-1115.0, 30.0, 50)])


def test_self_play_no_pcr_marks_all_full() -> None:
    # Default (no fast_search): every position is a full-search position.
    backend = MLPBackend((16,))
    samples, _, _ = self_play(
        n_samples=8, batch_size=4, seed=0, **_jitted(_uniform_weights, backend)
    )
    assert np.all(samples["train_policy"] == 1.0)


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
