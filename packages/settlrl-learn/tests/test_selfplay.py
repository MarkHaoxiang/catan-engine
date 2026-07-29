"""Self-play data generation: what a call records, what it reports, and the
persistent carry's stream semantics -- all under a uniform-policy stand-in for
the search (see ``_selfplay_stubs``), so these stay seconds-fast.

Expect tests: the inline snapshot is the contract; regenerate with
``EXPECTTEST_ACCEPT=1 pytest``."""

from __future__ import annotations

import hashlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from _selfplay_stubs import (
    jitted,
    uniform_legal_dist,
    uniform_weights,
    uniform_weights_value,
)
from expecttest import assert_expected_inline
from jaxtyping import Array
from settlrl_engine.env import N_FLAT
from settlrl_learn.nn.graphnet import PRESETS
from settlrl_learn.training import GNNBackend, MLPBackend
from settlrl_learn.training.carry import from_padded, to_padded
from settlrl_learn.training.gnn_backend import _SETUP_ROWS
from settlrl_learn.training.selfplay import Samples, self_play


def test_self_play_samples_shape_under_uniform_policy() -> None:
    # Drives the real generic self-play (env stepping, pending flush, outcome
    # credit) with the MLP observation but a trivial policy -- fast, no search.
    backend = MLPBackend((16,))
    samples, _, _ = self_play(
        n_samples=8, batch_size=4, seed=0,
        **jitted(uniform_weights, backend),
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
        **jitted(uniform_weights, backend),
    )  # fmt: skip
    assert stats.env_steps > 0
    assert stats.recorded == samples["value"].shape[0]
    assert stats.discarded >= 0


def test_self_play_records_root_value_when_asked() -> None:
    backend = MLPBackend((16,))
    samples, _, _ = self_play(
        n_samples=8, batch_size=4, seed=0, record_value=True,
        **jitted(uniform_weights_value, backend),
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
        **jitted(uniform_weights_value, backend),
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
        **jitted(uniform_weights_value, backend),
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
    j = jitted(uniform_legal_dist, backend)
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
    j = jitted(uniform_legal_dist, backend)
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
    j = jitted(uniform_legal_dist, backend)
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
    j = jitted(uniform_legal_dist, backend)
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
        **jitted(uniform_weights, backend),
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
        **jitted(uniform_legal_dist, backend),
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
        **jitted(uniform_weights, backend),
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
# Playout-cap randomization (PCR)                                              #
# --------------------------------------------------------------------------- #


def test_self_play_pcr_marks_fast_positions() -> None:
    # With a fast_search + full_prob < 1, each step is full (train_policy 1) or
    # fast (0); the data side of PCR. value is recorded for both (fast positions
    # still train the value head).
    backend = MLPBackend((16,))
    j = jitted(uniform_legal_dist, backend)
    samples, _, _ = self_play(
        n_samples=64, batch_size=8, seed=1,
        fast_search=j["search"], full_prob=0.5, **j,
    )  # fmt: skip
    tp = samples["train_policy"]
    assert set(np.unique(tp)).issubset({0.0, 1.0})
    assert tp.min() == 0.0 and tp.max() == 1.0  # both full and fast steps occurred
    assert tp.shape == samples["value"].shape  # a flag per recorded position


def test_self_play_no_pcr_marks_all_full() -> None:
    # Default (no fast_search): every position is a full-search position.
    backend = MLPBackend((16,))
    samples, _, _ = self_play(
        n_samples=8, batch_size=4, seed=0, **jitted(uniform_weights, backend)
    )
    assert np.all(samples["train_policy"] == 1.0)
