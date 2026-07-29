"""The self-play carry's padded (checkpointable) form: the lossless round trip,
what the pad must cover, and its ride through the eqx run-state file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _selfplay_stubs import jitted, uniform_weights_value
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_learn.training import MLPBackend, RunState
from settlrl_learn.training.backend import load_run_state, save_run_state
from settlrl_learn.training.carry import (
    PaddedEnv,
    SelfPlayCarry,
    empty_padded,
    from_padded,
    to_padded,
)
from settlrl_learn.training.selfplay import self_play


def _mid_game_carry(
    *, batch_size: int = 2, n_samples: int = 60, track_ordering: bool = False
) -> SelfPlayCarry:
    """A carry with games in flight (non-empty pending buffers) and a `q` key."""
    backend = MLPBackend((16,))
    _, _, carry = self_play(
        n_samples=n_samples, batch_size=batch_size, seed=1, temperature=1.0,
        persistent=True, record_value=True, track_ordering=track_ordering,
        **jitted(uniform_weights_value, backend),
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
    j = jitted(uniform_weights_value, backend)
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
