"""The live self-play pool and its fixed-shape checkpointable form.

:class:`SelfPlayCarry` is what :func:`~settlrl_learn.training.selfplay.self_play`
hands back under ``persistent`` and takes back to resume the games in flight; it
holds the env *object*, so it cannot be serialised as a pytree.
:class:`PaddedCarry` is the eqx-serialisable projection of it that rides in a
:class:`~settlrl_learn.training.backend.RunState` -- host-numpy pads of fixed
shape, PRNG keys as raw uint32 -- with ``to_padded``/``from_padded`` the lossless
pair between them and ``empty_padded``/``carry_template`` the zero template a
deserialisation needs.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int, UInt32
from settlrl_engine.belief import BeliefState
from settlrl_engine.board import make_board
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.env import N_FLAT, BatchedSettlrlEnv
from settlrl_engine.env.batched import (
    AgentSelectionArray,
    DoneArray,
    RewardArray,
    VPArray,
)
from settlrl_engine.mechanics.common import ResultCode
from settlrl_engine.mechanics.flat import FlatMaskArray

from settlrl_learn.training.backend import Backend
from settlrl_learn.training.config import LearnConfig

PendingRow = tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int, float, float]
"""One recorded, not-yet-credited position: (observation, policy target, legality
mask, acting seat, root value, ``train_policy`` flag)."""


class SelfPlayCarry(NamedTuple):
    """A live self-play pool between :func:`self_play` calls: the games in
    flight, their uncredited positions, and the RNG stream to continue.

    ``surplus`` is how many samples past the cumulative request the pool has
    already handed out (negative when a call ended short of it), so a resumed
    call stops where one long call would have. ``spec`` is the per-key trailing
    shape and dtype, carried so a call that records nothing still returns the
    same keys with the same dtypes.
    """

    env: BatchedSettlrlEnv
    pending: list[list[PendingRow]]
    key: Array
    surplus: int
    spec: dict[str, tuple[tuple[int, ...], np.dtype[Any]]]


DERIVED_KEYS = ("policy", "mask", "train_policy", "q")
"""The recorded keys that are not part of the backend's observation."""


def recorded_spec(
    obs_spec: dict[str, tuple[tuple[int, ...], np.dtype[Any]]],
    *,
    n_flat: int,
    record_value: bool,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """``obs_spec`` (the backend's per-key trailing shape+dtype) plus the
    derived keys self-play records over it -- the single source of truth for
    :data:`DERIVED_KEYS`, so a call-site's spec and the padding code's notion
    of "derived" cannot drift apart. ``q`` is included only when
    ``record_value``."""
    spec = {**obs_spec}
    f32 = np.dtype(np.float32)
    spec["policy"] = ((n_flat,), f32)
    spec["mask"] = ((n_flat,), np.dtype(np.bool_))
    spec["train_policy"] = ((), f32)
    if record_value:
        spec["q"] = ((), f32)
    assert set(spec) - set(obs_spec) <= set(DERIVED_KEYS)
    return spec


class PaddedEnv(NamedTuple):
    """:class:`BatchedSettlrlEnv`'s complete array state, PRNG keys as raw uint32
    (eqx cannot serialise typed key arrays).

    ``state`` is a :class:`BoardState` whose ``key`` field holds raw key data;
    ``category`` is zeros when the env does not track action ordering.
    """

    layout: BoardLayout
    state: Any  # BoardState, `key` as raw uint32
    reward: RewardArray
    terminations: DoneArray
    truncations: DoneArray
    result: ResultCode
    vps: VPArray
    avail: FlatMaskArray
    agent_sel: AgentSelectionArray
    belief: BeliefState
    category: Int[Array, " batch"]
    key: UInt32[Array, " key"]


class PaddedCarry(NamedTuple):
    """A :class:`SelfPlayCarry` as a fixed-shape pytree, so it can ride in an
    eqx-serialised :class:`~settlrl_learn.training.backend.RunState`.

    Each recorded key of ``pending`` is host numpy padded to
    ``(batch, max_game_len, *trailing)``, live rows per lane in ``pending_len``;
    ``seat`` joins the recorded keys. ``present`` is 0 for the zero template a
    run with no live pool checkpoints, so "no pool was stored" and "an empty
    pool" stay distinguishable.
    """

    env: PaddedEnv
    pending: dict[str, np.ndarray]
    pending_len: Int[np.ndarray, " batch"]
    key: UInt32[np.ndarray, " key"]
    surplus: Int[np.ndarray, ""]
    track_ordering: Int[np.ndarray, ""]
    present: Int[np.ndarray, ""]


def _raw_keys(tree: Any) -> Any:
    return jax.tree.map(
        lambda x: (
            jax.random.key_data(x)
            if jnp.issubdtype(x.dtype, jax.dtypes.prng_key)
            else x
        ),
        tree,
    )


def _typed_keys(template: Any, tree: Any) -> Any:
    return jax.tree.map(
        lambda t, x: (
            jax.random.wrap_key_data(jnp.asarray(x))
            if jnp.issubdtype(t.dtype, jax.dtypes.prng_key)
            else jnp.asarray(x)
        ),
        template,
        tree,
    )


def make_env(
    *, batch_size: int, seed: int, n_players: int, track_ordering: bool
) -> BatchedSettlrlEnv:
    """Self-play's env. The single construction site: a carried pool is only
    restorable if the env it is rebuilt into is built exactly like the original."""
    return BatchedSettlrlEnv(
        batch_size=batch_size, seed=seed, reward="sparse", n_players=n_players,
        track_beliefs=True, track_ordering=track_ordering,
    )  # fmt: skip


def _env_arrays(env: BatchedSettlrlEnv) -> PaddedEnv:
    return PaddedEnv(
        layout=env._layout,
        state=_raw_keys(env._state),
        reward=env._reward,
        terminations=env._terminations,
        truncations=env._truncations,
        result=env._result,
        vps=env._vps,
        avail=env._avail,
        agent_sel=env._agent_sel,
        belief=env.beliefs,
        category=(
            env._category
            if env._category is not None
            else jnp.zeros((env.batch_size,), jnp.int32)
        ),
        key=jax.random.key_data(env._key),
    )


def _restore_env(padded: PaddedEnv, *, track_ordering: bool) -> BatchedSettlrlEnv:
    """A live env equivalent to the one ``padded`` was taken from."""
    batch_size, n_players = padded.reward.shape
    env = make_env(
        batch_size=batch_size, seed=0, n_players=n_players,
        track_ordering=track_ordering,
    )  # fmt: skip
    env._layout = jax.tree.map(jnp.asarray, padded.layout)
    env._state = _typed_keys(env._state, padded.state)
    env._reward = jnp.asarray(padded.reward)
    env._terminations = jnp.asarray(padded.terminations)
    env._truncations = jnp.asarray(padded.truncations)
    env._result = jnp.asarray(padded.result)
    env._vps = jnp.asarray(padded.vps)
    env._avail = jnp.asarray(padded.avail)
    env._agent_sel = jnp.asarray(padded.agent_sel)
    env._belief = jax.tree.map(jnp.asarray, padded.belief)
    env._category = jnp.asarray(padded.category) if track_ordering else None
    env._key = jax.random.wrap_key_data(jnp.asarray(padded.key))
    return env


def _empty_pending(
    batch_size: int,
    max_game_len: int,
    spec: dict[str, tuple[tuple[int, ...], np.dtype[Any]]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    pending = {
        k: np.zeros((batch_size, max_game_len, *shape), dt)
        for k, (shape, dt) in spec.items()
    }
    pending["seat"] = np.zeros((batch_size, max_game_len), np.int32)
    return pending, np.zeros((batch_size,), np.int32)


def empty_padded(
    *,
    batch_size: int,
    n_players: int,
    track_ordering: bool,
    max_game_len: int,
    spec: dict[str, tuple[tuple[int, ...], np.dtype[Any]]],
) -> PaddedCarry:
    """The zero :class:`PaddedCarry` ``spec`` implies (``present`` = 0): the eqx
    deserialisation template, and what a run with no live pool checkpoints."""
    env = make_env(
        batch_size=batch_size, seed=0, n_players=n_players,
        track_ordering=track_ordering,
    )  # fmt: skip
    pending, pending_len = _empty_pending(batch_size, max_game_len, spec)
    return PaddedCarry(
        env=_env_arrays(env),
        pending=pending,
        pending_len=pending_len,
        key=np.asarray(jax.random.key_data(jax.random.key(0))),
        surplus=np.zeros((), np.int32),
        track_ordering=np.asarray(int(track_ordering), np.int32),
        present=np.zeros((), np.int32),
    )


def to_padded(carry: SelfPlayCarry, max_game_len: int) -> PaddedCarry:
    """``carry`` in its fixed-shape checkpointable form. ``max_game_len`` must be
    the one self-play trims at, so no lane can overflow the pad."""
    assert carry.spec, "an unplayed carry has no recorded-field spec to pad"
    obs_keys = [k for k in carry.spec if k not in DERIVED_KEYS]
    pend, lens = _empty_pending(len(carry.pending), max_game_len, carry.spec)
    for lane, rows in enumerate(carry.pending):
        n = len(rows)
        assert n <= max_game_len, f"lane {lane} holds {n} rows past the pad"
        lens[lane] = n
        if n == 0:
            continue
        for k in obs_keys:
            pend[k][lane, :n] = np.stack([r[0][k] for r in rows])
        pend["policy"][lane, :n] = np.stack([r[1] for r in rows])
        pend["mask"][lane, :n] = np.stack([r[2] for r in rows])
        pend["seat"][lane, :n] = [r[3] for r in rows]
        if "q" in pend:
            pend["q"][lane, :n] = [r[4] for r in rows]
        pend["train_policy"][lane, :n] = [r[5] for r in rows]
    return PaddedCarry(
        env=_env_arrays(carry.env),
        pending=pend,
        pending_len=lens,
        key=np.asarray(jax.random.key_data(carry.key)),
        surplus=np.asarray(carry.surplus, np.int32),
        track_ordering=np.asarray(int(carry.env.track_ordering), np.int32),
        present=np.ones((), np.int32),
    )


def from_padded(padded: PaddedCarry, *, track_ordering: bool) -> SelfPlayCarry:
    """The live carry ``padded`` holds, ready to pass back to :func:`self_play`.

    Raises ``ValueError`` when ``track_ordering`` differs from the run that wrote
    it -- the one self-play semantic the padded shapes do not pin.
    """
    if bool(padded.track_ordering) != track_ordering:
        raise ValueError(
            f"checkpointed self-play carry was recorded with "
            f"track_ordering={bool(padded.track_ordering)}, but this run asks for "
            f"{track_ordering}; resume with the original setting (or turn "
            "selfplay.persistent off to start a fresh pool)"
        )
    pend = padded.pending
    spec = {k: (v.shape[2:], v.dtype) for k, v in pend.items() if k != "seat"}
    obs_keys = [k for k in spec if k not in DERIVED_KEYS]
    pending: list[list[PendingRow]] = []
    for lane in range(padded.pending_len.shape[0]):
        pending.append(
            [
                (
                    {k: pend[k][lane, t].copy() for k in obs_keys},
                    pend["policy"][lane, t].copy(),
                    pend["mask"][lane, t].copy(),
                    int(pend["seat"][lane, t]),
                    float(pend["q"][lane, t]) if "q" in pend else 0.0,
                    float(pend["train_policy"][lane, t]),
                )
                for t in range(int(padded.pending_len[lane]))
            ]
        )
    return SelfPlayCarry(
        env=_restore_env(padded.env, track_ordering=track_ordering),
        pending=pending,
        key=jax.random.wrap_key_data(jnp.asarray(padded.key)),
        surplus=int(padded.surplus),
        spec=spec,
    )


_SELFPLAY_N_PLAYERS = 2
"""``run_selfplay`` never overrides :func:`self_play`'s seat count."""


def carry_template(backend: Backend, cfg: LearnConfig) -> PaddedCarry:
    """The zero :class:`PaddedCarry` a run of ``cfg`` checkpoints: the eqx
    template resume deserialises into, and what stands in without a live pool.
    A non-persistent run pads to zero rows."""
    layout, state = make_board(batch_size=1, seed=0, n_players=_SELFPLAY_N_PLAYERS)
    one = jax.tree.map(lambda x: x[0], (layout, state))
    obs = jax.eval_shape(backend.observe, one[0], one[1], jnp.int32(0))
    obs_spec = {k: (tuple(v.shape), np.dtype(v.dtype)) for k, v in obs.items()}
    spec = recorded_spec(obs_spec, n_flat=N_FLAT, record_value=cfg.value_blend.max > 0)
    return empty_padded(
        batch_size=cfg.selfplay.batch,
        n_players=_SELFPLAY_N_PLAYERS,
        track_ordering=cfg.search.ordered,
        max_game_len=cfg.selfplay.max_game_len if cfg.selfplay.persistent else 0,
        spec=spec,
    )
