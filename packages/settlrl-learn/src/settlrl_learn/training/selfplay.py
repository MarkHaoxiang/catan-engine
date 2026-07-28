"""Net-agnostic self-play data generation for the training loop.

Drives batched n-player self-play with a pre-built jitted+vmapped ``search``
callable (the net's, or a fixed teacher's) and records, per acting move, the
backend's observation of the *true* board, the search's improved-policy target,
the legality mask, and the eventual game outcome. Features are on the true board
(no hidden state in the net's inputs), so the net learns the belief-averaged
value; determinization stays inside the search.

The jitted callables (``search``, ``observe_of``, ``view_of``, ``setup_search``)
are built once by the caller and passed in -- the net's array params are threaded
into ``search`` as a traced argument (the caller closes them over via
``equinox.partition``/``combine``), so a weight update is a new *value* of a
same-shaped input and the search is compiled once and reused across iterations.

``setup_search`` (when given) plays the **setup phase** (initial placements) with
a fixed policy instead of the net, and those positions are *not recorded* -- so
the net's value/policy only ever train on (and act in) the main game loop. The
setup placements are rare, high-leverage, and structurally distinct; handing them
to a strong fixed policy keeps a weak net's bad opening from dooming every game.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int, UInt32
from settlrl_engine.belief import BeliefState, BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, GamePhase
from settlrl_engine.env import BatchedSettlrlEnv, flat_to_action
from settlrl_engine.env.batched import (
    AgentSelectionArray,
    DoneArray,
    RewardArray,
    VPArray,
)
from settlrl_engine.mechanics.common import ResultCode
from settlrl_engine.mechanics.flat import FlatMaskArray
from settlrl_search import PolicyWeights, PolicyWeightsValue
from settlrl_search.policy import BeliefPolicy

ObserveFn = Callable[[BoardLayout, BoardState, Array], dict[str, Array]]
Samples = dict[str, np.ndarray]
"""A batch of training positions: the backend's observation keys plus ``policy``,
``mask``, and ``value``, each stacked on a leading sample axis."""


class SelfPlayStats(NamedTuple):
    """What a :func:`self_play` call did.

    ``env_steps`` counts batched env steps (each advances all ``batch_size``
    lanes); ``recorded`` is the returned sample count; ``discarded`` is the
    positions generated but never returned -- the pending positions of games
    still unfinished when the call exited (the iteration-boundary waste), plus
    any trimmed by ``max_game_len``. Under ``persistent`` the unfinished games
    survive in the carry, so ``discarded`` counts only the trims.
    """

    env_steps: int
    recorded: int
    discarded: int


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


_DERIVED_KEYS = ("policy", "mask", "train_policy", "q")
"""The recorded keys that are not part of the backend's observation."""


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


def _make_env(
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
    env = _make_env(
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
    env = _make_env(
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
    obs_keys = [k for k in carry.spec if k not in _DERIVED_KEYS]
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
    obs_keys = [k for k in spec if k not in _DERIVED_KEYS]
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


def _sample_moves(key: Array, weights: Array, mask: Array, temperature: float) -> Array:
    """One legal move per lane from the masked improved policy (argmax at
    ``temperature`` 0, else a tempered categorical draw)."""
    if temperature <= 0.0:
        return jnp.argmax(jnp.where(mask, weights, -jnp.inf), axis=-1)
    logits = jnp.where(mask, jnp.log(jnp.clip(weights, 1e-8)) / temperature, -jnp.inf)
    return jax.random.categorical(key, logits, axis=-1)


def self_play(
    search: PolicyWeights | PolicyWeightsValue,
    *,
    observe_of: ObserveFn,
    view_of: Callable[[BoardState, BeliefState, Array], BeliefView],
    setup_search: BeliefPolicy | None = None,
    fast_search: PolicyWeights | PolicyWeightsValue | None = None,
    full_prob: float = 1.0,
    n_samples: int,
    n_players: int = 2,
    batch_size: int = 16,
    temperature: float = 1.0,
    seed: int = 0,
    max_steps: int = 100_000,
    max_game_len: int = 800,
    record_value: bool = False,
    track_ordering: bool = False,
    persistent: bool = False,
    carry: SelfPlayCarry | None = None,
) -> tuple[Samples, SelfPlayStats, SelfPlayCarry | None]:
    """Collect >= ``n_samples`` self-play positions, the moves and policy targets
    drawn from ``search``. Positions from finished games are credited with the
    acting seat's win (1) / loss (0); unfinished games are discarded (counted in
    the returned :class:`SelfPlayStats`).

    ``search``, ``observe_of``, ``view_of`` and ``setup_search`` are pre-built
    jitted+vmapped callables (see the module docstring): the search is compiled
    once by the caller and reused, with the net's params threaded in as a traced
    argument so a weight update does not recompile.

    ``record_value`` expects ``search`` to also return the searched root value
    (a :data:`~settlrl_search.PolicyWeightsValue`) and stores it under the
    ``q`` key (searcher frame, [-1, 1]) -- the value-blend target's ``q`` term.

    Playout-cap randomization (when ``fast_search`` is given and ``full_prob`` <
    1): each step is *full* (the deep ``search``) with probability ``full_prob``,
    else *fast* (the cheap ``fast_search``). Every position records its outcome
    value, but the ``train_policy`` flag is 1 only on full-search positions (0 on
    fast) so the policy loss trains on deep targets only. ``full_prob`` = 1
    disables it (every position ``train_policy`` = 1).

    ``max_steps`` caps the env-step budget and ``max_game_len`` each lane's
    retained pending positions -- a cold/degenerate net can drag a game out
    indefinitely, so without these the pending buffer grows unbounded. A capped
    lane keeps its most recent positions.

    ``persistent`` returns the live :class:`SelfPlayCarry` instead of dropping
    the games still in flight; passing it back as ``carry`` resumes them, so a
    sequence of persistent calls of ``n_samples`` each yields exactly what one
    call of their total would (same env stepping, same RNG stream, same flush
    order -- the surplus of a call that overshot is credited to the next), as
    long as no call exhausts its own ``max_steps`` (a per-call budget).
    ``seed`` then seeds only the *first* call: the RNG lives in the carry, so a
    persistent call's output is a function of (``seed``, carried state) rather
    than of ``seed`` alone."""
    pcr = fast_search is not None and full_prob < 1.0
    out: dict[str, list[np.ndarray]] = {}
    vals: list[float] = []
    env_steps = 0
    trimmed = 0
    if carry is None:
        env = _make_env(
            batch_size=batch_size, seed=seed,
            n_players=n_players, track_ordering=track_ordering,
        )  # fmt: skip
        pending: list[list[PendingRow]] = [[] for _ in range(batch_size)]
        spec: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        key = jax.random.key(seed)
        surplus = 0
    else:
        assert len(carry.pending) == batch_size, "carry was built at another batch size"
        env, pending, key, surplus = carry.env, carry.pending, carry.key, carry.surplus
        spec = carry.spec
        out = {k: [] for k in spec}

    for _step in range(max_steps):
        if surplus + len(vals) >= n_samples:
            break
        layout, state = env.board
        beliefs = env.beliefs
        assert beliefs is not None  # track_beliefs=True
        sel = jnp.asarray(env.agent_selection)
        mask = env.flat_mask()
        view = view_of(state, beliefs, sel)
        # Playout-cap: pick full (deep, policy-training) vs fast (cheap, value-only)
        # for this step. The extra key split is taken only when PCR is on, so the
        # off path's RNG stream is unchanged (bit-exact resume).
        if pcr:
            key, k_pcr = jax.random.split(key)
            full = bool(jax.random.uniform(k_pcr) < full_prob)
        else:
            full = True
        step_search = search if full else fast_search
        assert step_search is not None
        key, k_search, k_move, k_setup = jax.random.split(key, 4)
        result = step_search(
            jax.random.split(k_search, batch_size), layout, view, sel, mask
        )
        q_np = np.zeros(batch_size, np.float32)  # overwritten when recording value
        if record_value:
            weights, qv = result
            q_np = np.asarray(qv)
        else:
            weights = result
        move = _sample_moves(k_move, weights, mask, temperature)
        # Setup-phase lanes play (unrecorded) via the fixed setup policy.
        is_setup = (
            np.asarray(state.phase <= int(GamePhase.SETUP_ROAD))
            if setup_search is not None
            else np.zeros(batch_size, bool)
        )
        if setup_search is not None and is_setup.any():
            setup_move = setup_search(
                jax.random.split(k_setup, batch_size), layout, view, sel, mask
            )
            move = jnp.where(jnp.asarray(is_setup), setup_move, move)

        obs = {k: np.asarray(v) for k, v in observe_of(layout, state, sel).items()}
        w_np, sel_np, m_np = np.asarray(weights), np.asarray(sel), np.asarray(mask)
        if not spec:  # capture the per-key trailing shape+dtype once (empty case)
            f32 = np.dtype(np.float32)
            spec = {k: (v.shape[1:], v.dtype) for k, v in obs.items()}
            spec["policy"] = (w_np.shape[1:], w_np.dtype)
            spec["mask"] = (m_np.shape[1:], m_np.dtype)
            spec["train_policy"] = ((), f32)
            if record_value:
                spec["q"] = ((), f32)
            out = {k: [] for k in (*spec,)}
        tp_val = 1.0 if full else 0.0  # 1 = full-search position (trains policy)
        for lane in range(batch_size):
            if is_setup[lane]:  # the net does not train on setup positions
                continue
            row = (
                {k: obs[k][lane] for k in obs},
                w_np[lane],
                m_np[lane],
                int(sel_np[lane]),
                float(q_np[lane]),
                tp_val,
            )
            pending[lane].append(row)
            if len(pending[lane]) > max_game_len:
                trimmed += len(pending[lane]) - max_game_len
                del pending[lane][:-max_game_len]

        env.step(*flat_to_action(move))
        env_steps += 1
        rewards = np.asarray(env.rewards)
        for lane in np.flatnonzero(np.asarray(env.terminations).any(axis=1)).tolist():
            for obs_l, pol_l, mask_l, seat, q_l, tp_l in pending[lane]:
                for k, v in obs_l.items():
                    out[k].append(v)
                out["policy"].append(pol_l)
                out["mask"].append(mask_l)
                out["train_policy"].append(np.asarray(tp_l, np.float32))
                if record_value:
                    out["q"].append(np.asarray(q_l, np.float32))
                vals.append(float(rewards[lane, seat] > 0))
            pending[lane] = []

    stats = SelfPlayStats(
        env_steps=env_steps,
        recorded=len(vals),
        # In-flight games survive in the carry, so only trims are lost there.
        discarded=trimmed if persistent else trimmed + sum(len(p) for p in pending),
    )
    new_carry = (
        SelfPlayCarry(env, pending, key, surplus + len(vals) - n_samples, spec)
        if persistent
        else None
    )
    if not vals:  # no game finished within the budget, or a zero-step resume
        empty: Samples = {k: np.zeros((0, *s), dt) for k, (s, dt) in spec.items()}
        empty["value"] = np.zeros((0,), np.float32)
        return empty, stats, new_carry
    samples: Samples = {k: np.stack(out[k]) for k in out}
    samples["value"] = np.asarray(vals, np.float32)
    return samples, stats, new_carry
