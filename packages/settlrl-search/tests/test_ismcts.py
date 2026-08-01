"""Behaviour contracts for the SO-ISMCTS search (``search.ismcts`` driven by
``search.make_search``).

Correctness invariants only, at tiny budgets: the move is always legal, the
returned weights are a distribution supported on the legal set, the search is
reproducible from its key, and a self-played game reaches a terminal. The
per-determinization legality property -- the search only ever returns an action
legal in the true position -- is what the legality/support tests pin (an illegal
return would mean the descent leaked an action illegal under the real board).

Strength is settled by matches, not unit tests.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _heuristic_leaf import heuristic_value
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, KeyScalar, Player
from settlrl_engine.env import N_FLAT, BatchedSettlrlEnv, flat_to_action
from settlrl_search import make_search, make_search_weights
from settlrl_search.policy import ValuePrior
from settlrl_search.value import Value


@functools.cache
def _policy(num_simulations: int) -> Any:
    return jax.jit(make_search(heuristic_value, num_simulations=num_simulations))


@functools.cache
def _weights_fn(num_simulations: int) -> Any:
    return jax.jit(
        make_search_weights(heuristic_value, num_simulations=num_simulations)
    )


def _move(
    key: KeyScalar,
    layout: BoardLayout,
    view: BeliefView,
    p: int,
    mask: np.ndarray,
    num_simulations: int,
) -> int:
    return int(
        _policy(num_simulations)(key, layout, view, jnp.int32(p), jnp.asarray(mask))
    )


def _weights(
    key: KeyScalar,
    layout: BoardLayout,
    view: BeliefView,
    p: int,
    mask: np.ndarray,
    num_simulations: int,
) -> np.ndarray:
    w = _weights_fn(num_simulations)(key, layout, view, jnp.int32(p), jnp.asarray(mask))
    return np.asarray(w)


def _position(seed: int, steps: int, n_players: int = 2) -> tuple:
    """A single-game mid-game position with the acting seat's belief view."""
    env = BatchedSettlrlEnv(
        batch_size=1, seed=seed, n_players=n_players, track_beliefs=True
    )
    env.rollout(jax.random.key(seed), steps)
    layout = jax.tree.map(lambda x: x[0], env.board[0])
    p = int(env.agent_selection[0])
    view: BeliefView = jax.tree.map(lambda x: x[0], env.belief_view(p))
    mask = np.asarray(env.flat_mask()[0])
    return layout, view, p, mask


@pytest.mark.parametrize("seed", [0, 1])
def test_move_is_legal(seed: int) -> None:
    layout, view, p, mask = _position(seed, steps=100 + seed * 20)
    if mask.sum() == 0:
        pytest.skip("no legal move (stalled lane)")
    a = _move(jax.random.key(seed), layout, view, p, mask, num_simulations=12)
    assert mask[a] > 0


def test_weights_are_a_legal_distribution() -> None:
    layout, view, p, mask = _position(7, steps=130)
    w = _weights(jax.random.key(1), layout, view, p, mask, num_simulations=16)
    assert np.all(w >= 0.0)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert float(w[mask == 0].sum()) == 0.0  # support is exactly the legal set


def test_reproducible_from_key() -> None:
    layout, view, p, mask = _position(3, steps=110)
    a1 = _move(jax.random.key(9), layout, view, p, mask, num_simulations=12)
    a2 = _move(jax.random.key(9), layout, view, p, mask, num_simulations=12)
    assert a1 == a2


@functools.cache
def _chance_fn(num_simulations: int) -> Any:
    """A search with explicit dice + dev-card chance nodes."""
    from settlrl_search import make_search_weights_value

    return jax.jit(
        make_search_weights_value(
            heuristic_value,
            num_simulations=num_simulations,
            chance_nodes=True,
            dev_chance=True,
        )
    )


@pytest.mark.parametrize("seed", [0, 2])
def test_chance_nodes_weights_are_a_legal_distribution(seed: int) -> None:
    # The explicit-chance-node descent (dice + dev draws resolved in-tree) still
    # returns a legal improved-policy distribution and a finite searched root value
    # -- the contract that the decision/chance state machine never leaks an illegal
    # action or diverges.
    layout, view, p, mask = _position(seed, steps=120 + seed * 10)
    if mask.sum() == 0:
        pytest.skip("no legal move (stalled lane)")
    w, q = _chance_fn(16)(
        jax.random.key(seed), layout, view, jnp.int32(p), jnp.asarray(mask)
    )
    w = np.asarray(w)
    assert np.all(w >= 0.0) and abs(float(w.sum()) - 1.0) < 1e-6
    assert float(w[mask == 0].sum()) == 0.0  # support is exactly the legal set
    assert bool(np.isfinite(q)) and -1.0 <= float(q) <= 1.0  # searched root value


def test_chance_nodes_reproducible_from_key() -> None:
    layout, view, p, mask = _position(2, steps=140)
    args = (jax.random.key(4), layout, view, jnp.int32(p), jnp.asarray(mask))
    w1, q1 = _chance_fn(16)(*args)
    w2, q2 = _chance_fn(16)(*args)
    assert np.array_equal(np.asarray(w1), np.asarray(w2)) and float(q1) == float(q2)


@functools.cache
def _ordered_fn(num_simulations: int) -> Any:
    """A search with the action-ordering lock-out applied in-tree."""
    return jax.jit(
        make_search_weights(
            heuristic_value, num_simulations=num_simulations, ordered=True
        )
    )


@pytest.mark.parametrize("seed", [0, 3])
def test_ordered_weights_are_a_legal_distribution(seed: int) -> None:
    # The ordering lock-out applied in the descent still yields a legal
    # improved-policy distribution over the (env-supplied) root mask -- the
    # contract that threading `category` never leaks an illegal action.
    layout, view, p, mask = _position(seed, steps=120 + seed * 10)
    if mask.sum() == 0:
        pytest.skip("no legal move (stalled lane)")
    w = np.asarray(
        _ordered_fn(16)(
            jax.random.key(seed), layout, view, jnp.int32(p), jnp.asarray(mask)
        )
    )
    assert np.all(w >= 0.0) and abs(float(w.sum()) - 1.0) < 1e-6
    assert float(w[mask == 0].sum()) == 0.0  # support is exactly the legal set


@dataclasses.dataclass(frozen=True)
class _FusedPrior:
    """A :class:`ValuePrior`: uniform logits plus a value head that is ``gain``
    times the contract leaf — so a fused leaf (the prior's own value) and an
    unfused one (the ``value`` argument) agree exactly iff ``gain == 1``."""

    gain: float = 1.0

    def __call__(self, layout: BoardLayout, state: BoardState, player: Player) -> Any:
        return jnp.zeros((N_FLAT,), jnp.float32)

    def with_value(
        self, layout: BoardLayout, state: BoardState, player: Player
    ) -> tuple[Value, Any]:
        v: Value = self.gain * heuristic_value(layout, state, player)
        return v, self(layout, state, player)


def _prior_weights(prior: Any, *, fused_leaf: bool) -> np.ndarray:
    """Weights from a search whose interior prior is ``prior``."""
    layout, view, p, mask = _position(1, steps=120)
    fn = make_search_weights(
        heuristic_value, prior=prior, num_simulations=12, fused_leaf=fused_leaf
    )
    w = jax.jit(fn)(jax.random.key(5), layout, view, jnp.int32(p), jnp.asarray(mask))
    return np.asarray(w)


def test_value_prior_serves_the_leaf_only_under_fused_leaf() -> None:
    # `isinstance(prior, ValuePrior)` is what selects the one-forward leaf path,
    # so a prior whose value head *disagrees* with `value` shows which one ran.
    off_gain = _FusedPrior(gain=0.5)
    two_seam = _prior_weights(off_gain, fused_leaf=False)
    assert isinstance(off_gain, ValuePrior)  # the protocol is structural
    assert not np.array_equal(_prior_weights(off_gain, fused_leaf=True), two_seam)
    # Paired seams: taking both off one call changes nothing but the op count.
    paired = _FusedPrior(gain=1.0)
    assert np.array_equal(
        _prior_weights(paired, fused_leaf=True),
        _prior_weights(paired, fused_leaf=False),
    )


def test_visits_concentrate_above_uniform() -> None:
    # A healthy search is neither degenerate (all mass on one action) nor a
    # round-robin: the top action takes clearly more than a uniform share of the
    # visits, while more than one action is explored.
    layout, view, p, mask = _position(5, steps=120)
    n_legal = int(mask.sum())
    if n_legal < 4:
        pytest.skip("trivial decision")
    w = _weights(jax.random.key(2), layout, view, p, mask, num_simulations=48)
    assert float(w.max()) > 1.5 / n_legal  # concentrates above uniform
    assert int((w > 0).sum()) > 1  # but explores more than one action


@pytest.mark.parametrize("steps", [2, 40, 230])  # setup phase ... late game
def test_move_legal_across_game_stages(steps: int) -> None:
    # Edge cases: the setup phase (settle/road action types) and a near-end
    # position exercise different legal sets than the mid-game.
    layout, view, p, mask = _position(11, steps)
    if mask.sum() == 0:
        pytest.skip("no legal move (stalled lane)")
    a = _move(jax.random.key(11), layout, view, p, mask, num_simulations=12)
    assert mask[a] > 0


def test_four_player_move_legal() -> None:
    # The paranoid frame at 4 players (searcher vs three): still a legal move.
    layout, view, p, mask = _position(2, steps=150, n_players=4)
    if mask.sum() == 0:
        pytest.skip("no legal move")
    a = _move(jax.random.key(2), layout, view, p, mask, num_simulations=16)
    assert mask[a] > 0


def test_no_legal_actions_does_not_crash() -> None:
    # Degenerate input (empty mask): no crash, the move is the documented
    # arbitrary index (the engine rejects it).
    layout, view, p, mask = _position(7, steps=120)
    empty = np.zeros_like(mask)
    a = _move(jax.random.key(0), layout, view, p, empty, num_simulations=8)
    # `-> int` is already enforced (mypy + the beartype hook); the real property
    # is that the degenerate fallback is still an in-range action index.
    assert 0 <= a < empty.shape[-1]


@pytest.mark.slow
def test_self_play_completes_a_game() -> None:
    env = BatchedSettlrlEnv(batch_size=1, seed=4, n_players=2, track_beliefs=True)
    key = jax.random.key(0)
    for _ in range(400):
        if bool(env.terminations[0].any()):
            break
        layout = jax.tree.map(lambda x: x[0], env.board[0])
        p = int(env.agent_selection[0])
        view = jax.tree.map(lambda x: x[0], env.belief_view(p))
        mask = np.asarray(env.flat_mask()[0])
        if mask.sum() == 0:
            break
        key, k = jax.random.split(key)
        mv = _move(k, layout, view, p, mask, num_simulations=8)
        assert mask[mv] > 0
        env.step(*flat_to_action(jnp.asarray([mv])))
    assert bool(env.terminations[0].any())
