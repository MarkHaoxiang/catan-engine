"""The nets' :class:`~settlrl_search.policy.ValuePrior` seam: one forward per leaf.

The search needs a leaf's value *and* its expansion prior. XLA does not merge
two separate calls of a shared-trunk net on the same state (measured: two whole
trunks in the simulation loop), so the nets hand the search a single seam that
returns both. The structural test counts the trunk's node dots inside the
search's ``while`` body against a deliberately split seam -- the duplication
cannot come back silently.
"""

from __future__ import annotations

import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_learn.nn.board_gnn import BoardGNN, gnn_seams
from settlrl_learn.nn.graphnet import PRESETS
from settlrl_learn.nn.mlp import init_az_params, make_az
from settlrl_search import make_search_weights
from settlrl_search.policy import PolicyPrior

_W, _L = 8, 2  # a tiny trunk: the census counts ops, not FLOPs


def _position() -> tuple:
    env = BatchedSettlrlEnv(batch_size=1, seed=0, n_players=2, track_beliefs=True)
    env.rollout(jax.random.key(0), 60)
    return (
        jax.tree.map(lambda x: x[0], env.board[0]),
        jax.tree.map(lambda x: x[0], env.belief_view(0)),
        jnp.int32(0),
        env.flat_mask()[0],
    )


def _in_loop_node_dots(text: str) -> int:
    """Trunk node dots (the ``(54, width)`` per-vertex matmuls) emitted inside the
    search's simulation ``while`` body."""
    shape = rf"f32\[(?:54,{_W}|{_W},54)\]"
    return sum(
        "while" in m.group(1)
        for m in re.finditer(rf'{shape}\S* dot\(.*?op_name="([^"]+)"', text)
    )


def _search_text(*, split: bool) -> str:
    layout, view, player, mask = _position()
    net = BoardGNN(
        jax.random.key(0), PRESETS["gn_global"]._replace(width=_W, layers=_L)
    )
    params, static = eqx.partition(net, eqx.is_array)

    def fn(params, key, layout, view, player, mask):  # type: ignore[no-untyped-def]
        value, value_prior = gnn_seams(eqx.combine(params, static))
        # A plain closure hides `with_value`, so the search falls back to the two
        # separate forwards -- the shape this test exists to keep out.
        prior: PolicyPrior = (
            (lambda lay, st, pl: value_prior(lay, st, pl)) if split else value_prior
        )
        weights = make_search_weights(
            value, prior=prior, value_scale=2.0, num_simulations=2,
            max_depth=4, expected_rolls=False,
        )  # fmt: skip
        return weights(key, layout, view, player, mask)

    lowered = jax.jit(fn).lower(params, jax.random.key(1), layout, view, player, mask)
    text = lowered.compile().as_text()
    assert text is not None
    return text


def test_leaf_runs_one_trunk_per_simulation() -> None:
    shared = _in_loop_node_dots(_search_text(split=False))
    split = _in_loop_node_dots(_search_text(split=True))
    assert shared > 0
    assert split == 2 * shared, (
        f"expected the split seam to run two trunks per simulation and the shared "
        f"one, got {split} vs {shared} in-loop node dots"
    )


def test_gnn_with_value_matches_the_separate_seams() -> None:
    layout, _, player, _ = _position()
    env = BatchedSettlrlEnv(batch_size=1, seed=0, n_players=2)
    env.rollout(jax.random.key(0), 60)
    state = jax.tree.map(lambda x: x[0], env.board[1])
    net = BoardGNN(
        jax.random.key(0), PRESETS["gn_global"]._replace(width=_W, layers=_L)
    )
    value, prior = gnn_seams(net)
    v, logits = prior.with_value(layout, state, player)
    assert np.array_equal(v, value(layout, state, player))
    assert np.array_equal(logits, prior(layout, state, player))


def test_az_with_value_matches_the_separate_seams() -> None:
    layout, _, player, _ = _position()
    env = BatchedSettlrlEnv(batch_size=1, seed=0, n_players=2)
    env.rollout(jax.random.key(0), 60)
    state = jax.tree.map(lambda x: x[0], env.board[1])
    value, prior = make_az(init_az_params(jax.random.key(0), (8,)))
    v, logits = prior.with_value(layout, state, player)
    assert np.array_equal(v, value(layout, state, player))
    assert np.array_equal(logits, prior(layout, state, player))
