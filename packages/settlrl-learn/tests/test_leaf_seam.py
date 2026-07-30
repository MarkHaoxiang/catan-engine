"""The nets' :class:`~settlrl_search.policy.ValuePrior` seam: one forward per leaf.

The search needs a leaf's value *and* its expansion prior. XLA does not merge
two separate calls of a shared-trunk net on the same state (measured: two whole
trunks in the simulation loop), so the nets hand the search a single seam that
returns both. The structural test counts the trunk's node dots inside the
search's ``while`` body against a deliberately split seam -- the duplication
cannot come back silently -- and pins that the two emit the same weights.

Fusing makes the *prior's* value head score the leaf, which is only what a
caller wants when the two seams are paired; ``fused_leaf=False`` is the escape
for an unpaired pairing (a heuristic value under a net's policy head), and is
covered here too.
"""

from __future__ import annotations

import re
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from settlrl_agents.value import heuristic_value
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_learn.nn.board_gnn import BoardGNN, gnn_seams
from settlrl_learn.nn.graphnet import PRESETS
from settlrl_learn.nn.mlp import init_az_params, make_az
from settlrl_search import make_search_weights
from settlrl_search.policy import PolicyPrior

_W, _L = 8, 2  # a tiny trunk: the census counts ops, not FLOPs


def _position() -> tuple:
    """A mid-game position with a *branching* decision (19 legal moves): a forced
    move makes the weights degenerate, so the leaf could not be observed there."""
    env = BatchedSettlrlEnv(batch_size=1, seed=0, n_players=2, track_beliefs=True)
    env.rollout(jax.random.key(0), 120)
    player = int(env.agent_selection[0])
    mask = env.flat_mask()[0]
    assert int(mask.sum()) > 4
    return (
        jax.tree.map(lambda x: x[0], env.board[0]),
        jax.tree.map(lambda x: x[0], env.belief_view(player)),
        jnp.int32(player),
        mask,
    )


def _tiny_net() -> BoardGNN:
    return BoardGNN(
        jax.random.key(0), PRESETS["gn_global"]._replace(width=_W, layers=_L)
    )


def _in_loop_node_dots(text: str) -> int:
    """Trunk node dots (the ``(54, width)`` per-vertex matmuls) emitted inside the
    search's simulation ``while`` body."""
    shape = rf"f32\[(?:54,{_W}|{_W},54)\]"
    return sum(
        "while" in m.group(1)
        for m in re.finditer(rf'{shape}\S* dot\(.*?op_name="([^"]+)"', text)
    )


def _compiled_search(*, split: bool) -> tuple[str, np.ndarray]:
    """The lowered text and the weights of a tiny-net search, with the seam either
    shared (a ``ValuePrior``) or split into two forwards."""
    layout, view, player, mask = _position()
    params, static = eqx.partition(_tiny_net(), eqx.is_array)

    def fn(
        params: Any, key: Any, layout: Any, view: Any, player: Any, mask: Any
    ) -> Any:
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

    args = (params, jax.random.key(1), layout, view, player, mask)
    compiled = jax.jit(fn).lower(*args).compile()
    text = compiled.as_text()
    assert text is not None
    return text, np.asarray(compiled(*args))


def test_leaf_runs_one_trunk_per_simulation() -> None:
    shared_text, shared_w = _compiled_search(split=False)
    split_text, split_w = _compiled_search(split=True)
    shared, split = _in_loop_node_dots(shared_text), _in_loop_node_dots(split_text)
    assert shared > 0
    # Observed at width 8 / 2 layers: 5 shared, 10 split. The ratio is the
    # contract; the absolute counts move with the architecture.
    assert split == 2 * shared, (
        f"expected the split seam to run two trunks per simulation and the shared "
        f"one, got {split} vs {shared} in-loop node dots"
    )
    # Fusing paired seams is semantics-preserving, not just cheaper.
    assert np.array_equal(shared_w, split_w)


def _unpaired_weights(*, split: bool, fused_leaf: bool) -> np.ndarray:
    """Weights for an *unpaired* pairing: the heuristic value under the net's
    policy head, with the net's seam either exposed as a ``ValuePrior`` or hidden
    behind a plain closure."""
    layout, view, player, mask = _position()
    params, static = eqx.partition(_tiny_net(), eqx.is_array)

    def fn(
        params: Any, key: Any, layout: Any, view: Any, player: Any, mask: Any
    ) -> Any:
        _, value_prior = gnn_seams(eqx.combine(params, static))
        prior: PolicyPrior = (
            (lambda lay, st, pl: value_prior(lay, st, pl)) if split else value_prior
        )
        weights = make_search_weights(
            heuristic_value, prior=prior, value_scale=2.0, num_simulations=8,
            max_depth=6, expected_rolls=False, fused_leaf=fused_leaf,
        )  # fmt: skip
        return weights(key, layout, view, player, mask)

    return np.asarray(
        jax.jit(fn)(params, jax.random.key(1), layout, view, player, mask)
    )


def test_fused_leaf_off_keeps_the_explicit_value_for_an_unpaired_prior() -> None:
    two_seam = _unpaired_weights(split=True, fused_leaf=True)  # no ValuePrior at all
    assert np.array_equal(_unpaired_weights(split=False, fused_leaf=False), two_seam)


def test_fused_leaf_on_scores_the_leaf_with_the_priors_value_head() -> None:
    two_seam = _unpaired_weights(split=True, fused_leaf=True)
    fused = _unpaired_weights(split=False, fused_leaf=True)
    # The net's untrained value head is not the heuristic, so the leaf changed.
    assert not np.array_equal(fused, two_seam)


def test_gnn_with_value_matches_the_separate_seams() -> None:
    layout, _, player, _ = _position()
    env = BatchedSettlrlEnv(batch_size=1, seed=0, n_players=2)
    env.rollout(jax.random.key(0), 60)
    state = jax.tree.map(lambda x: x[0], env.board[1])
    value, prior = gnn_seams(_tiny_net())
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
