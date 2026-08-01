"""The message/update MLPs' first Linears are weight-blocked
(``graphnet._blocked_mlp``): one matmul per unique input row, gathered onto the
pair set. These contracts pin that composition against a naive
gather-concatenate-transform reference built here from the SAME modules -- the
naive matmul consumes each first Linear's weight whole, the blocked form its
column slices, so closeness is also the parameter-identity proof (a
checkpoint's message-MLP parameters serve both formulations unchanged). Pinned
at float32 on a real board featurization, and at float64 where the only
difference left -- float summation order -- must leave a residual below 1e-12.
One arm per preset: the feature_version/incidence axes only resize encoder
inputs and never touch the blocked paths. The symmetry contracts
(D3/relabel/equivariance) live in ``test_architectures.py``."""

from __future__ import annotations

import functools
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from settlrl_engine.board import Board
from settlrl_engine.board.layout import N_TILES, N_VERTICES
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_learn.nn import graphnet
from settlrl_learn.nn.board_gnn import BoardGNN
from settlrl_learn.nn.graph import SENDERS, Sample, board_sample, dims
from settlrl_learn.nn.graphnet import PRESETS, GraphNetConfig, GraphTrunk

_ARMS = [
    # Between them the two arms cover every blocked call site: message,
    # node-update and global blocks (gn_global), the two hetero messages
    # (gn_hetero).
    ("gn_global", 2, False),
    ("gn_hetero", 2, True),
]


def _naive_mlp(
    mlp: eqx.nn.MLP,
    blocks: Sequence[tuple[jax.Array, jax.Array | None]],
) -> jax.Array:
    """Gather-then-transform reference: gather every block onto the output rows,
    concatenate, and ``jax.vmap`` the MLP whole -- the un-sliced weight of the
    same first Linear the blocked form reads column slices of."""
    n_rows = next(
        x[index].shape[0] if index is not None else x.shape[0]
        for x, index in blocks
        if index is not None or x.ndim == 2
    )
    rows = [
        x[index]
        if index is not None
        else (x if x.ndim == 2 else jnp.broadcast_to(x, (n_rows, x.shape[0])))
        for x, index in blocks
    ]
    concat = jnp.concatenate(rows, axis=-1)
    # the blocked slices tile this weight exactly: same parameters, whole.
    assert concat.shape[-1] == mlp.layers[0].weight.shape[1]
    return jax.vmap(mlp)(concat)


def _cfg(preset: str, version: int, incidence: bool) -> GraphNetConfig:
    return PRESETS[preset]._replace(
        width=16,
        layers=2,
        head_depth=1,
        feature_version=version,
        incidence=incidence,
    )


def _random_sample(key: jax.Array, version: int, incidence: bool) -> Sample:
    node_dim, edge_dim, glob_dim, tile_dim = dims(version, incidence)
    kn, ke, kg, kt = jax.random.split(key, 4)
    n_edges = SENDERS.shape[0]
    return Sample(
        nodes=jax.random.normal(kn, (N_VERTICES, node_dim)),
        edges=jax.random.normal(ke, (n_edges, edge_dim)),
        glob=jax.random.normal(kg, (glob_dim,)),
        tiles=jax.random.normal(kt, (N_TILES, tile_dim)),
    )


@functools.cache
def _mid_game(n_players: int = 2, steps: int = 120, seed: int = 7) -> Board:
    """A single-game position with real ownership (past setup), random play."""
    env = BatchedSettlrlEnv(
        batch_size=1, n_players=n_players, seed=seed, auto_reset=False
    )
    key = jax.random.key(seed)
    for _ in range(steps):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
    layout = jax.tree.map(lambda x: x[0], env.board[0])
    state = jax.tree.map(lambda x: x[0], env.board[1])
    return layout, state


@pytest.mark.parametrize(("preset", "version", "incidence"), _ARMS)
def test_blocked_matches_the_naive_reference_on_real_boards(
    preset: str,
    version: int,
    incidence: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, state = _mid_game()
    s = board_sample(layout, state, jnp.int32(0), version=version, incidence=incidence)
    cfg = _cfg(preset, version, incidence)
    trunk = GraphTrunk(jax.random.key(0), cfg)
    net = BoardGNN(jax.random.key(1), cfg)
    trunk_out = trunk(s)
    v, pol = net(s)
    # the SAME modules forwarded through the naive reference: parameter
    # identity by construction, only the summation composition differs.
    monkeypatch.setattr(graphnet, "_blocked_mlp", _naive_mlp)
    for a, b in zip(trunk_out, trunk(s), strict=True):
        if a is None or b is None:
            assert a is None and b is None
            continue
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5, rtol=1e-5)
    v_ref, pol_ref = net(s)
    # Vacuity guard: the reference must have actually rerouted the forward --
    # reassociation guarantees a float32 bit difference between the two paths.
    assert not np.array_equal(np.asarray(v), np.asarray(v_ref))
    np.testing.assert_allclose(np.asarray(v), np.asarray(v_ref), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(pol), np.asarray(pol_ref), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(("preset", "version", "incidence"), _ARMS)
def test_blocked_is_the_same_function_at_float64(
    preset: str,
    version: int,
    incidence: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The linear-algebra identity, not "close enough": at float64 the only
    # difference left is the reassociated summation, whose residual is ~machine
    # epsilon -- 1e-12 rules out any weight/bias/segment misalignment, which
    # would surface at O(1).
    with jax.enable_x64():

        def to64(tree: object) -> object:
            return jax.tree.map(
                lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x,
                tree,
            )

        trunk = to64(GraphTrunk(jax.random.key(0), _cfg(preset, version, incidence)))
        assert isinstance(trunk, GraphTrunk)
        s = to64(_random_sample(jax.random.key(0), version, incidence))
        assert isinstance(s, Sample)
        blocked_out = trunk(s)
        monkeypatch.setattr(graphnet, "_blocked_mlp", _naive_mlp)
        for a, b in zip(blocked_out, trunk(s), strict=True):
            if a is None:
                assert b is None
                continue
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), atol=1e-12, rtol=1e-12
            )
