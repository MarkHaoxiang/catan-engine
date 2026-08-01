"""Contracts for ``GraphNetConfig.blocked_linear``: the flag-on trunk computes
the SAME function as flag-off with the SAME parameters -- the two compositions
differ only in float summation order (weight-blocked first Linears). Pinned
three ways: parameter identity, float32 closeness on the real featurization,
and a float64 run where the reassociation residual must vanish to ~1e-12. The
symmetry contracts (D3 board invariance, player-relabel invariance, policy
equivariance) are re-asserted with the flag on. One arm per preset: the
feature_version/incidence axes only resize encoder inputs and never touch the
blocked paths."""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _symmetry import (
    action_permutation,
    apply_symmetry,
    board_symmetries,
    relabel_players,
)
from settlrl_engine.board import Board
from settlrl_engine.board.layout import N_TILES, N_VERTICES
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_learn.nn.board_gnn import BoardGNN
from settlrl_learn.nn.graph import SENDERS, Sample, board_sample, dims
from settlrl_learn.nn.graphnet import PRESETS, GraphNetConfig, GraphTrunk
from settlrl_learn.training import GNNBackend

_ARMS = [
    ("gn_global", 2, False),
    ("gn_hetero", 2, True),
    # gat: the conv is untouched by the flag; blocking applies to the node
    # update (and would apply to the hetero messages).
    ("gn_gat", 1, False),
]


def _cfg(preset: str, version: int, incidence: bool, blocked: bool) -> GraphNetConfig:
    return PRESETS[preset]._replace(
        width=16,
        layers=2,
        head_depth=1,
        feature_version=version,
        incidence=incidence,
        blocked_linear=blocked,
    )


def _trunk_pair(
    preset: str, version: int, incidence: bool
) -> tuple[GraphTrunk, GraphTrunk]:
    """Flag-off and flag-on trunks from the same key: identical parameters."""
    key = jax.random.key(0)
    return (
        GraphTrunk(key, _cfg(preset, version, incidence, False)),
        GraphTrunk(key, _cfg(preset, version, incidence, True)),
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


def _assert_trunk_close(
    off: GraphTrunk, on: GraphTrunk, s: Sample, atol: float, rtol: float
) -> None:
    for a, b in zip(off(s), on(s), strict=True):
        if a is None or b is None:
            assert a is None and b is None
            continue
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


def test_flag_does_not_touch_parameters() -> None:
    # A checkpoint's weights load identically under either flag: init consumes
    # the same keys and builds the same arrays regardless of `blocked_linear`.
    off, on = _trunk_pair("gn_hetero", 2, True)
    leaves_off = jax.tree.leaves(eqx.filter(off, eqx.is_array))
    leaves_on = jax.tree.leaves(eqx.filter(on, eqx.is_array))
    for a, b in zip(leaves_off, leaves_on, strict=True):
        assert a.shape == b.shape and bool((a == b).all())


@pytest.mark.parametrize(("preset", "version", "incidence"), _ARMS)
def test_blocked_matches_unblocked_on_real_boards(
    preset: str, version: int, incidence: bool
) -> None:
    layout, state = _mid_game()
    s = board_sample(layout, state, jnp.int32(0), version=version, incidence=incidence)
    off, on = _trunk_pair(preset, version, incidence)
    _assert_trunk_close(off, on, s, atol=1e-5, rtol=1e-5)
    # the full value+policy net over the same trunk config.
    key = jax.random.key(1)
    net_off = BoardGNN(key, _cfg(preset, version, incidence, False))
    net_on = BoardGNN(key, _cfg(preset, version, incidence, True))
    v_off, p_off = net_off(s)
    v_on, p_on = net_on(s)
    np.testing.assert_allclose(
        np.asarray(v_off), np.asarray(v_on), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(p_off), np.asarray(p_on), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(("preset", "version", "incidence"), _ARMS)
def test_blocked_is_the_same_function_at_float64(
    preset: str, version: int, incidence: bool
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

        off, on = _trunk_pair(preset, version, incidence)
        off64, on64 = to64(off), to64(on)
        assert isinstance(off64, GraphTrunk) and isinstance(on64, GraphTrunk)
        s = to64(_random_sample(jax.random.key(0), version, incidence))
        assert isinstance(s, Sample)
        for a, b in zip(off64(s), on64(s), strict=True):
            if a is None:
                assert b is None
                continue
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), atol=1e-12, rtol=1e-12
            )


@pytest.mark.parametrize(("preset", "version", "incidence"), _ARMS)
def test_blocked_trunk_keeps_both_invariances(
    preset: str, version: int, incidence: bool
) -> None:
    # The flag-on composition reuses only symmetric aggregations (matmul over
    # all rows + segment gathers), so the D3/relabel contracts must still hold
    # directly, not merely via closeness to the flag-off net.
    layout, state = _mid_game(n_players=4)
    net = BoardGNN(jax.random.key(0), _cfg(preset, version, incidence, True))

    def sample(lo: object, st: object, q: int) -> Sample:
        return board_sample(lo, st, jnp.int32(q), version=version, incidence=incidence)  # type: ignore[arg-type]

    base_v, base_pol = net(sample(layout, state, 0))
    v0, pol0 = np.asarray(base_v), np.asarray(base_pol)
    for sym in board_symmetries():
        l2, s2 = apply_symmetry(layout, state, sym)
        v, pol = net(sample(l2, s2, 0))
        assert np.allclose(np.asarray(v), v0, atol=1e-3)  # value invariant
        perm = action_permutation(sym)
        assert np.allclose(np.asarray(pol)[perm], pol0, atol=1e-3)  # equivariant
    relabel = np.array([1, 2, 3, 0])
    v, _ = net(sample(layout, relabel_players(state, relabel), int(relabel[0])))
    assert np.allclose(np.asarray(v), v0, atol=1e-3)  # relabel invariant


def test_backend_composes_with_the_flag() -> None:
    # The distill guard (experiments/0003) reaches the flag through
    # GNNBackend(netcfg): a flag-on config must build, observe and forward.
    cfg = _cfg("gn_hetero", 2, False, True)
    backend = GNNBackend(cfg)
    net = backend.init(jax.random.key(0))
    layout, state = _mid_game()
    obs = backend.observe(layout, state, jnp.int32(0))
    s = Sample(obs["nodes"], obs["edges"], obs["glob"], obs["tiles"], None)
    v, pol = net(s)
    assert bool(jnp.isfinite(v)) and bool(jnp.isfinite(pol).all())
