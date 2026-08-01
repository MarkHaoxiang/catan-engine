"""Symmetry contracts for the board architectures, and the featurization-version
contracts (bottom of the file).

Two symmetries leave a position's *meaning* unchanged, so a sound representation
must score it identically:

- **player relabeling** -- swap who is who (and the perspective); the
  player-relative featurization is exactly invariant, so any model is too;
- **board rotation/reflection** -- the 12 hexagon automorphisms; a structure-
  aware readout (GNN message passing, DeepSet pooling) is invariant, while the
  structure-blind flat MLP is not (it reads nodes in fixed vertex order).
"""

from __future__ import annotations

import hashlib

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
from settlrl_engine.board.dev_cards import DevCard
from settlrl_engine.board.layout import N_VERTICES, BoardLayout
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.board.state import BoardState
from settlrl_engine.board.tile import Tile
from settlrl_engine.env import N_FLAT, ActionType, BatchedSettlrlEnv
from settlrl_learn.features import features
from settlrl_learn.nn import graph, graphnet
from settlrl_learn.nn.architectures import DeepSetModel, GNNModel, MLPModel
from settlrl_learn.nn.board_gnn import BoardGNN
from settlrl_learn.nn.graph import Sample, board_sample
from settlrl_learn.nn.graphnet import PRESETS, GraphNet
from settlrl_learn.training.backends.gnn import gnn_loss
from settlrl_search.rows import ROW_TYPE

_OUT, _W = 4, 8


def _mid_game(n_players: int, steps: int = 150, seed: int = 7) -> Board:
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


_V2_BLOCK = 14  # the version-2 tail of `glob` (own hand, dev hand, turn state)


def _engage_v2(state: BoardState) -> BoardState:
    """Set every version-2 global to a nonzero value from seat 0's view (and the
    opponent Longest Road slot from seat 1's), so an invariance test cannot pass on
    a block of zeros: mid-game random play leaves 9 of the 14 at 0."""
    resources = np.zeros_like(np.asarray(state.player_resources))
    resources[0], resources[1] = [1, 2, 1, 3, 4], [2, 1, 1, 1, 1]
    dev = np.zeros_like(np.asarray(state.dev_hand))
    dev[0], dev[1] = [2, 1, 1, 1, 1], [1, 1, 1, 1, 1]
    discard = np.zeros_like(np.asarray(state.pending_discard))
    discard[0], discard[1] = 4, 2
    u8 = state.free_roads.dtype
    return state._replace(
        player_resources=jnp.asarray(resources),
        dev_hand=jnp.asarray(dev),
        pending_discard=jnp.asarray(discard),
        free_roads=jnp.asarray(2, u8),
        current_player=jnp.asarray(0, state.current_player.dtype),
        longest_road_owner=jnp.asarray(0, state.longest_road_owner.dtype),
        longest_road_len=jnp.asarray(7, state.longest_road_len.dtype),
    )


def _position(version: int, n_players: int = 4) -> Board:
    """The invariance fixture: v1 as played, v2 with its added globals engaged."""
    layout, state = _mid_game(n_players)
    return layout, (_engage_v2(state) if version >= 2 else state)


def _gnn() -> GNNModel:
    return GNNModel(jax.random.key(0), out_dim=_OUT, width=_W, depth=1, layers=2)


def test_board_symmetries_form_the_order_6_group() -> None:
    # The harbors are 3-fold symmetric, so the full board's group is order 6.
    syms = board_symmetries()
    assert len(syms) == 6
    keys = {s.vertices.tobytes() for s in syms}
    assert len(keys) == 6  # all distinct
    assert any(np.array_equal(s.vertices, np.arange(s.vertices.shape[0])) for s in syms)


def test_player_relabel_leaves_features_and_gnn_invariant() -> None:
    layout, state = _mid_game(4)
    gnn = _gnn()
    perms = [np.array([1, 0, 2, 3]), np.array([1, 2, 3, 0]), np.array([3, 2, 1, 0])]
    for p in range(4):
        base = board_sample(layout, state, jnp.int32(p))
        base_out = gnn(base)
        for perm in perms:
            other = board_sample(
                layout, relabel_players(state, perm), jnp.int32(perm[p])
            )
            # the graph featurization is exactly relabeling-invariant ...
            for a, b in zip(base[:3], other[:3], strict=True):
                assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)
            # ... so the model is too.
            assert np.allclose(np.asarray(base_out), np.asarray(gnn(other)), atol=1e-4)


def test_board_symmetry_leaves_structured_models_invariant() -> None:
    layout, state = _mid_game(2)
    p = jnp.int32(0)
    base = board_sample(layout, state, p)
    key = jax.random.key(1)
    structured = (
        _gnn(),
        DeepSetModel(key, out_dim=_OUT, width=_W, depth=1),
    )
    rotated = [apply_symmetry(layout, state, sym) for sym in board_symmetries()]
    for model in structured:
        base_out = np.asarray(model(base))
        for l2, s2 in rotated:
            out = np.asarray(model(board_sample(l2, s2, p)))
            assert np.allclose(base_out, out, atol=1e-4)


@pytest.mark.parametrize(
    ("preset", "version", "incidence"),
    [
        ("gn_multi", 1, False),
        ("gn_graphnorm", 1, False),
        ("gn_gat", 1, False),
        ("gn_full", 1, False),
        ("gn_hetero", 1, False),
        ("gn_global", 2, False),
        ("gn_hetero", 2, False),
        ("gn_hetero_dnorm", 2, False),
        ("gn_global", 2, True),
        ("gn_hetero", 2, True),
    ],
)
def test_graphnet_presets_are_invariant(
    preset: str, version: int, incidence: bool
) -> None:
    # The configurable GraphNet keeps both invariances across every lever
    # (attention, GraphNorm spanning the node axis, the global node, JK) -- it
    # uses only symmetric aggregations and relative features, no absolute PE.
    layout, state = _position(version)
    cfg = PRESETS[preset]._replace(
        width=8, layers=2, head_depth=1, feature_version=version, incidence=incidence
    )
    model = GraphNet(jax.random.key(0), out_dim=_OUT, cfg=cfg)

    def sample(lo: BoardLayout, st: BoardState, q: int) -> Sample:
        return board_sample(lo, st, jnp.int32(q), version=version, incidence=incidence)

    base = np.asarray(model(sample(layout, state, 0)))
    for sym in board_symmetries():
        l2, s2 = apply_symmetry(layout, state, sym)
        assert np.allclose(base, np.asarray(model(sample(l2, s2, 0))), atol=1e-3)
    perm = np.array([1, 2, 3, 0])
    relabeled = sample(layout, relabel_players(state, perm), int(perm[0]))
    assert np.allclose(base, np.asarray(model(relabeled)), atol=1e-3)


@pytest.mark.parametrize("net_seed", [0, 3])
def test_v1_ctx_unbounded_v2_layernorm_bounds_it(net_seed: int) -> None:
    # Correctness-audit evidence, direct: on a real board, v1's raw ctx
    # (pooled readout ++ the global node ``g``) sits at ~25x the L2 norm a
    # random-init head is scaled for -- an artifact of sum-pooling over
    # N=54 nodes, not learned signal (the audit's own measurement: block
    # norms ~373/10/7 vs g~34, before this fix). ``feature_version>=2``'s
    # LayerNorm provably (weight=1, bias=0 at init) bounds the WHOLE vector
    # to unit per-element variance, so its norm sits at ``sqrt(dim)``
    # regardless of net init; v1 clears many times that -- the control this
    # test asserts must fail to stay bounded. Parametrized over two net-init
    # seeds: the margin is huge either way (raw norm ~200-210 vs a ~40
    # threshold, checked across 6 seeds), so this is not noise-sensitive.
    layout, state = _mid_game(4)
    p = jnp.int32(0)
    net1, net2 = _aznet("gn_global", 1, net_seed), _aznet("gn_global", 2, net_seed)
    _h1, g1, readout1, _t1 = net1.trunk(board_sample(layout, state, p, version=1))
    ctx1 = jnp.concatenate([readout1, g1])
    _h2, g2, readout2, _t2 = net2.trunk(board_sample(layout, state, p, version=2))
    ctx2 = net2.trunk.ctx(g2, readout2)
    dim1, dim2 = ctx1.shape[0], ctx2.shape[0]
    assert float(jnp.linalg.norm(ctx1)) > 5 * dim1**0.5  # v1: unbounded (the control)
    assert float(jnp.linalg.norm(ctx2)) == pytest.approx(
        dim2**0.5, rel=0.05
    )  # v2: bounded


def test_v2_ctx_norm_keeps_init_value_logit_calibrated() -> None:
    # The plan's literal operationalization (std of value logits over ~32
    # positions vs |mean|) does not discriminate at this width/depth: checked
    # across 10 net-init seeds, v1's std/|mean| ratio is *not* reliably lower
    # than v2's (sometimes higher) -- raw ctx magnitude dominates the head's
    # random weights enough to inflate both the mean *and* the spread, not
    # just the mean. The behavioral consequence that DOES discriminate,
    # robustly, is calibration: the value logit is read as ``P(win)`` via
    # ``tanh(logit/2)`` (``value_scale=2``, package CLAUDE.md), and v1's
    # oversized ctx (previous test) pushes the untrained logit's magnitude far
    # outside the unsaturated band regardless of the actual position, while
    # v2's bounded ctx keeps it inside -- at the project's canonical test net
    # seed (``_aznet``'s ``jax.random.key(0)``, not selected for this result).
    # Not parametrized over a second net seed (unlike the previous test): of
    # the 10 seeds checked, one (net_seed=1) has v1's |mean logit| at 0.418 --
    # under the saturation bar -- so a second seed here is a coin flip, not a
    # cheap robustness fold.
    positions = [_mid_game(2, steps=150, seed=s) for s in range(32)]
    saturated_bar = 2.0  # |logit|/2 > 1 already compresses tanh toward +-1
    for version, must_saturate in ((1, True), (2, False)):
        net = _aznet("gn_global", version)
        logits = np.asarray(
            [
                float(net(board_sample(lo, st, jnp.int32(0), version=version))[0])
                for lo, st in positions
            ]
        )
        assert (abs(float(logits.mean())) > saturated_bar) == must_saturate


def test_v2_readout_std_finite_at_zero_variance() -> None:
    # The std readout block (`graphnet._pool`) computes `sqrt(var + eps)`;
    # without the eps guard, `sqrt`'s gradient is infinite at exact zero
    # variance (`sqrt(0) == 0` is already fine forward -- the hazard is in
    # the backward pass). A real forward's post-message-passing `h` never
    # hits *exact* zero variance -- even with every node's encoded embedding
    # zeroed, edge features + degree reintroduce ~1e-4..1e-3 cross-node
    # variance -- so a black-box forward test cannot exercise the guard.
    # Exercises `_pool` directly on a manufactured all-equal `h`
    # (`var(0) == 0` exactly), checking both the forward value and the
    # gradient. Falsified: fails (NaN grad) with `_STD_EPS` locally patched
    # to `0.0` -- see the Task 2 fix report.
    h = jnp.full((6, 4), 3.0)
    assert bool(jnp.all(h.var(0) == 0.0))  # the exact-zero-variance precondition

    def loss(x: jax.Array) -> jax.Array:
        return graphnet._pool("multi", x, 2).sum() ** 2

    val, grad = jax.value_and_grad(loss)(h)
    assert bool(jnp.isfinite(val))
    assert bool(jnp.isfinite(grad).all())


def test_hetero_off_ignores_tiles() -> None:
    # `board_sample(with_tiles=False)`'s prose promise: a non-hetero trunk's
    # graph is free of tile ops, so garbage `tiles` must score identically to
    # the constant-zero default.
    layout, state = _mid_game(2)
    p = jnp.int32(0)
    base = board_sample(layout, state, p, with_tiles=False)
    garbage = base._replace(
        tiles=jax.random.normal(jax.random.key(0), base.tiles.shape)
    )
    model = GraphNet(
        jax.random.key(1), out_dim=_OUT,
        cfg=PRESETS["gn_global"]._replace(width=_W, layers=1, head_depth=1),
    )  # fmt: skip
    assert np.array_equal(np.asarray(model(base)), np.asarray(model(garbage)))


def test_mlp_engineered_forward() -> None:
    # `Sample.extra` contract: `board_sample(features=...)` populates it, and
    # `MLPModel(engineered=True)` reads it -- pinned once inside the package
    # (currently only witnessed indirectly by experiment 0003's smoke).
    layout, state = _mid_game(2)
    p = jnp.int32(0)
    sample = board_sample(layout, state, p, features=features)
    model = MLPModel(
        jax.random.key(0), out_dim=_OUT, width=_W, depth=1, engineered=True
    )
    out = np.asarray(model(sample))
    assert out.shape == (_OUT,)
    assert bool(np.all(np.isfinite(out)))


def test_flat_mlp_is_not_symmetry_invariant() -> None:
    # The contrast that motivates structure: reordering nodes moves the flat
    # input vector, so the structure-blind MLP cannot be rotation-invariant.
    layout, state = _mid_game(2)
    p = jnp.int32(0)
    flat = MLPModel(
        jax.random.key(0), out_dim=_OUT, width=_W, depth=1, engineered=False
    )
    base = np.asarray(flat(board_sample(layout, state, p)))
    moved = max(
        float(np.abs(np.asarray(flat(board_sample(l2, s2, p))) - base).max())
        for l2, s2 in (apply_symmetry(layout, state, sym) for sym in board_symmetries())
    )
    assert moved > 1e-3


def _aznet(
    preset: str = "gn_global",
    version: int = 1,
    net_seed: int = 0,
    incidence: bool = False,
) -> BoardGNN:
    cfg = PRESETS[preset]._replace(
        width=16,
        layers=2,
        head_depth=1,
        feature_version=version,
        incidence=incidence,
    )
    return BoardGNN(jax.random.key(net_seed), cfg)


_FEATURE_ARMS = [(1, False), (2, False), (2, True)]
"""The featurization arms every net-level symmetry contract is checked over:
v1, v2, v2 + the incidence block."""


@pytest.mark.parametrize(("version", "incidence"), _FEATURE_ARMS)
@pytest.mark.parametrize("preset", ["gn_global", "gn_hetero", "gn_hetero_dnorm"])
def test_aznet_value_invariant_policy_equivariant_under_board_symmetry(
    preset: str, version: int, incidence: bool
) -> None:
    # The factored value+policy net: the value is invariant under a board
    # symmetry, and the policy is *equivariant* -- a settlement-at-v action maps
    # to settlement-at-(sigma v), road-at-e to road-at-(sigma e), robber-tile-t
    # to sigma(t) -- so policy(sigma . board)[action_permutation] == policy(board).
    layout, state = _position(version)
    net = _aznet(preset, version, incidence=incidence)
    p = jnp.int32(0)
    vv, pp = net(board_sample(layout, state, p, version=version, incidence=incidence))
    v0, pol0 = np.asarray(vv), np.asarray(pp)
    for sym in board_symmetries():
        l2, s2 = apply_symmetry(layout, state, sym)
        v, pol = net(board_sample(l2, s2, p, version=version, incidence=incidence))
        assert np.allclose(np.asarray(v), v0, atol=1e-3)  # value invariant
        perm = action_permutation(sym)
        assert np.allclose(np.asarray(pol)[perm], pol0, atol=1e-3)  # policy equivariant


@pytest.mark.parametrize(("version", "incidence"), _FEATURE_ARMS)
@pytest.mark.parametrize("preset", ["gn_global", "gn_hetero", "gn_hetero_dnorm"])
def test_aznet_value_and_policy_invariant_under_player_relabel(
    preset: str, version: int, incidence: bool
) -> None:
    layout, state = _position(version)
    net = _aznet(preset, version, incidence=incidence)
    vv, pp = net(
        board_sample(layout, state, jnp.int32(0), version=version, incidence=incidence)
    )
    v0, pol0 = np.asarray(vv), np.asarray(pp)
    perm = np.array([1, 2, 3, 0])
    relabeled = board_sample(
        layout,
        relabel_players(state, perm),
        jnp.int32(perm[0]),
        version=version,
        incidence=incidence,
    )
    v, pol = net(relabeled)
    assert np.allclose(np.asarray(v), v0, atol=1e-3)  # value invariant
    # Spatial (vertex/edge/tile, with the robber victim collapsed to no-steal vs
    # steal) and non-player-indexed actions are relabel-invariant; PROPOSE_TRADE's
    # partner index inherits the opponent-collapse limitation, so it is excluded.
    keep = np.asarray(ROW_TYPE) != int(ActionType.PROPOSE_TRADE)
    assert np.allclose(np.asarray(pol)[keep], pol0[keep], atol=1e-3)


def test_hetero_net_consumes_tile_features() -> None:
    # Equivariance/invariance tests alone would pass on a net that ignores
    # `h_t` entirely (a constant is trivially both). Pin genuine consumption:
    # zeroing the real per-hex features (vs. leaving them as computed) must
    # move both heads materially -- both read `h_t`, the policy tile logits
    # directly (board_gnn.py's `_FactoredPolicy`) and the value via the
    # trunk's hex-pooled readout.
    layout, state = _mid_game(4)
    net = _aznet("gn_hetero")
    p = jnp.int32(0)
    real = board_sample(layout, state, p)
    zeroed = real._replace(tiles=jnp.zeros_like(real.tiles))
    v_real, pol_real = net(real)
    v_zero, pol_zero = net(zeroed)
    assert float(jnp.abs(v_real - v_zero)) > 1e-3
    assert float(jnp.max(jnp.abs(pol_real - pol_zero))) > 1e-3


def test_aznet_runs_on_random_play_boards() -> None:
    # Fast net check (no MCTS): drive the board with a random policy and run the
    # net forward -- correct shapes, finite values.
    net = _aznet()
    env = BatchedSettlrlEnv(batch_size=8, n_players=2, seed=0)
    fwd = jax.jit(jax.vmap(lambda lo, st, p: net(board_sample(lo, st, p))))
    key = jax.random.key(0)
    for _ in range(40):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
        lo, st = env.board
        v, pol = fwd(lo, st, env.agent_selection)
        assert v.shape == (8,) and pol.shape == (8, N_FLAT)
        assert bool(jnp.isfinite(v).all()) and bool(jnp.isfinite(pol).all())


def test_gnn_loss_masked_is_finite() -> None:
    # The masked policy CE must stay finite (no 0 * -inf on illegal slots) for a
    # legal-supported target -- checked on real random-play boards + their masks.
    net = _aznet()
    env = BatchedSettlrlEnv(batch_size=4, n_players=2, seed=1)
    key = jax.random.key(0)
    for _ in range(25):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
    lo, st = env.board
    mask = jnp.asarray(env.flat_mask(), jnp.float32)  # (4, N_FLAT)
    samples = jax.vmap(board_sample)(lo, st, env.agent_selection)
    target = mask / jnp.clip(mask.sum(-1, keepdims=True), 1.0)  # uniform over legal
    loss, aux = gnn_loss(net, samples, target, jnp.zeros(4), mask, jnp.ones(4))
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.isfinite(v)) for v in aux.values())


def _pcr_batch(batch_size: int, seed: int) -> tuple[Sample, jax.Array, jax.Array]:
    env = BatchedSettlrlEnv(batch_size=batch_size, n_players=2, seed=seed)
    key = jax.random.key(seed)
    for _ in range(20):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
    lo, st = env.board
    mask = jnp.asarray(env.flat_mask(), jnp.float32)
    samples = jax.vmap(board_sample)(lo, st, env.agent_selection)
    target = mask / jnp.clip(mask.sum(-1, keepdims=True), 1.0)  # uniform over legal
    return samples, target, mask


def test_gnn_loss_masks_policy_by_train_policy() -> None:
    # The loss side of PCR (mirrors test_mlp_loss_masks_policy_by_train_policy):
    # the policy CE averages over train_policy=1 positions only (so it equals
    # the loss computed on that subset alone), while value loss spans all.
    net = _aznet()
    samples, target, mask = _pcr_batch(6, seed=2)
    value = jnp.zeros(6)
    full = jnp.ones(6, jnp.float32)
    half = jnp.array([1, 1, 1, 0, 0, 0], jnp.float32)
    first3 = jax.tree.map(lambda x: x[:3], samples)

    _, a_full = gnn_loss(net, samples, target, value, mask, full)
    _, a_half = gnn_loss(net, samples, target, value, mask, half)
    _, a_first3 = gnn_loss(
        net, first3, target[:3], value[:3], mask[:3], jnp.ones(3, jnp.float32)
    )
    # value loss spans every position -> unchanged by the policy mask.
    assert abs(float(a_full["value_loss"]) - float(a_half["value_loss"])) < 1e-5
    # masked policy loss == the policy loss over the unmasked subset alone.
    assert abs(float(a_half["policy_loss"]) - float(a_first3["policy_loss"])) < 1e-4


def test_gnn_loss_all_zero_train_policy_is_finite() -> None:
    # PCR can produce a batch with no full-search (train_policy=1) rows -- the
    # `jnp.maximum(sum(train_policy), 1.0)` guard (backends/gnn.py) must keep the
    # policy loss at a defined 0, not divide 0/0 into NaN.
    net = _aznet()
    samples, target, mask = _pcr_batch(4, seed=3)
    loss, aux = gnn_loss(
        net, samples, target, jnp.zeros(4), mask, jnp.zeros(4, jnp.float32)
    )
    assert bool(jnp.isfinite(loss))
    assert float(aux["policy_loss"]) == 0.0


# --------------------------------------------------------------------------- #
# Featurization versions                                                       #
# --------------------------------------------------------------------------- #

# Captured from commit 57a4e08's `graph.py` (the only featurization there is what
# `version=1` must reproduce): sha256 prefixes of each array of
# `board_sample(_mid_game(n), p=0)`. Frozen constants, not a regenerable snapshot
# -- the az0 anchor and every v1 checkpoint read exactly these bytes.
_V1_GOLDEN = {
    2: {
        "nodes": ((54, 17), "6332d47fcfae01d5"),
        "edges": ((144, 3), "983a355d434685a0"),
        "glob": ((40,), "8d8914685524b298"),
        "tiles": ((19, 9), "3654e9f9f523b789"),
    },
    4: {
        "nodes": ((54, 17), "ffe99efcb7e74df4"),
        "edges": ((144, 3), "658ca8b2746119cb"),
        "glob": ((40,), "3b6acd18ca0ac383"),
        "tiles": ((19, 9), "819be6de7b7bb3a6"),
    },
}


def _flat(s: Sample) -> np.ndarray:
    return np.concatenate([np.asarray(x).ravel() for x in s[:4]])


@pytest.mark.parametrize("n_players", [2, 4])
def test_v1_features_match_the_frozen_golden(n_players: int) -> None:
    layout, state = _mid_game(n_players)
    sample = board_sample(layout, state, jnp.int32(0), version=1)
    for name, (shape, digest) in _V1_GOLDEN[n_players].items():
        v = np.asarray(getattr(sample, name))
        assert v.shape == shape and str(v.dtype) == "float32"
        assert hashlib.sha256(v.tobytes()).hexdigest()[:16] == digest


# Captured from the featurization-v2 Tasks 1/2 landing (`board_sample(_mid_game(n),
# p=0, version=2)`, incidence off): sha256 prefixes of each array. `edges`/`tiles`
# match `_V1_GOLDEN` byte-for-byte (v2 doesn't touch them); `nodes` differs (the
# pips/5 scaling fix, Task 1) and `glob` differs (wider + the pips-consistent
# scaling) -- pinning both closes the gap `_V1_GOLDEN` leaves: v2-sans-incidence
# is the featurization every study arm except v2_incidence trains on, and until
# this test it had no byte-level pin of its own, only the appended-block's
# numeric values (`test_v2_block_pins_its_layout_and_scaling`) and shape/dims
# checks. `nn/graphnet.py`'s v2-gated readout/LayerNorm (Task 2) is a net-level
# change, not a `board_sample` one, so it cannot perturb this golden.
_V2_GOLDEN = {
    2: {
        "nodes": ((54, 17), "2278ef52689927ef"),
        "edges": ((144, 3), "983a355d434685a0"),
        "glob": ((54,), "28692c3860ed6229"),
        "tiles": ((19, 9), "3654e9f9f523b789"),
    },
    4: {
        "nodes": ((54, 17), "7132cdee90f1749b"),
        "edges": ((144, 3), "658ca8b2746119cb"),
        "glob": ((54,), "9707e255be1f231b"),
        "tiles": ((19, 9), "819be6de7b7bb3a6"),
    },
}


@pytest.mark.parametrize("n_players", [2, 4])
def test_v2_features_match_the_frozen_golden(n_players: int) -> None:
    layout, state = _mid_game(n_players)
    sample = board_sample(layout, state, jnp.int32(0), version=2)
    for name, (shape, digest) in _V2_GOLDEN[n_players].items():
        v = np.asarray(getattr(sample, name))
        assert v.shape == shape and str(v.dtype) == "float32"
        assert hashlib.sha256(v.tobytes()).hexdigest()[:16] == digest
        # edges/tiles are v1's bytes exactly -- v2 never touches them.
        if name in ("edges", "tiles"):
            assert digest == _V1_GOLDEN[n_players][name][1]


def test_feature_version_dims() -> None:
    v1, v2 = graph.dims(1, False), graph.dims(2, False)
    assert v1 == (graph.NODE_DIM, graph.EDGE_DIM, graph.GLOBAL_DIM, graph.TILE_DIM)
    # v2 only widens the globals (own hand 5 + dev hand 5 + free roads, own and
    # opponent longest-road length, pending discard); node/edge/tile keep theirs.
    assert v2[2] - v1[2] == 14
    assert (v2[0], v2[1], v2[3]) == (v1[0], v1[1], v1[3])
    with pytest.raises(ValueError, match="unknown feature version"):
        board_sample(*_mid_game(2), jnp.int32(0), version=3)


def _hand_pair(
    state: BoardState, a: list[int], b: list[int]
) -> tuple[BoardState, BoardState]:
    """Two states where players 0 and 1 hold ``a``/``b`` and then ``b``/``a``: the
    per-resource totals (hence the bank) and both hand *sizes* are identical, so v1
    -- which sees only sizes and the bank -- cannot tell them apart."""
    rows = np.zeros_like(np.asarray(state.player_resources))
    rows[0], rows[1] = a, b
    swapped = rows.copy()
    swapped[0], swapped[1] = b, a
    return (
        state._replace(player_resources=jnp.asarray(rows)),
        state._replace(player_resources=jnp.asarray(swapped)),
    )


def test_v2_reveals_own_hand_composition() -> None:
    # The 2026-07-30 audit's blindness probe, inverted. v1's globals carry hand
    # *sizes* and the bank only, so "I hold 2 wheat + 3 ore" and "I hold 5 wool"
    # (opponent holding the complement) featurize byte-identically -- the net could
    # not see what it can afford. v2 must separate them.
    layout, state = _mid_game(2)
    p = jnp.int32(0)
    for a, b in (
        ([0, 2, 0, 0, 3], [5, 0, 0, 0, 0]),
        ([2, 2, 1, 0, 0], [0, 0, 1, 2, 2]),
    ):
        s_a, s_b = _hand_pair(state, a, b)
        v1 = [board_sample(layout, s, p, version=1) for s in (s_a, s_b)]
        v2 = [board_sample(layout, s, p, version=2) for s in (s_a, s_b)]
        assert np.array_equal(_flat(v1[0]), _flat(v1[1]))  # the proven blindness
        assert not np.array_equal(_flat(v2[0]), _flat(v2[1]))  # ... now resolved


def test_v2_reveals_dev_card_composition() -> None:
    # Same probe on the dev hand: 2 knights vs 2 monopolies (neither a VP card, so
    # the VP totals match too) are byte-identical in v1, which carries only the
    # dev-card count.
    layout, state = _mid_game(2)
    p = jnp.int32(0)
    hands = []
    for kind in (int(DevCard.KNIGHT), int(DevCard.MONOPOLY)):
        row = np.zeros_like(np.asarray(state.dev_hand))
        row[0, kind] = 2
        hands.append(state._replace(dev_hand=jnp.asarray(row)))
    v1 = [board_sample(layout, s, p, version=1) for s in hands]
    v2 = [board_sample(layout, s, p, version=2) for s in hands]
    assert np.array_equal(_flat(v1[0]), _flat(v1[1]))
    assert not np.array_equal(_flat(v2[0]), _flat(v2[1]))


@pytest.mark.parametrize("field", ["free_roads", "pending_discard", "longest_road_len"])
def test_v2_reveals_turn_state(field: str) -> None:
    # Three more quantities v1 never encoded: the Road Building counter, the cards
    # owed after a 7, and *how long* the Longest Road is (v1 has only the award
    # holder flag, so the pair below fixes the holder and moves only the length).
    layout, state = _mid_game(2)
    p = jnp.int32(0)
    state = state._replace(
        current_player=jnp.uint8(0),  # free_roads belongs to the player on turn
        longest_road_owner=jnp.uint8(0),
    )
    if field == "pending_discard":
        lo, hi = (jnp.zeros_like(state.pending_discard), jnp.uint8([4, 0]))
    else:
        dtype = getattr(state, field).dtype
        lo, hi = jnp.asarray(0, dtype), jnp.asarray(6, dtype)
    pair = [state._replace(**{field: v}) for v in (lo, hi)]
    v1 = [board_sample(layout, s, p, version=1) for s in pair]
    v2 = [board_sample(layout, s, p, version=2) for s in pair]
    assert np.array_equal(_flat(v1[0]), _flat(v1[1]))
    assert not np.array_equal(_flat(v2[0]), _flat(v2[1]))


def test_v2_features_are_symmetry_and_relabel_invariant() -> None:
    # The v2 additions are all player-relative, so the featurization keeps both
    # invariances exactly (the nets built on it are covered above).
    layout, state = _position(version=2)
    base = board_sample(layout, state, jnp.int32(0), version=2)
    # vacuity guard: no added dim may sit at 0 in *both* seat views, or a broken
    # current-player mask / Longest-Road holder mapping would pass unnoticed.
    seat1 = board_sample(layout, state, jnp.int32(1), version=2)
    engaged = (np.asarray(base.glob)[-_V2_BLOCK:] != 0) | (
        np.asarray(seat1.glob)[-_V2_BLOCK:] != 0
    )
    assert bool(engaged.all())
    for sym in board_symmetries():
        l2, s2 = apply_symmetry(layout, state, sym)
        rot = board_sample(l2, s2, jnp.int32(0), version=2)
        assert np.allclose(np.asarray(base.glob), np.asarray(rot.glob), atol=1e-6)
    for perm in (np.array([1, 0, 2, 3]), np.array([1, 2, 3, 0])):
        for q in range(4):
            mine = board_sample(layout, state, jnp.int32(q), version=2)
            other = board_sample(
                layout, relabel_players(state, perm), jnp.int32(perm[q]), version=2
            )
            assert np.allclose(_flat(mine), _flat(other), atol=1e-6)


def test_v2_block_pins_its_layout_and_scaling() -> None:
    # The 14 appended values, spelled out: this pins the divisors (hand /5, dev /3,
    # free roads /2, road length /MAX_ROADS, discard /5), the slot order, the
    # "my turn" mask on free_roads (seat 1 reads 0) and the own/opponent split of
    # the Longest Road length (seat 0 holds it at 7 roads; seat 1 sees it opposite).
    # Tasks 2/3 must not perturb these silently.
    layout, state = _position(version=2)
    third = 1.0 / 3.0
    expected = {
        # own hand /5 ++ own dev hand /3 ++ [free roads /2, own LR, opponent LR, discard /5]
        0: [*[0.2, 0.4, 0.2, 0.6, 0.8], *[2 * third, third, third, third, third],
            1.0, 7.0 / 15.0, 0.0, 0.8],
        1: [*[0.4, 0.2, 0.2, 0.2, 0.2], *[third, third, third, third, third],
            0.0, 0.0, 7.0 / 15.0, 0.4],
    }  # fmt: skip
    for p, want in expected.items():
        block = np.asarray(board_sample(layout, state, jnp.int32(p), version=2).glob)
        np.testing.assert_allclose(block[-_V2_BLOCK:], want, atol=1e-6)


# --------------------------------------------------------------------------- #
# The incidence block (v2 option): per-tile identity without hex nodes          #
# --------------------------------------------------------------------------- #


def _incidence(nodes: np.ndarray) -> np.ndarray:
    """The incidence tail of a node-feature matrix, as (V, slots, per-hex)."""
    tail = nodes[:, -graph.INCIDENCE_DIM :]
    return tail.reshape(N_VERTICES, graph.MAX_VERTEX_TILES, graph.INCIDENT_TILE_DIM)


def test_incidence_reveals_number_identity() -> None:
    # The headline: `tile_pips` (6 - |7 - n|) collapses 6 with 8, and nothing else
    # in v1/v2 carries the number -- so swapping a 6-tile's number with an 8-tile's
    # produces a *different board* (different same-number income correlation, a real
    # strategic difference) that featurizes byte-identically, `tiles` block included.
    # The incidence block's number one-hot is the fix.
    layout, state = _mid_game(2)
    nums = np.asarray(layout.tile_number)
    six, eight = int(np.flatnonzero(nums == 6)[0]), int(np.flatnonzero(nums == 8)[0])
    swapped = np.asarray(layout.tile_number).copy()
    swapped[six], swapped[eight] = swapped[eight], swapped[six]
    pair = (layout, layout._replace(tile_number=jnp.asarray(swapped)))
    p = jnp.int32(0)

    for version in (1, 2):
        a, b = (board_sample(lo, state, p, version=version) for lo in pair)
        assert np.array_equal(_flat(a), _flat(b))  # the proven collapse
    a, b = (
        board_sample(lo, state, p, version=2, incidence=True) for lo in pair
    )  # ... now resolved
    assert not np.array_equal(np.asarray(a.nodes), np.asarray(b.nodes))
    # and it is the number one-hot that separates them: the other per-hex terms
    # (resource, pips, robber) are untouched by a number swap between equal-pip
    # hexes, slot order included -- reordering a vertex's slots would take an
    # incident hex of the same resource numbered strictly between 6 and 8.
    inc_a, inc_b = _incidence(np.asarray(a.nodes)), _incidence(np.asarray(b.nodes))
    n_oh = graph.INCIDENT_TILE_DIM - graph.N_TILE_NUMBERS
    assert np.array_equal(inc_a[..., :n_oh], inc_b[..., :n_oh])
    assert not np.array_equal(inc_a[..., n_oh:], inc_b[..., n_oh:])


def test_incidence_sort_key_is_injective_on_the_hex_payload() -> None:
    # The load-bearing invariant behind equivariance: slots are ordered by a key
    # computed from the hex's own (resource, number, robber) -- never its index --
    # so two hexes tie only when their whole feature row is identical, making the
    # sorted sequence a function of the *multiset* alone. Injectivity is what turns
    # "ties are harmless" into a proof rather than a hope.
    seen: dict[int, tuple[int, int, int]] = {}
    for res in range(N_RESOURCES + 1):
        for num in (0, *range(2, 13)):
            for robber in (0, 1):
                k = int(
                    graph._tile_sort_key(*(jnp.int32(x) for x in (res, num, robber)))
                )
                assert k not in seen, (seen.get(k), (res, num, robber))
                assert k < graph._ABSENT_TILE_KEY  # pads must sort strictly last
                seen[k] = (res, num, robber)


def _incidence_stress(layout: BoardLayout, state: BoardState) -> Board:
    """A board that engages every incidence hazard at one interior vertex: its
    three hexes are the desert plus a *near-tie* (two hexes sharing resource and
    number), with the robber on one of the near-tied hexes -- the robber term of
    ``_tile_sort_key`` is what breaks it, so this exercises the tie-breaking path
    itself, not just a resolved tie."""
    slots = np.asarray(graph.VERTEX_TILES)
    deg = np.asarray(graph.VERTEX_TILE_PRESENT).sum(1)
    v = int(np.flatnonzero(deg == graph.MAX_VERTEX_TILES)[0])
    t0, t1, t2 = (int(t) for t in slots[v])
    res, num = (
        np.asarray(a).copy() for a in (layout.tile_resource, layout.tile_number)
    )
    res[t0], num[t0] = int(Tile.DESERT), 0
    res[t1] = res[t2] = int(Tile.WOOD)
    num[t1] = num[t2] = 6
    return layout._replace(
        tile_resource=jnp.asarray(res), tile_number=jnp.asarray(num)
    ), state._replace(robber=jnp.asarray(t1, state.robber.dtype))


def test_incidence_features_are_symmetry_equivariant() -> None:
    # The gate. A *fixed* per-vertex hex order (canonical hex index) is NOT
    # D3-equivariant: for symmetry #1, vertex 0's hexes [0, 3, 4] map to
    # [16, 12, 13], while vertex 40 = sigma(0) lists its own as [12, 13, 16] --
    # slot k of v and slot k of sigma(v) hold different hexes. Sorting each
    # vertex's slots by the hexes' own attributes instead is equivariant: a
    # symmetry carries each hex's attributes with it, so the *multiset* of
    # incident payloads at sigma(v) equals the one at v, and sorting a multiset
    # is well defined. Checked on the hostile board (desert, duplicate hexes with
    # the robber on one of them) -- the near-tie, its robber-driven break, and the
    # pad are exactly where a slot-order bug would hide. The robber sits on one of
    # the near-tied hexes (not a fourth, uninvolved one) so the robber term of the
    # sort key is the thing under test, not just an already-broken tie.
    layout, state = _incidence_stress(*_mid_game(4))
    p = jnp.int32(0)
    base = np.asarray(board_sample(layout, state, p, version=2, incidence=True).nodes)
    slots = np.asarray(graph.VERTEX_TILES)
    deg = np.asarray(graph.VERTEX_TILE_PRESENT).sum(1)
    v = int(np.flatnonzero(deg == graph.MAX_VERTEX_TILES)[0])
    rows = _incidence(base)[v]
    # vacuity: the desert slot is present, and exactly one pair agrees on every
    # column except the robber flag -- the near-tie the robber term must break.
    assert rows[:, N_RESOURCES].sum() == 1.0  # exactly one desert slot
    robber_col = N_RESOURCES + 2
    other_cols = [c for c in range(graph.INCIDENT_TILE_DIM) if c != robber_col]
    near_ties = [
        (i, j)
        for i, j in ((0, 1), (0, 2), (1, 2))
        if np.array_equal(rows[i, other_cols], rows[j, other_cols])
    ]
    assert len(near_ties) == 1
    i, j = near_ties[0]
    assert rows[i, robber_col] != rows[j, robber_col]  # robber is what breaks it
    for sym in board_symmetries():
        l2, s2 = apply_symmetry(layout, state, sym)
        rot = np.asarray(board_sample(l2, s2, p, version=2, incidence=True).nodes)
        # per-vertex equivariance, the statement the net-level invariance rests on
        np.testing.assert_allclose(rot[sym.vertices], base, atol=1e-6)
    assert slots.shape == (N_VERTICES, graph.MAX_VERTEX_TILES)


def test_incidence_pads_low_degree_vertices_with_no_hex() -> None:
    # Coast vertices touch 1 or 2 hexes. The pad is an all-zero slot placed last
    # (absent hexes sort after every real key), and it is unambiguous *because*
    # the resource one-hot is 6 wide with the desert as its own column: every real
    # hex sets exactly one of those columns, so no presence flag is needed.
    layout, state = _mid_game(2)
    inc = _incidence(
        np.asarray(
            board_sample(layout, state, jnp.int32(0), version=2, incidence=True).nodes
        )
    )
    deg = np.asarray(graph.VERTEX_TILE_PRESENT).sum(1).astype(int)
    assert sorted(set(deg.tolist())) == [1, 2, 3]  # coast / edge / interior
    for v in range(N_VERTICES):
        assert not inc[v, deg[v] :].any()  # pads last, exactly zero
        real = inc[v, : deg[v]]
        assert np.array_equal(real[:, : N_RESOURCES + 1].sum(1), np.ones(deg[v]))


def test_incidence_dims_and_v2_block_untouched() -> None:
    v2, v2i = graph.dims(2, False), graph.dims(2, True)
    assert graph.INCIDENCE_DIM == graph.MAX_VERTEX_TILES * graph.INCIDENT_TILE_DIM == 57
    assert v2i[0] - v2[0] == graph.INCIDENCE_DIM
    assert (v2i[1], v2i[2], v2i[3]) == (v2[1], v2[2], v2[3])  # nodes only
    # The block is *appended*, so the v2 node columns (and every other array) keep
    # their bytes -- a v2 golden's scope is unaffected by turning incidence on.
    layout, state = _mid_game(2)
    p = jnp.int32(0)
    plain = board_sample(layout, state, p, version=2)
    with_inc = board_sample(layout, state, p, version=2, incidence=True)
    assert np.array_equal(
        np.asarray(with_inc.nodes)[:, : v2[0]], np.asarray(plain.nodes)
    )
    for name in ("edges", "glob", "tiles"):
        assert np.array_equal(
            np.asarray(getattr(plain, name)), np.asarray(getattr(with_inc, name))
        )
    with pytest.raises(ValueError, match="incidence"):
        board_sample(layout, state, p, version=1, incidence=True)


# The v2_incidence study arm's own node block: sha256 of just the appended
# incidence tail (nodes[:, -INCIDENCE_DIM:]) on the canonical `_mid_game`
# fixture. `test_incidence_dims_and_v2_block_untouched` already pins that the
# v2-sans-incidence prefix and every other array are untouched by the flag, so
# this is the one remaining unpinned byte range across the four study arms'
# feature paths -- v1 (`_V1_GOLDEN`), v2-sans-incidence (`_V2_GOLDEN`, feeds
# v2_base/v2_deep/v2_hetero), and now v2_incidence's own tail.
_V2_INCIDENCE_GOLDEN = {
    2: ((54, 57), "986011fbff3b35f7"),
    4: ((54, 57), "2a3b5d0f9de68800"),
}


@pytest.mark.parametrize("n_players", [2, 4])
def test_v2_incidence_block_matches_the_frozen_golden(n_players: int) -> None:
    layout, state = _mid_game(n_players)
    sample = board_sample(layout, state, jnp.int32(0), version=2, incidence=True)
    tail = np.asarray(sample.nodes)[:, -graph.INCIDENCE_DIM :]
    shape, digest = _V2_INCIDENCE_GOLDEN[n_players]
    assert tail.shape == shape and str(tail.dtype) == "float32"
    assert hashlib.sha256(tail.tobytes()).hexdigest()[:16] == digest
