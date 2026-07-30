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
from settlrl_engine.board.state import BoardState
from settlrl_engine.env import N_FLAT, ActionType, BatchedSettlrlEnv
from settlrl_learn.features import features
from settlrl_learn.nn import graph, graphnet
from settlrl_learn.nn.architectures import DeepSetModel, GNNModel, MLPModel
from settlrl_learn.nn.board_gnn import BoardGNN
from settlrl_learn.nn.graph import Sample, board_sample
from settlrl_learn.nn.graphnet import PRESETS, GraphNet
from settlrl_learn.training.gnn_backend import gnn_loss
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
    ("preset", "version"),
    [
        ("gn_multi", 1),
        ("gn_graphnorm", 1),
        ("gn_gat", 1),
        ("gn_full", 1),
        ("gn_hetero", 1),
        ("gn_global", 2),
        ("gn_hetero", 2),
    ],
)
def test_graphnet_presets_are_invariant(preset: str, version: int) -> None:
    # The configurable GraphNet keeps both invariances across every lever
    # (attention, GraphNorm spanning the node axis, the global node, JK) -- it
    # uses only symmetric aggregations and relative features, no absolute PE.
    layout, state = _position(version)
    cfg = PRESETS[preset]._replace(
        width=8, layers=2, head_depth=1, feature_version=version
    )
    model = GraphNet(jax.random.key(0), out_dim=_OUT, cfg=cfg)
    base = np.asarray(model(board_sample(layout, state, jnp.int32(0), version=version)))
    for sym in board_symmetries():
        l2, s2 = apply_symmetry(layout, state, sym)
        rot = np.asarray(model(board_sample(l2, s2, jnp.int32(0), version=version)))
        assert np.allclose(base, rot, atol=1e-3)
    perm = np.array([1, 2, 3, 0])
    relabeled = board_sample(
        layout, relabel_players(state, perm), jnp.int32(perm[0]), version=version
    )
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


def _aznet(preset: str = "gn_global", version: int = 1, net_seed: int = 0) -> BoardGNN:
    cfg = PRESETS[preset]._replace(
        width=16, layers=2, head_depth=1, feature_version=version
    )
    return BoardGNN(jax.random.key(net_seed), cfg)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("preset", ["gn_global", "gn_hetero"])
def test_aznet_value_invariant_policy_equivariant_under_board_symmetry(
    preset: str, version: int
) -> None:
    # The factored value+policy net: the value is invariant under a board
    # symmetry, and the policy is *equivariant* -- a settlement-at-v action maps
    # to settlement-at-(sigma v), road-at-e to road-at-(sigma e), robber-tile-t
    # to sigma(t) -- so policy(sigma . board)[action_permutation] == policy(board).
    layout, state = _position(version)
    net = _aznet(preset, version)
    p = jnp.int32(0)
    vv, pp = net(board_sample(layout, state, p, version=version))
    v0, pol0 = np.asarray(vv), np.asarray(pp)
    for sym in board_symmetries():
        l2, s2 = apply_symmetry(layout, state, sym)
        v, pol = net(board_sample(l2, s2, p, version=version))
        assert np.allclose(np.asarray(v), v0, atol=1e-3)  # value invariant
        perm = action_permutation(sym)
        assert np.allclose(np.asarray(pol)[perm], pol0, atol=1e-3)  # policy equivariant


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("preset", ["gn_global", "gn_hetero"])
def test_aznet_value_and_policy_invariant_under_player_relabel(
    preset: str, version: int
) -> None:
    layout, state = _position(version)
    net = _aznet(preset, version)
    vv, pp = net(board_sample(layout, state, jnp.int32(0), version=version))
    v0, pol0 = np.asarray(vv), np.asarray(pp)
    perm = np.array([1, 2, 3, 0])
    relabeled = board_sample(
        layout, relabel_players(state, perm), jnp.int32(perm[0]), version=version
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
    # `jnp.maximum(sum(train_policy), 1.0)` guard (gnn_backend.py) must keep the
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


def test_feature_version_dims() -> None:
    v1, v2 = graph.dims(1), graph.dims(2)
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
