"""The board value+policy GNN: a shared :class:`~settlrl_learn.nn.graphnet.GraphTrunk`
feeding a value head (win-prob logit, ``value_scale=2``) and a structure-factored
policy head. The training loop (:mod:`settlrl_learn.training`) and the search seams
(:func:`gnn_seams`) build on it.

Training-side (equinox/jraph): not imported by the package root.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from settlrl_agents.value import Value, ValueFunction
from settlrl_engine.board.layout import EDGE_V, TILE_V, BoardLayout
from settlrl_engine.board.state import BoardState, IntScalar
from settlrl_engine.env import N_FLAT
from settlrl_search.policy import ValuePrior

from settlrl_learn.nn import action_layout as al
from settlrl_learn.nn.graph import Sample, board_sample
from settlrl_learn.nn.graphnet import GraphNetConfig, GraphTrunk, readout_dim

# undirected-edge endpoints and tile corner-vertices, for the spatial heads.
_EDGE_U, _EDGE_W = EDGE_V[:, 0], EDGE_V[:, 1]
_TILE_V = jnp.asarray(TILE_V)
_SCATTER = jnp.asarray(al.SCATTER)
_TYPE_ID = jnp.asarray(al.TYPE_ID)


class _FactoredPolicy(eqx.Module):
    """A structure-aware policy head: spatial actions get their logit from the
    matching board embedding (vertex / undirected-edge / tile), the rest from a
    dense head, plus a per-type bias (the class-balance knob). Equivariant under
    board symmetry (shared per-slot heads), invariant under player relabeling."""

    vertex: eqx.nn.Linear  # node emb -> {setup-settle, settle, city}
    edge: eqx.nn.Linear  # symmetric endpoint emb -> {setup-road, road}
    tile: eqx.nn.Linear  # mean corner emb -> {robber, knight} x {no-steal, steal}
    other: eqx.nn.MLP  # global ctx -> the non-spatial actions
    type_bias: eqx.nn.Linear  # global ctx -> per-type bias

    def __init__(self, key: Array, cfg: GraphNetConfig, ctx_dim: int) -> None:
        w = cfg.width
        ks = jax.random.split(key, 5)
        self.vertex = eqx.nn.Linear(w, al.N_VCLASS, key=ks[0])
        self.edge = eqx.nn.Linear(w, al.N_ECLASS, key=ks[1])
        self.tile = eqx.nn.Linear(w, al.N_TCLASS, key=ks[2])
        self.other = eqx.nn.MLP(ctx_dim, al.N_OTHER, w, cfg.head_depth, key=ks[3])
        self.type_bias = eqx.nn.Linear(ctx_dim, al.N_TYPES, key=ks[4])

    def __call__(
        self,
        h: Float[Array, "v w"],
        ctx: Float[Array, "ctx"],
        h_t: Float[Array, "t w"] | None = None,
    ) -> Float[Array, f"flat={N_FLAT}"]:
        v = jax.vmap(self.vertex)(h)  # (V, N_VCLASS)
        e = jax.vmap(self.edge)(h[_EDGE_U] + h[_EDGE_W])  # (E, N_ECLASS), symmetric
        # hetero: tile logits come from the hex embeddings; else pool the corners.
        if h_t is not None:
            t = jax.vmap(self.tile)(h_t)  # (T, N_TCLASS)
        else:
            t = jax.vmap(self.tile)(h[_TILE_V].mean(axis=1))  # (T, N_TCLASS)
        big = jnp.concatenate(
            [v.reshape(-1), e.reshape(-1), t.reshape(-1), self.other(ctx)]
        )
        return big[_SCATTER] + self.type_bias(ctx)[_TYPE_ID]


class BoardGNN(eqx.Module):
    """The board value+policy net: a shared :class:`GraphTrunk`, then the heads
    **split right after message passing** -- a value head (pooled readout ->
    win-prob logit, ``value_scale=2``) and the structure-factored policy head
    (:class:`_FactoredPolicy`). Returns ``(value_logit, policy_logits)``."""

    trunk: GraphTrunk
    value: eqx.nn.MLP
    policy: _FactoredPolicy

    def __init__(self, key: Array, cfg: GraphNetConfig) -> None:
        kt, kv, kp = jax.random.split(key, 3)
        ctx_dim = readout_dim(cfg) + cfg.width
        self.trunk = GraphTrunk(kt, cfg)
        self.value = eqx.nn.MLP(ctx_dim, 1, cfg.width, cfg.head_depth, key=kv)
        self.policy = _FactoredPolicy(kp, cfg, ctx_dim)

    def __call__(self, s: Sample) -> tuple[Value, Float[Array, f"flat={N_FLAT}"]]:
        h, g, readout, h_t = self.trunk(s)
        ctx = jnp.concatenate([readout, g])
        return self.value(ctx)[0], self.policy(h, ctx, h_t)


@dataclasses.dataclass(frozen=True)
class _GNNPrior:
    """The GNN's policy seam as a :class:`~settlrl_search.policy.ValuePrior`, so
    the search takes both heads off one forward where it needs both (the leaf)."""

    model: BoardGNN
    het: bool
    version: int = 1

    def _sample(
        self, layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> Sample:
        return board_sample(
            layout, state, player, with_tiles=self.het, version=self.version
        )

    def __call__(
        self, layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> Float[Array, f"flat={N_FLAT}"]:
        return self.model(self._sample(layout, state, player))[1]

    def with_value(
        self, layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> tuple[Value, Float[Array, f"flat={N_FLAT}"]]:
        return self.model(self._sample(layout, state, player))


def gnn_seams(model: BoardGNN) -> tuple[ValueFunction, ValuePrior]:
    """Adapt the GNN onto the search seams as ``(value, prior)``; both run the
    board-graph forward. Build the search with ``value_scale=2``. Tiles are
    featurized only for a heterogeneous net (else the trunk ignores them, so we
    keep them out of the graph); the featurization version is the net's own."""
    het = model.trunk.tile_enc is not None
    version = model.trunk.cfg.feature_version

    def value(layout: BoardLayout, state: BoardState, player: IntScalar) -> Value:
        return model(
            board_sample(layout, state, player, with_tiles=het, version=version)
        )[0]

    return value, _GNNPrior(model, het, version)
