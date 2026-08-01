"""Architectures over a :class:`~settlrl_learn.nn.graph.Sample`, in equinox.

All four read the *same* position and emit ``out_dim`` logits/regressands, so a
benchmark isolates representation x architecture:

- ``mlp_engineered`` -- MLP over the hand-tuned feature vector (the baseline to
  beat);
- ``mlp_flat`` -- MLP over the flattened graph (sees every node feature in fixed
  vertex order, no structure);
- ``deepset`` -- permutation-invariant pool over node features + globals (set, no
  edges);
- ``gnn`` -- message passing over the board graph (jraph ``GraphNetwork``), then a
  global readout (uses topology and edge ownership);
- the configurable :class:`~settlrl_learn.nn.graphnet.GraphNet` presets
  (``gn_*`` in ``graphnet.PRESETS``) -- the strong, ablatable graph net (attention
  vs message passing, normalisation, global node, count-preserving readout).

The forward takes one un-batched ``Sample``; callers ``vmap`` it over a
minibatch. ``deepset`` and ``gnn`` are invariant under board symmetries and the
player relabeling (the readout pools over nodes / reads ownership relatively);
``mlp_flat`` is not, by design -- see ``tests/test_architectures.py``.

A training-side module (equinox/jraph): not imported by the package root.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
from jaxtyping import Array, Float
from settlrl_engine.board.layout import N_VERTICES
from settlrl_engine.board.state import KeyScalar

from settlrl_learn.features import FEATURE_DIM
from settlrl_learn.nn.graph import (
    N_DIR_EDGES,
    RECEIVERS,
    SENDERS,
    Sample,
    dims,
)
from settlrl_learn.nn.graphnet import PRESETS, GraphNet

Model = Callable[[Sample], Float[Array, "out"]]


class MLPModel(eqx.Module):
    net: eqx.nn.MLP
    engineered: bool = eqx.field(static=True)

    def __init__(
        self,
        key: KeyScalar,
        *,
        out_dim: int,
        width: int,
        depth: int,
        engineered: bool,
        feature_version: int = 1,
        incidence: bool = False,
    ) -> None:
        node_dim, _, global_dim, _ = dims(feature_version, incidence)
        in_dim = FEATURE_DIM if engineered else N_VERTICES * node_dim + global_dim
        self.net = eqx.nn.MLP(in_dim, out_dim, width, depth, key=key)
        self.engineered = engineered

    def __call__(self, s: Sample) -> Float[Array, "out"]:
        if self.engineered:
            assert s.extra is not None, (
                "engineered MLP needs board_sample(features=...)"
            )
            x = s.extra
        else:
            x = jnp.concatenate([s.nodes.reshape(-1), s.glob])
        return self.net(x)


class DeepSetModel(eqx.Module):
    phi: eqx.nn.MLP  # per-node encoder
    rho: eqx.nn.MLP  # head over pooled nodes + globals

    def __init__(
        self,
        key: KeyScalar,
        *,
        out_dim: int,
        width: int,
        depth: int,
        feature_version: int = 1,
        incidence: bool = False,
    ) -> None:
        k1, k2 = jax.random.split(key)
        node_dim, _, global_dim, _ = dims(feature_version, incidence)
        self.phi = eqx.nn.MLP(node_dim, width, width, depth, key=k1)
        self.rho = eqx.nn.MLP(width + global_dim, out_dim, width, depth, key=k2)

    def __call__(self, s: Sample) -> Float[Array, "out"]:
        h = jax.vmap(self.phi)(s.nodes).mean(axis=0)
        return self.rho(jnp.concatenate([h, s.glob]))


class _GNNLayer(eqx.Module):
    edge_mlp: eqx.nn.MLP
    node_mlp: eqx.nn.MLP

    def __init__(self, key: KeyScalar, hidden: int) -> None:
        k1, k2 = jax.random.split(key)
        # edge update sees [edge, sender, receiver, global]; node update sees
        # [node, aggregated-received-edges, global] (all width `hidden`).
        self.edge_mlp = eqx.nn.MLP(4 * hidden, hidden, hidden, 1, key=k1)
        self.node_mlp = eqx.nn.MLP(3 * hidden, hidden, hidden, 1, key=k2)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        def update_edge_fn(
            edges: Array, senders: Array, receivers: Array, globals_: Array
        ) -> Array:
            x = jnp.concatenate([edges, senders, receivers, globals_], axis=-1)
            return edges + jax.vmap(self.edge_mlp)(x)  # residual

        def update_node_fn(
            nodes: Array, sent: Array, received: Array, globals_: Array
        ) -> Array:
            x = jnp.concatenate([nodes, received, globals_], axis=-1)
            return nodes + jax.vmap(self.node_mlp)(x)  # residual

        net = jraph.GraphNetwork(update_edge_fn, update_node_fn)
        return net(graph)


class GNNModel(eqx.Module):
    node_enc: eqx.nn.Linear
    edge_enc: eqx.nn.Linear
    glob_enc: eqx.nn.Linear
    layers: tuple[_GNNLayer, ...]
    head: eqx.nn.MLP

    def __init__(
        self,
        key: KeyScalar,
        *,
        out_dim: int,
        width: int,
        depth: int,
        layers: int,
        feature_version: int = 1,
        incidence: bool = False,
    ) -> None:
        keys = jax.random.split(key, 4 + layers)
        node_dim, edge_dim, global_dim, _ = dims(feature_version, incidence)
        self.node_enc = eqx.nn.Linear(node_dim, width, key=keys[0])
        self.edge_enc = eqx.nn.Linear(edge_dim, width, key=keys[1])
        self.glob_enc = eqx.nn.Linear(global_dim, width, key=keys[2])
        self.layers = tuple(_GNNLayer(keys[4 + i], width) for i in range(layers))
        # readout: mean-pooled nodes + final globals.
        self.head = eqx.nn.MLP(2 * width, out_dim, width, depth, key=keys[3])

    def __call__(self, s: Sample) -> Float[Array, "out"]:
        graph = jraph.GraphsTuple(
            nodes=jax.vmap(self.node_enc)(s.nodes),
            edges=jax.vmap(self.edge_enc)(s.edges),
            senders=SENDERS,
            receivers=RECEIVERS,
            globals=self.glob_enc(s.glob)[None, :],
            n_node=jnp.asarray([N_VERTICES]),
            n_edge=jnp.asarray([N_DIR_EDGES]),
        )
        for layer in self.layers:
            graph = layer(graph)
        pooled = jnp.concatenate([graph.nodes.mean(axis=0), graph.globals[0]])
        return self.head(pooled)


def make_model(
    arch: str,
    key: KeyScalar,
    *,
    out_dim: int,
    width: int,
    depth: int,
    layers: int,
    feature_version: int = 1,
    incidence: bool = False,
) -> eqx.Module:
    """Build the named architecture: a fixed baseline (``mlp_engineered`` /
    ``mlp_flat`` / ``deepset`` / ``gnn``) or a configurable ``GraphNet`` preset
    (``settlrl_learn.nn.graphnet.PRESETS``), whose ``width`` / ``layers`` /
    ``head_depth`` are taken from ``width`` / ``layers`` / ``depth`` here so the
    experiment's capacity knobs apply across presets. ``feature_version`` /
    ``incidence`` size the model for that feature set (``graph.dims``);
    ``mlp_engineered`` reads ``s.extra`` and is feature-set independent."""
    if arch == "mlp_engineered":
        return MLPModel(key, out_dim=out_dim, width=width, depth=depth, engineered=True)
    if arch == "mlp_flat":
        return MLPModel(
            key, out_dim=out_dim, width=width, depth=depth, engineered=False,
            feature_version=feature_version, incidence=incidence,
        )  # fmt: skip
    if arch == "deepset":
        return DeepSetModel(
            key, out_dim=out_dim, width=width, depth=depth,
            feature_version=feature_version, incidence=incidence,
        )  # fmt: skip
    if arch == "gnn":
        return GNNModel(
            key, out_dim=out_dim, width=width, depth=depth, layers=layers,
            feature_version=feature_version, incidence=incidence,
        )  # fmt: skip
    if arch in PRESETS:
        cfg = PRESETS[arch]._replace(
            width=width, layers=layers, head_depth=depth,
            feature_version=feature_version, incidence=incidence,
        )  # fmt: skip
        return GraphNet(key, out_dim=out_dim, cfg=cfg)
    raise SystemExit(f"unknown arch {arch!r}")
