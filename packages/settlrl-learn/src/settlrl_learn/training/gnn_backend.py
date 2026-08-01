"""The board-GNN backend: a :class:`~settlrl_learn.nn.board_gnn.BoardGNN` over the
board *graph* (``settlrl_learn.nn.graph.board_sample``), with a **masked** policy CE
(the softmax is restricted to the legal set, so the net never spends capacity on
per-position illegality) + value logistic loss. A fixed policy plays the setup
phase; the net trains/acts only on the main loop.

A training-side module (equinox/jraph/optax): not imported by the package root.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float
from settlrl_agents.value import ValueFunction, heuristic_value
from settlrl_engine.board.layout import N_TILES, N_VERTICES, BoardLayout
from settlrl_engine.board.state import BoardState, IntScalar
from settlrl_engine.env import N_FLAT
from settlrl_engine.mechanics.action import ActionType
from settlrl_search import make_search
from settlrl_search.expectimax import make_setup_lookahead, make_setup_search
from settlrl_search.policy import BeliefPolicy, PolicyPrior
from settlrl_search.rows import ROW_TYPE as _ROW_TYPE

from settlrl_learn.nn.board_gnn import BoardGNN, gnn_seams
from settlrl_learn.nn.graph import (
    N_DIR_EDGES,
    Sample,
    board_sample,
    dims,
)
from settlrl_learn.nn.graphnet import GraphNetConfig
from settlrl_learn.training.backend import Metrics, StepFn
from settlrl_learn.training.selfplay import Samples

# Flat rows that are setup placements -- a non-setup legal action marks the main
# loop (the net's search plays it; setup is delegated to a fixed policy).
_SETUP_ROWS = (int(ActionType.SETUP_SETTLEMENT) == _ROW_TYPE) | (
    int(ActionType.SETUP_ROAD) == _ROW_TYPE
)


def setup_policy(
    n_players: int = 2,
    *,
    setup_depth: int = 1,
    setup_temperature: float = 2.0,
    setup_beam: int = 4,
) -> BeliefPolicy:
    """The fixed policy for the setup phase. ``setup_depth <= 1`` is
    ``lookahead(heuristic)`` -- a 1-ply pip-maxing opener; the default, since at
    2p a depth-6 search ties it (the heuristic value is ~additive, so greedy ≈
    optimal pairing) at a fraction of the cost. ``setup_depth >= 2`` switches to
    the probabilistic-expectimax setup search
    (:func:`search.expectimax.make_setup_search`, opponents Boltzmann-rational
    at ``setup_temperature``) -- kept for >= 3 players and complementarity-aware
    values, where the deeper opening may pay off."""
    if setup_depth <= 1:
        return make_setup_lookahead(heuristic_value)
    return make_setup_search(
        heuristic_value, n_players=n_players, depth=setup_depth,
        temperature=setup_temperature, beam=setup_beam,
    )  # fmt: skip


def make_net_agent(
    value_fn: ValueFunction,
    prior_fn: PolicyPrior,
    *,
    num_simulations: int,
    max_num_considered_actions: int,
    n_players: int = 2,
    setup_depth: int = 1,
    setup_temperature: float = 2.0,
    setup_beam: int = 4,
    chance_nodes: bool = False,
    dev_chance: bool = True,
    ordered: bool = False,
) -> BeliefPolicy:
    """The net at play: the setup phase from :func:`setup_policy`, the main loop
    from the net's search. The phase is read off the mask (setup ⇔ the only legal
    actions are placements)."""
    net = make_search(
        value_fn, prior=prior_fn, value_scale=2.0,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        chance_nodes=chance_nodes, dev_chance=dev_chance, ordered=ordered,
    )  # fmt: skip
    setup = setup_policy(
        n_players, setup_depth=setup_depth,
        setup_temperature=setup_temperature, setup_beam=setup_beam,
    )  # fmt: skip

    def policy(
        key: Array, layout: BoardLayout, view: Any, player: IntScalar, mask: Array
    ) -> Array:
        main_legal = (mask & ~_SETUP_ROWS).any()  # a non-setup action is legal
        return jnp.where(
            main_legal,
            net(key, layout, view, player, mask),
            setup(key, layout, view, player, mask),
        )

    return policy


class GNNItem(NamedTuple):
    """One replay item: the board graph (nodes/edges/glob/tiles), the search's
    improved policy, the legality mask, the acting seat's win (1) / loss (0), and
    the ``train_policy`` flag (1 = train the policy head here; 0 = value-only, a
    playout-cap fast-search position). ``tiles`` is the per-hex node features
    (consumed only by the heterogeneous trunk)."""

    nodes: Array
    edges: Array
    glob: Array
    tiles: Array
    policy: Array
    mask: Array
    value: Array
    train_policy: Array


def _masked_logp(logits: Array, mask: Array) -> Array:
    """log-softmax over the legal actions only (illegal -> log-prob 0 weight)."""
    return jax.nn.log_softmax(jnp.where(mask > 0, logits, -jnp.inf), axis=-1)


def gnn_loss(
    model: BoardGNN,
    sample: Sample,
    policy: Array,
    value: Array,
    mask: Array,
    train_policy: Array,
    value_weight: float = 1.0,
) -> tuple[Float[Array, ""], Metrics]:
    """Masked policy cross-entropy (against the search target, over *legal*
    actions) + ``value_weight`` x the value logistic loss. The softmax is masked
    to the legal set so the net never spends capacity on per-position illegality
    (the search masks at play time). The policy CE is averaged over
    ``train_policy`` = 1 positions only (value-only playout-cap positions don't
    train the policy); value trains on all. With ``train_policy`` all 1 this is
    the plain mean."""
    vs, logits = jax.vmap(model)(sample)
    logp = _masked_logp(logits, mask)
    # guard 0 * -inf on illegal slots (target is 0 there anyway).
    ce = -jnp.sum(jnp.where(mask > 0, policy * logp, 0.0), axis=-1)  # per position
    policy_loss = jnp.sum(train_policy * ce) / jnp.maximum(jnp.sum(train_policy), 1.0)
    value_loss = jnp.mean(jax.nn.softplus(vs) - value * vs)
    return policy_loss + value_weight * value_loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }


def _sample_of(item: GNNItem) -> Sample:
    """A batched graph-only :class:`Sample` for the GNN forward (no extra)."""
    return Sample(item.nodes, item.edges, item.glob, item.tiles, None)


class GNNBackend:
    """A :class:`~settlrl_learn.training.backend.Backend` over a ``BoardGNN`` net.

    The setup phase is delegated to a fixed policy (configured here) both during
    self-play and in the arena."""

    def __init__(
        self,
        cfg: GraphNetConfig,
        *,
        value_weight: float = 1.0,
        setup_depth: int = 1,
        setup_temperature: float = 2.0,
        setup_beam: int = 4,
        chance_nodes: bool = False,
        dev_chance: bool = True,
        ordered: bool = False,
    ) -> None:
        self.cfg = cfg
        self.value_weight = value_weight
        self.setup_depth = setup_depth
        self.setup_temperature = setup_temperature
        self.setup_beam = setup_beam
        self.chance_nodes = chance_nodes
        self.dev_chance = dev_chance
        self.ordered = ordered

    def init(self, key: Array) -> BoardGNN:
        return BoardGNN(key, self.cfg)

    def seams(self, net: BoardGNN) -> tuple[ValueFunction, PolicyPrior]:
        return gnn_seams(net)

    def setup_policy(self) -> BeliefPolicy:
        return setup_policy(
            2, setup_depth=self.setup_depth,
            setup_temperature=self.setup_temperature, setup_beam=self.setup_beam,
        )  # fmt: skip

    def play_agent(
        self, net: BoardGNN, *, num_simulations: int, max_num_considered_actions: int
    ) -> BeliefPolicy:
        value_fn, prior_fn = gnn_seams(net)
        return make_net_agent(
            value_fn, prior_fn,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            setup_depth=self.setup_depth,
            setup_temperature=self.setup_temperature, setup_beam=self.setup_beam,
            chance_nodes=self.chance_nodes, dev_chance=self.dev_chance,
            ordered=self.ordered,
        )  # fmt: skip

    def observe(
        self, layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> dict[str, Array]:
        # Tiles are featurized only for a heterogeneous net; else a constant zero,
        # so the non-hetero forward graph has no tile ops.
        s = board_sample(
            layout,
            state,
            player,
            with_tiles=self.cfg.hetero,
            version=self.cfg.feature_version,
            incidence=self.cfg.incidence,
        )
        return {"nodes": s.nodes, "edges": s.edges, "glob": s.glob, "tiles": s.tiles}

    def to_item(self, samples: Samples) -> GNNItem:
        return GNNItem(
            jnp.asarray(samples["nodes"], jnp.float32),
            jnp.asarray(samples["edges"], jnp.float32),
            jnp.asarray(samples["glob"], jnp.float32),
            jnp.asarray(samples["tiles"], jnp.float32),
            jnp.asarray(samples["policy"], jnp.float32),
            jnp.asarray(samples["mask"], jnp.float32),
            jnp.asarray(samples["value"], jnp.float32),
            jnp.asarray(samples["train_policy"], jnp.float32),
        )

    def empty_item(self) -> GNNItem:
        node_dim, edge_dim, global_dim, tile_dim = dims(
            self.cfg.feature_version, self.cfg.incidence
        )
        return GNNItem(
            jnp.zeros((N_VERTICES, node_dim), jnp.float32),
            jnp.zeros((N_DIR_EDGES, edge_dim), jnp.float32),
            jnp.zeros((global_dim,), jnp.float32),
            jnp.zeros((N_TILES, tile_dim), jnp.float32),
            jnp.zeros((N_FLAT,), jnp.float32),
            jnp.zeros((N_FLAT,), jnp.float32),
            jnp.float32(0.0),
            jnp.float32(1.0),
        )

    def init_opt(
        self, optimizer: optax.GradientTransformation, net: BoardGNN
    ) -> optax.OptState:
        return optimizer.init(eqx.filter(net, eqx.is_inexact_array))

    def make_step(self, optimizer: optax.GradientTransformation) -> StepFn:
        value_weight = self.value_weight

        @eqx.filter_jit
        def step(
            net: BoardGNN, opt_state: optax.OptState, item: GNNItem
        ) -> tuple[BoardGNN, optax.OptState, Metrics]:
            (loss, aux), grads = eqx.filter_value_and_grad(_loss_item, has_aux=True)(
                net, item, value_weight
            )
            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(net, eqx.is_inexact_array)
            )
            net = eqx.apply_updates(net, updates)
            return net, opt_state, {
                "loss": loss, "grad_norm": optax.global_norm(grads),
                "update_norm": optax.global_norm(updates), **aux,
            }  # fmt: skip

        return step

    def eval_metrics(self, net: BoardGNN, item: GNNItem) -> Metrics:
        return _eval(net, item)


def _loss_item(
    net: BoardGNN, item: GNNItem, value_weight: float
) -> tuple[Float[Array, ""], Metrics]:
    return gnn_loss(
        net, _sample_of(item), item.policy, item.value, item.mask,
        item.train_policy, value_weight,
    )  # fmt: skip


@eqx.filter_jit
def _eval(net: BoardGNN, item: GNNItem) -> Metrics:
    vs, logits = jax.vmap(net)(_sample_of(item))
    msk = item.mask
    logp = _masked_logp(logits, msk)  # over legal actions only
    p = jnp.where(msk > 0, jnp.exp(logp), 0.0)
    legal = jnp.where(msk > 0, p * logp, 0.0)
    return {
        "val_policy_loss": -jnp.mean(
            jnp.sum(jnp.where(msk > 0, item.policy * logp, 0.0), -1)
        ),
        "val_value_loss": jnp.mean(jax.nn.softplus(vs) - item.value * vs),
        "val_value_acc": jnp.mean((vs > 0).astype(jnp.float32) == item.value),
        # policy-head health: legal-set entropy (collapse -> ~0) + top prob.
        "policy_entropy": -jnp.mean(jnp.sum(legal, axis=-1)),
        "policy_top_prob": jnp.mean(jnp.max(p, axis=-1)),
        # value-head health: logit spread + mean predicted P(win) (~0.5 sane).
        "value_logit_mean": jnp.mean(vs),
        "value_logit_std": jnp.std(vs),
        "pred_winrate": jnp.mean(jax.nn.sigmoid(vs)),
        "value_label_mean": jnp.mean(item.value),
    }
