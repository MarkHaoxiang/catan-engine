"""The search orchestration (:func:`_run`) and the :func:`make_tree` factory."""

from __future__ import annotations

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import IntScalar, KeyScalar, Player
from settlrl_engine.env import N_FLAT

from settlrl_search._common import _Weights
from settlrl_search.policy import PolicyPrior
from settlrl_search.sample import sample_world
from settlrl_search.value import Value, ValueFunction

from ._types import TreeSearch, _LegalMask, _PriorLogits
from .config import (
    SearchConfig,
    _Cfg,
    _considered_table,
    _count_dtype,
    _node_dtype,
)
from .descent import (
    _descend,
    _determinize,
    _evaluate,
    _legal_mask,
    _value,
)
from .tree import _backup, _completed_q, _expand, _root_select, _Tree

# --- the search: one re-determinizing tree, built over the engine ---


def _run(
    cfg: _Cfg,
    key: KeyScalar,
    layout: BoardLayout,
    view: BeliefView,
    player: Player,
    mask: _LegalMask,
    root_logits: _PriorLogits,
) -> tuple[_Weights, Value]:
    player = player.astype(jnp.int32)
    key, k_gumbel = jax.random.split(key)
    keys = jax.random.split(key, cfg.num_simulations + 1)
    # Root mover is the searcher and its legal set is invariant to the hidden state
    # (its own move), so `mask` fixes the candidate set for halving; one
    # determinization gives the searcher-frame root value for the Q mix.
    root_legal = mask.astype(jnp.float32)
    root_value = _value(cfg, layout, sample_world(keys[0], view, player), player)
    gumbel = jax.random.gumbel(k_gumbel, (N_FLAT,))
    num_considered = jnp.minimum(
        cfg.max_considered, (root_legal > 0).sum().astype(jnp.int32)
    )
    node_dtype = _node_dtype(cfg.n_nodes)
    count_dtype = _count_dtype(cfg.num_simulations)
    tree = _Tree(
        mover=jnp.zeros((cfg.n_nodes,), jnp.int8).at[0].set(player.astype(jnp.int8)),
        children=-jnp.ones((cfg.n_nodes, N_FLAT), node_dtype),
        n=jnp.zeros((cfg.n_nodes, N_FLAT), count_dtype),
        w=jnp.zeros((cfg.n_nodes, N_FLAT), jnp.float32),
        prior=jnp.zeros((cfg.n_nodes, N_FLAT), jnp.float32).at[0].set(root_logits),
        raw=jnp.zeros((cfg.n_nodes,), jnp.float32).at[0].set(root_value),
        kind=jnp.zeros((cfg.n_nodes,), jnp.int8),  # root + all nodes default DECISION
        size=jnp.int32(1),
    )

    def simulate(s: IntScalar, tree: _Tree) -> _Tree:
        # DETERMINIZE: sample a world and seed the root descent over it.
        walk = _determinize(cfg, keys[s + 1], layout, view, player)

        # SELECT (root): the Sequential-Halving action for this simulation.
        world_legal = _legal_mask(layout, walk.state)
        a_root = _root_select(
            tree, cfg, gumbel, num_considered, root_legal, world_legal, player
        )

        # SELECT (descend): walk to an unexpanded leaf (no value work in the loop).
        walk = _descend(cfg, tree, a_root, walk, layout, player)

        # EVALUATE: score the leaf once, here, not per descent step — value and
        # the node's interior prior from one evaluation of the leaf state.
        value, mover, leaf_prior = _evaluate(cfg, layout, walk, player)

        # EXPAND: attach the new leaf node (a no-op when the descent grew no node).
        tree = _expand(tree, walk, value, mover, leaf_prior)

        # BACKUP: add the leaf value to every edge on the path.
        return _backup(tree, walk, value, cfg.max_depth)

    tree = cast(_Tree, jax.lax.fori_loop(0, cfg.num_simulations, simulate, tree))
    # action_weights = softmax(root_logits + completed_Q) over the legal set.
    cq, legal_logits = _completed_q(tree, jnp.int32(0), root_legal, player)
    weights = jax.nn.softmax(legal_logits + cq)
    # root value = visit-weighted mean of the root edges' backed-up values (the
    # root mover is the searcher, so w[0] is already in the searcher's frame).
    root_q = tree.w[0].sum() / jnp.maximum(tree.n[0].astype(jnp.float32).sum(), 1.0)
    return weights, root_q


# --- factory ---


def make_tree(
    value: ValueFunction,
    prior: PolicyPrior | None,
    *,
    num_simulations: int,
    max_depth: int,
    max_considered: int,
    value_scale: float,
    expected_rolls: bool = True,
    chance_nodes: bool = False,
    dev_chance: bool = True,
    ordered: bool = False,
) -> TreeSearch:
    """Build one SO-ISMCTS tree (a :data:`TreeSearch` over :func:`_run`).
    ``prior`` (when given) is the *interior* node prior — a learned policy head;
    otherwise interior nodes use the tier table. The root prior is supplied per
    call (``root_logits``). The returned function is pure (the caller
    ``jit``/``vmap``s it).

    ``expected_rolls`` scores a dice-edge leaf by the exact 11-roll expectation
    (variance reduction at 11x the value-fn calls); False uses the single sampled
    post-roll state already produced by the step (much cheaper, noisier leaf).

    ``chance_nodes`` resolves stochastic transitions through explicit chance nodes
    in the tree (dice always, dev-card buys when ``dev_chance``) — nature's move
    sampled at its true probability and applied via the engine's forced-outcome
    seam, so the search can plan *past* a roll. It supersedes ``expected_rolls``
    (a leaf-only roll EV), so the two are mutually exclusive."""
    sc = SearchConfig(
        num_simulations=num_simulations,
        max_depth=max_depth,
        max_considered=max_considered,
        value_scale=value_scale,
        expected_rolls=expected_rolls,
        chance_nodes=chance_nodes,
        dev_chance=dev_chance,
        ordered=ordered,
    )
    cfg = _Cfg(
        value=value,
        prior=prior,
        num_simulations=sc.num_simulations,
        max_depth=sc.max_depth,
        max_considered=sc.max_considered,
        value_scale=sc.value_scale,
        expected_rolls=sc.expected_rolls,
        chance_nodes=sc.chance_nodes,
        dev_chance=sc.dev_chance,
        ordered=sc.ordered,
        n_nodes=sc.num_simulations + 1,
        table=jnp.asarray(_considered_table(sc.max_considered, sc.num_simulations)),
    )
    return cast(TreeSearch, partial(_run, cfg))
