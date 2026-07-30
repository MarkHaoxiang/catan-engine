"""The search over the engine: re-determinizing Single-Observer ISMCTS.

The public wrapper around :func:`make_tree`: it assembles the root prior (the
one-step value sweep, a learned ``prior``, or trade-scored proposals), runs the
``num_simulations=0`` *lookahead* special case (the bare root sweep, the only
configuration that offers trades), and averages ``num_trees`` independent trees.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, KeyScalar, Player
from settlrl_engine.env import N_FLAT
from settlrl_engine.mechanics.action import (
    ActionParams,
    ActionType,
    action_available,
    apply_action,
)
from settlrl_engine.mechanics.trade import _PARTNER_BITS

from settlrl_search.policy import (
    BeliefPolicy,
    FlatAction,
    FlatMask,
    PolicyPrior,
    ValuePrior,
)
from settlrl_search.rows import ROW_PARAMS as _ROW_PARAMS
from settlrl_search.rows import ROW_TYPE as _ROW_TYPE
from settlrl_search.sample import sample_world
from settlrl_search.value import Value, ValueFunction

from ._common import (
    _ILLEGAL,
    _NO_PROPOSE,
    PolicyWeights,
    PolicyWeightsValue,
    _Weights,
)
from .ismcts import make_tree

__all__ = [
    "PolicyWeights",
    "PolicyWeightsValue",
    "ValuePrior",
    "make_search",
    "make_search_weights",
    "make_search_weights_value",
    "make_tree",
]

# Trade-proposal rows and their partners (the low bits of the packed target —
# see trade.pack_trade): a proposal's successor is material-neutral (cards move
# only on the partner's accept), so the root scores them by their *accepted*
# outcome under a partner model.
_PROPOSE_ROWS = jnp.asarray(
    np.flatnonzero(np.asarray(_ROW_TYPE) == int(ActionType.PROPOSE_TRADE))
)
_PROPOSE_PARTNER = _ROW_PARAMS.target[_PROPOSE_ROWS] & ((1 << _PARTNER_BITS) - 1)
_ACCEPT = jnp.int32(ActionType.ACCEPT_TRADE)
_NO_PARAMS = ActionParams(idx=jnp.int32(0), target=jnp.int32(0))


def make_search_weights_value(
    value: ValueFunction,
    *,
    prior: PolicyPrior | None = None,
    num_trees: int = 1,
    num_simulations: int = 32,
    max_depth: int = 12,
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
    propose_rate: float = 0.0,
    trade_penalty: float = 0.25,
    expected_rolls: bool = True,
    chance_nodes: bool = False,
    dev_chance: bool = True,
    ordered: bool = False,
    fused_leaf: bool = True,
) -> PolicyWeightsValue:
    """Re-determinizing SO-ISMCTS, returning ``(improved-policy weights, root
    value)`` — the AlphaZero policy *and* value (``q``) targets. The root value is
    the searched estimate in the searcher's frame (2·P(win)-1), averaged over the
    ``num_trees`` trees alongside the weights.

    ``value`` drives the leaf and, when ``prior`` is None, the root one-step
    sweep; a ``prior`` replaces both the root and interior priors with learned
    logits. A :class:`~settlrl_search.policy.ValuePrior` ``prior`` also *supplies
    the leaf value* from its own head (one forward for both); ``fused_leaf=False``
    keeps ``value`` in that role, which is what an unpaired ``value``/``prior``
    wants. ``num_trees`` independent trees are averaged. ``num_simulations=0`` is
    the *lookahead* special case (the bare root sweep, root value = the best legal
    successor value). ``propose_rate`` > 0 lets the root offer trades, scored by
    their accepted outcome under a partner model minus ``trade_penalty``; offers
    are root-only.
    """
    tree = make_tree(
        value,
        prior,
        num_simulations=num_simulations,
        max_depth=max_depth,
        max_considered=max_num_considered_actions,
        value_scale=value_scale,
        expected_rolls=expected_rolls,
        chance_nodes=chance_nodes,
        dev_chance=dev_chance,
        ordered=ordered,
        fused_leaf=fused_leaf,
    )

    def value_sweep(
        layout: BoardLayout, world: BoardState, player: Player, mask: FlatMask
    ) -> tuple[_Weights, BoardState]:
        """Per-row successor values (and the successors) over one concrete
        world: the one-step value sweep."""
        successors, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
            layout, world, _ROW_TYPE, _ROW_PARAMS, mask
        )
        values = jax.vmap(value, in_axes=(None, 0, None))(layout, successors, player)
        return values, successors

    def root_logits(
        key: KeyScalar,
        layout: BoardLayout,
        world: BoardState,
        player: Player,
        mask: FlatMask,
    ) -> _Weights:
        """The root prior over one concrete world: the one-step value sweep
        (proposals excluded), optionally with trade offers priced in."""
        if prior is not None:
            return prior(layout, world, player)
        values, successors = value_sweep(layout, world, player, mask)
        logits = values / prior_scale + _NO_PROPOSE
        if propose_rate <= 0.0:
            return logits
        # Score each proposal by its accepted outcome, if the modeled partner
        # would take it and proposing is allowed this move (see trade_penalty).
        offered = jax.tree.map(lambda x: x[_PROPOSE_ROWS], successors)
        accepted = jax.vmap(_accepted_outcome, in_axes=(None, 0))(layout, offered)
        mine = jax.vmap(value, in_axes=(None, 0, None))(layout, accepted, player)
        partner_reject = jax.vmap(value, in_axes=(None, 0, 0))(
            layout, offered, _PROPOSE_PARTNER
        )
        partner_accept = jax.vmap(value, in_axes=(None, 0, 0))(
            layout, accepted, _PROPOSE_PARTNER
        )
        allowed = jax.random.uniform(key) < propose_rate
        propose_score = jnp.where(
            (partner_accept > partner_reject) & allowed, mine - trade_penalty, _ILLEGAL
        )
        return logits.at[_PROPOSE_ROWS].set(propose_score / prior_scale)

    def search_tree(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: Player,
        mask: FlatMask,
    ) -> tuple[_Weights, Value]:
        """Improved-policy weights + root value from one re-determinizing tree."""
        k_root, k_gate, k_search = jax.random.split(key, 3)
        # The root prior + value use one sampled world (the prior only orders the
        # first expansions; per-simulation resampling carries the rest).
        root_world = sample_world(k_root, view, player)
        logits = jnp.where(
            mask, root_logits(k_gate, layout, root_world, player, mask), _ILLEGAL
        )
        return tree(k_search, layout, view, player, mask.astype(jnp.float32), logits)

    def lookahead(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: Player,
        mask: FlatMask,
    ) -> tuple[_Weights, Value]:
        """The root one-step sweep over one sampled world (num_simulations=0); the
        root value is the best legal successor value mapped to [-1, 1]."""
        k_world, k_gate = jax.random.split(key)
        world = sample_world(k_world, view, player)
        w = jnp.where(mask, root_logits(k_gate, layout, world, player, mask), _ILLEGAL)
        if prior is None:
            # w is value/prior_scale on the non-proposal rows; recover the value.
            best = jnp.max(jnp.where(mask, w, -jnp.inf)) * prior_scale
        else:
            # A learned prior's logits are not values, so value the successors
            # directly (proposals excluded, as in the prior-less sweep).
            values, _ = value_sweep(layout, world, player, mask)
            best = jnp.max(jnp.where(mask, values + _NO_PROPOSE, -jnp.inf))
        return w, jnp.tanh(best / value_scale)

    def weights(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: Player,
        mask: FlatMask,
    ) -> tuple[_Weights, Value]:
        leaf = lookahead if num_simulations == 0 else search_tree
        w, v = jax.vmap(leaf, in_axes=(0, None, None, None, None))(
            jax.random.split(key, num_trees), layout, view, player, mask
        )
        return w.mean(axis=0), v.mean(axis=0)

    return weights


def make_search_weights(
    value: ValueFunction,
    *,
    prior: PolicyPrior | None = None,
    num_trees: int = 1,
    num_simulations: int = 32,
    max_depth: int = 12,
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
    propose_rate: float = 0.0,
    trade_penalty: float = 0.25,
    expected_rolls: bool = True,
    chance_nodes: bool = False,
    dev_chance: bool = True,
    ordered: bool = False,
    fused_leaf: bool = True,
) -> PolicyWeights:
    """The improved-policy weights alone (the AlphaZero policy target;
    :func:`make_search` argmaxes these) — :func:`make_search_weights_value` with
    the root value dropped. Parameters are identical, including ``fused_leaf``:
    a :class:`~settlrl_search.policy.ValuePrior` prior's value head scores the
    leaf unless it is off."""
    wv = make_search_weights_value(
        value,
        prior=prior,
        num_trees=num_trees,
        num_simulations=num_simulations,
        max_depth=max_depth,
        max_num_considered_actions=max_num_considered_actions,
        value_scale=value_scale,
        prior_scale=prior_scale,
        propose_rate=propose_rate,
        trade_penalty=trade_penalty,
        expected_rolls=expected_rolls,
        chance_nodes=chance_nodes,
        dev_chance=dev_chance,
        ordered=ordered,
        fused_leaf=fused_leaf,
    )

    def weights(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: Player,
        mask: FlatMask,
    ) -> _Weights:
        return wv(key, layout, view, player, mask)[0]

    return weights


def make_search(
    value: ValueFunction,
    *,
    prior: PolicyPrior | None = None,
    num_trees: int = 1,
    num_simulations: int = 32,
    max_depth: int = 12,
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
    propose_rate: float = 0.0,
    trade_penalty: float = 0.25,
    expected_rolls: bool = True,
    chance_nodes: bool = False,
    dev_chance: bool = True,
    ordered: bool = False,
    fused_leaf: bool = True,
) -> BeliefPolicy:
    """Re-determinizing search as a :class:`BeliefPolicy`: the masked argmax of
    the improved policy. Parameters are :func:`make_search_weights`'; tiny noise
    breaks the lookahead sweep's exact-value ties. A
    :class:`~settlrl_search.policy.ValuePrior` prior's value head scores the leaf
    unless ``fused_leaf`` is off."""
    weights = make_search_weights(
        value,
        prior=prior,
        num_trees=num_trees,
        num_simulations=num_simulations,
        max_depth=max_depth,
        max_num_considered_actions=max_num_considered_actions,
        value_scale=value_scale,
        prior_scale=prior_scale,
        propose_rate=propose_rate,
        trade_penalty=trade_penalty,
        expected_rolls=expected_rolls,
        chance_nodes=chance_nodes,
        dev_chance=dev_chance,
        ordered=ordered,
        fused_leaf=fused_leaf,
    )

    def policy(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: Player,
        mask: FlatMask,
    ) -> FlatAction:
        noise = jax.random.uniform(key, (N_FLAT,)) * 1e-4
        w = weights(key, layout, view, player, mask)
        return jnp.argmax(jnp.where(mask, w + noise, -jnp.inf))

    return policy


def _accepted_outcome(layout: BoardLayout, succ: BoardState) -> BoardState:
    """The state after the partner accepts ``succ``'s pending proposal.

    ``succ`` unchanged when accepting is illegal there (no pending trade, or
    the partner lacks the asked-for card in this world).
    """
    avail = action_available(layout, succ, _ACCEPT, _NO_PARAMS)
    accepted, _ = apply_action(layout, succ, _ACCEPT, _NO_PARAMS, avail)
    return accepted
