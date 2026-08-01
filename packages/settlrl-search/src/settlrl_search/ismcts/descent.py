"""The simulation's forward walk: determinize, descend (decision and chance
state machines), and evaluate the leaf — plus the engine-interface facts the
walk reads off one determinized state."""

from __future__ import annotations

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.dev_cards import N_DEV_CARD_TYPES
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import (
    BoardState,
    BoolScalar,
    IntScalar,
    KeyScalar,
    Player,
)
from settlrl_engine.env import flat_to_action
from settlrl_engine.mechanics.action import (
    ActionParams,
    ActionType,
    action_available,
    apply_action,
)
from settlrl_engine.mechanics.awards import current_player_won
from settlrl_engine.mechanics.common import agent_selection_single
from settlrl_engine.mechanics.dice import distribute_resources
from settlrl_engine.mechanics.flat import flat_available_for
from settlrl_engine.ordering import next_category, ordering_mask

from settlrl_search._common import _ILLEGAL, _ROLL_P, _ROLLS, _TIER_LOGITS
from settlrl_search.policy import ValuePrior
from settlrl_search.rows import ROW_TYPE
from settlrl_search.sample import sample_world
from settlrl_search.value import Value

from ._types import (
    _Action,
    _LegalMask,
    _Node,
    _PathI,
    _PriorLogits,
)
from .config import _Cfg
from .tree import _interior_select, _Tree

_ROLL_T = jnp.int32(ActionType.ROLL_DICE)
_BUY_T = jnp.int32(ActionType.BUY_DEVELOPMENT_CARD)

# Explicit chance nodes: a decision node's stochastic action (a dice roll, or a
# dev-card buy when `dev_chance`) leads to a *chance* node whose children are the
# real outcomes, sampled at their true probability and applied via the engine's
# forced-outcome seam. `_N_OUTCOMES` bounds the per-chance-node child axis: 11
# dice outcomes (2..12) or `N_DEV_CARD_TYPES` card types.
_DECISION = jnp.int32(0)
_CHANCE = jnp.int32(1)
_N_DICE = 11
_N_OUTCOMES = max(_N_DICE, N_DEV_CARD_TYPES)
_ROLL_LOGITS = jnp.log(_ROLL_P)  # the true two-dice outcome distribution


class _Descent(NamedTuple):
    """One simulation's forward walk; carried through the descent ``while_loop``.
    ``exp_parent`` >= 0 marks the node a new leaf attaches to (the expansion). The
    leaf is scored once *after* the loop (:func:`_evaluate`); ``prev_state`` carries
    the pre-step parent for a dice leaf's roll expectation, and is ``None`` (not
    carried) when ``expected_rolls`` is off — the single-sample leaf needs no
    parent."""

    state: BoardState
    prev_state: BoardState | None  # step's parent for _roll_ev; None if single-sample
    key: KeyScalar  # descent RNG, for sampling chance outcomes (chance_nodes)
    cur: _Node
    depth: IntScalar  # edges taken so far
    path_node: _PathI
    path_act: _PathI
    done: BoolScalar
    exp_parent: _Node
    exp_act: _Action
    exp_kind: _Node  # kind of the leaf node being expanded (chance_nodes)
    category: IntScalar  # action-ordering category reached this turn (ordered)


# --- engine interface: facts about one determinized state ---


def _legal_mask(layout: BoardLayout, state: BoardState) -> _LegalMask:
    return flat_available_for(layout, state).astype(jnp.float32)


def _squash(cfg: _Cfg, state: BoardState, player: Player, raw: Value) -> Value:
    """A raw value-function score in the searcher's frame: ±1 once the mover has
    won, else tanh-squashed."""
    terminal = current_player_won(state)
    win = state.current_player.astype(jnp.int32) == player
    return jnp.where(
        terminal, jnp.where(win, 1.0, -1.0), jnp.tanh(raw / cfg.value_scale)
    )


def _value(cfg: _Cfg, layout: BoardLayout, state: BoardState, player: Player) -> Value:
    """Searcher-frame value of a state: ±1 once the mover has won, else the
    tanh-squashed value function."""
    return _squash(cfg, state, player, cfg.value(layout, state, player))


def _step(layout: BoardLayout, state: BoardState, action: _Action) -> BoardState:
    atype, aparams = flat_to_action(action)
    avail = action_available(layout, state, atype, aparams)
    next_state, _ = apply_action(layout, state, atype, aparams, avail)
    return next_state


def _value_and_logits(
    cfg: _Cfg, layout: BoardLayout, state: BoardState, player: Player
) -> tuple[Value, _PriorLogits]:
    """A leaf's searcher-frame value and the prior over its actions (a learned
    policy head if one was supplied, else the constant tier table).

    Both come from a single evaluation when the prior is a
    :class:`~settlrl_search.policy.ValuePrior` — the shared-trunk nets' seam —
    instead of one forward per head. That makes the *prior's* value head score
    the leaf; ``cfg.fused_leaf`` off keeps ``cfg.value`` in that role."""
    if cfg.prior is None:
        return _value(cfg, layout, state, player), _TIER_LOGITS
    if cfg.fused_leaf and isinstance(cfg.prior, ValuePrior):
        raw, logits = cfg.prior.with_value(layout, state, player)
        return _squash(cfg, state, player, raw), logits
    return _value(cfg, layout, state, player), cfg.prior(layout, state, player)


def _roll_ev(
    cfg: _Cfg, layout: BoardLayout, state: BoardState, player: Player
) -> Value:
    """Expected post-payout value of a pre-roll ``state`` over the 11 dice rolls —
    the leaf value of a ``ROLL_DICE`` edge."""
    vals = jax.vmap(
        lambda r: jnp.tanh(
            cfg.value(layout, distribute_resources(layout, state, r), player)
            / cfg.value_scale
        )
    )(_ROLLS)
    return _ROLL_P @ vals


# --- explicit chance nodes: stochastic actions resolve through nature nodes ---


def _select_state(pred: BoolScalar, a: BoardState, b: BoardState) -> BoardState:
    """``a`` where ``pred`` else ``b`` (a state pytree ``where``)."""
    return cast(BoardState, jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b))


def _is_stochastic(cfg: _Cfg, action: _Action) -> BoolScalar:
    """Whether taking ``action`` lands in a chance node: a dice roll always, a
    dev-card buy when ``dev_chance``."""
    atype = ROW_TYPE[action]
    return cast(BoolScalar, (atype == _ROLL_T) | (cfg.dev_chance & (atype == _BUY_T)))


def _sample_outcome(
    cfg: _Cfg, key: KeyScalar, pending: _Action, state: BoardState
) -> IntScalar:
    """Sample a chance outcome index for the ``pending`` stochastic action: a dice
    outcome ``0..10`` (roll-2) at the true probabilities, or a dev-card type at the
    deck's current composition."""
    is_buy = cfg.dev_chance & (ROW_TYPE[pending] == _BUY_T)
    deck = state.dev_deck.astype(jnp.float32)
    deck_logits = jnp.where(
        deck > 0, jnp.log(deck / jnp.maximum(deck.sum(), 1.0)), _ILLEGAL
    )
    buy_logits = jnp.concatenate(
        [deck_logits, jnp.full((_N_OUTCOMES - N_DEV_CARD_TYPES,), _ILLEGAL)]
    )
    roll_logits = jnp.concatenate(
        [_ROLL_LOGITS, jnp.full((_N_OUTCOMES - _N_DICE,), _ILLEGAL)]
    )
    logits = jnp.where(is_buy, buy_logits, roll_logits)
    return jax.random.categorical(key, logits).astype(jnp.int32)


def _resolve_chance(
    cfg: _Cfg,
    layout: BoardLayout,
    state: BoardState,
    pending: _Action,
    outcome: IntScalar,
) -> BoardState:
    """Apply the ``pending`` stochastic action with its sampled ``outcome`` forced
    through the engine seam (roll ``outcome+2``, or dev-card type ``outcome``)."""
    atype, _ = flat_to_action(pending)
    is_buy = cfg.dev_chance & (ROW_TYPE[pending] == _BUY_T)
    forced = jnp.where(is_buy, outcome + 1, outcome + 2).astype(jnp.int32)
    params = ActionParams(idx=forced, target=jnp.int32(0))
    avail = action_available(layout, state, atype, params)
    next_state, _ = apply_action(layout, state, atype, params, avail)
    return next_state


# --- the simulation phases: determinize, then select (descend) ---


def _determinize(
    cfg: _Cfg, key: KeyScalar, layout: BoardLayout, view: BeliefView, player: Player
) -> _Descent:
    # DETERMINIZE: sample a world consistent with the belief, and seed the walk
    # over it at the root (depth 0, node 0, nothing expanded yet).
    k_world, k_desc = jax.random.split(key)
    state = sample_world(k_world, view, player)
    return _Descent(
        state=state,
        prev_state=state if cfg.expected_rolls else None,
        key=k_desc,
        cur=jnp.int32(0),
        depth=jnp.int32(0),
        path_node=jnp.zeros((cfg.max_depth,), jnp.int32),
        path_act=jnp.zeros((cfg.max_depth,), jnp.int32),
        done=current_player_won(state),
        exp_parent=jnp.int32(-1),
        exp_act=jnp.int32(0),
        exp_kind=_DECISION,
        category=jnp.int32(0),  # reset at the root; the env supplies the root mask
    )


def _descend(
    cfg: _Cfg,
    tree: _Tree,
    a_root: _Action,
    walk: _Descent,
    layout: BoardLayout,
    player: Player,
) -> _Descent:
    # SELECT: from the root, follow already-expanded edges (root action by
    # Sequential Halving, interior by the improved-policy rule) until an unexpanded
    # edge or a terminal state. No value function runs in the loop — the leaf is
    # scored once after it by `_evaluate`; the body does only the cheap terminal
    # test needed to stop.
    def cond(walk: _Descent) -> BoolScalar:
        return (~walk.done) & (walk.depth < cfg.max_depth)

    def selected(walk: _Descent) -> _Action:
        # The step's action: the improved-policy rule over this world's legal
        # set (the ordering lock-out ANDed in when on). Runs only past the
        # peeled depth-0 step, so the root (whose action is `a_root`) never
        # reaches it.
        legal = _legal_mask(layout, walk.state)
        if cfg.ordered:
            legal = jnp.where(ordering_mask(walk.state, walk.category), legal, 0.0)
        return _interior_select(tree, walk.cur, legal, player)

    def advance(walk: _Descent, action: _Action) -> _Descent:
        # One decision step by `action`; `tree` is read-only here.
        next_state = _step(layout, walk.state, action)
        is_leaf = tree.children[walk.cur, action] < 0  # unexpanded edge -> stop here
        category = (
            next_category(
                walk.category,
                ROW_TYPE[action],
                walk.state.current_player != next_state.current_player,
            )
            if cfg.ordered
            else walk.category
        )
        return _Descent(
            state=next_state,
            prev_state=walk.state if cfg.expected_rolls else None,
            key=walk.key,
            cur=jnp.where(is_leaf, walk.cur, tree.children[walk.cur, action]),
            depth=walk.depth + 1,
            path_node=walk.path_node.at[walk.depth].set(walk.cur),
            path_act=walk.path_act.at[walk.depth].set(action),
            done=is_leaf | current_player_won(next_state),
            exp_parent=jnp.where(is_leaf, walk.cur, jnp.int32(-1)),
            exp_act=jnp.where(is_leaf, action, jnp.int32(0)),
            exp_kind=_DECISION,
            category=category,
        )

    def body(walk: _Descent) -> _Descent:
        return advance(walk, selected(walk))

    def advance_chance(
        walk: _Descent, action: _Action, root_decision: bool
    ) -> _Descent:
        # Decision/chance state machine step. A decision node's *stochastic*
        # `action` (roll, or dev-buy under dev_chance) defers to a chance node
        # (the afterstate, action unapplied). A chance node samples its outcome at
        # the true probability and applies the forced transition. Both branches
        # are computed every step (vmap runs both) and `where`-selected by node
        # kind — except under `root_decision` (static): the root is always a
        # decision node (`tree.kind[0]` stays ``_DECISION``), so the chance
        # branch drops out.
        key, k_out = jax.random.split(walk.key)
        stoch = _is_stochastic(cfg, action)
        # decision step: defer a stochastic action (afterstate = current state),
        # else apply it.
        dec_state = _select_state(stoch, walk.state, _step(layout, walk.state, action))
        if root_decision:
            is_chance: BoolScalar = jnp.bool_(False)
            next_state = dec_state
            edge = action
        else:
            is_chance = tree.kind[walk.cur] == _CHANCE
            # chance step: resolve the pending action's forced outcome.
            pending = walk.path_act[walk.depth - 1]  # the action creating this node
            outcome = _sample_outcome(cfg, k_out, pending, walk.state)
            chance_state = _resolve_chance(cfg, layout, walk.state, pending, outcome)
            next_state = _select_state(is_chance, chance_state, dec_state)
            edge = jnp.where(is_chance, outcome, action)
        next_kind = jnp.where(
            is_chance, _DECISION, jnp.where(stoch, _CHANCE, _DECISION)
        )
        is_leaf = tree.children[walk.cur, edge] < 0
        # Only a decision-node action raises the ordering category; a chance
        # resolution (nature's outcome) is uncategorised.
        if cfg.ordered:
            turn_changed = walk.state.current_player != next_state.current_player
            act_type = jnp.where(
                is_chance, jnp.int32(ActionType.END_TURN), ROW_TYPE[action]
            )
            category = next_category(walk.category, act_type, turn_changed)
        else:
            category = walk.category
        return _Descent(
            state=next_state,
            prev_state=None,
            key=key,
            cur=jnp.where(is_leaf, walk.cur, tree.children[walk.cur, edge]),
            depth=walk.depth + 1,
            path_node=walk.path_node.at[walk.depth].set(walk.cur),
            path_act=walk.path_act.at[walk.depth].set(edge),
            done=is_leaf | current_player_won(next_state),
            exp_parent=jnp.where(is_leaf, walk.cur, jnp.int32(-1)),
            exp_act=jnp.where(is_leaf, edge, jnp.int32(0)),
            exp_kind=next_kind,
            category=category,
        )

    def body_chance(walk: _Descent) -> _Descent:
        return advance_chance(walk, selected(walk), root_decision=False)

    step = body_chance if cfg.chance_nodes else body

    # The depth-0 step is peeled: the root action is already fixed (`a_root`),
    # so its legality sweep and interior select never run. The peel is guarded
    # exactly like `cond` (a terminal root takes no step; depth 0 < max_depth
    # holds statically).
    def root_step(walk: _Descent) -> _Descent:
        if cfg.chance_nodes:
            return advance_chance(walk, a_root, root_decision=True)
        return advance(walk, a_root)

    walk = jax.lax.cond(walk.done, lambda walk: walk, root_step, walk)
    return jax.lax.while_loop(cond, step, walk)


def _evaluate(
    cfg: _Cfg, layout: BoardLayout, walk: _Descent, player: Player
) -> tuple[Value, Player, _PriorLogits]:
    # EVALUATE: score the descent's leaf once, here, rather than on every step —
    # value and expansion prior together, so a shared-trunk net evaluates the leaf
    # state once. The leaf's mover is stored on the new node for the backup frame.
    #
    # `_step` already applies one sampled dice outcome, so the plain leaf value is
    # a single-sample roll. With `expected_rolls`, a roll-edge leaf instead takes
    # the exact 11-roll expectation from the pre-roll parent (`prev_state`) —
    # variance reduction at 11x the value-fn calls. Keying off the last path action
    # (not just a grown edge) also covers a `max_depth` truncation onto a roll.
    leaf = walk.state
    mover = agent_selection_single(leaf).astype(jnp.int32)
    value, logits = _value_and_logits(cfg, layout, leaf, player)
    if not cfg.expected_rolls:
        return value, mover, logits
    # prev_state is carried (non-None) iff cfg.expected_rolls, which holds here.
    prev_state = walk.prev_state
    assert prev_state is not None
    last_action = walk.path_act[walk.depth - 1]  # edge into the leaf
    roll_leaf = (walk.depth > 0) & (ROW_TYPE[last_action] == _ROLL_T)
    value = jnp.where(
        roll_leaf & ~current_player_won(leaf),
        _roll_ev(cfg, layout, prev_state, player),
        value,
    )
    return value, mover, logits
