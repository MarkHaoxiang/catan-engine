"""A *batched* PettingZoo-AEC-style Settlrl env.

Two layers: the functional ``step`` / ``available`` over the action dispatch,
and ``BatchedSettlrlEnv``, which adapts the `AEC model
<https://pettingzoo.farama.org/api/aec/>`_ to ``batch_size`` parallel games --
the batch axis is independent games, the acting agent in lane ``b`` is that
game's ``current_player`` (the next owing discarder during DISCARD; the
proposed-to partner during TRADE_RESPONSE), and the
per-agent AEC dicts become batched arrays (see the attribute docstrings).
Terminated lanes auto-reset by default. Spaces are the lightweight
``Discrete`` / ``Box`` descriptors below -- no gymnasium dependency.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import ClassVar, Literal, NamedTuple, TypedDict, TypeVar, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, UInt8

from settlrl_engine.belief import (
    BeliefState,
    BeliefView,
    PlayerBelief,
    belief_view,
    make_belief,
    update_belief,
)
from settlrl_engine.board import Board
from settlrl_engine.board.dev_cards import N_DEV_CARD_TYPES
from settlrl_engine.board.layout import (
    N_EDGES,
    N_PORTS,
    N_TILES,
    N_VERTICES,
    BoardLayout,
    desert_tile,
    make_layout,
)
from settlrl_engine.board.resources import (
    BANK_INITIAL,
    N_PLAYERS,
    N_RESOURCES,
    compute_bank_resources,
)
from settlrl_engine.board.state import (
    VICTORY_POINTS_TO_WIN,
    BoardState,
    GamePhase,
    KeyScalar,
    make_board_state,
)
from settlrl_engine.mechanics.action import (
    N_ACTION_TYPES,
    ActionParams,
    ActionResult,
    ActionType,
    action_available,
    apply_action,
)
from settlrl_engine.mechanics.common import (
    ActionTypeArray,
    Mask,
    ResultCode,
    agent_selection_single,
    player_total_vp,
)
from settlrl_engine.mechanics.flat import (
    INDEX_MASKS,
    N_FLAT,
    FlatMaskArray,
    TypeMaskArray,
    flat_available_b,
    flat_available_for,
    flat_legality,
    flat_to_action,
    random_flat,
    type_mask_from_flat,
)
from settlrl_engine.mechanics.trade import _propose_trade_avail
from settlrl_engine.ordering import next_category, ordering_mask

__all__ = [
    "N_ACTION_TYPES",
    "N_FLAT",
    "ActionParams",
    "ActionResult",
    "ActionType",
    "Actor",
    "BatchedSettlrlEnv",
    "BeliefState",
    "BeliefView",
    "Box",
    "Discrete",
    "Infos",
    "Observation",
    "PlayerBelief",
    "available",
    "flat_available",
    "flat_to_action",
    "observe_for",
    "random_flat",
    "step",
]

# ---------------------------------------------------------------------------
# Functional core: one batched (ActionType, ActionParams) action per game.
# ---------------------------------------------------------------------------
_apply_action_v: Callable[
    [BoardLayout, BoardState, ActionTypeArray, ActionParams, Mask],
    tuple[BoardState, ResultCode],
] = jax.vmap(apply_action, in_axes=(0, 0, 0, 0, 0))
_step = jax.jit(_apply_action_v)
_available: Callable[[BoardLayout, BoardState, ActionTypeArray, ActionParams], Mask] = (
    jax.jit(jax.vmap(action_available, in_axes=(0, 0, 0, 0)))
)


def step(
    board: Board, action_type: ActionTypeArray, params: ActionParams
) -> tuple[BoardState, ResultCode]:
    """Apply one (batched) action per game; return (new state, ActionResult codes).

    Self-validating for arbitrary params: legality is computed internally, and
    an illegal action leaves its lane unchanged with ``INVALID``.
    """
    available = _available(board[0], board[1], action_type, params)
    new_state, result = _step(board[0], board[1], action_type, params, available)
    return new_state, result


def available(board: Board, action_type: ActionTypeArray, params: ActionParams) -> Mask:
    """``(batch,)`` legality mask for the chosen action per game (no state change)."""
    return _available(board[0], board[1], action_type, params)


def flat_available(board: Board) -> FlatMaskArray:
    """``(batch, N_FLAT)`` legality of every flat action for each game's acting
    player (the same sweep behind :meth:`BatchedSettlrlEnv.flat_mask`)."""
    return flat_available_b(board[0], board[1])


# ---------------------------------------------------------------------------
# Env-surface aliases and observation structure.
# ---------------------------------------------------------------------------
AgentSelectionArray = Int[Array, "batch"]
RewardArray = Float[Array, "batch players"]
DoneArray = Bool[Array, "batch players"]
VPArray = Int[Array, "batch players"]


class Observation(TypedDict):
    """One player's partial view (:meth:`BatchedSettlrlEnv.observe`).

    ``*batch`` is the env's batch axis; the single-game policies in
    settlrl-agents consume the same structure with it stripped (one lane).
    Dimension sizes are pinned by :meth:`BatchedSettlrlEnv.observation_space`.
    """

    # public board
    tile_resource: UInt8[Array, "*batch tiles"]
    tile_number: UInt8[Array, "*batch tiles"]
    port_allocation: UInt8[Array, "*batch ports"]
    vertex_owner: UInt8[Array, "*batch vertices"]
    vertex_type: UInt8[Array, "*batch vertices"]
    edge_road: UInt8[Array, "*batch edges"]
    robber: UInt8[Array, "*batch"]
    # public player info
    victory_points: UInt8[Array, "*batch players"]
    knights_played: UInt8[Array, "*batch players"]
    hand_size: Int[Array, "*batch players"]
    dev_card_count: Int[Array, "*batch players"]
    longest_road_owner: UInt8[Array, "*batch"]
    largest_army_owner: UInt8[Array, "*batch"]
    longest_road_len: UInt8[Array, "*batch"]
    bank: UInt8[Array, "*batch resources"]
    # turn / flow
    phase: UInt8[Array, "*batch"]
    current_player: UInt8[Array, "*batch"]
    dice_roll: UInt8[Array, "*batch"]
    has_rolled: UInt8[Array, "*batch"]
    trade_partner: UInt8[Array, "*batch"]
    trade_give: UInt8[Array, "*batch resources"]
    trade_receive: UInt8[Array, "*batch resources"]
    # private (observer)
    self: Int[Array, "*batch"]
    self_resources: UInt8[Array, "*batch resources"]
    self_dev_hand: UInt8[Array, "*batch dev_card_types"]
    self_pending_discard: UInt8[Array, "*batch"]


class Infos(TypedDict):
    """The batched AEC ``infos``: action mask, acting / current player, last
    :class:`ActionResult` per lane."""

    action_mask: TypeMaskArray
    agent_selection: AgentSelectionArray
    current_player: Int[Array, "batch"]
    result: ResultCode


# ---------------------------------------------------------------------------
# Batched derived quantities used by the environment.
# ---------------------------------------------------------------------------
def _total_vp_single(state: BoardState) -> Int[Array, "players"]:
    """Total VP (buildings + awards + VP cards) for every player in one game."""
    players = jnp.arange(state.n_players, dtype=jnp.int32)
    return jax.vmap(lambda p: player_total_vp(state, p))(players)


_total_vp_v: Callable[[BoardState], VPArray] = jax.vmap(_total_vp_single)
_total_vp_b: Callable[[BoardState], VPArray] = jax.jit(_total_vp_v)


_agent_selection_v: Callable[[BoardState], AgentSelectionArray] = jax.vmap(
    agent_selection_single
)
_agent_selection_b: Callable[[BoardState], AgentSelectionArray] = jax.jit(
    _agent_selection_v
)

# Batched action-ordering helpers (opt-in `track_ordering`): advance the per-turn
# category and build the legality overlay across the batch.
_next_category_b = jax.jit(jax.vmap(next_category, in_axes=(0, 0, 0)))
_ordering_mask_b = jax.jit(jax.vmap(ordering_mask, in_axes=(0, 0)))


def _fresh_board(
    k_layout: KeyScalar,
    k_state: KeyScalar,
    batch_size: int,
    number_placement: Literal["random", "spiral"],
    n_players: int,
) -> tuple[BoardLayout, BoardState]:
    """A fresh batch of random boards, robber seeded on the desert (rulebook)."""
    layout = make_layout(batch_size, key=k_layout, number_placement=number_placement)
    state = make_board_state(batch_size, key=k_state, n_players=n_players)
    return layout, state._replace(robber=desert_tile(layout.tile_resource))


def _random_actions_core(
    avail_flat: FlatMaskArray, key: KeyScalar
) -> tuple[ActionTypeArray, ActionParams]:
    """One :func:`random_flat` draw per lane (``key`` split per lane), decoded."""
    keys = jax.random.split(key, avail_flat.shape[0])
    return flat_to_action(jax.vmap(random_flat)(keys, avail_flat))


_random_actions_b: Callable[
    [FlatMaskArray, KeyScalar], tuple[ActionTypeArray, ActionParams]
] = jax.jit(_random_actions_core)


# ---------------------------------------------------------------------------
# Belief tracking (optional; see settlrl_engine.belief).
# ---------------------------------------------------------------------------
_update_belief_v = jax.vmap(update_belief)
_belief_view_b: Callable[[BoardState, BeliefState, int], BeliefView] = jax.jit(
    jax.vmap(belief_view, in_axes=(0, 0, None)), static_argnums=2
)


# ---------------------------------------------------------------------------
# Lightweight space descriptors (no gymnasium dependency).
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Discrete:
    """A discrete space of ``n`` values ``{0, ..., n - 1}``."""

    n: int


@dataclasses.dataclass(frozen=True)
class Box:
    """A bounded integer array space (shape excludes the leading batch axis)."""

    shape: tuple[int, ...]
    dtype: str
    low: int
    high: int


Space = Discrete | Box


# ---------------------------------------------------------------------------
# The batched AEC environment.
# ---------------------------------------------------------------------------
class BatchedSettlrlEnv:
    """A batch of Settlrl games behind a (batched) PettingZoo-AEC interface.

    Args:
        batch_size: number of games run in parallel (the leading array axis).
        seed: PRNG seed for the initial boards and auto-reset randomness.
        n_players: players seated per game (2..4, default 4; same for every
            lane). Sizes the per-player axis of the state, observations, and
            ``rewards`` / ``terminations`` / ``truncations``.
        victory_points_to_win: total VP that ends a game (default 10; same for
            every lane). A lane terminates when its current player reaches it.
        reward: ``"sparse"`` (+1 to the winner on the terminal step, 0 else) or
            ``"vp_delta"`` (each player's change in total VP this step).
        auto_reset: when True (default), a lane that terminates is replaced with a
            fresh game on the same step; when False, terminated lanes freeze (and
            keep reporting ``terminations`` True) for callers that manage episode
            boundaries themselves (e.g. the single-game AEC wrapper).
        number_placement: ``"random"`` (default) shuffles the number tokens
            uniformly; ``"spiral"`` lays them in the rulebook's variable set-up
            spiral (tournament-style balanced boards). Applied to the initial
            boards and to auto-reset replacements.
        track_beliefs: when True, maintain a :class:`BeliefState` (per-observer
            card counting; see ``settlrl_engine.belief``) across steps and
            auto-resets, read via :attr:`beliefs` / :meth:`belief_view`.

    The action consumed by :meth:`step` is the engine's
    ``(action_type, ActionParams)`` pair with a leading batch axis -- one action
    per lane, applied to that lane's acting player. See module docstring for the
    batched adaptations of the AEC attributes.
    """

    metadata: ClassVar[dict[str, str]] = {"name": "settlrl_batched_aec_v0"}

    def __init__(
        self,
        batch_size: int = 1,
        seed: int = 0,
        reward: str = "sparse",
        auto_reset: bool = True,
        number_placement: Literal["random", "spiral"] = "random",
        n_players: int = N_PLAYERS,
        victory_points_to_win: int = VICTORY_POINTS_TO_WIN,
        track_beliefs: bool = False,
        track_ordering: bool = False,
    ) -> None:
        if reward not in ("sparse", "vp_delta"):
            raise ValueError(f"reward must be 'sparse' or 'vp_delta', got {reward!r}")
        if number_placement not in ("random", "spiral"):
            raise ValueError(
                "number_placement must be 'random' or 'spiral', "
                f"got {number_placement!r}"
            )
        if not 2 <= n_players <= N_PLAYERS:
            raise ValueError(f"n_players must be in [2, {N_PLAYERS}], got {n_players}")
        if victory_points_to_win < 1:
            raise ValueError(
                f"victory_points_to_win must be >= 1, got {victory_points_to_win}"
            )
        self.batch_size = batch_size
        self.reward_mode = reward
        self.auto_reset = auto_reset
        self.number_placement = number_placement
        self.n_players = n_players
        self.victory_points_to_win = victory_points_to_win
        self.track_beliefs = track_beliefs
        self.track_ordering = track_ordering
        self.possible_agents = [f"player_{i}" for i in range(n_players)]
        self.agents = list(self.possible_agents)
        self.num_agents = n_players
        self._seed = seed
        self.reset(seed)

    # -- AEC lifecycle ----------------------------------------------------

    def reset(self, seed: int | None = None, options: object = None) -> None:
        """Start a fresh batch of games (AEC ``reset``; returns ``None``).

        Read the starting position with :meth:`observe` / :meth:`last`.
        """
        if seed is not None:
            self._seed = seed
        self._key = jax.random.key(self._seed)
        self._key, k_layout, k_state = jax.random.split(self._key, 3)
        self._layout, self._state = _fresh_board(
            k_layout, k_state, self.batch_size, self.number_placement, self.n_players
        )
        B, P = self.batch_size, self.n_players
        self._reward = jnp.zeros((B, P), dtype=jnp.float32)
        self._terminations = jnp.zeros((B, P), dtype=jnp.bool_)
        self._truncations = jnp.zeros((B, P), dtype=jnp.bool_)
        self._result = jnp.full((B,), int(ActionResult.SUCCESS), dtype=jnp.int32)
        self._vps = _total_vp_b(self._state)
        # Flat legality of every action for each lane's acting player, computed
        # once and reused by step (to gate the chosen action), random_actions, and
        # action_mask -- the single legality source. Refreshed after every step.
        self._avail = flat_available_b(self._layout, self._state)
        self._agent_sel = _agent_selection_b(self._state)
        self._belief: BeliefState | None = (
            make_belief(self.batch_size, self.n_players) if self.track_beliefs else None
        )
        # Per-turn action-ordering category (the lock-out's one int of state); the
        # SETUP start is non-MAIN so the overlay is a no-op until the first turn.
        self._category = jnp.zeros((B,), jnp.int32) if self.track_ordering else None
        if self.track_ordering:
            self._avail = self._avail & _ordering_mask_b(self._state, self._category)
        self.agents = list(self.possible_agents)

    def step(self, action_type: ActionTypeArray, params: ActionParams) -> None:
        """Apply one action per lane to its acting player (AEC ``step``).

        Terminated lanes auto-reset; the resulting reward / termination reflect
        the transition that just happened, while the next observation is the
        reset game's. DISCARD is one card per step: ``idx`` is the resource the
        acting discarder gives up one card of; the action repeats until every
        owed count reaches zero, then the phase advances to MOVE_ROBBER.
        """
        at = jnp.asarray(action_type, dtype=jnp.int32)
        if self.track_ordering:
            pre_player = self._state.current_player
            pre_main = self._state.phase == int(GamePhase.MAIN)
            legal = flat_legality(self._avail, at, params.idx, params.target)
        out = _env_step_core(
            self._layout,
            self._state,
            at,
            params,
            self._avail,
            self._vps,
            self._belief,
            self._key,
            self.batch_size,
            self.reward_mode,
            self.auto_reset,
            self.number_placement,
            self.n_players,
            self.victory_points_to_win,
        )
        self._layout, self._state = out.layout, out.state
        self._reward, self._terminations = out.reward, out.terminations
        self._result, self._vps = out.result, out.vps
        self._avail, self._agent_sel, self._key = out.avail, out.agent_sel, out.key
        self._belief = out.belief
        if self.track_ordering:
            assert self._category is not None
            # Reset the category on a turn change (current player moved, or the
            # lane auto-reset); else raise it by a *legal, main-phase* action's
            # category (an illegal / sub-phase action counts as uncategorised).
            turn_changed = (
                pre_player != self._state.current_player
            ) | out.terminations.any(-1)
            safe_at = jnp.where(legal & pre_main, at, jnp.int32(ActionType.END_TURN))
            self._category = _next_category_b(self._category, safe_at, turn_changed)
            self._avail = self._avail & _ordering_mask_b(self._state, self._category)

    def last(
        self,
    ) -> tuple[
        Observation,
        Float[Array, "batch"],
        Bool[Array, "batch"],
        Bool[Array, "batch"],
        Infos,
    ]:
        """``(obs, reward, termination, truncation, info)`` for the acting agents.

        Each quantity is gathered per lane for that lane's ``agent_selection``.
        """
        sel = self.agent_selection
        rows = jnp.arange(self.batch_size)
        obs = self._obs_for(sel)
        reward = self._reward[rows, sel]
        termination = self._terminations[rows, sel]
        truncation = self._truncations[rows, sel]
        return obs, reward, termination, truncation, self.infos

    def close(self) -> None:
        """No resources to release."""

    def render(self, lane: int = 0) -> str:
        """One-line textual status for ``lane`` (full rendering lives in tests)."""
        s = self._state
        phase = GamePhase(int(s.phase[lane]))
        return (
            f"[lane {lane}] {phase}  player={int(s.current_player[lane]) + 1}  "
            f"dice={int(s.dice_roll[lane])}  vp={[int(x) for x in self._vps[lane]]}"
        )

    # -- AEC attributes ---------------------------------------------------

    @property
    def agent_selection(self) -> AgentSelectionArray:
        """``(B,)`` int array of the acting player per lane (batched AEC).

        Cached: refreshed by :meth:`reset` / :meth:`step` / :meth:`rollout`, so
        reading it costs nothing.
        """
        return self._agent_sel

    @property
    def rewards(self) -> RewardArray:
        """``(B, n_players)`` reward from the last :meth:`step`."""
        return self._reward

    @property
    def terminations(self) -> DoneArray:
        """``(B, n_players)`` per-lane game-over flags (broadcast across players)."""
        return self._terminations

    @property
    def truncations(self) -> DoneArray:
        """``(B, n_players)`` truncation flags (always False -- no time limit)."""
        return self._truncations

    @property
    def infos(self) -> Infos:
        """Batched info: action mask, acting player, current player, last result."""
        return {
            "action_mask": self.action_mask(),
            "agent_selection": self.agent_selection,
            "current_player": self._state.current_player.astype(jnp.int32),
            "result": self._result,
        }

    @property
    def board(self) -> Board:
        """The underlying batched ``(BoardLayout, BoardState)``."""
        return self._layout, self._state

    @property
    def beliefs(self) -> BeliefState:
        """The tracked batched :class:`BeliefState` (requires ``track_beliefs``)."""
        if self._belief is None:
            raise RuntimeError("belief tracking is off; pass track_beliefs=True")
        return self._belief

    def belief_view(self, agent: int | str) -> BeliefView:
        """``agent``'s honest :class:`BeliefView` across all lanes (batched;
        requires ``track_beliefs``): the public board fields plus everything
        the seat knows about hidden hands -- nothing it couldn't know. See
        :func:`settlrl_engine.belief.belief_view`.
        """
        me = self._agent_index(agent)
        return _belief_view_b(self._state, self.beliefs, me)

    # -- Spaces -----------------------------------------------------------

    def action_space(self, agent: object = None) -> dict[str, Space]:
        """Descriptor of the ``(action_type, ActionParams)`` action (per lane)."""
        return {
            "action_type": Discrete(N_ACTION_TYPES),
            # vertex / edge / tile / resource — or ProposeTrade's bit-packed
            # give counts (see mechanics.trade.pack_trade).
            "idx": Discrete(1 << 25),
            # victim / receive / 2nd resource — or packed receive counts + partner.
            "target": Discrete(1 << 27),
        }

    def observation_space(self, agent: object = None) -> dict[str, Space]:
        """Descriptor of one lane's observation (see :meth:`observe`)."""
        return {
            "tile_resource": Box((N_TILES,), "uint8", 0, 5),
            "tile_number": Box((N_TILES,), "uint8", 0, 12),
            "port_allocation": Box((N_PORTS,), "uint8", 0, 5),
            "vertex_owner": Box((N_VERTICES,), "uint8", 0, self.n_players),
            "vertex_type": Box((N_VERTICES,), "uint8", 0, 2),
            "edge_road": Box((N_EDGES,), "uint8", 0, self.n_players),
            "robber": Discrete(N_TILES),
            "victory_points": Box(
                (self.n_players,), "uint8", 0, self.victory_points_to_win
            ),
            "knights_played": Box((self.n_players,), "uint8", 0, 14),
            "hand_size": Box((self.n_players,), "int32", 0, 255),
            "dev_card_count": Box((self.n_players,), "int32", 0, 25),
            "longest_road_owner": Discrete(self.n_players + 1),
            "largest_army_owner": Discrete(self.n_players + 1),
            "longest_road_len": Discrete(N_EDGES + 1),
            "bank": Box((N_RESOURCES,), "uint8", 0, BANK_INITIAL),
            "phase": Discrete(len(GamePhase)),
            "current_player": Discrete(self.n_players),
            "dice_roll": Discrete(13),
            "has_rolled": Discrete(2),
            "trade_partner": Discrete(256),  # player, or NO_INDEX (255)
            "trade_give": Box((N_RESOURCES,), "uint8", 0, 31),  # offered counts
            "trade_receive": Box((N_RESOURCES,), "uint8", 0, 31),  # asked counts
            "self": Discrete(self.n_players),
            "self_resources": Box((N_RESOURCES,), "uint8", 0, BANK_INITIAL),
            "self_dev_hand": Box((N_DEV_CARD_TYPES,), "uint8", 0, 25),
            "self_pending_discard": Discrete(256),
        }

    # -- Observations -----------------------------------------------------

    def observe(self, agent: int | str) -> Observation:
        """Partial observation from ``agent``'s point of view, across all lanes.

        ``agent`` may be a player index (0..n_players-1) or ``"player_i"``. The
        observer sees its own hand / dev cards in full but only public counts for
        opponents.
        """
        me = self._agent_index(agent)
        sel = jnp.full((self.batch_size,), me, dtype=jnp.int32)
        return self._obs_for(sel)

    def action_mask(self) -> TypeMaskArray:
        """``(B, N_ACTION_TYPES)`` -- which action types the acting player can use."""
        return type_mask_from_flat(self._avail)

    def flat_mask(self) -> FlatMaskArray:
        """``(B, N_FLAT)`` -- legality of every concrete flat action for the
        acting player per lane (decode a chosen index with :func:`flat_to_action`)."""
        return self._avail

    def available_indices(
        self, action_type: int | ActionType
    ) -> Bool[Array, "batch domain"]:
        """``(B, D)`` legality over the primary index of an index-parameterized action.

        Supported: SETUP_SETTLEMENT / BUILD_SETTLEMENT / BUILD_CITY (vertices),
        SETUP_ROAD / BUILD_ROAD (edges), DISCARD / PLAY_MONOPOLY (resources),
        and MOVE_ROBBER / PLAY_KNIGHT (tiles, legal if some victim choice works).
        Other action types have no single index domain -- use :meth:`action_mask`
        or :func:`available`.
        """
        at = ActionType(int(action_type))
        fn = INDEX_MASKS.get(at)
        if fn is None:
            raise ValueError(
                f"{at.name} has no single primary-index domain; "
                "use action_mask() or available()"
            )
        return fn(self._layout, self._state)

    def random_actions(self, key: KeyScalar) -> tuple[ActionTypeArray, ActionParams]:
        """A random *legal* action per lane (the random-rollout driver): one
        :func:`random_flat` type-first draw per lane.

        A lane with no legal action yields an INVALID action and simply stalls
        until its next auto-reset. ``key`` is a JAX PRNG key, split per lane.
        """
        return _random_actions_b(self._avail, key)

    def rollout(
        self, key: KeyScalar, n_steps: int, actor: Actor | None = None
    ) -> RewardArray:
        """Advance every lane ``n_steps`` steps as one fused ``lax.scan``.

        Actions come from ``actor`` (see :data:`Actor`), or random legal play
        (:meth:`random_actions`'s type-first sampling) when it is None — in
        which case the trajectory is identical to a :meth:`random_actions` +
        :meth:`step` loop for the same ``key``;
        afterwards the env reflects the final step. Returns the reward summed
        over the window (under ``"sparse"`` reward: each player's win count).
        Compiles once per distinct ``(n_steps, actor identity)``.
        """
        (
            self._layout,
            self._state,
            self._avail,
            self._vps,
            self._belief,
            (self._reward, self._terminations, self._result, self._agent_sel),
            self._key,
            cum_reward,
        ) = _rollout_core(
            self._layout,
            self._state,
            self._avail,
            self._vps,
            self._belief,
            (self._reward, self._terminations, self._result, self._agent_sel),
            self._key,
            key,
            n_steps,
            actor,
            self.batch_size,
            self.reward_mode,
            self.auto_reset,
            self.number_placement,
            self.n_players,
            self.victory_points_to_win,
        )
        return cast(RewardArray, cum_reward)

    # -- Internals --------------------------------------------------------

    def _agent_index(self, agent: int | str) -> int:
        if isinstance(agent, str):
            return self.possible_agents.index(agent)
        if not 0 <= agent < self.n_players:
            raise ValueError(f"agent index {agent} out of range [0, {self.n_players})")
        return agent

    def _obs_for(self, sel: AgentSelectionArray) -> Observation:
        """Observation with per-lane ``self`` index ``sel`` (``(B,)`` int array)."""
        return observe_for(self._layout, self._state, sel)


def observe_for(
    layout: BoardLayout, state: BoardState, sel: AgentSelectionArray
) -> Observation:
    """:class:`Observation` of a batched board with per-lane observer ``sel``.

    The pure builder behind :meth:`BatchedSettlrlEnv.observe`, usable inside a
    trace (e.g. an :data:`Actor` running under :meth:`BatchedSettlrlEnv.rollout`).
    """
    res = state.player_resources
    self_res = jnp.take_along_axis(res, sel[:, None, None], axis=1)[:, 0, :]
    self_dev = jnp.take_along_axis(state.dev_hand, sel[:, None, None], axis=1)[:, 0, :]
    self_pending = jnp.take_along_axis(state.pending_discard, sel[:, None], axis=1)[
        :, 0
    ]
    return {
        # public board
        "tile_resource": layout.tile_resource,
        "tile_number": layout.tile_number,
        "port_allocation": layout.port_allocation,
        "vertex_owner": state.vertex_owner,
        "vertex_type": state.vertex_type,
        "edge_road": state.edge_road,
        "robber": state.robber,
        # public player info
        "victory_points": state.victory_points,
        "knights_played": state.knights_played,
        "hand_size": res.astype(jnp.int32).sum(axis=2),
        "dev_card_count": state.dev_hand.astype(jnp.int32).sum(axis=2),
        "longest_road_owner": state.longest_road_owner,
        "largest_army_owner": state.largest_army_owner,
        "longest_road_len": state.longest_road_len,
        "bank": compute_bank_resources(res),
        # turn / flow
        "phase": state.phase,
        "current_player": state.current_player,
        "dice_roll": state.dice_roll,
        "has_rolled": state.has_rolled,
        "trade_partner": state.trade_partner,
        "trade_give": state.trade_give,
        "trade_receive": state.trade_receive,
        # private (observer)
        "self": sel,
        "self_resources": self_res,
        "self_dev_hand": self_dev,
        "self_pending_discard": self_pending,
    }


def _where_lane(mask: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """Per-lane ``where``: pick ``a`` where ``mask`` (``(B,)``) is set, else ``b``."""
    m = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
    return jnp.where(m, a, b)


def _select_key(mask: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """``_where_lane`` for typed PRNG key arrays (select on the raw key data)."""
    ad, bd = jax.random.key_data(a), jax.random.key_data(b)
    m = mask.reshape((mask.shape[0],) + (1,) * (ad.ndim - 1))
    return cast(jax.Array, jax.random.wrap_key_data(jnp.where(m, ad, bd)))


_Tree = TypeVar("_Tree")


def _tree_where_lane(mask: Bool[Array, "batch"], a: _Tree, b: _Tree) -> _Tree:
    """Per-leaf ``_where_lane`` over two matching pytrees of batched arrays
    (PRNG-key leaves route through ``_select_key``)."""

    def sel(x: jax.Array, y: jax.Array) -> jax.Array:
        if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            return _select_key(mask, x, y)
        return _where_lane(mask, x, y)

    return cast("_Tree", jax.tree.map(sel, a, b))


@functools.partial(
    jax.jit, static_argnames=("batch_size", "number_placement", "n_players")
)
def _auto_reset_core(
    layout: BoardLayout,
    state: BoardState,
    done_lane: jax.Array,
    key: jax.Array,
    batch_size: int,
    number_placement: Literal["random", "spiral"],
    n_players: int,
) -> tuple[BoardLayout, BoardState]:
    """Replace finished lanes (``done_lane``) with fresh games, fully on device."""

    def do_reset(
        operand: tuple[BoardLayout, BoardState],
    ) -> tuple[BoardLayout, BoardState]:
        layout, state = operand
        k_layout, k_state = jax.random.split(key)
        fresh = _fresh_board(k_layout, k_state, batch_size, number_placement, n_players)
        return _tree_where_lane(done_lane, fresh, (layout, state))

    def no_reset(
        operand: tuple[BoardLayout, BoardState],
    ) -> tuple[BoardLayout, BoardState]:
        return operand

    return cast(
        "tuple[BoardLayout, BoardState]",
        jax.lax.cond(jnp.any(done_lane), do_reset, no_reset, (layout, state)),
    )


class _StepOut(NamedTuple):
    """Everything one ``BatchedSettlrlEnv.step`` refreshes (the env's caches)."""

    layout: BoardLayout
    state: BoardState
    reward: RewardArray
    terminations: DoneArray
    result: ResultCode
    vps: VPArray
    avail: FlatMaskArray
    agent_sel: AgentSelectionArray
    belief: BeliefState | None
    key: KeyScalar


@functools.partial(
    jax.jit,
    static_argnames=(
        "batch_size",
        "reward_mode",
        "auto_reset",
        "number_placement",
        "n_players",
        "victory_points_to_win",
    ),
)
def _env_step_core(
    layout: BoardLayout,
    state: BoardState,
    action_type: ActionTypeArray,
    params: ActionParams,
    avail_flat: FlatMaskArray,
    vps_before: VPArray,
    belief: BeliefState | None,
    key: KeyScalar,
    batch_size: int,
    reward_mode: str,
    auto_reset: bool,
    number_placement: Literal["random", "spiral"],
    n_players: int,
    victory_points_to_win: int,
) -> _StepOut:
    """One whole ``BatchedSettlrlEnv.step``: gate the chosen action with the
    cached flat legality, apply it, score reward / termination, advance the
    belief (when tracked), auto-reset finished lanes, and refresh the VP /
    legality / acting-player caches -- one fused dispatch.

    ``key`` is threaded: split for auto-reset, returned unchanged when
    ``auto_reset`` is False.
    """
    legal = flat_legality(avail_flat, action_type, params.idx, params.target)
    # Trade proposals live outside the flat reverse lookup (their bundle
    # params are bit-packed over a domain the table only samples at 1:1), so
    # they are validated with the trade core directly.
    propose_ok = jax.vmap(_propose_trade_avail)(
        layout, state, (params.idx, params.target)
    )
    legal = jnp.where(action_type == ActionType.PROPOSE_TRADE, propose_ok, legal)
    # A lane whose current player comes into the step already at the win
    # threshold is terminal (the win can only have happened on that player's own
    # turn). Gating its action to INVALID freezes it -- the state, belief diff,
    # and reward all no-op -- which is how ``auto_reset=False`` lanes stay put
    # past termination. Under ``auto_reset`` finished lanes were replaced with
    # fresh 0-VP boards last step, so ``was_done`` is always False there and this
    # gate never fires.
    was_done = jnp.any(
        (state.current_player[:, None] == jnp.arange(n_players))
        & (vps_before >= victory_points_to_win),
        axis=1,
    )  # (B,)
    legal = legal & ~was_done
    applied, result = jax.vmap(
        functools.partial(apply_action, victory_points_to_win=victory_points_to_win),
        in_axes=(0, 0, 0, 0, 0),
    )(layout, state, action_type, params, legal)

    vps_after = _total_vp_v(applied)
    # Rulebook p.5: a player only wins during their own turn, so a lane ends
    # when its *current* player is at the threshold (an opponent crowned with
    # Longest Road by a settlement break keeps playing until their turn starts;
    # END_TURN's rotation makes this the turn-start claim). Mirrors
    # ``awards.current_player_won``, which upgrades the result code.
    is_current = applied.current_player[:, None] == jnp.arange(n_players)  # (B, P)
    done_lane = jnp.any(is_current & (vps_after >= victory_points_to_win), axis=1)

    if belief is not None:
        # Diff the pre-reset transition (sound for every lane); auto-reset
        # lanes then restart from the empty-board belief. A frozen lane's gated
        # action is INVALID (state unchanged), so its diff is a belief no-op.
        belief = _update_belief_v(belief, state, applied, action_type, params)
        if auto_reset:
            belief = _tree_where_lane(
                done_lane, make_belief(batch_size, n_players), belief
            )

    if reward_mode == "vp_delta":
        reward = (vps_after - vps_before).astype(jnp.float32)
    else:  # "sparse": +1 to the winner, only on the step that reaches the win.
        # ``done_lane & ~was_done`` is the *transition* into terminal, so a
        # frozen lane (already won) is not re-credited each step it sits there.
        reward = (is_current & (done_lane & ~was_done)[:, None]).astype(jnp.float32)
    terminations = jnp.broadcast_to(done_lane[:, None], (batch_size, n_players))

    if auto_reset:
        key, subkey = jax.random.split(key)
        new_layout, new_state = _auto_reset_core(
            layout, applied, done_lane, subkey, batch_size, number_placement, n_players
        )
        # A reset lane is a fresh board (0 VP everywhere); the others kept
        # ``applied``, whose VPs were just computed.
        new_vps = jnp.where(done_lane[:, None], 0, vps_after)
    else:
        # Finished lanes are frozen, not reset: their action was gated to INVALID
        # above (``was_done``), so ``applied`` already equals the terminal state
        # and stays there for every further step. Callers manage episode
        # boundaries themselves (e.g. aec.py).
        new_layout, new_state = layout, applied
        new_vps = vps_after

    new_avail = jax.vmap(flat_available_for)(new_layout, new_state)
    agent_sel = _agent_selection_v(new_state)
    return _StepOut(
        layout=new_layout,
        state=new_state,
        reward=reward,
        terminations=terminations,
        result=result,
        vps=new_vps,
        avail=new_avail,
        agent_sel=agent_sel,
        belief=belief,
        key=key,
    )


# The per-step env fields refreshed by ``_env_step_core``, bundled so the scan
# carry and the env assignment stay in one place: (reward, terminations,
# result, agent_sel).
_StepExtras = tuple[RewardArray, DoneArray, ResultCode, AgentSelectionArray]

Actor = Callable[
    [
        KeyScalar,
        BoardLayout,
        BoardState,
        "BeliefState | None",
        FlatMaskArray,
        AgentSelectionArray,
    ],
    tuple[ActionTypeArray, ActionParams],
]
"""A traceable per-step action source for :meth:`BatchedSettlrlEnv.rollout`:
``(step key, layout, state, belief, flat legality, acting players) -> one
(action_type, params) per lane``. ``belief`` is None unless the env tracks
beliefs. Runs inside the rollout's ``lax.scan``, so it must be pure."""


@functools.partial(
    jax.jit,
    static_argnames=(
        "n_steps",
        "batch_size",
        "reward_mode",
        "auto_reset",
        "number_placement",
        "n_players",
        "victory_points_to_win",
        "actor",
    ),
)
def _rollout_core(
    layout: BoardLayout,
    state: BoardState,
    avail_flat: FlatMaskArray,
    vps: VPArray,
    belief: BeliefState | None,
    extras: _StepExtras,
    env_key: KeyScalar,
    sample_key: KeyScalar,
    n_steps: int,
    actor: Actor | None,
    batch_size: int,
    reward_mode: str,
    auto_reset: bool,
    number_placement: Literal["random", "spiral"],
    n_players: int,
    victory_points_to_win: int,
) -> tuple[
    BoardLayout,
    BoardState,
    FlatMaskArray,
    VPArray,
    BeliefState | None,
    _StepExtras,
    KeyScalar,
    RewardArray,
]:
    """``n_steps`` random-action env steps as one ``lax.scan``.

    Each iteration replays the per-step driver exactly -- split ``sample_key``,
    sample a legal action per lane (``_random_actions_b``), run
    ``_env_step_core`` -- so a rollout matches the equivalent
    ``random_actions`` + ``step`` loop trajectory for the same key. Returns
    the final carry plus the ``(B, n_players)`` reward summed over the window.
    """
    Carry = tuple[
        BoardLayout,
        BoardState,
        FlatMaskArray,
        VPArray,
        BeliefState | None,
        _StepExtras,
        KeyScalar,
        KeyScalar,
        RewardArray,
    ]

    def body(carry: Carry, _: None) -> tuple[Carry, None]:
        layout, state, avail, vps, belief, _extras, env_key, sample_key, cum = carry
        sample_key, k_act = jax.random.split(sample_key)
        if actor is None:
            atype, params = _random_actions_b(avail, k_act)
        else:
            atype, params = actor(k_act, layout, state, belief, avail, _extras[3])
        out = _env_step_core(
            layout,
            state,
            atype,
            params,
            avail,
            vps,
            belief,
            env_key,
            batch_size,
            reward_mode,
            auto_reset,
            number_placement,
            n_players,
            victory_points_to_win,
        )
        new_extras: _StepExtras = (
            out.reward,
            out.terminations,
            out.result,
            out.agent_sel,
        )
        new_carry: Carry = (
            out.layout,
            out.state,
            out.avail,
            out.vps,
            out.belief,
            new_extras,
            out.key,
            sample_key,
            cum + out.reward,
        )
        return new_carry, None

    cum0 = jnp.zeros((batch_size, n_players), dtype=jnp.float32)
    init: Carry = (
        layout,
        state,
        avail_flat,
        vps,
        belief,
        extras,
        env_key,
        sample_key,
        cum0,
    )
    carry, _ = jax.lax.scan(body, init, None, length=n_steps)
    layout, state, avail, vps, belief, extras, env_key, _sample_key, cum = carry
    return layout, state, avail, vps, belief, extras, env_key, cum
