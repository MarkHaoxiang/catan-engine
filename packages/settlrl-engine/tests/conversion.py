"""Translate the engine's batched ``Board`` into ``settlrl-reference`` games.

The reference (``settlrl_game.reference``) is an independent, plain-Python statement of
the Settlrl rules used as the differential oracle. The engine stores a *batch* of
games as JAX arrays with its own vertex/edge/tile indexing; the reference is one
game at a time with its own indexing. The two share only the cube-coordinate
*convention* for the physical board, so positions are translated by cube
coordinate (the engine's ``board/layout.py`` host-side lookups give the cube of
any engine index; ``settlrl_game.reference.board`` maps a cube back to its own index).

This module exposes:

- ``to_reference(board)`` -> one ``Game`` per batch lane (the headline
  conversion: an engine board becomes a *list* of reference games).
- ``to_reference_single`` / ``state_to_game`` -- single-lane helpers.
- ``assert_states_match`` -- a full structural comparison of one engine lane
  against its reference game, used by the equivalence test.
- ``assert_legality_match`` -- engine flat legality vs reference ``is_legal``,
  row for row over the whole flat table.
- thin oracle wrappers (``distance_rule_ok``, ``longest_road_length``,
  ``port_ratio``, ``recompute_longest_road``, ...) used by the differential
  tests. Each converts then delegates to the reference.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import settlrl_game.reference as ref
from settlrl_engine.board import Board
from settlrl_engine.board import layout as eng_layout
from settlrl_engine.board.dev_cards import N_DEV_CARD_TYPES
from settlrl_engine.board.layout import (
    N_EDGES,
    N_PORTS,
    N_TILES,
    N_VERTICES,
    PORT_V,
    BoardLayout,
)
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.board.state import NO_INDEX, BoardState, GamePhase
from settlrl_engine.mechanics.action import ActionType
from settlrl_engine.mechanics.flat import (
    FLAT_ATYPE,
    FLAT_IDX,
    FLAT_TARGET,
    N_FLAT,
    flat_available_b,
)
from settlrl_engine.mechanics.trade import (
    _COUNT_BITS,
    _COUNT_MASK,
    _PARTNER_BITS,
    pack_trade,
)
from settlrl_game.reference import board as ref_board

# --- index bridges (engine index -> reference index, via cube coords) -------

_ENG2REF_VERTEX: list[int] = [
    ref_board.cube_to_vertex(eng_layout.vertex_cube(v)) for v in range(N_VERTICES)
]
_ENG2REF_TILE: list[int] = [
    ref_board.cube_to_tile(eng_layout.tile_cube(t)) for t in range(N_TILES)
]


def _eng2ref_edge(e: int) -> int:
    a_cube, b_cube = eng_layout.edge_cubes(e)
    return ref_board.edge_between(
        ref_board.cube_to_vertex(a_cube), ref_board.cube_to_vertex(b_cube)
    )


_ENG2REF_EDGE: list[int] = [_eng2ref_edge(e) for e in range(N_EDGES)]


def _invert(forward: list[int], size: int) -> list[int]:
    out = [0] * size
    for engine_idx, ref_idx in enumerate(forward):
        out[ref_idx] = engine_idx
    return out


# reference index -> engine index, for translating reference actions back.
_REF2ENG_VERTEX: list[int] = _invert(_ENG2REF_VERTEX, N_VERTICES)
_REF2ENG_EDGE: list[int] = _invert(_ENG2REF_EDGE, N_EDGES)
_REF2ENG_TILE: list[int] = _invert(_ENG2REF_TILE, N_TILES)

# --- enum bridges -----------------------------------------------------------

_PHASE: dict[int, ref.Phase] = {
    int(GamePhase.SETUP_SETTLEMENT): ref.Phase.SETUP_SETTLEMENT,
    int(GamePhase.SETUP_ROAD): ref.Phase.SETUP_ROAD,
    int(GamePhase.ROLL): ref.Phase.ROLL,
    int(GamePhase.DISCARD): ref.Phase.DISCARD,
    int(GamePhase.MOVE_ROBBER): ref.Phase.MOVE_ROBBER,
    int(GamePhase.MAIN): ref.Phase.MAIN,
    int(GamePhase.TRADE_RESPONSE): ref.Phase.TRADE_RESPONSE,
    int(GamePhase.GAME_OVER): ref.Phase.GAME_OVER,
}


def _resource(tile_value: int) -> ref.Resource | None:
    """Engine ``Tile`` value -> reference ``Resource`` (``None`` for the desert)."""
    return None if tile_value >= N_RESOURCES else ref.Resource(tile_value)


def _port_type(port_value: int) -> ref.PortType:
    """Engine ``Port`` value -> reference ``PortType`` (5 = the generic 3:1)."""
    if port_value >= N_RESOURCES:
        return ref.PortType.GENERIC
    return ref.PortType(ref.Resource(port_value))


def _owner_or_none(value: int) -> int | None:
    return None if value == NO_INDEX else value


def _trade_counts(partner: int, counts: np.ndarray) -> tuple[int, ...] | None:
    """A pending trade's per-resource counts (``None`` when no proposal is
    pending: the engine zeroes the count vectors, the reference uses ``None``)."""
    return None if partner == NO_INDEX else tuple(int(c) for c in counts)


# --- layout conversion ------------------------------------------------------

# An all-desert, harbour-free layout for the rules that don't read the board
# allocation (placement, longest road, awards, the victim mask). Lets a bare
# ``BoardState`` become a ``Game`` without a paired ``BoardLayout``.
_PLACEHOLDER_LAYOUT = ref.Layout(
    tile_resource=tuple([None] * N_TILES),
    tile_number=tuple([0] * N_TILES),
    ports=(),
)


def _layout_from_engine(layout: BoardLayout, b: int) -> ref.Layout:
    tile_resource_eng = np.asarray(layout.tile_resource[b])
    tile_number_eng = np.asarray(layout.tile_number[b])
    port_alloc = np.asarray(layout.port_allocation[b])
    port_v = np.asarray(PORT_V)

    tile_resource: list[ref.Resource | None] = [None] * N_TILES
    tile_number: list[int] = [0] * N_TILES
    for t in range(N_TILES):
        rt = _ENG2REF_TILE[t]
        tile_resource[rt] = _resource(int(tile_resource_eng[t]))
        tile_number[rt] = int(tile_number_eng[t])

    ports = tuple(
        ref.Port(
            _port_type(int(port_alloc[p])),
            (
                _ENG2REF_VERTEX[int(port_v[p, 0])],
                _ENG2REF_VERTEX[int(port_v[p, 1])],
            ),
        )
        for p in range(N_PORTS)
    )
    return ref.Layout(tuple(tile_resource), tuple(tile_number), ports)


# --- state conversion -------------------------------------------------------


def _build_game(layout: ref.Layout, state: BoardState, b: int) -> ref.Game:
    vertex_owner = np.asarray(state.vertex_owner[b])
    vertex_type = np.asarray(state.vertex_type[b])
    edge_road = np.asarray(state.edge_road[b])
    player_resources = np.asarray(state.player_resources[b])
    dev_hand = np.asarray(state.dev_hand[b])
    dev_bought = np.asarray(state.dev_bought[b])
    dev_deck = np.asarray(state.dev_deck[b])
    knights = np.asarray(state.knights_played[b])
    pending = np.asarray(state.pending_discard[b])
    current_player = int(state.current_player[b])
    n_players = state.n_players

    players: list[ref.Player] = []
    for p in range(n_players):
        resources = {
            ref.Resource(r): int(player_resources[p, r]) for r in range(N_RESOURCES)
        }
        held = {ref.DevCard(c): int(dev_hand[p, c]) for c in range(N_DEV_CARD_TYPES)}
        # ``dev_bought`` tracks only the *current* player's buys this turn.
        bought = {
            ref.DevCard(c): (int(dev_bought[c]) if p == current_player else 0)
            for c in range(N_DEV_CARD_TYPES)
        }
        players.append(
            ref.Player(
                resources=resources,
                dev_cards=held,
                dev_bought_this_turn=bought,
                knights_played=int(knights[p]),
            )
        )

    buildings: dict[int, tuple[int, ref.Building]] = {}
    for v in range(N_VERTICES):
        owner = int(vertex_owner[v])
        if owner != 0:
            vt = int(vertex_type[v])
            # Randomized-occupancy unit tests set owners without a building kind;
            # those rules only read presence/owner, so default an unset kind to a
            # settlement. Real states always carry a valid 1/2 here.
            kind = ref.Building(vt) if vt in (1, 2) else ref.Building.SETTLEMENT
            buildings[_ENG2REF_VERTEX[v]] = (owner - 1, kind)

    roads: dict[int, int] = {}
    for e in range(N_EDGES):
        r = int(edge_road[e])
        if r != 0:
            roads[_ENG2REF_EDGE[e]] = r - 1

    return ref.Game(
        layout=layout,
        robber=_ENG2REF_TILE[int(state.robber[b])],
        players=players,
        buildings=buildings,
        roads=roads,
        dev_deck={ref.DevCard(c): int(dev_deck[c]) for c in range(N_DEV_CARD_TYPES)},
        phase=_PHASE[int(state.phase[b])],
        current_player=current_player,
        n_players=n_players,
        setup_index=int(state.setup_index[b]),
        dice_roll=int(state.dice_roll[b]),
        has_rolled=bool(int(state.has_rolled[b])),
        dev_played_this_turn=bool(int(state.dev_played[b])),
        free_roads=int(state.free_roads[b]),
        pending_discard=[int(pending[p]) for p in range(n_players)],
        trade_partner=_owner_or_none(int(state.trade_partner[b])),
        trade_give=_trade_counts(
            int(state.trade_partner[b]), np.asarray(state.trade_give[b])
        ),
        trade_receive=_trade_counts(
            int(state.trade_partner[b]), np.asarray(state.trade_receive[b])
        ),
        longest_road_owner=_owner_or_none(int(state.longest_road_owner[b])),
        largest_army_owner=_owner_or_none(int(state.largest_army_owner[b])),
        longest_road_len=int(state.longest_road_len[b]),
    )


def to_reference(board: Board) -> list[ref.Game]:
    """Convert a batched engine ``Board`` into one reference ``Game`` per lane."""
    layout, state = board
    batch = state.phase.shape[0]
    return [_build_game(_layout_from_engine(layout, b), state, b) for b in range(batch)]


def to_reference_single(board: Board, b: int = 0) -> ref.Game:
    layout, state = board
    return _build_game(_layout_from_engine(layout, b), state, b)


def state_to_game(state: BoardState, b: int = 0) -> ref.Game:
    """A reference game from a bare ``BoardState`` (placeholder all-desert layout).

    For the rules that ignore the board allocation: placement legality, longest
    road, the award reassignments, and the robber victim mask.
    """
    return _build_game(_PLACEHOLDER_LAYOUT, state, b)


# ===========================================================================
# Oracle wrappers (convert, then delegate to the reference)
# ===========================================================================


def distance_rule_ok(state: BoardState, vertex: int, b: int = 0) -> bool:
    return state_to_game(state, b).distance_rule_ok(_ENG2REF_VERTEX[vertex])


def settlement_connected(
    state: BoardState, player: int, vertex: int, b: int = 0
) -> bool:
    return state_to_game(state, b).settlement_connected(player, _ENG2REF_VERTEX[vertex])


def road_placeable(state: BoardState, player: int, edge: int, b: int = 0) -> bool:
    return state_to_game(state, b).road_placeable(player, _ENG2REF_EDGE[edge])


def longest_road_length(state: BoardState, player: int, b: int = 0) -> int:
    return state_to_game(state, b).longest_road_length(player)


def robber_victims(state: BoardState, tile: int, current: int, b: int = 0) -> list[int]:
    return state_to_game(state, b).robber_victims(_ENG2REF_TILE[tile], current)


def player_total_vp(state: BoardState, player: int, b: int = 0) -> int:
    return state_to_game(state, b).total_vp(player)


def port_ratio(
    state: BoardState, layout: BoardLayout, player: int, give: int, b: int = 0
) -> int:
    game = to_reference_single((layout, state), b)
    return game.port_ratio(player, ref.Resource(give))


def _award_owner_value(owner: int | None) -> int:
    return NO_INDEX if owner is None else owner


def recompute_largest_army(state: BoardState, b: int = 0) -> BoardState:
    """Reassign Largest Army; returns a ``BoardState`` with the holder updated."""
    game = state_to_game(state, b)
    game.recompute_largest_army()
    owner = _award_owner_value(game.largest_army_owner)
    return state._replace(largest_army_owner=state.largest_army_owner.at[b].set(owner))


def recompute_longest_road(state: BoardState, b: int = 0) -> BoardState:
    """Reassign Longest Road; returns a ``BoardState`` with holder + length set."""
    game = state_to_game(state, b)
    game.recompute_longest_road()
    owner = _award_owner_value(game.longest_road_owner)
    return state._replace(
        longest_road_owner=state.longest_road_owner.at[b].set(owner),
        longest_road_len=state.longest_road_len.at[b].set(game.longest_road_len),
    )


def distribute_resources(
    layout: BoardLayout, state: BoardState, roll: int, b: int = 0
) -> BoardState:
    """Pay out production for ``roll``; returns a ``BoardState`` with new hands."""
    game = to_reference_single((layout, state), b)
    gains = game.production(roll)
    hands = np.asarray(state.player_resources[b]).astype(np.int64)
    for p, per_resource in gains.items():
        for resource, amount in per_resource.items():
            hands[p, int(resource)] += amount
    new_row = jnp.asarray(hands.astype(np.uint8))
    return state._replace(player_resources=state.player_resources.at[b].set(new_row))


def grant_setup_resources(
    layout: BoardLayout, state: BoardState, vertex: int, player: int, b: int = 0
) -> BoardState:
    """Grant the 2nd-settlement bonus; returns a ``BoardState`` with new hands."""
    game = to_reference_single((layout, state), b)
    game.grant_setup_resources(_ENG2REF_VERTEX[vertex], player)
    hand = [game.players[player].resources[ref.Resource(r)] for r in range(N_RESOURCES)]
    new_row = jnp.asarray(np.asarray(hand, dtype=np.uint8))
    return state._replace(
        player_resources=state.player_resources.at[b, player].set(new_row)
    )


# ===========================================================================
# Reference action -> engine (ActionType, params) translation
# ===========================================================================


def to_engine_action(action: ref.Action) -> tuple[int, int, int]:
    """Translate a reference action into the engine's ``(action_type, idx,
    target)`` tuple (engine indexing). ``target`` is ``-1`` when unused / for
    "steal from no one". Discard carries only its resource -- the engine derives
    the discarder (the first owing player) from the state."""
    match action:
        case ref.SetupSettlement(vertex=v):
            return int(ActionType.SETUP_SETTLEMENT), _REF2ENG_VERTEX[v], -1
        case ref.SetupRoad(edge=e):
            return int(ActionType.SETUP_ROAD), _REF2ENG_EDGE[e], -1
        case ref.Roll():
            return int(ActionType.ROLL_DICE), 0, -1
        case ref.Discard(resource=r):
            return int(ActionType.DISCARD), int(r), -1
        case ref.MoveRobber(tile=t, victim=victim):
            return (
                int(ActionType.MOVE_ROBBER),
                _REF2ENG_TILE[t],
                -1 if victim is None else victim,
            )
        case ref.BuildRoad(edge=e):
            return int(ActionType.BUILD_ROAD), _REF2ENG_EDGE[e], -1
        case ref.BuildSettlement(vertex=v):
            return int(ActionType.BUILD_SETTLEMENT), _REF2ENG_VERTEX[v], -1
        case ref.BuildCity(vertex=v):
            return int(ActionType.BUILD_CITY), _REF2ENG_VERTEX[v], -1
        case ref.BuyDevelopmentCard():
            return int(ActionType.BUY_DEVELOPMENT_CARD), 0, -1
        case ref.PlayKnight(tile=t, victim=victim):
            return (
                int(ActionType.PLAY_KNIGHT),
                _REF2ENG_TILE[t],
                -1 if victim is None else victim,
            )
        case ref.PlayRoadBuilding():
            return int(ActionType.PLAY_ROAD_BUILDING), 0, -1
        case ref.PlayYearOfPlenty(first=a, second=b):
            return int(ActionType.PLAY_YEAR_OF_PLENTY), int(a), int(b)
        case ref.PlayMonopoly(resource=r):
            return int(ActionType.PLAY_MONOPOLY), int(r), -1
        case ref.MaritimeTrade(give=g, receive=r):
            return int(ActionType.MARITIME_TRADE), int(g), int(r)
        case ref.ProposeTrade(partner=p, give=g, receive=r):
            idx, target = pack_trade(list(g), list(r), p)
            return int(ActionType.PROPOSE_TRADE), idx, target
        case ref.AcceptTrade():
            return int(ActionType.ACCEPT_TRADE), 0, -1
        case ref.RejectTrade():
            return int(ActionType.REJECT_TRADE), 0, -1
        case ref.EndTurn():
            return int(ActionType.END_TURN), 0, -1
        case _:  # pragma: no cover - the match above is exhaustive over Action
            raise AssertionError(f"unhandled reference action: {action!r}")


# ===========================================================================
# Flat-table row -> reference action, and the legality cross-check
# ===========================================================================

_FLAT_ATYPE = np.asarray(FLAT_ATYPE)
_FLAT_IDX = np.asarray(FLAT_IDX)
_FLAT_TARGET = np.asarray(FLAT_TARGET)


def _unpack_counts(packed: int) -> tuple[int, ...]:
    return tuple(
        (packed >> (_COUNT_BITS * r)) & _COUNT_MASK for r in range(N_RESOURCES)
    )


def _victim(target: int) -> int | None:
    return None if target == -1 else target  # -1 = steal from no one


# Row (idx, target) -> reference action, the inverse of ``to_engine_action``
# (Discard is handled in ``flat_row_action``: it also needs the game).
_ROW_DECODE: dict[int, Callable[[int, int], ref.Action]] = {
    ActionType.SETUP_SETTLEMENT: lambda i, t: ref.SetupSettlement(_ENG2REF_VERTEX[i]),
    ActionType.SETUP_ROAD: lambda i, t: ref.SetupRoad(_ENG2REF_EDGE[i]),
    ActionType.ROLL_DICE: lambda i, t: ref.Roll(),
    ActionType.MOVE_ROBBER: lambda i, t: ref.MoveRobber(_ENG2REF_TILE[i], _victim(t)),
    ActionType.BUILD_ROAD: lambda i, t: ref.BuildRoad(_ENG2REF_EDGE[i]),
    ActionType.BUILD_SETTLEMENT: lambda i, t: ref.BuildSettlement(_ENG2REF_VERTEX[i]),
    ActionType.BUILD_CITY: lambda i, t: ref.BuildCity(_ENG2REF_VERTEX[i]),
    ActionType.BUY_DEVELOPMENT_CARD: lambda i, t: ref.BuyDevelopmentCard(),
    ActionType.PLAY_KNIGHT: lambda i, t: ref.PlayKnight(_ENG2REF_TILE[i], _victim(t)),
    ActionType.PLAY_ROAD_BUILDING: lambda i, t: ref.PlayRoadBuilding(),
    ActionType.PLAY_YEAR_OF_PLENTY: lambda i, t: ref.PlayYearOfPlenty(
        ref.Resource(i), ref.Resource(t)
    ),
    ActionType.PLAY_MONOPOLY: lambda i, t: ref.PlayMonopoly(ref.Resource(i)),
    ActionType.MARITIME_TRADE: lambda i, t: ref.MaritimeTrade(
        ref.Resource(i), ref.Resource(t)
    ),
    ActionType.PROPOSE_TRADE: lambda i, t: ref.ProposeTrade(
        partner=t & ((1 << _PARTNER_BITS) - 1),
        give=_unpack_counts(i),
        receive=_unpack_counts(t >> _PARTNER_BITS),
    ),
    ActionType.ACCEPT_TRADE: lambda i, t: ref.AcceptTrade(),
    ActionType.REJECT_TRADE: lambda i, t: ref.RejectTrade(),
    ActionType.END_TURN: lambda i, t: ref.EndTurn(),
}


def flat_row_action(row: int, game: ref.Game) -> ref.Action:
    """The reference action named by flat-table row ``row``.

    Stochastic outcome fields are left unset (``is_legal`` ignores them). A
    Discard row names the first owing player, matching the engine's acting
    player during DISCARD (player 0 when nobody owes: the phase gate makes the
    row illegal on both sides regardless).
    """
    atype, idx, target = (
        int(_FLAT_ATYPE[row]),
        int(_FLAT_IDX[row]),
        int(_FLAT_TARGET[row]),
    )
    if atype == ActionType.DISCARD:
        owing = (p for p in range(game.n_players) if game.pending_discard[p] > 0)
        return ref.Discard(next(owing, 0), ref.Resource(idx))
    return _ROW_DECODE[atype](idx, target)


def assert_legality_match(board: Board, game: ref.Game, b: int = 0) -> None:
    """Assert the engine's flat legality mask equals reference ``is_legal``
    row for row.

    The differential playout only ever drives reference-legal actions, so on
    its own it cannot see an engine that is *more* permissive than the
    reference, nor one that rejects some reference-legal moves while accepting
    another. This closes both directions for every move the flat table names
    (arbitrary trade bundles, which the table cannot name, are probed by the
    bundle fuzz in ``test_reference_equivalence``).
    """
    mask = np.asarray(flat_available_b(board[0], board[1])[b])
    for row in range(N_FLAT):
        action = flat_row_action(row, game)
        expected = game.is_legal(action)
        assert bool(mask[row]) == expected, (
            f"legality mismatch on flat row {row} ({action!r}) in phase "
            f"{game.phase}: engine={bool(mask[row])} reference={expected}"
        )


# ===========================================================================
# Full-state comparison (for the equivalence playout test)
# ===========================================================================


def assert_states_match(
    board: Board, game: ref.Game, b: int = 0, ignore_phase: bool = False
) -> None:
    """Assert engine lane ``b`` and the reference ``game`` describe one state.

    ``ignore_phase`` is for the winning step: the engine signals a completed game
    through its action *result code* and leaves ``phase`` untouched, whereas the
    reference moves to ``GAME_OVER``. Every other field still matches.
    """
    expected = to_reference_single(board, b)

    def check(name: str, lhs: object, rhs: object) -> None:
        assert lhs == rhs, f"{name} mismatch: engine={lhs!r} reference={rhs!r}"

    if not ignore_phase:
        check("phase", expected.phase, game.phase)
    check("n_players", expected.n_players, game.n_players)
    check("current_player", expected.current_player, game.current_player)
    check("setup_index", expected.setup_index, game.setup_index)
    check("dice_roll", expected.dice_roll, game.dice_roll)
    check("has_rolled", expected.has_rolled, game.has_rolled)
    check("dev_played", expected.dev_played_this_turn, game.dev_played_this_turn)
    check("free_roads", expected.free_roads, game.free_roads)
    check("pending_discard", expected.pending_discard, game.pending_discard)
    check("trade_partner", expected.trade_partner, game.trade_partner)
    check("trade_give", expected.trade_give, game.trade_give)
    check("trade_receive", expected.trade_receive, game.trade_receive)
    check("robber", expected.robber, game.robber)
    check("buildings", expected.buildings, game.buildings)
    check("roads", expected.roads, game.roads)
    check("dev_deck", expected.dev_deck, game.dev_deck)
    check("longest_road_owner", expected.longest_road_owner, game.longest_road_owner)
    check("longest_road_len", expected.longest_road_len, game.longest_road_len)
    check("largest_army_owner", expected.largest_army_owner, game.largest_army_owner)
    for p in range(expected.n_players):
        check(
            f"resources[{p}]", expected.players[p].resources, game.players[p].resources
        )
        check(
            f"dev_cards[{p}]", expected.players[p].dev_cards, game.players[p].dev_cards
        )
        check(
            f"dev_bought[{p}]",
            expected.players[p].dev_bought_this_turn,
            game.players[p].dev_bought_this_turn,
        )
        check(
            f"knights[{p}]",
            expected.players[p].knights_played,
            game.players[p].knights_played,
        )
