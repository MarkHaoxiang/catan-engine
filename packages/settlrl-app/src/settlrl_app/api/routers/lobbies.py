"""Lobby routes (``/api/lobbies*``): a pre-game table you configure, others
join, and the host starts — at which point it materialises into a game.

A lobby never holds a ``GameSession``; the room renders a board *preview* from
the seed. ``POST /api/lobbies/{id}/start`` builds the engine, copies the claimed
seats into a fresh game (so everyone keeps their seat token), and hands back the
game id. Start is allowed only when no human seat is still open, so a half-empty
table can never begin.
"""

from collections.abc import AsyncIterator
from random import Random
from typing import Annotated, Literal

import anyio.to_thread
import settlrl_game.reference as ref
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from settlrl_game.convert import board_to_model
from settlrl_game.models import BoardModel
from settlrl_game.session import HUMAN, GameSession
from sse_starlette.sse import EventSourceResponse
from starlette.responses import Response

from settlrl_app.api.deps import ClientId, Deps, SeatTokens, needs_driver, tokens, uid
from settlrl_app.game.games import QueuePosition, RegistryFullError, win_threshold
from settlrl_app.game.lobbies import HOTSEAT, ONLINE, Lobby
from settlrl_app.storage.db import User


class _CreateLobbyRequest(BaseModel):
    mode: Literal["hotseat", "online"] = "online"
    n_players: Literal[2, 4] = 4
    seed: int = 0
    number_placement: Literal["random", "spiral"] = "random"
    victory_points_to_win: int | None = None
    listed: bool = False
    searchable: bool = False


class _ConfigureLobbyRequest(BaseModel):
    seed: int | None = None
    n_players: Literal[2, 4] | None = None
    number_placement: Literal["random", "spiral"] | None = None
    victory_points_to_win: int | None = None
    listed: bool | None = None
    searchable: bool | None = None


class _SeatRequest(BaseModel):
    seat: int
    kind: str  # "human" (open / host-held) or a bot kind from GET /api/bots


class _JoinRequest(BaseModel):
    seat: int | None = None


class _ChatRequest(BaseModel):
    text: str
    player: int | None = None


class _StartRequest(BaseModel):
    # Re-sent each poll when the server was at its game cap (a queued start).
    ticket: str | None = None


class _CreatedLobbyModel(BaseModel):
    id: str
    tokens: dict[int, str]  # the host's seat tokens


class _LobbyModel(BaseModel):
    """One lobby as a requester sees it: its config, seats, and preview board."""

    id: str
    mode: str
    seed: int
    number_placement: str
    n_players: int
    victory_points_to_win: int
    kinds: list[str]
    seats_claimed: list[int]
    seat_names: list[str | None]
    your_seats: list[int]
    listed: bool
    searchable: bool
    ready: bool
    started_game_id: str | None
    board: BoardModel
    chat: list[dict[str, object]]


class _LobbyListModel(BaseModel):
    id: str
    n_players: int
    number_placement: str
    open_seats: int
    searchable: bool
    created_at: float


class _StartedModel(BaseModel):
    game_id: str


class _QueuedModel(BaseModel):
    queued: Literal[True] = True
    ticket: str
    position: int
    total: int


def _preview_board(lobby: Lobby) -> BoardModel:
    layout = ref.random_layout(Random(lobby.seed), lobby.number_placement)
    game = ref.Game.new(layout, ref.desert_tile(layout), n_players=lobby.n_players)
    return board_to_model(game)


def _lobby_model(lobby: Lobby, owned: set[int]) -> _LobbyModel:
    return _LobbyModel(
        id=lobby.id,
        mode=lobby.mode,
        seed=lobby.seed,
        number_placement=lobby.number_placement,
        n_players=lobby.n_players,
        victory_points_to_win=lobby.victory_points_to_win,
        kinds=list(lobby.kinds),
        seats_claimed=sorted(lobby.seats.tokens),
        seat_names=[lobby.seats.names.get(i) for i in range(lobby.n_players)],
        your_seats=sorted(owned),
        listed=lobby.listed,
        searchable=lobby.searchable,
        ready=lobby.ready(),
        started_game_id=lobby.started_game_id,
        board=_preview_board(lobby),
        chat=[{"player": p, "text": t} for p, t in lobby.chat],
    )


def build(deps: Deps) -> APIRouter:
    router = APIRouter()
    registry, lobbies, bots = deps.registry, deps.lobbies, deps.bots
    CurrentUser = Annotated[User | None, Depends(deps.auth.optional_user)]

    def lobby_of(lobby_id: str) -> Lobby:
        lobby = lobbies.get(lobby_id)
        if lobby is None:
            raise HTTPException(status_code=404, detail="no such lobby")
        return lobby

    def owner_label(user: User | None) -> str | None:
        return user.email.split("@", 1)[0] if user is not None else None

    @router.get("/api/lobbies")
    def list_lobbies() -> list[_LobbyListModel]:
        """Listed, joinable online lobbies (newest first)."""
        return [
            _LobbyListModel(
                id=lobby.id,
                n_players=lobby.n_players,
                number_placement=lobby.number_placement,
                open_seats=len(lobby.open_human_seats()),
                searchable=lobby.searchable,
                created_at=lobby.created_at,
            )
            for lobby in lobbies.open()
        ]

    @router.post("/api/lobbies")
    def create_lobby(
        req: _CreateLobbyRequest,
        user: CurrentUser = None,
        x_client_id: ClientId = None,
    ) -> _CreatedLobbyModel:
        """Open a lobby. The host takes seat 0 (online) or every human seat
        (hotseat). Listing or Quick-Match visibility needs a signed-in account."""
        if (req.listed or req.searchable) and user is None:
            raise HTTPException(status_code=401, detail="sign in to list a game")
        deps.guard_one_game(user, x_client_id)
        lobby = lobbies.create(
            mode=req.mode,
            seed=req.seed,
            number_placement=req.number_placement,
            n_players=req.n_players,
            victory_points_to_win=(
                req.victory_points_to_win or win_threshold(req.n_players)
            ),
            listed=req.listed,
            searchable=req.searchable,
        )
        # Hotseat: the host drives every human seat. Online: the host takes seat 0
        # and the rest stay open for joiners.
        claiming = (
            lobby.human_seats() if req.mode == HOTSEAT else lobby.human_seats()[:1]
        )
        held = dict(
            lobby.claim(seat, uid(user), x_client_id, owner_label(user))
            for seat in claiming
        )
        return _CreatedLobbyModel(id=lobby.id, tokens=held)

    @router.get("/api/lobbies/{lobby_id}")
    def get_lobby(
        lobby_id: str, user: CurrentUser = None, x_seat_tokens: SeatTokens = None
    ) -> _LobbyModel:
        lobby = lobby_of(lobby_id)
        return _lobby_model(lobby, lobby.seats.owned(tokens(x_seat_tokens), uid(user)))

    @router.get("/api/lobbies/{lobby_id}/events")
    async def lobby_events(
        lobby_id: str, user: CurrentUser = None, x_seat_tokens: SeatTokens = None
    ) -> EventSourceResponse:
        """The requester's lobby snapshot now, then again on every change (joins,
        seat edits, config, chat) and once more carrying ``started_game_id``."""
        lobby = lobby_of(lobby_id)
        seat_tokens, user_id = tokens(x_seat_tokens), uid(user)

        async def stream() -> AsyncIterator[str]:
            seen = -1
            while True:
                changed = lobby._changed
                if lobby.closed:
                    return
                if lobby.version != seen:
                    seen = lobby.version
                    owned = lobby.seats.owned(seat_tokens, user_id)
                    yield _lobby_model(lobby, owned).model_dump_json()
                else:
                    await changed.wait()

        return EventSourceResponse(stream(), ping=15)

    @router.post("/api/lobbies/{lobby_id}/configure")
    def configure_lobby(
        lobby_id: str,
        req: _ConfigureLobbyRequest,
        user: CurrentUser = None,
        x_seat_tokens: SeatTokens = None,
    ) -> _LobbyModel:
        """Host-only (seat-0 owner) pre-start config. Changing the player count
        rebuilds the seat list, keeping seat 0 (the host)."""
        lobby = lobby_of(lobby_id)
        owned = lobby.seats.owned(tokens(x_seat_tokens), uid(user))
        if 0 not in owned:
            raise HTTPException(status_code=403, detail="only the host can configure")
        if req.listed and user is None:
            raise HTTPException(status_code=401, detail="sign in to list a game")
        if req.n_players is not None and req.n_players != lobby.n_players:
            _resize(lobby, req.n_players)
        if req.seed is not None:
            lobby.seed = req.seed
        if req.number_placement is not None:
            lobby.number_placement = req.number_placement
        if req.victory_points_to_win is not None:
            lobby.victory_points_to_win = req.victory_points_to_win
        if req.listed is not None:
            lobby.listed = req.listed
        if req.searchable is not None:
            lobby.searchable = req.searchable
        lobby.bump()
        return _lobby_model(lobby, lobby.seats.owned(tokens(x_seat_tokens), uid(user)))

    def _resize(lobby: Lobby, n: int) -> None:
        """Change the seat count, keeping claims on seats that survive; in a
        hotseat the host fills every new human seat too."""
        survivors = {
            s: (
                lobby.seats.tokens[s],
                lobby.seats.users.get(s),
                lobby.seats.clients.get(s),
                lobby.seats.names.get(s),
            )
            for s in list(lobby.seats.tokens)
            if s < n
        }
        lobby.n_players = n
        lobby.kinds = [
            lobby.kinds[i] if i < len(lobby.kinds) else HUMAN for i in range(n)
        ]
        for attr in (
            lobby.seats.tokens,
            lobby.seats.users,
            lobby.seats.clients,
            lobby.seats.names,
        ):
            attr.clear()
        for seat, (tok, user_id, client_id, name) in survivors.items():
            lobby.seats.record(seat, tok, user_id, client_id, name)
        if lobby.mode == HOTSEAT:
            host = _host_identity(lobby)
            for seat in lobby.open_human_seats():
                lobby.claim(seat, *host)

    @router.post("/api/lobbies/{lobby_id}/seats")
    def set_seat(
        lobby_id: str,
        req: _SeatRequest,
        user: CurrentUser = None,
        x_seat_tokens: SeatTokens = None,
    ) -> _LobbyModel:
        """Host-only: retarget a seat — a bot kind, or open it (online) / take it
        (hotseat) as human. The host can change its own seats (the whole table in
        a hotseat) but not seat 0 (its anchor) nor a seat another player holds."""
        lobby = lobby_of(lobby_id)
        owned = lobby.seats.owned(tokens(x_seat_tokens), uid(user))
        if 0 not in owned:
            raise HTTPException(status_code=403, detail="only the host can set seats")
        if req.seat == 0:
            raise HTTPException(status_code=409, detail="the host keeps seat 0")
        if req.seat in lobby.seats.tokens and req.seat not in owned:
            raise HTTPException(status_code=409, detail="seat is taken by a player")
        if req.kind != HUMAN and req.kind not in bots.catalog():
            raise HTTPException(
                status_code=422, detail=f"unknown bot kind: {req.kind!r}"
            )
        lobby.set_kind(req.seat, req.kind)
        if req.kind == HUMAN and lobby.mode == HOTSEAT:
            lobby.claim(req.seat, *_host_identity(lobby))
        lobby.bump()
        return _lobby_model(lobby, lobby.seats.owned(tokens(x_seat_tokens), uid(user)))

    def _host_identity(lobby: Lobby) -> tuple[str | None, str | None, str | None]:
        return (
            lobby.seats.users.get(0),
            lobby.seats.clients.get(0),
            lobby.seats.names.get(0),
        )

    @router.post("/api/lobbies/{lobby_id}/join")
    def join_lobby(
        lobby_id: str,
        req: _JoinRequest,
        user: CurrentUser = None,
        x_client_id: ClientId = None,
    ) -> _CreatedLobbyModel:
        """Take an open human seat in an online lobby."""
        lobby = lobby_of(lobby_id)
        if lobby.mode != ONLINE:
            raise HTTPException(status_code=409, detail="this lobby is not joinable")
        deps.guard_one_game(user, x_client_id, allow=lobby_id)
        try:
            seat, token = lobby.claim(
                req.seat, uid(user), x_client_id, owner_label(user)
            )
        except (LookupError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        lobby.bump()
        return _CreatedLobbyModel(id=lobby.id, tokens={seat: token})

    @router.post("/api/lobbies/{lobby_id}/leave")
    def leave_lobby(
        lobby_id: str, user: CurrentUser = None, x_seat_tokens: SeatTokens = None
    ) -> Response:
        """The host (seat 0) closes the whole lobby; anyone else frees their
        seat. 403 if you hold no seat here."""
        lobby = lobby_of(lobby_id)
        owned = lobby.seats.owned(tokens(x_seat_tokens), uid(user))
        if not owned:
            raise HTTPException(status_code=403, detail="you are not in this lobby")
        if 0 in owned:
            lobbies.remove(lobby_id)
        else:
            for seat in owned:
                lobby.seats.release(seat)
            lobby.bump()
        return Response(status_code=204)

    @router.post("/api/lobbies/{lobby_id}/chat")
    def lobby_chat(
        lobby_id: str,
        req: _ChatRequest,
        user: CurrentUser = None,
        x_seat_tokens: SeatTokens = None,
    ) -> _LobbyModel:
        text = req.text.strip()
        if not text or len(text) > 500:
            raise HTTPException(status_code=422, detail="chat must be 1-500 chars")
        owned = lobby_of(lobby_id).seats.owned(tokens(x_seat_tokens), uid(user))
        if req.player is not None and req.player not in owned:
            raise HTTPException(status_code=403, detail="not your seat")
        lobby = lobby_of(lobby_id)
        lobby.chat.append((req.player, text))
        lobby.bump()
        return _lobby_model(lobby, owned)

    @router.post("/api/lobbies/{lobby_id}/start")
    async def start_lobby(
        lobby_id: str,
        req: _StartRequest,
        user: CurrentUser = None,
        x_seat_tokens: SeatTokens = None,
    ) -> _StartedModel | _QueuedModel:
        """Host-only: materialise the lobby into a game. 409 while any human seat
        is still open (start a table only once every seat is decided). A queued
        body with a place in line when the server is at its game cap."""
        lobby = lobby_of(lobby_id)
        if 0 not in lobby.seats.owned(tokens(x_seat_tokens), uid(user)):
            raise HTTPException(status_code=403, detail="only the host can start")
        if not lobby.ready():
            raise HTTPException(status_code=409, detail="seats are still open")
        if lobby.started_game_id is not None:
            return _StartedModel(game_id=lobby.started_game_id)

        session = GameSession(seed=lobby.seed, n_players=lobby.n_players)
        await anyio.to_thread.run_sync(
            lambda: session.reset(
                lobby.seed,
                number_placement=lobby.number_placement,
                seats=lobby.kinds,
                external_kinds=bots.remote_kinds(),
                victory_points_to_win=lobby.victory_points_to_win,
            )
        )
        try:
            seated = registry.admit(session, req.ticket)
        except RegistryFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(seated, QueuePosition):
            return _QueuedModel(
                ticket=seated.ticket, position=seated.position, total=seated.total
            )

        # Move the claimed seats over verbatim, so every lobby seat token keeps
        # working in the game, then seed the chat and start play.
        seated.claims = dict(lobby.seats.tokens)
        seated.claim_users = dict(lobby.seats.users)
        seated.claim_clients = dict(lobby.seats.clients)
        seated.claim_names = dict(lobby.seats.names)
        seated.listed = lobby.listed
        seated.searchable = lobby.searchable
        if seated.journal is not None:
            for seat, token in seated.claims.items():
                seated.journal.claim(seat, token, seated.claim_users.get(seat))
        for player, msg in lobby.chat:
            seated.session.add_chat(player, msg)
            if seated.journal is not None:
                seated.journal.chat(player, msg)
        if needs_driver(seated, deps.turn_timeout):
            deps.spawn_driver(seated)
        seated.bump()
        lobby.started_game_id = seated.id  # the SSE carries everyone into the game
        lobby.bump()
        return _StartedModel(game_id=seated.id)

    return router
