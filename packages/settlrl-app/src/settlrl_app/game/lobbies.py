"""Pre-game lobbies: a configurable staging table that becomes a game on start.

A :class:`Lobby` holds only configuration and membership — never a
``GameSession``. The board shown in the room is a preview derived from the
seed; the engine is built exactly once, at start, by materialising the lobby
into a :class:`~settlrl_app.game.games.GameHandle`.

Keeping the two apart makes the illegal states unrepresentable: you cannot
configure a game that has begun, and you cannot start a table that still has open
seats. The mode fixes whether seats can be open:

- ``"online"`` — human seats start open; others join them and changes stream live.
- ``"hotseat"`` — one browser drives every human seat (all claimed at create), so
  there are no open seats and it can start at once.

Lobbies are in-memory only: they are pre-game and disposable (a lost lobby just
means re-hosting), so unlike games they are never journalled.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Literal

from settlrl_game.session import HUMAN

NumberPlacement = Literal["random", "spiral"]

_ID_BYTES = 8
_TOKEN_BYTES = 16

# A lobby no one has touched for this long is abandoned and may be evicted.
_IDLE_TTL_S = 1800.0

HOTSEAT = "hotseat"
ONLINE = "online"


@dataclass
class SeatClaims:
    """Who owns each human seat: the bearer token, plus the account / browser /
    display name behind it. The seat-ownership unit a lobby hands to the game it
    becomes — ownership is proven by ``token`` or ``user_id``; ``client_id`` is
    only the guest one-game-guard key, never an ownership proof."""

    tokens: dict[int, str] = field(default_factory=dict)
    users: dict[int, str] = field(default_factory=dict)
    clients: dict[int, str] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)

    def record(
        self,
        seat: int,
        token: str,
        user_id: str | None = None,
        client_id: str | None = None,
        name: str | None = None,
    ) -> None:
        self.tokens[seat] = token
        if user_id is not None:
            self.users[seat] = user_id
        elif client_id is not None:
            self.clients[seat] = client_id
        if name is not None:
            self.names[seat] = name

    def release(self, seat: int) -> None:
        self.tokens.pop(seat, None)
        self.users.pop(seat, None)
        self.clients.pop(seat, None)
        self.names.pop(seat, None)

    def owned(self, tokens: list[str], user_id: str | None = None) -> set[int]:
        """Seats the requester owns: any proven by a presented token, plus any
        tied to their account."""
        owned = {s for s, t in self.tokens.items() if t in set(tokens)}
        if user_id is not None:
            owned |= {s for s, u in self.users.items() if u == user_id}
        return owned

    def for_user(self, user_id: str) -> list[int]:
        return sorted(s for s, u in self.users.items() if u == user_id)

    def for_client(self, client_id: str) -> list[int]:
        return sorted(s for s, c in self.clients.items() if c == client_id)


class Lobby:
    """One pre-game table: its configuration, seat kinds, and claimed seats."""

    def __init__(
        self,
        lobby_id: str,
        *,
        mode: str,
        seed: int,
        number_placement: NumberPlacement,
        n_players: int,
        victory_points_to_win: int,
        listed: bool = False,
        searchable: bool = False,
    ) -> None:
        self.id = lobby_id
        self.mode = mode
        self.seed = seed
        self.number_placement = number_placement
        self.n_players = n_players
        self.victory_points_to_win = victory_points_to_win
        self.listed = listed
        self.searchable = searchable
        # Per-seat controller: HUMAN (claimed or, online, open) or a bot kind.
        self.kinds: list[str] = [HUMAN] * n_players
        self.seats = SeatClaims()
        self.chat: list[tuple[int | None, str]] = []
        # Set once started — the room's SSE carries it so everyone follows.
        self.started_game_id: str | None = None
        self.created_at = time.time()
        self.touched = time.monotonic()
        self.version = 0
        self.closed = False
        self.lock = asyncio.Lock()
        self._changed = asyncio.Event()

    def touch(self) -> None:
        self.touched = time.monotonic()

    def human_seats(self) -> list[int]:
        return [i for i, kind in enumerate(self.kinds) if kind == HUMAN]

    def open_human_seats(self) -> list[int]:
        """Human seats no one holds yet — joinable (online) and the reason a
        lobby isn't ready to start."""
        return [s for s in self.human_seats() if s not in self.seats.tokens]

    def ready(self) -> bool:
        """Every seat decided: each is a bot or a claimed human, none open. Start
        is allowed only here, so a half-empty table can never begin."""
        return not self.open_human_seats()

    def claim(
        self,
        seat: int | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
        name: str | None = None,
    ) -> tuple[int, str]:
        """Take ``seat`` (or the first open human seat) and mint its token.

        ``LookupError`` when no seat is open, ``ValueError`` when the seat is not
        an open human seat."""
        free = self.open_human_seats()
        if seat is None:
            if not free:
                raise LookupError("no open human seat")
            seat = free[0]
        elif seat not in self.human_seats():
            raise ValueError(f"seat {seat} is not a human seat")
        elif seat in self.seats.tokens:
            raise ValueError(f"seat {seat} is already claimed")
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        self.seats.record(seat, token, user_id, client_id, name)
        return seat, token

    def set_kind(self, seat: int, kind: str) -> None:
        """Retarget a seat (host control): a bot kind, or HUMAN (open online, or
        re-held by the host in a hotseat). Frees any prior claim on it."""
        if not 0 <= seat < self.n_players:
            raise ValueError(f"no seat {seat}")
        self.seats.release(seat)
        self.kinds[seat] = kind

    def bump(self) -> None:
        self.version += 1
        self.touch()
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()


class LobbyRegistry:
    """In-memory id-addressed lobbies. Mutated only on the event loop, so the
    dict needs no lock."""

    def __init__(self, max_lobbies: int = 64) -> None:
        self._lobbies: dict[str, Lobby] = {}
        self._max = max_lobbies

    def create(self, **config: object) -> Lobby:
        self._evict()
        lobby_id = secrets.token_urlsafe(_ID_BYTES)
        while lobby_id in self._lobbies:
            lobby_id = secrets.token_urlsafe(_ID_BYTES)
        lobby = Lobby(lobby_id, **config)  # type: ignore[arg-type]
        self._lobbies[lobby_id] = lobby
        return lobby

    def get(self, lobby_id: str) -> Lobby | None:
        lobby = self._lobbies.get(lobby_id)
        if lobby is not None:
            lobby.touch()
        return lobby

    def remove(self, lobby_id: str) -> None:
        """Drop a lobby and wake its waiters so the SSE/streams exit."""
        lobby = self._lobbies.pop(lobby_id, None)
        if lobby is not None:
            lobby.closed = True
            lobby.bump()

    def all(self) -> list[Lobby]:
        return list(self._lobbies.values())

    def open(self) -> list[Lobby]:
        """Listed, joinable lobbies for the public list — online, with an open
        seat, not yet started. Newest first."""
        rooms = [
            lobby
            for lobby in self._lobbies.values()
            if lobby.listed
            and lobby.mode == ONLINE
            and lobby.started_game_id is None
            and lobby.open_human_seats()
        ]
        return sorted(rooms, key=lambda lobby: lobby.created_at, reverse=True)

    def live_for_user(self, user_id: str) -> Lobby | None:
        """An un-started lobby this account holds a seat in (the one-game guard)."""
        for lobby in self._lobbies.values():
            if lobby.started_game_id is None and lobby.seats.for_user(user_id):
                return lobby
        return None

    def live_for_client(self, client_id: str) -> Lobby | None:
        for lobby in self._lobbies.values():
            if lobby.started_game_id is None and lobby.seats.for_client(client_id):
                return lobby
        return None

    def _evict(self) -> None:
        while len(self._lobbies) >= self._max:
            now = time.monotonic()
            stale = [
                lobby_id
                for lobby_id, lobby in self._lobbies.items()
                if lobby.started_game_id is not None
                or lobby.touched < now - _IDLE_TTL_S
            ]
            if not stale:
                # Nothing reclaimable; drop the oldest-touched to bound memory.
                stale = [min(self._lobbies, key=lambda i: self._lobbies[i].touched)]
            for lobby_id in stale:
                self.remove(lobby_id)
