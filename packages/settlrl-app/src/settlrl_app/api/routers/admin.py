"""Admin status: a superuser-only health view of the running server — uptime,
the live games, and the registered bot services."""

import time
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from settlrl_app.api.deps import Deps


class _GameSummary(BaseModel):
    """One live game's at-a-glance state for the admin list."""

    id: str
    n_players: int
    phase: str
    terminal: bool
    moves: int  # the change counter (one per applied move)
    seats: list[str]  # seat kinds
    open_seats: int  # unclaimed human seats
    listed: bool
    searchable: bool
    created_at: float


class _AdminStatus(BaseModel):
    uptime_seconds: float
    games_active: int  # non-terminal games held
    games_total: int  # active + finished retained for history
    games_capacity: int  # the registry cap
    bot_providers: list[dict[str, object]]  # registered remote services
    bot_kinds: list[str]  # seatable kinds from the catalog
    games: list[_GameSummary]


def build(deps: Deps) -> APIRouter:
    router = APIRouter()
    registry, bots, auth = deps.registry, deps.bots, deps.auth

    @router.get("/api/admin/status")
    def status(_: Annotated[object, Depends(auth.admin_user)]) -> _AdminStatus:
        """Server health for the admin page (superuser only)."""
        handles = sorted(
            registry.all_handles(), key=lambda h: h.created_at, reverse=True
        )
        games = [
            _GameSummary(
                id=h.id,
                n_players=h.session.n_players,
                phase=h.session.status().phase,
                terminal=h.session.terminal(),
                moves=h.version,
                seats=list(h.session.seats),
                open_seats=len(h.open_human_seats()),
                listed=h.listed,
                searchable=h.searchable,
                created_at=h.created_at,
            )
            for h in handles
        ]
        return _AdminStatus(
            uptime_seconds=time.time() - deps.started_at,
            games_active=registry.active_count(),
            games_total=len(handles),
            games_capacity=registry.max_games,
            bot_providers=bots.providers(),
            bot_kinds=sorted(bots.catalog()),
            games=games,
        )

    return router
