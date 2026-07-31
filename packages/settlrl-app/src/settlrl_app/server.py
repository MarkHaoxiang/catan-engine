"""The FastAPI app: the composition root. ``create_app`` wires the one async
:class:`~settlrl_app.storage.db.Database`, the game registry and its journal store,
the auth system, the bot providers, and the bot/timeout driver tasks into a
shared :class:`~settlrl_app.api.deps.Deps`, then mounts the routers
(:mod:`settlrl_app.api.routers`) and the SPA. Tests build isolated apps with
their own registries instead of sharing module state.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response
from starlette.types import Scope

from settlrl_app.api import routers
from settlrl_app.api.deps import Deps, ReplaySlot, needs_driver
from settlrl_app.bots.providers import ProviderRegistry
from settlrl_app.config import Settings
from settlrl_app.game.driver import start_game_driver
from settlrl_app.game.games import GameHandle, GameRegistry, restore_games
from settlrl_app.game.lobbies import LobbyRegistry
from settlrl_app.game.matchmaking import Matchmaker
from settlrl_app.storage.auth import Auth
from settlrl_app.storage.db import Database
from settlrl_app.storage.store import GameStore

log = logging.getLogger(__name__)


def _build_registry(
    games: GameRegistry | None,
    database: Database,
    state_dir: str | None,
    max_active: int,
) -> tuple[GameRegistry, GameStore | None]:
    """The app's registry and (when persistence is on) its journal store: the
    registry passed in by tests, a fresh persistent one backed by ``database``,
    or a fresh in-memory one. The store's games are replayed in at startup."""
    if games is not None:
        return games, None
    if state_dir:
        store = GameStore(database)
        return GameRegistry(max_active=max_active, store=store), store
    return GameRegistry(max_active=max_active), None


def _database(user_db: str | None, state_dir: str | None) -> Database:
    """The one async db: an explicit path, ``settlrl.db`` under the state dir
    when persistence is on, or an ephemeral in-memory db (tests / stateless)."""
    if user_db is not None:
        return Database(user_db)
    if state_dir:
        return Database(str(Path(state_dir) / "settlrl.db"))
    return Database(None)


def create_app(
    games: GameRegistry | None = None,
    bot_delay: float = 0.65,
    root_path: str = "",
    max_body_bytes: int = 2 * 1024 * 1024,
    state_dir: str | None = None,
    turn_timeout: float = 0.0,
    max_active: int = 16,
    user_db: str | None = None,
    admin_emails: frozenset[str] = frozenset(),
    providers: ProviderRegistry | None = None,
    bot_providers: frozenset[str] = frozenset(),
) -> FastAPI:
    """Build the app around its own registry (tests pass theirs in).

    ``bot_delay`` paces the server-side bot driver (seconds between bot moves,
    so clients can animate each one). ``root_path`` is the proxy prefix the app
    is served under. ``max_body_bytes`` rejects oversized request bodies before
    they are parsed. ``state_dir`` turns on persistence: games are journalled
    there and replayed back on the next startup (ignored when ``games`` is
    passed). ``turn_timeout`` (seconds, 0 = off) auto-plays a human turn that has
    gone idle that long, so an abandoned game still finishes. ``max_active`` caps
    how many games run at once; beyond it, new creators are queued
    (``POST /api/games`` returns their place in line). ``user_db`` overrides the
    shared SQLite path (defaults to ``settlrl.db`` under ``state_dir``, else
    in-memory); ``admin_emails`` are granted admin on register / login.
    ``bot_providers`` are bot-service base URLs registered on startup (retried
    until each is reachable), so their bots are seatable without a manual admin
    call — equivalent to ``POST /api/admin/bot-providers`` for each.
    """
    database = _database(user_db, state_dir)
    registry, store = _build_registry(games, database, state_dir, max_active)
    auth = Auth(database, admin_emails=admin_emails)
    bots = providers if providers is not None else ProviderRegistry()
    bots_owned = providers is None  # only close the HTTP client we created
    # The live bot/timeout driver tasks, so the lifespan can cancel them on
    # shutdown; each removes itself when it ends (game over or evicted).
    drivers: set[asyncio.Task[None]] = set()

    def spawn_driver(handle: GameHandle) -> None:
        # Idempotent: create, restart, and a seat turned into a bot may all ask,
        # but a game must have exactly one driver.
        if handle.driver_started:
            return
        handle.driver_started = True
        task = start_game_driver(handle, bot_delay, turn_timeout, bots)
        drivers.add(task)
        task.add_done_callback(drivers.discard)

    matchmaker = Matchmaker(
        registry,
        bots,
        spawn_driver,
        store=store,
        turn_timeout=turn_timeout,
    )
    deps = Deps(
        registry=registry,
        bots=bots,
        auth=auth,
        replays=ReplaySlot(),
        spawn_driver=spawn_driver,
        turn_timeout=turn_timeout,
        lobbies=LobbyRegistry(),
        store=store,
        matchmaker=matchmaker,
    )

    async def register_provider(url: str) -> None:
        # The bot service may still be starting (JAX import is slow), so retry
        # with capped backoff until it answers /info; cancelled on shutdown.
        delay = 0.5
        while True:
            try:
                provider = await bots.register(url)
                log.info("registered bot provider %r at %s", provider.name, url)
                return
            except Exception as exc:
                log.warning("bot provider %s not ready (%s); retrying", url, exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await database.init()
        if store is not None:
            store.start()
            await restore_games(registry, store)
        # Resume pacing/timeouts for any games replayed in from the store.
        for handle in registry.all_handles():
            if needs_driver(handle, turn_timeout):
                spawn_driver(handle)
        registrations = [
            asyncio.create_task(register_provider(u)) for u in bot_providers
        ]
        yield
        for task in (*drivers, *registrations):
            task.cancel()
        if store is not None:
            await store.aclose()  # flush queued writes before the engine closes
        if bots_owned:
            await bots.aclose()
        await database.dispose()

    app = FastAPI(title="Settlrl", lifespan=lifespan, root_path=root_path)

    @app.middleware("http")
    async def _limit_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declared = request.headers.get("content-length")
        if (
            declared is not None
            and declared.isdigit()
            and int(declared) > max_body_bytes
        ):
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    for module in (
        routers.games,
        routers.replay,
        routers.bots,
        routers.me,
        routers.leaderboard,
        routers.quickmatch,
        routers.lobbies,
        routers.admin,
    ):
        app.include_router(module.build(deps))
    app.include_router(auth.router)

    if _dist.exists():
        app.mount("/", _SPAStaticFiles(directory=_dist, html=True), name="static")
    return app


class _SPAStaticFiles(StaticFiles):
    """StaticFiles that falls back to index.html for unknown paths.

    The frontend uses client-side routing (/play, /replay/:id), so a deep link
    or a refresh on those paths must still return the SPA entry point rather
    than a 404. Only extension-less paths fall back; a request for a missing
    file (e.g. a stale /assets/*.js) still returns 404.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and "." not in path.rsplit("/", 1)[-1]:
                return FileResponse(_dist / "index.html")
            raise


# Serve built frontend when it exists (src/settlrl_app/server.py -> package root)
_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"


# The uvicorn entry point (settlrl_app.server:app), configured from the
# environment (see settlrl_app.config.Settings).
_settings = Settings()
app = create_app(
    root_path=_settings.root_path,
    state_dir=_settings.state_dir,
    turn_timeout=_settings.turn_timeout_s,
    max_active=_settings.max_active,
    user_db=_settings.user_db,
    admin_emails=_settings.admin_emails,
    bot_providers=_settings.bot_providers,
)
