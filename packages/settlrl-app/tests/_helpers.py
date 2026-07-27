"""Shared test helpers.

The game server runs no bots itself, so any test that seats or plays a bot needs
in-process bot services registered with it. Each service hosts ONE bot, so
``bot_app`` mounts a one-bot service per kind under ``/<kind>`` (one ASGI client
then reaches them all), and ``bot_registry`` stuffs a provider per bot straight
into a registry (white-box, no admin/HTTP/event-loop dance at setup time).
``BOT_KINDS`` is what such seats validate against when a ``GameSession`` is built
directly.
"""

from collections.abc import Sequence

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from settlrl_agents.service.app import create_app as create_bot_app
from settlrl_agents.service.bots import BUNDLED, make_bot
from settlrl_app.bots.providers import ProviderRegistry, RemoteBotProvider
from settlrl_game.session import GameSession

# The bot kinds the in-process services offer (used as ``external_kinds`` when a
# test constructs a GameSession with bot seats directly).
BOT_KINDS = frozenset(BUNDLED)


def play_out(session: GameSession) -> None:
    """Random legal moves until the game ends (a fast terminal position) --
    direct in-process stepping, no wall clock or HTTP driver involved."""
    for _ in range(50_000):
        if session.auto_step() is None:
            return
    raise AssertionError("game did not terminate")


def start_game(
    client: TestClient,
    seats: Sequence[str],
    *,
    headers: dict[str, str] | None = None,
    seed: int = 0,
    vp: int | None = None,
) -> tuple[str, dict[str, str]]:
    """Start a human game through a lobby (the only way to make one now).

    Opens a hotseat lobby the host holds entirely, retargets the non-human
    ``seats`` to their bot kinds, and starts it. Returns the new game id and a
    header carrying the host's seat tokens (plus any account ``headers``), so the
    caller can act on the game as the human seat(s)."""
    body: dict[str, object] = {"mode": "hotseat", "n_players": len(seats), "seed": seed}
    if vp is not None:
        body["victory_points_to_win"] = vp
    created = client.post("/api/lobbies", json=body, headers=headers).json()
    lobby_id, held = created["id"], created["tokens"]
    hdrs = {**(headers or {}), "X-Seat-Tokens": ",".join(held.values())}
    for seat, kind in enumerate(seats):
        if kind != "human":
            client.post(
                f"/api/lobbies/{lobby_id}/seats",
                json={"seat": seat, "kind": kind},
                headers=hdrs,
            )
    game_id = client.post(
        f"/api/lobbies/{lobby_id}/start", json={}, headers=hdrs
    ).json()["game_id"]
    return game_id, hdrs


def bot_game(client: TestClient, seats: Sequence[str], *, seed: int = 0) -> str:
    """Create an all-bot game (the one direct game create that remains) and
    return its id — for the bot-driver / finished-game tests."""
    return str(
        client.post("/api/games", json={"seed": seed, "seats": list(seats)}).json()[
            "id"
        ]
    )


def asgi_client(app: FastAPI) -> httpx.AsyncClient:
    """An async client dispatching in-process to ``app`` (no sockets)."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://svc"
    )


def bot_app(kinds: Sequence[str] = BUNDLED) -> FastAPI:
    """One ASGI app mounting a one-bot service per kind under ``/<kind>``."""
    root = FastAPI()
    for kind in kinds:
        root.mount(f"/{kind}", create_bot_app(make_bot(kind)))
    return root


def bot_registry(kinds: Sequence[str] = BUNDLED) -> ProviderRegistry:
    """A provider registry with an in-process one-bot service per kind, so a game
    server built with it can seat and play those kinds (random, greedy, …)."""
    client = asgi_client(bot_app(kinds))
    reg = ProviderRegistry(client=client)
    for kind in kinds:
        reg._remotes[kind] = RemoteBotProvider(
            f"http://svc/{kind}", make_bot(kind).info(), client
        )
    return reg
