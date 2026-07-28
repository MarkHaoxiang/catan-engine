"""Tests for the bot-provider layer: the registry, the admin API that mutates
it, the incremental/resync wire, and end-to-end remote dispatch through the
driver.

In-process remote services are reached with an ``httpx.AsyncClient`` over an
``ASGITransport`` bound to a one-bot service app, so no sockets are opened. The
registry's ``register`` is async; the unit tests drive it under ``asyncio.run``,
while the end-to-end tests register through the admin route so everything runs on
the app's own event loop.
"""

import asyncio
import time
from collections.abc import Iterator
from typing import Any

import pytest
from _helpers import asgi_client, bot_app
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from settlrl_agents.service.app import create_app as create_bot_app
from settlrl_agents.service.bots import make_bot
from settlrl_app.bots.providers import (
    ProviderRegistry,
    RemoteBotError,
    RemoteBotProvider,
)
from settlrl_app.game.games import GameRegistry
from settlrl_app.server import create_app
from settlrl_game.actions import move_for_flat
from settlrl_game.botproto import ActRequest, BotInfo
from settlrl_game.session import GameSession


def _bot_service(kind: str) -> FastAPI:
    """A real one-bot service app for ``kind``."""
    return create_bot_app(make_bot(kind))


def _info_only_app(name: str, counts: tuple[int, ...] = (2, 3, 4)) -> FastAPI:
    """A minimal remote service that only self-identifies (no real play)."""
    app = FastAPI()

    @app.get("/info")
    def info() -> dict[str, object]:
        return {"name": name, "title": name.title(), "counts": list(counts)}

    return app


# -- registry -----------------------------------------------------------------


def test_register_routes_the_bot_name() -> None:
    async def go() -> None:
        reg = ProviderRegistry(client=asgi_client(_info_only_app("alphacatan")))
        prov = await reg.register("http://svc")
        assert prov.name == "alphacatan"
        assert "alphacatan" in reg.catalog()  # the remote bot joined
        assert reg.remote_for("alphacatan") is not None
        assert reg.remote_for("nope") is None  # unknown kind
        assert reg.providers()[0] == {"name": "alphacatan", "base_url": "http://svc"}
        await reg.aclose()

    asyncio.run(go())


def test_register_unreachable_service_errors() -> None:
    async def go() -> None:
        # A client pointed at an app with no /info route -> 404 -> error.
        reg = ProviderRegistry(client=asgi_client(FastAPI()))
        with pytest.raises(RemoteBotError):
            await reg.register("http://svc")
        await reg.aclose()

    asyncio.run(go())


def test_unregister() -> None:
    async def go() -> None:
        reg = ProviderRegistry(client=asgi_client(_info_only_app("alphacatan")))
        await reg.register("http://svc")
        assert reg.unregister("alphacatan")
        assert "alphacatan" not in reg.catalog()
        assert not reg.unregister("alphacatan")
        await reg.aclose()

    asyncio.run(go())


def test_offers_only_registered_kinds() -> None:
    async def go() -> None:
        reg = ProviderRegistry(client=asgi_client(_info_only_app("alphacatan")))
        assert reg.catalog() == {}  # nothing until a provider registers
        await reg.register("http://svc")
        assert set(reg.catalog()) == {"alphacatan"}
        await reg.aclose()

    asyncio.run(go())


def test_two_services_coexist() -> None:
    async def go() -> None:
        reg = ProviderRegistry(client=asgi_client(bot_app(["random", "greedy"])))
        await reg.register("http://svc/random")
        await reg.register("http://svc/greedy")
        assert set(reg.catalog()) == {"random", "greedy"}
        await reg.aclose()

    asyncio.run(go())


# -- incremental / resync wire ------------------------------------------------


def test_provider_resyncs_when_service_reports_a_different_count() -> None:
    """A service that answers 409 with the count it actually holds makes the
    provider replay from there; the second request carries that ``base``."""
    # A short real opening, so the replayed moves are valid structured actions.
    sess = GameSession(seed=1, n_players=2, seats=["human", "human"])
    history: list[int] = []
    for _ in range(3):
        f = sess.legal_flat()[0]
        history.append(f)
        sess.apply(f)

    calls: list[dict[str, int]] = []
    app = FastAPI()

    @app.post("/act")
    def act(req: ActRequest) -> dict[str, Any]:
        calls.append({"base": req.base, "moves": len(req.moves)})
        if len(calls) == 1:
            raise HTTPException(409, detail={"resync": True, "have": 2})
        return {"move": move_for_flat(history[0]).model_dump()}

    async def go() -> None:
        client = asgi_client(app)
        prov = RemoteBotProvider(
            "http://svc", BotInfo(name="stub", title="Stub"), client
        )
        flat = await prov.act("g", {}, history, seat=0)
        assert flat == history[0]
        assert calls[0]["base"] == 0 and calls[1]["base"] == 2
        assert calls[1]["moves"] == len(history) - 2
        await client.aclose()

    asyncio.run(go())


# -- admin API ----------------------------------------------------------------


def _login(client: TestClient, email: str) -> str:
    client.post("/api/auth/register", json={"email": email, "password": "password1"})
    return str(
        client.post(
            "/api/auth/login", data={"username": email, "password": "password1"}
        ).json()["access_token"]
    )


def test_admin_register_provider_requires_admin() -> None:
    reg = ProviderRegistry(client=asgi_client(_info_only_app("alphacatan")))
    with TestClient(
        create_app(
            GameRegistry(),
            providers=reg,
            admin_emails=frozenset({"a@x.com"}),
        )
    ) as client:
        body = {"base_url": "http://svc"}
        # Anonymous and non-admin are refused.
        assert client.post("/api/admin/bot-providers", json=body).status_code == 401
        utok = _login(client, "u@x.com")
        assert (
            client.post(
                "/api/admin/bot-providers",
                json=body,
                headers={"Authorization": f"Bearer {utok}"},
            ).status_code
            == 403
        )
        # The admin can register; the bot then shows in the public catalog.
        h = {"Authorization": f"Bearer {_login(client, 'a@x.com')}"}
        resp = client.post("/api/admin/bot-providers", json=body, headers=h)
        assert resp.status_code == 201 and resp.json()["name"] == "alphacatan"
        assert "alphacatan" in client.get("/api/bots").json()
        listed = client.get("/api/admin/bot-providers", headers=h).json()
        assert listed[0] == {"name": "alphacatan", "base_url": "http://svc"}
        assert (
            client.delete("/api/admin/bot-providers/alphacatan", headers=h).status_code
            == 204
        )
        assert (
            client.delete("/api/admin/bot-providers/alphacatan", headers=h).status_code
            == 404
        )
        assert "alphacatan" not in client.get("/api/bots").json()
    asyncio.run(reg.aclose())


# -- end-to-end dispatch ------------------------------------------------------


def _register(client: TestClient, base_url: str = "http://svc") -> None:
    tok = _login(client, "a@x.com")
    resp = client.post(
        "/api/admin/bot-providers",
        json={"base_url": base_url},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert resp.status_code == 201, resp.text


def _drive_to_terminal(client: TestClient, body: dict[str, object]) -> dict[str, Any]:
    gid = client.post("/api/games", json=body).json()["id"]
    snap: dict[str, Any] = {}
    # Wall-clock deadline, not a poll count: loaded CI runners need far
    # longer than a dev machine to drive a full game.
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        snap = client.get(f"/api/games/{gid}").json()
        if snap["status"]["terminal"]:
            break
        time.sleep(0.05)
    return snap


@pytest.fixture()
def remote_only_client() -> Iterator[TestClient]:
    """A game server with no local agent execution; all bots run in a separate
    in-process one-bot service, registered via the admin route once the loop is
    up."""
    reg = ProviderRegistry(client=asgi_client(_bot_service("random")))
    with TestClient(
        create_app(
            GameRegistry(),
            providers=reg,
            bot_delay=0.0,
            admin_emails=frozenset({"a@x.com"}),
        )
    ) as client:
        _register(client)
        yield client
    asyncio.run(reg.aclose())


def test_remote_bots_play_a_game_to_completion(remote_only_client: TestClient) -> None:
    snap = _drive_to_terminal(
        remote_only_client,
        {"seed": 1, "n_players": 2, "seats": ["random", "random"], "claim": "none"},
    )
    assert snap["status"]["terminal"] and snap["status"]["winner"] in (0, 1)


def test_create_rejects_unknown_remote_kind(remote_only_client: TestClient) -> None:
    # With only "random" registered, an unadvertised kind is unknown -> 422.
    resp = remote_only_client.post(
        "/api/games", json={"seed": 1, "n_players": 2, "seats": ["nope", "random"]}
    )
    assert resp.status_code == 422


def test_remote_failure_falls_back_and_game_progresses() -> None:
    """A remote service that errors on every move must not stall the game: the
    driver falls back to a local random move, so moves keep coming."""
    app = _info_only_app("flaky")

    @app.post("/act")
    def act(_: ActRequest) -> dict[str, int]:
        raise HTTPException(status_code=500, detail="boom")

    reg = ProviderRegistry(client=asgi_client(app))
    with TestClient(
        create_app(
            GameRegistry(),
            providers=reg,
            bot_delay=0.0,
            admin_emails=frozenset({"a@x.com"}),
        )
    ) as client:
        _register(client)
        gid = client.post(
            "/api/games",
            json={
                "seed": 2,
                "n_players": 2,
                "seats": ["flaky", "flaky"],
                "claim": "none",
            },
        ).json()["id"]
        snap: dict[str, Any] = {}
        for _ in range(200):
            snap = client.get(f"/api/games/{gid}").json()
            if snap["status"]["terminal"] or len(snap["log"]) > 12:
                break
            time.sleep(0.05)
        assert snap["status"]["terminal"] or len(snap["log"]) > 12
    asyncio.run(reg.aclose())
