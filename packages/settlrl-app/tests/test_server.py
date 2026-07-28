"""Route-level tests for the FastAPI server.

Each test builds its own app around its own registry (``create_app``), so
nothing is shared between tests. These cover what the routes themselves own —
auth and status codes, locking, request plumbing — plus the SPA fallback. Human
games are staged through a lobby (``_human_game``); the all-bot create route
backs live-game tests (``bot_game``). Finished-game tests (record/replay) play
a session out directly (``_finished_game``), not through the HTTP bot driver —
driving a whole game to completion over the wire is wall-clock bound and too
flaky for CI.
"""

import json
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from _helpers import BOT_KINDS, bot_game, bot_registry, play_out
from fastapi import FastAPI
from fastapi.testclient import TestClient
from settlrl_app.game.games import GameRegistry
from settlrl_app.server import create_app
from settlrl_game.session import GameSession


@pytest.fixture()
def registry() -> GameRegistry:
    return GameRegistry()


@pytest.fixture()
def client(registry: GameRegistry) -> Iterator[TestClient]:
    # The ``with`` form runs the lifespan and keeps one event loop alive for the
    # client's lifetime, so background driver tasks and the per-game asyncio
    # locks share a single loop across requests. Bot-backed so the bot seats
    # these tests use validate and play (the server runs no bots itself).
    with TestClient(create_app(registry, providers=bot_registry())) as client:
        yield client


def _human_game(
    client: TestClient, seats: Sequence[str], *, seed: int = 0
) -> tuple[str, dict[str, str]]:
    """Start a human game through a hotseat lobby (one browser holds every human
    seat); return its id and the per-seat tokens for the human seats."""
    created = client.post(
        "/api/lobbies", json={"mode": "hotseat", "n_players": len(seats), "seed": seed}
    ).json()
    lobby_id, held = created["id"], created["tokens"]
    hdr = {"X-Seat-Tokens": ",".join(held.values())}
    for seat, kind in enumerate(seats):
        if kind != "human":
            client.post(
                f"/api/lobbies/{lobby_id}/seats",
                json={"seat": seat, "kind": kind},
                headers=hdr,
            )
    game = client.post(f"/api/lobbies/{lobby_id}/start", json={}, headers=hdr).json()[
        "game_id"
    ]
    tokens = {s: t for s, t in held.items() if seats[int(s)] == "human"}
    return game, tokens


def _hdr(tokens: dict[str, str]) -> dict[str, str]:
    return {"X-Seat-Tokens": ",".join(tokens.values())}


def test_two_player_games_play_to_fifteen_others_to_ten(client: TestClient) -> None:
    game2, t2 = _human_game(client, ["human", "human"])
    st2 = client.get(f"/api/games/{game2}", headers=_hdr(t2)).json()["status"]
    assert st2["victory_points_to_win"] == 15

    game4, t4 = _human_game(client, ["human", "random", "random", "random"])
    st4 = client.get(f"/api/games/{game4}", headers=_hdr(t4)).json()["status"]
    assert st4["victory_points_to_win"] == 10


def test_all_bot_create_validates_its_seats(client: TestClient) -> None:
    assert (
        client.post("/api/games", json={"seats": ["random", "random"]}).status_code
        == 200
    )
    # A human seat must be hosted through a lobby; unknown / wrong-count bots 422.
    assert (
        client.post("/api/games", json={"seats": ["human", "random"]}).status_code
        == 422
    )
    assert (
        client.post("/api/games", json={"seats": ["nope", "random"]}).status_code == 422
    )
    assert client.post("/api/games", json={"seats": ["random"]}).status_code == 422


def test_full_registry_of_active_games_returns_503() -> None:
    with TestClient(
        create_app(GameRegistry(max_games=1), providers=bot_registry())
    ) as c:
        assert (
            c.post("/api/games", json={"seats": ["random", "random"]}).status_code
            == 200
        )
        assert (
            c.post("/api/games", json={"seats": ["random", "random"]}).status_code
            == 503
        )


def test_unknown_game_404s(client: TestClient) -> None:
    assert client.get("/api/games/nope").status_code == 404
    assert client.post("/api/games/nope/action", json={"flat": 0}).status_code == 404


def test_action_requires_the_acting_seats_token(client: TestClient) -> None:
    game, tokens = _human_game(client, ["human", "human"])
    a, b = tokens["0"], tokens["1"]
    flat = client.get(f"/api/games/{game}", headers={"X-Seat-Tokens": a}).json()[
        "actions"
    ][0]["flat"]
    # No token, and the wrong seat's token: refused before legality.
    assert (
        client.post(f"/api/games/{game}/action", json={"flat": flat}).status_code == 403
    )
    assert (
        client.post(
            f"/api/games/{game}/action",
            json={"flat": flat},
            headers={"X-Seat-Tokens": b},
        ).status_code
        == 403
    )
    resp = client.post(
        f"/api/games/{game}/action", json={"flat": flat}, headers={"X-Seat-Tokens": a}
    )
    assert resp.status_code == 200


def test_illegal_action_returns_409(client: TestClient) -> None:
    game, tokens = _human_game(client, ["human", "random", "random", "random"])
    legal = {
        a["flat"]
        for a in client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()[
            "actions"
        ]
    }
    illegal = next(f for f in range(1000) if f not in legal)
    resp = client.post(
        f"/api/games/{game}/action", json={"flat": illegal}, headers=_hdr(tokens)
    )
    assert resp.status_code == 409


def test_same_seed_reproduces_the_board(client: TestClient) -> None:
    game = bot_game(client, ["random", "random"], seed=7)
    again = bot_game(client, ["random", "random"], seed=7)
    a = client.get(f"/api/games/{game}").json()["board"]["tiles"]
    b = client.get(f"/api/games/{again}").json()["board"]["tiles"]
    assert a == b


@contextmanager
def _live_server(app: FastAPI) -> Iterator[int]:
    """Run ``app`` on a real uvicorn server (ephemeral port), yielding the port.

    SSE streams need this: TestClient buffers whole responses and would block
    on an open stream forever.
    """
    config = uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=1)
    server = uvicorn.Server(config)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    thread = threading.Thread(target=server.run, args=([sock],), daemon=True)
    thread.start()
    try:
        while not server.started:
            time.sleep(0.02)
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _next_event(lines: Iterator[str]) -> dict[str, object]:
    """The next SSE data event (skipping keepalives), parsed."""
    for _, line in zip(range(50), lines, strict=False):
        if line.startswith("data:"):
            return dict(json.loads(line[5:].strip()))
    raise AssertionError("no data event arrived")


def _start_all_human(http: httpx.Client, n: int = 4) -> tuple[str, dict[str, str]]:
    """Host an all-human hotseat lobby and start it, over a live-server client."""
    created = http.post("/api/lobbies", json={"mode": "hotseat", "n_players": n}).json()
    hdr = _hdr(dict(created["tokens"]))
    game = http.post(
        f"/api/lobbies/{created['id']}/start", json={}, headers=hdr
    ).json()["game_id"]
    return game, hdr


def test_events_stream_snapshot_now_then_on_every_change() -> None:
    with (
        _live_server(create_app(GameRegistry())) as port,
        httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as http,
    ):
        # All-human so no bot driver mutates the game mid-test.
        game, hdr = _start_all_human(http)
        with http.stream("GET", f"/api/games/{game}/events", headers=hdr) as resp:
            lines = resp.iter_lines()
            first = _next_event(lines)
            assert first["status"]["your_turn"] is True  # type: ignore[index]
            flat = first["actions"][0]["flat"]  # type: ignore[index]
            http.post(f"/api/games/{game}/action", json={"flat": flat}, headers=hdr)
            second = _next_event(lines)
    assert second["version"] > first["version"]  # type: ignore[operator]
    assert len(second["log"]) == len(first["log"]) + 1  # type: ignore[arg-type]


def test_turn_timeout_auto_advances_an_idle_human_turn() -> None:
    # All human, but a turn timeout is set: nobody acts, so the driver auto-
    # plays the idle turn and the game advances on its own.
    with (
        _live_server(create_app(turn_timeout=0.2)) as port,
        httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as http,
    ):
        game, _ = _start_all_human(http)
        deadline = time.monotonic() + 10  # turn_timeout=0.2s; a generous margin
        body = http.get(f"/api/games/{game}").json()
        while time.monotonic() < deadline:
            body = http.get(f"/api/games/{game}").json()
            if any(e["kind"] == "move" for e in body["log"]):
                break
            time.sleep(0.05)
        assert any(e["kind"] == "move" for e in body["log"])


def test_no_turn_timeout_leaves_an_idle_human_turn_alone() -> None:
    # Default (no timeout): an all-human game has no driver and never self-plays.
    with (
        _live_server(create_app()) as port,
        httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as http,
    ):
        game, _ = _start_all_human(http)
        time.sleep(0.5)
        body = http.get(f"/api/games/{game}").json()
        assert not any(e["kind"] == "move" for e in body["log"])


def test_concurrent_duplicate_actions_apply_once(client: TestClient) -> None:
    game, tokens = _human_game(client, ["human", "random", "random", "random"])
    flat = client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()["actions"][0][
        "flat"
    ]
    codes: list[int] = []

    def post() -> None:
        codes.append(
            client.post(
                f"/api/games/{game}/action", json={"flat": flat}, headers=_hdr(tokens)
            ).status_code
        )

    threads = [threading.Thread(target=post) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # The per-game lock serialises: exactly one apply succeeds, the rest see
    # the move as no-longer-legal (409).
    assert sorted(codes) == [200, 409, 409, 409]


def test_chat_requires_seat_ownership(client: TestClient) -> None:
    game, tokens = _human_game(client, ["human", "random", "random", "random"])
    body = client.post(
        f"/api/games/{game}/chat",
        json={"text": "hi", "player": 0},
        headers=_hdr(tokens),
    ).json()
    assert body["log"][-1]["kind"] == "chat" and body["log"][-1]["player"] == 0
    # Unowned seat: refused. Spectator (no seat given): allowed.
    assert (
        client.post(
            f"/api/games/{game}/chat", json={"text": "hi", "player": 1}
        ).status_code
        == 403
    )
    assert (
        client.post(f"/api/games/{game}/chat", json={"text": "gl"}).json()["log"][-1][
            "player"
        ]
        is None
    )
    assert (
        client.post(f"/api/games/{game}/chat", json={"text": "   "}).status_code == 422
    )


@contextmanager
def _finished_game(seed: int = 0) -> Iterator[tuple[TestClient, str]]:
    """An all-bot app whose game is already finished, yielding the client and
    the finished game's id.

    Played out directly (``auto_step``, no bot service, no wall clock) before
    the app is even built: a finished game never spawns a driver, so the
    record/replay routes are reachable immediately -- no polling.
    """
    session = GameSession(
        seed=seed, n_players=2, seats=["random", "random"], external_kinds=BOT_KINDS
    )
    play_out(session)
    registry = GameRegistry()
    handle = registry.create(session)
    with TestClient(create_app(registry)) as client:
        yield client, handle.id


def test_record_and_replay_export_finished_games_only(client: TestClient) -> None:
    game = bot_game(client, ["random", "random"])
    # A live game's record would reconstruct hidden hands when replayed.
    assert client.get(f"/api/games/{game}/record").status_code == 409
    assert client.post(f"/api/games/{game}/replay").status_code == 409

    with _finished_game() as (fclient, fgame):
        doc = fclient.get(f"/api/games/{fgame}/record").json()
        assert doc["winner"] is not None and len(doc["moves"]) > 0
        opening = fclient.post(f"/api/games/{fgame}/replay").json()
        assert opening["move"] == 0 and opening["n_moves"] == len(doc["moves"])
        mid = fclient.get("/api/replay/state", params={"move": 5}).json()
        assert mid["move"] == 5
        bad = fclient.get("/api/replay/state", params={"move": 99999})
        assert bad.status_code == 422


def test_replay_upload_roundtrip() -> None:
    with _finished_game() as (client, game):
        doc = client.get(f"/api/games/{game}/record").json()
        assert client.post("/api/replay", json=doc).status_code == 200
        assert client.get("/api/replay/record").status_code == 200


def test_replay_state_404_until_loaded(client: TestClient) -> None:
    assert client.get("/api/replay/state").status_code == 404


def test_replay_probe_is_null_not_404_until_loaded() -> None:
    # The page's load-time probe must not 404 on a fresh visit; it returns null,
    # then the loaded opening state once a record is posted.
    with _finished_game() as (client, game):
        probe = client.get("/api/replay")
        assert probe.status_code == 200 and probe.json() is None
        doc = client.get(f"/api/games/{game}/record").json()
        client.post("/api/replay", json=doc)
        loaded = client.get("/api/replay")
        assert loaded.status_code == 200 and loaded.json()["move"] == 0


def test_replay_rejects_bad_records(client: TestClient) -> None:
    assert client.post("/api/replay", json={"seed": 1}).status_code == 422
    assert (
        client.post(
            "/api/replay",
            json={
                "seed": 1,
                "n_players": 4,
                "number_placement": "random",
                # An action illegal at the opening (a roll, not a setup move).
                "moves": [{"player": 0, "flat": 126}],
                "winner": None,
            },
        ).status_code
        == 422
    )


def test_oversized_request_body_is_rejected_before_parsing() -> None:
    with TestClient(create_app(GameRegistry(), max_body_bytes=100)) as c:
        big = c.post("/api/replay", json={"pad": "x" * 500})
        assert big.status_code == 413
        # A small body still reaches the route (and is rejected on its merits).
        assert c.post("/api/replay", json={"seed": 1}).status_code == 422


def test_replay_with_too_many_moves_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _finished_game() as (client, game):
        doc = client.get(f"/api/games/{game}/record").json()
        monkeypatch.setattr(
            "settlrl_app.api.routers.replay._MAX_REPLAY_MOVES", len(doc["moves"]) - 1
        )
        resp = client.post("/api/replay", json=doc)
        assert resp.status_code == 422 and "too many moves" in resp.json()["detail"]


def test_get_bots_lists_policies(client: TestClient) -> None:
    body = client.get("/api/bots").json()
    assert "random" in body and body["random"]["description"]
    assert all(
        "counts" in spec and "title" in spec and "description" in spec
        for spec in body.values()
    )


def test_openapi_schema_is_committed() -> None:
    # The committed schema generates the frontend's wire types; regen with
    # `npm run gen-api` in frontend/ whenever the models change.
    committed = json.loads(
        (Path(__file__).parent.parent / "frontend" / "openapi.json").read_text()
    )
    assert create_app().openapi() == committed, "schema drift: run `npm run gen-api`"


_DIST = (Path(__file__).parent.parent / "frontend" / "dist").exists()


@pytest.mark.skipif(not _DIST, reason="frontend/dist not built")
def test_spa_fallback_serves_index_for_client_route(client: TestClient) -> None:
    resp = client.get("/play")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.skipif(not _DIST, reason="frontend/dist not built")
def test_spa_fallback_404_for_missing_asset(client: TestClient) -> None:
    assert client.get("/assets/nope.js").status_code == 404
