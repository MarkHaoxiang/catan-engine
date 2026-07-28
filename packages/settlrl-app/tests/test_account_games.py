"""Seats follow the account that claimed them.

A signed-in player owns their seats by user id, not just by the per-device seat
token — so they are recognised on any device (and can list their games) without
carrying the token around. These drive that through the routes.
"""

import random
import tempfile
import time
from collections.abc import Iterator

import pytest
from _helpers import bot_registry, start_game
from fastapi.testclient import TestClient
from settlrl_app.game.games import GameRegistry
from settlrl_app.server import create_app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    # Bot-backed so the non-human seats these tests use validate and play.
    with TestClient(create_app(GameRegistry(), providers=bot_registry())) as client:
        yield client


def _token(client: TestClient, email: str, password: str = "password1") -> str:
    client.post("/api/auth/register", json={"email": email, "password": password})
    return str(
        client.post(
            "/api/auth/login", data={"username": email, "password": password}
        ).json()["access_token"]
    )


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_authenticated_creator_is_recognised_without_the_seat_token(
    client: TestClient,
) -> None:
    token = _token(client, "a@example.com")
    game, _ = start_game(
        client, ["human", "random", "random", "random"], headers=_bearer(token)
    )

    # The account owns seat 0 with only the bearer token — no seat token sent.
    snap = client.get(f"/api/games/{game}", headers=_bearer(token)).json()
    assert snap["your_seats"] == [0]
    assert snap["status"]["your_turn"] and snap["actions"]

    # And can act on that seat with just the account token.
    flat = snap["actions"][0]["flat"]
    assert (
        client.post(
            f"/api/games/{game}/action", json={"flat": flat}, headers=_bearer(token)
        ).status_code
        == 200
    )


def test_other_users_and_anonymous_own_nothing(client: TestClient) -> None:
    owner = _token(client, "owner@example.com")
    other = _token(client, "other@example.com")
    game, _ = start_game(
        client, ["human", "random", "random", "random"], headers=_bearer(owner)
    )

    assert (
        client.get(f"/api/games/{game}", headers=_bearer(other)).json()["your_seats"]
        == []
    )
    assert client.get(f"/api/games/{game}").json()["your_seats"] == []


def test_seat_names_show_account_holders_and_bots(client: TestClient) -> None:
    token = _token(client, "alice@example.com")
    game, _ = start_game(
        client, ["human", "random", "random", "random"], headers=_bearer(token)
    )

    # The account's seat carries its label; bot seats have no name.
    snap = client.get(f"/api/games/{game}", headers=_bearer(token)).json()
    assert snap["seat_names"] == ["alice", None, None, None]
    assert snap["status"]["seats"] == ["human", "random", "random", "random"]


def test_my_games_lists_only_the_users_games(client: TestClient) -> None:
    a = _token(client, "a@example.com")
    b = _token(client, "b@example.com")
    game_a, _ = start_game(
        client, ["human", "random", "random", "random"], headers=_bearer(a)
    )
    start_game(
        client, ["human", "random", "random", "random"], headers=_bearer(b), seed=1
    )

    mine = client.get("/api/me/games", headers=_bearer(a)).json()
    assert [g["id"] for g in mine] == [game_a]
    assert mine[0]["seats"] == [0]
    assert client.get("/api/me/games").status_code == 401  # requires sign-in


def test_join_ties_the_seat_to_the_account(client: TestClient) -> None:
    a = _token(client, "a@example.com")
    b = _token(client, "b@example.com")
    # a hosts an online table (holding seat 0); b joins the open seat; starting
    # carries that claim into the game, tied to b's account.
    lobby = client.post(
        "/api/lobbies", json={"mode": "online", "n_players": 2}, headers=_bearer(a)
    ).json()
    lid = lobby["id"]
    a_hdr = {**_bearer(a), "X-Seat-Tokens": ",".join(lobby["tokens"].values())}
    joined = client.post(f"/api/lobbies/{lid}/join", json={}, headers=_bearer(b)).json()
    seat = int(next(iter(joined["tokens"])))
    game = client.post(f"/api/lobbies/{lid}/start", json={}, headers=a_hdr).json()[
        "game_id"
    ]

    snap = client.get(f"/api/games/{game}", headers=_bearer(b)).json()
    assert snap["your_seats"] == [seat]


def test_finished_game_updates_history_and_leaderboard() -> None:
    """A finished game the account owns a seat in shows up in /api/me/history
    (record downloadable, for the owner only) and rates both the account (by
    handle) and the bot on the public per-player-count leaderboard -- one
    drive to terminal covers both journalled write-behind effects."""
    rng = random.Random(0)
    with (
        tempfile.TemporaryDirectory() as state_dir,
        TestClient(
            create_app(state_dir=state_dir, bot_delay=0.0, providers=bot_registry())
        ) as client,
    ):
        h = _bearer(_token(client, "champ@example.com"))
        # A low win threshold: the finish/rating machinery under test doesn't
        # depend on the standard 15-VP duel length, only on reaching terminal.
        game, _ = start_game(client, ["human", "random"], headers=h, seed=3, vp=5)

        # Play the human seat (random legal moves) while the driver plays the
        # bot, until the game ends.
        for _ in range(6000):
            snap = client.get(f"/api/games/{game}", headers=h).json()
            if snap["status"]["terminal"]:
                break
            if snap["status"]["your_turn"] and snap["actions"]:
                flat = rng.choice(snap["actions"])["flat"]
                client.post(f"/api/games/{game}/action", json={"flat": flat}, headers=h)
            else:
                time.sleep(0.01)  # the bot's turn — let the driver advance it
        assert client.get(f"/api/games/{game}", headers=h).json()["status"]["terminal"]

        # The finish is journalled write-behind; poll until history reflects it.
        hist: list[dict[str, object]] = []
        for _ in range(100):
            hist = client.get("/api/me/history", headers=h).json()
            if hist:
                break
            time.sleep(0.02)
        assert [g["id"] for g in hist] == [game]
        assert hist[0]["seats"] == [0] and hist[0]["winner"] in (0, 1)
        assert hist[0]["n_players"] == 2

        # Another user sees no history, and it requires sign-in.
        other = _bearer(_token(client, "o@example.com"))
        assert client.get("/api/me/history", headers=other).json() == []
        assert client.get("/api/me/history").status_code == 401

        # The record + replay are served for the finished game.
        assert client.get(f"/api/games/{game}/record").status_code == 200
        assert client.post(f"/api/games/{game}/replay").status_code == 200

        # The rating update is journalled write-behind too; poll until it lands.
        board: list[dict[str, object]] = []
        for _ in range(100):
            board = client.get("/api/leaderboard").json()  # public: no auth
            if board:
                break
            time.sleep(0.02)
        assert {e["name"] for e in board} == {"champ", "random"}  # handle + bot
        assert {e["kind"] for e in board} == {"account", "bot"}
        assert all(e["n_players"] == 2 and e["games"] == 1 for e in board)
        assert sum(int(e["wins"]) for e in board) == 1  # type: ignore[call-overload]  # exactly one winner


def test_account_seat_ownership_survives_a_restart() -> None:
    """The seat<->account tie is journalled, so a restored game still knows it."""
    with tempfile.TemporaryDirectory() as state_dir:
        with TestClient(
            create_app(state_dir=state_dir, providers=bot_registry())
        ) as first:
            token = _token(first, "a@example.com")
            game, _ = start_game(
                first, ["human", "random", "random", "random"], headers=_bearer(token)
            )

        # A fresh app restores games + accounts from the same state dir.
        with TestClient(
            create_app(state_dir=state_dir, providers=bot_registry())
        ) as restored:
            token2 = str(
                restored.post(
                    "/api/auth/login",
                    data={"username": "a@example.com", "password": "password1"},
                ).json()["access_token"]
            )
            snap = restored.get(f"/api/games/{game}", headers=_bearer(token2)).json()
            assert snap["your_seats"] == [0]
