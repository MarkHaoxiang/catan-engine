"""Persistence: journals on the shared async DB replay back into live games on
restart. The write-behind store is exercised through the HTTP surface (a clean
``with`` shutdown drains queued writes); the eviction unit test drives the store
and registry directly under ``asyncio.run``.
"""

import asyncio
import time
from pathlib import Path

import pytest
from _helpers import bot_registry, play_out, start_game
from fastapi.testclient import TestClient
from settlrl_app.game.games import GameRegistry, _rebuild_handle
from settlrl_app.server import create_app
from settlrl_app.storage.db import Database, GameLog
from settlrl_app.storage.store import FinishedGame, GameStore, RatingEntry
from settlrl_game.convert import board_to_model
from settlrl_game.session import GameSession, GameSetup
from sqlalchemy import select


def test_game_setup_round_trips_through_a_dict() -> None:
    setup = GameSetup(
        seed=3,
        n_players=2,
        number_placement="spiral",
        seats=["human", "random"],
    )
    # from_dict ignores the journal's framing keys (id, t).
    assert GameSetup.from_dict({**setup.to_dict(), "id": "x", "t": "header"}) == setup


def test_session_setup_captures_its_seats() -> None:
    session = GameSession(
        seed=5,
        n_players=2,
        seats=["human", "random"],
        external_kinds=frozenset({"random"}),
    )
    assert session.setup == GameSetup(5, 2, "random", ["human", "random"])


def test_restart_resumes_an_in_progress_game(tmp_path: Path) -> None:
    # First boot: create a game (with a bot seat), play a move, leave a chat line.
    with TestClient(
        create_app(state_dir=str(tmp_path), providers=bot_registry())
    ) as c1:
        game, hdr = start_game(c1, ["human", "random"])
        flat = c1.get(f"/api/games/{game}", headers=hdr).json()["actions"][0]["flat"]
        before = c1.post(
            f"/api/games/{game}/action", json={"flat": flat}, headers=hdr
        ).json()
        c1.post(
            f"/api/games/{game}/chat", json={"text": "gg", "player": 0}, headers=hdr
        )

    # Second boot from the same dir: the game is back, at the same position.
    with TestClient(
        create_app(state_dir=str(tmp_path), providers=bot_registry())
    ) as c2:
        after = c2.get(f"/api/games/{game}", headers=hdr).json()
        assert after["id"] == game
        assert after["board"] == before["board"]  # replayed to the same state
        assert after["seats_claimed"] == [0]  # the claim (and its kinds) survived
        assert after["status"]["seats"] == ["human", "random"]
        assert any(e["kind"] == "chat" and e["text"] == "gg" for e in after["log"])


def test_eviction_drops_the_game_from_the_store(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, str, set[str]]:
        db = Database(str(tmp_path / "settlrl.db"))
        await db.init()
        store = GameStore(db)
        store.start()
        reg = GameRegistry(max_games=1, store=store)
        a = reg.create(GameSession(seed=0, n_players=2, seats=["human", "human"]))
        # Idle well past the TTL (relative to monotonic(), which need not be large).
        a.touched = time.monotonic() - 100_000
        b = reg.create(GameSession(seed=1, n_players=2, seats=["human", "human"]))
        await store.aclose()  # flush the header writes and the eviction removal
        stored = {str(header["id"]) for header, _ in await store.load()}
        await db.dispose()
        return a.id, b.id, stored

    a_id, b_id, stored = asyncio.run(scenario())
    assert a_id not in stored and b_id in stored


def test_finished_game_is_kept_as_history_not_restored(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, int | None, set[str], list[FinishedGame]]:
        db = Database(str(tmp_path / "settlrl.db"))
        await db.init()
        store = GameStore(db)
        store.start()
        reg = GameRegistry(max_games=1, store=store)
        h = reg.create(GameSession(seed=0, n_players=2, seats=["human", "human"]))
        h.claim(0, user_id="acc-1")  # an account owns seat 0
        play_out(h.session)
        h.bump()  # journals the moves and fires the finish hook
        # At cap=1, creating another game evicts the finished one from the
        # registry — but it stays in the store as history.
        reg.create(GameSession(seed=1, n_players=2, seats=["human", "human"]))
        await store.aclose()
        winner = h.session.winner()
        live = {str(hdr["id"]) for hdr, _ in await store.load()}
        history = await store.history()
        rec = await store.finished_record(h.id)
        await db.dispose()
        # Rebuilt from the journalled outcomes (no re-sampling), so it is faithful
        # even though this game was played out via the random fallback.
        assert rec is not None and rec.winner == winner and len(rec.moves) > 0
        return h.id, winner, live, list(history)

    gid, winner, live, history = asyncio.run(scenario())
    assert gid not in live  # finished games are not restored as live games
    assert [g.id for g in history] == [gid]  # but kept as history
    assert history[0].owners == {"acc-1": [0]} and history[0].winner == winner


def test_history_is_capped_to_the_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("settlrl_app.storage.store._HISTORY_CAP", 1)

    async def scenario() -> tuple[list[str], list[str]]:
        db = Database(str(tmp_path / "settlrl.db"))
        await db.init()
        store = GameStore(db)
        store.start()
        reg = GameRegistry(max_games=10, store=store)
        ids = []
        for seed in (0, 1):
            h = reg.create(GameSession(seed=seed, n_players=2, seats=["human"] * 2))
            play_out(h.session)
            h.bump()
            ids.append(h.id)
        await store.aclose()
        kept = [g.id for g in await store.history()]
        await db.dispose()
        return ids, kept

    ids, kept = asyncio.run(scenario())
    assert len(kept) == 1 and kept[0] in ids  # pruned to the cap


def test_restore_is_faithful_for_a_random_fallback_game(tmp_path: Path) -> None:
    """A game advanced by the random fallback (``auto_step`` draws the seed RNG to
    *pick* moves, not just resolve them) must still rebuild to the same state —
    restore re-applies the stored outcomes rather than re-sampling."""

    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        db = Database(str(tmp_path / "settlrl.db"))
        await db.init()
        store = GameStore(db)
        store.start()
        reg = GameRegistry(store=store)
        h = reg.create(GameSession(seed=0, n_players=2, seats=["human", "human"]))
        # Play well past the opening so several rolls (stochastic) are journalled,
        # but stop before the end so the game restores as live.
        for _ in range(25):
            if h.session.auto_step() is None:
                break
            h.bump()
        before = board_to_model(h.session.game).model_dump()
        await store.aclose()
        (header, events) = next(
            (hdr, ev) for hdr, ev in await store.load() if hdr["id"] == h.id
        )
        restored = _rebuild_handle(store, header, events)
        assert restored is not None
        after = board_to_model(restored.session.game).model_dump()
        await db.dispose()
        return before, after

    before, after = asyncio.run(scenario())
    assert before == after  # identical, not a re-sampled divergence


def _bot_game(seed: int, kinds: list[str]) -> GameSession:
    return GameSession(
        seed=seed,
        n_players=len(kinds),
        seats=kinds,
        external_kinds=frozenset(kinds),
    )


def test_finished_game_updates_ratings(tmp_path: Path) -> None:
    async def scenario() -> tuple[int | None, list[RatingEntry]]:
        db = Database(str(tmp_path / "settlrl.db"))
        await db.init()
        store = GameStore(db)
        store.start()
        reg = GameRegistry(store=store)
        h = reg.create(_bot_game(0, ["alpha", "beta"]))
        play_out(h.session)
        winner = h.session.winner()
        h.bump()  # fires the finish hook -> enqueues the rating update
        await store.aclose()
        board = await store.leaderboard()
        await db.dispose()
        return winner, board

    winner, board = asyncio.run(scenario())
    assert winner is not None
    by_name = {e.name: e for e in board}
    assert set(by_name) == {"alpha", "beta"}
    assert all(e.kind == "bot" and e.n_players == 2 and e.games == 1 for e in board)
    won, lost = ["alpha", "beta"][winner], ["beta", "alpha"][winner]
    assert by_name[won].wins == 1 and by_name[lost].wins == 0
    assert by_name[won].rating > by_name[lost].rating  # the winner ranks higher
    assert board[0].rating >= board[1].rating  # best first


def test_ratings_are_bucketed_by_player_count(tmp_path: Path) -> None:
    async def scenario() -> list[RatingEntry]:
        db = Database(str(tmp_path / "settlrl.db"))
        await db.init()
        store = GameStore(db)
        store.start()
        reg = GameRegistry(store=store)
        for game in (
            _bot_game(0, ["alpha", "beta"]),
            _bot_game(1, ["alpha", "beta", "gamma", "delta"]),
        ):
            h = reg.create(game)
            play_out(h.session)
            h.bump()
        await store.aclose()
        board = await store.leaderboard()
        await db.dispose()
        return board

    board = asyncio.run(scenario())
    # "alpha" played both sizes -> one independent rating row per bucket.
    alpha = {e.n_players for e in board if e.name == "alpha"}
    assert alpha == {2, 4}
    assert [e.n_players for e in board] == sorted(e.n_players for e in board)


def test_log_is_stored_as_one_row_per_game(tmp_path: Path) -> None:
    """The whole event log (claims + moves) is a single ``game_log`` row, not a
    row per event."""

    async def scenario() -> tuple[int, int, int]:
        db = Database(str(tmp_path / "settlrl.db"))
        await db.init()
        store = GameStore(db)
        store.start()
        reg = GameRegistry(store=store)
        h = reg.create(GameSession(seed=0, n_players=2, seats=["human", "human"]))
        h.claim(0)
        h.claim(1)
        for _ in range(25):
            if h.session.auto_step() is None:
                break
            h.bump()
        await store.aclose()
        async with db.sessionmaker() as session:
            rows = (await session.execute(select(GameLog))).scalars().all()
        (_, events) = (await store.load())[0]
        await db.dispose()
        moves = sum(1 for e in events if e["t"] == "move")
        return len(rows), len(events), moves

    n_rows, n_events, n_moves = asyncio.run(scenario())
    assert n_rows == 1  # one row holds the entire log
    assert n_moves > 0 and n_events == 2 + n_moves  # two claims + the moves


def test_restored_bot_game_resumes_playing(tmp_path: Path) -> None:
    # First boot: an all-bot game plays a few moves, journalled, then we shut
    # down cleanly (draining the queued writes).
    with TestClient(
        create_app(state_dir=str(tmp_path), bot_delay=0.0, providers=bot_registry())
    ) as c1:
        game = c1.post(
            "/api/games",
            json={
                "seed": 0,
                "n_players": 2,
                "seats": ["random", "random"],
                "claim": "none",
            },
        ).json()["id"]
        deadline = time.monotonic() + 30
        moves_at_shutdown = 0
        while time.monotonic() < deadline:
            snap = c1.get(f"/api/games/{game}").json()
            moves_at_shutdown = sum(1 for e in snap["log"] if e["kind"] == "move")
            if snap["status"]["terminal"] or moves_at_shutdown >= 5:
                break
            time.sleep(0.05)

    # A fresh app restores the position and its startup restarts the driver,
    # which resumes playing (random fallback even before the bot service is
    # re-registered) -- we only need to see it advance, not finish, since
    # finishing a whole game is an unbounded wall-clock wait.
    with TestClient(
        create_app(state_dir=str(tmp_path), bot_delay=0.0, providers=bot_registry())
    ) as c2:
        deadline = time.monotonic() + 30
        body = c2.get(f"/api/games/{game}").json()
        while time.monotonic() < deadline:
            body = c2.get(f"/api/games/{game}").json()
            moves = sum(1 for e in body["log"] if e["kind"] == "move")
            if body["status"]["terminal"] or moves > moves_at_shutdown:
                break
            time.sleep(0.05)
        moves = sum(1 for e in body["log"] if e["kind"] == "move")
        assert body["status"]["terminal"] or moves > moves_at_shutdown
