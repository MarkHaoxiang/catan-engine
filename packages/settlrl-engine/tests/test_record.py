"""Game records: generation, JSON roundtrip, and validated deterministic replay.

A randomly-played game is recorded once per player count (module-scoped -- a
full game is thousands of steps) and shared across the tests; tampering tests
corrupt early moves so replay fails fast.
"""

import dataclasses
import json

import jax
import numpy as np
import pytest
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_engine.record import (
    GameRecord,
    ReplayError,
    initial_board,
    record_game,
    replay,
)

# Keeps every test in this module on one xdist worker, so the module-scoped
# record4/record2 fixtures (full games, thousands of steps) are computed once
# per run instead of once per worker that happens to draw a test from here.
pytestmark = pytest.mark.xdist_group("record")


@pytest.fixture(scope="module")
def record4() -> GameRecord:
    return record_game(seed=0, meta={"note": "test game"})


@pytest.fixture(scope="module")
def record2() -> GameRecord:
    return record_game(seed=1, n_players=2, number_placement="spiral")


def test_generated_game_completes(record4: GameRecord) -> None:
    assert record4.winner is not None
    assert len(record4.moves) > 10
    # The transcript opens with the setup snake and records dice outcomes.
    assert {m.player for m in record4.moves} == {0, 1, 2, 3}
    rolls = [m.dice for m in record4.moves if m.dice is not None]
    assert rolls and all(2 <= r <= 12 for r in rolls)


def test_json_roundtrip(record4: GameRecord) -> None:
    assert GameRecord.from_json(record4.to_json()) == record4


def test_json_is_readable(record4: GameRecord) -> None:
    doc = json.loads(record4.to_json())
    assert doc["version"] == 3
    assert doc["meta"] == {"note": "test game"}
    moves = doc["moves"]
    first = moves[0]
    # Setup opens with a settlement: annotated with its type and vertex.
    assert first["type"] == "setup_settlement" and isinstance(first["vertex"], int)
    # Moves carry the stable identifiers, not table positions.
    assert first["idx"] == first["vertex"] and "flat" not in first
    types = {m["type"] for m in moves}
    assert {"setup_road", "roll_dice", "end_turn"} <= types
    roll = next(m for m in moves if m["type"] == "roll_dice")
    assert 2 <= roll["dice"] <= 12


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_records_migrate_via_annotations(
    record4: GameRecord, version: int
) -> None:
    # v1 stored flat table positions (stale once the table grew); v2 stored
    # ProposeTrade params in a retired encoding. The loader must recover every
    # move from its annotations instead.
    doc = json.loads(record4.to_json())
    doc["version"] = version
    for move in doc["moves"]:
        if version == 1:
            del move["idx"], move["target"]
            move["flat"] = 0  # deliberately wrong: must be ignored
        elif move["type"] == "propose_trade":
            move["idx"], move["target"] = -1, -1  # retired encoding: ignored
    assert GameRecord.from_json(json.dumps(doc)) == record4


def test_from_json_rejects_unknown_moves(record2: GameRecord) -> None:
    doc = json.loads(record2.to_json())
    doc["moves"][0]["type"] = "build_spaceship"
    with pytest.raises(ValueError, match="unknown action type"):
        GameRecord.from_json(json.dumps(doc))
    doc["moves"][0]["type"] = "setup_settlement"
    doc["moves"][0]["idx"] = 99_999
    with pytest.raises(ValueError, match="current action table"):
        GameRecord.from_json(json.dumps(doc))


def test_replay_reproduces_the_game(record2: GameRecord) -> None:
    boards = list(replay(record2))
    assert len(boards) == len(record2.moves)
    # The final state is bit-identical to a second replay (determinism).
    final, again = boards[-1], list(replay(record2))[-1]
    assert all(
        bool((a == b).all())
        for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(again), strict=True)
    )


def test_replay_rejects_wrong_player(record4: GameRecord) -> None:
    bad = dataclasses.replace(
        record4,
        moves=(dataclasses.replace(record4.moves[0], player=3), *record4.moves[1:]),
    )
    with pytest.raises(ReplayError, match="player"):
        next(replay(bad))


def test_replay_rejects_illegal_move(record4: GameRecord) -> None:
    # A second action on the same vertex as move 0 cannot be legal at move 1.
    bad = dataclasses.replace(
        record4,
        moves=(record4.moves[0], record4.moves[0], *record4.moves[2:]),
    )
    steps = replay(bad)
    next(steps)
    with pytest.raises(ReplayError, match="not legal"):
        next(steps)


def test_replay_rejects_wrong_dice(record4: GameRecord) -> None:
    idx, roll = next((i, m) for i, m in enumerate(record4.moves) if m.dice is not None)
    assert roll.dice is not None
    wrong = dataclasses.replace(roll, dice=roll.dice % 12 + 2)
    bad = dataclasses.replace(
        record4, moves=(*record4.moves[:idx], wrong, *record4.moves[idx + 1 :])
    )
    with pytest.raises(ReplayError, match="rolled"):
        for _ in replay(bad):
            pass


def test_replay_rejects_wrong_winner(record2: GameRecord) -> None:
    assert record2.winner is not None
    bad = dataclasses.replace(record2, winner=(record2.winner + 1) % 2)
    with pytest.raises(ReplayError, match="winner"):
        for _ in replay(bad):
            pass


def test_record_game_rejects_illegal_act() -> None:
    # An act that picks the first currently-*illegal* flat action.
    def bad_act(key: jax.Array, env: BatchedSettlrlEnv) -> int:
        mask = np.asarray(env.flat_mask()[0]).astype(bool)
        return int(np.flatnonzero(~mask)[0])

    with pytest.raises(ValueError, match="illegal"):
        record_game(seed=0, act=bad_act)


def test_partial_record_replays_without_winner_check(record2: GameRecord) -> None:
    partial = dataclasses.replace(record2, moves=record2.moves[:8], winner=None)
    assert len(list(replay(partial))) == 8


def test_from_json_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="version"):
        GameRecord.from_json('{"version": 99}')


def test_initial_board_is_the_unplayed_opening(record2: GameRecord) -> None:
    layout, state = initial_board(record2)
    assert not bool((np.asarray(state.vertex_owner) != 0).any())  # nothing built
    # Same deterministic layout as the recorded game's replay.
    replayed_layout, _ = next(replay(record2))
    assert all(
        bool((a == b).all())
        for a, b in zip(
            jax.tree.leaves(layout), jax.tree.leaves(replayed_layout), strict=True
        )
    )
