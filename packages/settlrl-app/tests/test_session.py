"""Tests for :class:`session.GameSession` turn flow.

The game server runs no bots, so a seat is a human or a remote bot kind (stored
verbatim, played by a bot service). These check turn flow, the random
``auto_step`` liveness move, seat validation, the per-seat belief, and the log /
record export.
"""

import pytest
from settlrl_game.record import GameRecord, replay
from settlrl_game.session import HUMAN, GameSession, GameSetup, IllegalActionError

_BOTS = frozenset({"random"})  # an accepted "remote" kind for these tests


def _drive_to_completion(sess: GameSession) -> None:
    """Play random legal moves (``auto_step``) for whoever is acting until the
    game ends."""
    for _ in range(50_000):
        if sess.auto_step() is None:
            break


def test_acting_seat_is_human_after_construction() -> None:
    sess = GameSession(seed=0)
    # Seat 0 opens, and by default it's a human: the game waits for its move.
    assert sess.acting_seat() == 0
    assert sess.status().your_turn
    assert not sess.status().terminal


def test_auto_step_plays_a_random_legal_move() -> None:
    sess = GameSession(seed=0)
    flat = sess.auto_step()
    assert flat is not None
    (entry,) = sess.log()
    assert entry.kind == "move" and entry.player == 0


def test_auto_step_drives_a_game_to_a_winner() -> None:
    sess = GameSession(seed=0, n_players=2)  # 2p reaches a win the fastest
    _drive_to_completion(sess)
    status = sess.status()
    assert status.terminal
    assert status.winner is not None and 0 <= status.winner < 2


def test_external_bot_kinds_are_accepted_and_stored() -> None:
    sess = GameSession(
        seed=0,
        seats=[HUMAN, "random", "random", "random"],
        external_kinds=_BOTS,
    )
    assert sess.status().seats == [HUMAN, "random", "random", "random"]
    # The seats fold back into the reconstructable setup.
    assert sess.setup.seats == [HUMAN, "random", "random", "random"]
    assert sess.status().your_turn  # seat 0 (human) opens


def test_unknown_seat_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown seat kind"):
        GameSession(
            seed=0, n_players=2, seats=["human", "clever"], external_kinds=_BOTS
        )


def test_all_bot_game_waits_on_a_bot_and_plays_out() -> None:
    # A low win threshold: the property under test is that no human seat acts
    # and the game still plays to a terminal, not the standard 4p VP target
    # (which random play takes far longer to reach).
    sess = GameSession(
        seed=0, seats=["random"] * 4, external_kinds=_BOTS, victory_points_to_win=5
    )
    assert not sess.status().your_turn  # no human seat acts
    _drive_to_completion(sess)
    assert sess.status().terminal and sess.status().winner is not None


def test_belief_serves_the_hand_seat() -> None:
    # Fresh game: the human observer sees every opponent's (empty) hand exactly;
    # their own row is omitted. All-bot games have no human observer.
    sess = GameSession(seed=0)
    belief = sess.belief()
    assert belief is not None and belief.observer == 0
    assert [b.player for b in belief.players] == [1, 2, 3]
    for b in belief.players:
        assert b.res_lo == b.res_hi  # nothing dealt yet: bounds are exact

    assert (
        GameSession(seed=0, seats=["random"] * 4, external_kinds=_BOTS).belief() is None
    )


def test_reset_keeps_seat_count_unless_changed() -> None:
    sess = GameSession(seed=0, n_players=2)
    sess.reset(seed=1)  # no n_players -> keeps 2
    assert sess.n_players == 2
    sess.reset(seed=2, n_players=4)
    assert sess.n_players == 4


def test_invalid_seats_rejected() -> None:
    with pytest.raises(ValueError, match="unknown seat"):  # not human, not external
        GameSession(seed=0, seats=[HUMAN, "clever", HUMAN, HUMAN])
    with pytest.raises(ValueError, match="expected 4 seats"):
        GameSession(seed=0, seats=[HUMAN, HUMAN])


def test_log_records_moves_and_chat() -> None:
    sess = GameSession(seed=0)
    assert sess.log() == []
    sess.apply(int(sess.legal_flat()[0]))  # human setup settlement
    (entry,) = sess.log()
    assert entry.kind == "move"
    assert entry.player == 0
    assert entry.action_type == "setup_settlement"
    sess.add_chat(0, "hello")
    assert sess.log()[-1].kind == "chat" and sess.log()[-1].text == "hello"
    sess.add_chat(None, "just watching")  # a spectator line
    assert sess.log()[-1].player is None
    sess.reset(seed=1)
    assert sess.log() == []


def test_log_rolls_and_win() -> None:
    sess = GameSession(seed=0, n_players=2)  # 2p reaches a win the fastest
    _drive_to_completion(sess)
    entries = sess.log()
    # Rolls carry the rolled value...
    rolls = [e for e in entries if e.action_type == "roll_dice"]
    assert rolls and all(e.text.startswith("rolled ") for e in rolls)
    # ...and the game's end is logged exactly once, for the winning seat.
    wins = [e for e in entries if e.kind == "win"]
    assert len(wins) == 1
    assert wins[0].player == sess.status().winner


def test_record_exports_a_replayable_game() -> None:
    sess = GameSession(
        seed=3, n_players=2, seats=["random", "random"], external_kinds=_BOTS
    )
    _drive_to_completion(sess)
    rec = sess.record()
    assert rec.winner == sess.status().winner
    assert rec.meta == {"seats": ["random", "random"]}
    assert len(rec.moves) > 10  # the full trace, not the capped log
    # The JSON roundtrips and the engine replays it without complaint.
    rec2 = GameRecord.from_json(rec.to_json())
    boards = list(replay(rec2))
    assert len(boards) == len(rec.moves)


def test_record_of_running_game_has_no_winner() -> None:
    sess = GameSession(seed=0)
    sess.apply(int(sess.legal_flat()[0]))
    rec = sess.record()
    assert rec.winner is None
    assert [m.flat for m in rec.moves] == [rec.moves[0].flat]
    sess.reset(seed=1)
    assert sess.record().moves == ()  # reset starts a fresh trace


def test_set_seat_kind_relabels_before_play() -> None:
    sess = GameSession(seed=0, n_players=2, seats=["human", "human"])
    sess.set_seat_kind(1, "random")
    assert sess.seats == ["human", "random"]
    assert sess.status().seats == ["human", "random"]  # surfaced to the wire


def test_set_seat_kind_rejected_after_a_move() -> None:
    sess = GameSession(seed=0, n_players=2)
    sess.auto_step()
    with pytest.raises(IllegalActionError):
        sess.set_seat_kind(1, "random")


def test_victory_points_to_win_threads_into_game_status_and_record() -> None:
    sess = GameSession(seed=0, n_players=2, victory_points_to_win=15)
    assert sess.game.victory_points_to_win == 15
    assert sess.status().victory_points_to_win == 15
    # Persisted in the journal header and the replayable record.
    assert sess.setup.victory_points_to_win == 15
    assert GameSetup.from_dict(sess.setup.to_dict()).victory_points_to_win == 15
    rec = sess.record()
    assert rec.victory_points_to_win == 15
    assert GameRecord.from_json(rec.to_json()).victory_points_to_win == 15


def test_pre_threshold_records_default_to_ten() -> None:
    # A header / record written before the field defaults to the rulebook 10.
    setup = GameSetup.from_dict(
        {
            "seed": 0,
            "n_players": 4,
            "number_placement": "random",
            "seats": ["human"] * 4,
        }
    )
    assert setup.victory_points_to_win == 10
    rec = GameRecord.from_json('{"seed": 0, "n_players": 4}')
    assert rec.victory_points_to_win == 10
