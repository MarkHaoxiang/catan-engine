"""Win timing (rulebook p.5, "Ending the Game").

You can only win *during your own turn*. A player who reaches 10 VP out of
turn (e.g. a settlement break handing them Longest Road) keeps waiting; play
continues until any player is at 10 VP on their own turn, which they claim at
the turn's start.
"""

from __future__ import annotations

from conftest import make_game, make_layout, place
from settlrl_game.reference.game import EndTurn, Game, Roll
from settlrl_game.reference.types import Building, Phase


def _give_ten_vp(game: Game, player: int) -> None:
    """Ten victory points of buildings for ``player`` (3 cities + 4
    settlements, within the piece caps), on otherwise-unused vertices."""
    base = 20 + 10 * player  # vertex blocks far apart per player
    for v in range(base, base + 3):
        place(game, v, player, Building.CITY)
    for v in range(base + 3, base + 7):
        place(game, v, player, Building.SETTLEMENT)


def test_off_turn_ten_vp_claims_at_turn_start() -> None:
    game = make_game(make_layout())  # player 0's MAIN, has_rolled
    _give_ten_vp(game, 1)
    assert game.total_vp(1) == 10
    # Phase-typed locals here (and below) keep mypy from narrowing
    # `game.phase` across the mutating `apply` calls.
    phase: Phase = game.phase
    assert phase is Phase.MAIN  # 10 VP out of turn ends nothing

    game.apply(EndTurn())  # player 1's turn begins -> they claim

    assert game.phase is Phase.GAME_OVER
    assert game.current_player == 1


def test_game_continues_until_the_ten_vp_players_turn() -> None:
    game = make_game(make_layout())
    _give_ten_vp(game, 2)

    game.apply(EndTurn())  # player 1's turn: not the 10-VP player
    phase: Phase = game.phase
    assert phase is Phase.ROLL
    assert game.current_player == 1

    game.apply(Roll(value=4))  # all-desert layout: no production, no 7
    game.apply(EndTurn())  # player 2's turn begins -> they claim

    assert game.phase is Phase.GAME_OVER
    assert game.current_player == 2


def _give_settlements(game: Game, player: int, n: int) -> None:
    base = 20 + 10 * player
    for v in range(base, base + n):
        place(game, v, player, Building.SETTLEMENT)


def test_default_win_threshold_is_ten() -> None:
    game = make_game(make_layout())
    assert game.victory_points_to_win == 10
    _give_settlements(game, 1, 4)  # only 4 VP -- well short of 10
    game.apply(EndTurn())  # player 1's turn begins
    assert game.phase is Phase.ROLL  # no win: play continues
    assert game.current_player == 1


def test_lower_win_threshold_ends_the_game_sooner() -> None:
    game = make_game(make_layout())  # player 0's MAIN
    game.victory_points_to_win = 4
    _give_settlements(game, 1, 4)  # exactly the (lowered) threshold
    assert game.total_vp(1) == 4
    phase: Phase = game.phase
    assert phase is Phase.MAIN  # at the threshold but out of turn: no win

    game.apply(EndTurn())  # player 1's turn begins -> they claim

    assert game.phase is Phase.GAME_OVER
    assert game.current_player == 1
