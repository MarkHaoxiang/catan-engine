"""The randomness layer: a seed yields a standard board, and the reference can
drive a whole game on its own (the app's path) by sampling the stochastic
outcomes its actions take. Belief tracking rides along to show it holds up on a
purely reference-driven game -- no engine anywhere.
"""

from __future__ import annotations

import copy
from collections import Counter
from random import Random

import settlrl_game.reference as ref
from settlrl_game.reference.board import (
    _NUMBER_TOKENS,
    _PORT_VERTICES,
    _SPIRAL_TILE_ORDER,
    _TERRAIN,
    SPIRAL_NUMBERS,
)
from settlrl_game.reference.game import (
    Action,
    BuyDevelopmentCard,
    Game,
    MoveRobber,
    PlayKnight,
    Roll,
)
from settlrl_game.reference.types import RESOURCES, PortType


def _fill(game: Game, action: Action, rng: Random) -> Action:
    """Inject the realised stochastic outcome a live driver would sample."""
    if isinstance(action, Roll):
        return Roll(ref.roll_dice(rng))
    if isinstance(action, BuyDevelopmentCard):
        return BuyDevelopmentCard(ref.draw_dev_card(game, rng))
    if isinstance(action, MoveRobber | PlayKnight) and action.victim is not None:
        return type(action)(
            action.tile, action.victim, ref.steal(game, action.victim, rng)
        )
    return action


def test_random_layout_is_a_standard_board() -> None:
    layout = ref.random_layout(Random(0))
    assert Counter(layout.tile_resource) == Counter(_TERRAIN)
    # The desert carries no token; every other tile carries one from the supply.
    desert = ref.desert_tile(layout)
    assert layout.tile_number[desert] == 0
    placed = [layout.tile_number[t] for t in range(ref.N_TILES) if t != desert]
    assert Counter(placed) == Counter(_NUMBER_TOKENS)
    # Nine harbours on the fixed positions, the standard type multiset.
    assert {p.vertices for p in layout.ports} == set(_PORT_VERTICES)
    assert Counter(p.type for p in layout.ports) == Counter(
        {PortType.GENERIC: 4, **{PortType(r): 1 for r in RESOURCES}}
    )


def test_spiral_places_the_canonical_sequence() -> None:
    layout = ref.random_layout(Random(0), "spiral")
    # The non-desert tiles, in spiral order, carry SPIRAL_NUMBERS in order.
    placed = [
        layout.tile_number[t]
        for t in _SPIRAL_TILE_ORDER
        if layout.tile_resource[t] is not None
    ]
    assert tuple(placed) == SPIRAL_NUMBERS
    assert layout.tile_number[ref.desert_tile(layout)] == 0


def test_placement_mode_keeps_terrain_and_ports() -> None:
    # Toggling the number placement leaves the same map (terrain + harbours),
    # changing only where the tokens sit.
    rand = ref.random_layout(Random(3), "random")
    spiral = ref.random_layout(Random(3), "spiral")
    assert rand.tile_resource == spiral.tile_resource
    assert rand.ports == spiral.ports
    assert rand.tile_number != spiral.tile_number


def test_random_layout_is_deterministic_per_seed() -> None:
    assert ref.random_layout(Random(7)) == ref.random_layout(Random(7))
    assert ref.random_layout(Random(7)) != ref.random_layout(Random(8))


def _drive(seed: int, n_players: int, steps: int) -> None:
    """Play random legal moves, sampling outcomes, asserting invariants hold and
    the belief bounds bracket the truth every step."""
    rng = Random(seed)
    layout = ref.random_layout(rng)
    game = Game.new(layout, ref.desert_tile(layout), n_players=n_players)
    belief = ref.Belief.new(n_players)
    for _ in range(steps):
        legal = game.legal_actions()
        if not legal:
            break
        before = copy.deepcopy(game)
        action = _fill(game, rng.choice(legal), rng)
        game.apply(action)
        belief.update(before, game, action)
        # The bank never goes negative, and the bounds always bracket the truth.
        assert all(game.bank(r) >= 0 for r in RESOURCES)
        for o in range(n_players):
            for p in range(n_players):
                for ri, r in enumerate(RESOURCES):
                    held = game.players[p].resources[r]
                    assert belief.res_lo[o][p][ri] <= held <= belief.res_hi[o][p][ri]
        if n_players == 2:  # every flow is mutually visible: bounds stay exact
            assert belief.res_lo == belief.res_hi


def test_reference_drives_a_random_four_player_game() -> None:
    _drive(seed=6, n_players=4, steps=600)


def test_two_player_beliefs_stay_exact_under_self_play() -> None:
    _drive(seed=1, n_players=2, steps=600)
