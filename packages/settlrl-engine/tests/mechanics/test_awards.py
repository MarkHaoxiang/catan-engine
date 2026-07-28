"""Tests for awards.py award reassignment: Largest Army and Longest Road holder
selection (threshold, holder-keeps-on-tie, and the rulebook "set aside on a tie
among non-holders" rule), checked against the `settlrl-reference` oracle.

(The longest-road *length* DFS itself is covered in test_rules.py.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from settlrl_engine.board import make_board
from settlrl_engine.board.layout import N_EDGES
from settlrl_engine.board.resources import N_PLAYERS
from settlrl_engine.board.state import NO_INDEX, BoardState, make_board_state
from settlrl_engine.mechanics import awards, longest_road
from settlrl_engine.mechanics.action import ActionParams, ActionType, apply_action

from tests import conversion as reference
from tests.mechanics._occupancy import random_occupancy
from tests.mechanics._occupancy import single as _single

# More roads than the other suites: stress the longest-road award selection.
_EDGE_P = [0.4, 0.2, 0.16, 0.14, 0.1]
_VERTEX_P = [0.7, 0.1, 0.08, 0.07, 0.05]


def _army_state(knights: list[int], owner: int) -> BoardState:
    return make_board_state(1, key=jax.random.key(0))._replace(
        knights_played=jnp.asarray(knights, jnp.uint8)[None],
        largest_army_owner=jnp.asarray([owner], jnp.uint8),
    )


class TestLargestArmy:
    def test_no_qualifier_is_unclaimed(self) -> None:
        out = awards.recompute_largest_army(
            _single(_army_state([2, 2, 0, 1], NO_INDEX))
        )
        assert int(out.largest_army_owner) == NO_INDEX

    def test_awarded_at_three(self) -> None:
        out = awards.recompute_largest_army(
            _single(_army_state([0, 3, 0, 0], NO_INDEX))
        )
        assert int(out.largest_army_owner) == 1

    def test_tie_keeps_current_holder(self) -> None:
        # Player 0 already holds it; player 2 ties at 3 -> holder keeps it.
        out = awards.recompute_largest_army(_single(_army_state([3, 0, 3, 0], 0)))
        assert int(out.largest_army_owner) == 0

    def test_matches_reference(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            knights = rng.integers(0, 5, size=N_PLAYERS).tolist()
            # Reachable invariant: knights never decrease, so once anyone reaches
            # 3 there is a holder and it is among the leaders. (Engine and the
            # reference deliberately differ only on unreachable states where a
            # stale holder trails a tied lead -- not worth comparing.)
            top = max(knights)
            if top >= 3:
                leaders = [p for p in range(N_PLAYERS) if knights[p] == top]
                owner = int(rng.choice(leaders))
            else:
                owner = NO_INDEX
            state = _army_state(knights, owner)
            got = awards.recompute_largest_army(_single(state))
            ref = reference.recompute_largest_army(state, 0)
            assert int(got.largest_army_owner) == int(ref.largest_army_owner[0]), (
                f"knights={knights} owner={owner}"
            )


def _road_state(seed: int) -> BoardState:
    edge_road, vertex_owner = random_occupancy(seed, edge_p=_EDGE_P, vertex_p=_VERTEX_P)
    owner = int(np.random.default_rng(seed).choice([NO_INDEX, 0, 1, 2, 3]))
    return make_board_state(1, key=jax.random.key(0))._replace(
        edge_road=jnp.asarray(edge_road)[None],
        vertex_owner=jnp.asarray(vertex_owner)[None],
        longest_road_owner=jnp.asarray([owner], jnp.uint8),
    )


class TestLongestRoadAward:
    def test_recompute_matches_reference(self) -> None:
        for seed in range(10):
            state = _road_state(seed)
            got = awards.recompute_longest_road(_single(state))
            ref = reference.recompute_longest_road(state, 0)
            assert int(got.longest_road_owner) == int(ref.longest_road_owner[0]), (
                f"seed={seed}: owner"
            )
            assert int(got.longest_road_len) == int(ref.longest_road_len[0]), (
                f"seed={seed}: len"
            )

    def test_needed_false_keeps_stored_award(self) -> None:
        # With ``needed=False`` the stored holder/length survive untouched even
        # though a from-scratch recompute would reassign them; ``needed=True``
        # recomputes as usual.
        state = _single(_tie_state(owner=NO_INDEX))
        kept = awards.recompute_longest_road(state, jnp.bool_(False))
        assert int(kept.longest_road_owner) == NO_INDEX
        assert int(kept.longest_road_len) == 0
        # Same state, gate open: players 1 and 2 tie at 5 with no holder, so the
        # card stays unheld -- but a sole leader is taken.
        solo = state._replace(
            edge_road=state.edge_road.at[_ROAD_P2[-1]].set(0)  # break the tie
        )
        got = awards.recompute_longest_road(solo, jnp.bool_(True))
        assert int(got.longest_road_owner) == 1
        assert int(got.longest_road_len) == 5
        # And the gated version of the same call keeps the empty award.
        ungot = awards.recompute_longest_road(solo, jnp.bool_(False))
        assert int(ungot.longest_road_owner) == NO_INDEX


# Two vertex-disjoint 5-segment roads (chosen from the board geometry) plus a
# single segment for player 0, used to construct Longest Road ties.
_ROAD_P0 = [17]  # player 0: length 1
_ROAD_P1 = [0, 7, 4, 3, 5]  # player 1: length 5
_ROAD_P2 = [10, 13, 14, 18, 19]  # player 2: length 5


def _tie_state(owner: int) -> BoardState:
    """A board where players 1 and 2 tie for the longest road (5) and player 0
    trails with 1, with the Longest Road card currently held by ``owner``."""
    edge_road = np.zeros(N_EDGES, np.uint8)
    for e in _ROAD_P0:
        edge_road[e] = 1  # player 0
    for e in _ROAD_P1:
        edge_road[e] = 2  # player 1
    for e in _ROAD_P2:
        edge_road[e] = 3  # player 2
    return make_board_state(1, key=jax.random.key(0))._replace(
        edge_road=jnp.asarray(edge_road)[None],
        longest_road_owner=jnp.asarray([owner], jnp.uint8),
    )


class TestLongestRoadTie:
    """The rulebook tie rule (Almanac, "Longest Road", p.9): the holder keeps the
    card only while tied for the longest road; if it is beaten and two or more
    players tie for the new longest, the card is set aside (held by no one)."""

    def test_tie_lengths_are_as_expected(self) -> None:
        s = _single(_tie_state(NO_INDEX))
        lens = [
            int(
                longest_road.longest_road_length(
                    s.edge_road, s.vertex_owner, jnp.int32(p)
                )
            )
            for p in range(N_PLAYERS)
        ]
        assert lens == [1, 5, 5, 0]

    def test_beaten_holder_with_tie_sets_card_aside(self) -> None:
        # Player 0 holds the card but has been reduced to 1 segment; players 1
        # and 2 now tie at 5. Per the rulebook the card is set aside.
        out = awards.recompute_longest_road(_single(_tie_state(0)))
        assert int(out.longest_road_owner) == NO_INDEX
        assert int(out.longest_road_len) == 0

    def test_no_holder_with_tie_is_unclaimed(self) -> None:
        # No current holder and a 2-way tie for longest -> nobody holds it.
        out = awards.recompute_longest_road(_single(_tie_state(NO_INDEX)))
        assert int(out.longest_road_owner) == NO_INDEX

    def test_tied_holder_keeps_card(self) -> None:
        # If the holder is itself one of the tied leaders, it keeps the card.
        out = awards.recompute_longest_road(_single(_tie_state(1)))
        assert int(out.longest_road_owner) == 1
        assert int(out.longest_road_len) == 5


# Vertex 0 has degree 3 with incident edges 0, 1, 2; vertex 4 is the interior
# vertex shared by chain edges 7 and 4 of _ROAD_P1 (see the board geometry).
_DEG3_V, _DEG3_EDGES = 0, [0, 1, 2]
_P1_MID_V = 4


def _edges_state(owners: dict[int, int], **fields: jax.Array) -> BoardState:
    """A single-game state with ``edge -> owner code`` roads and extra fields."""
    edge_road = np.zeros(N_EDGES, np.uint8)
    for e, code in owners.items():
        edge_road[e] = code
    return _single(
        make_board_state(1, key=jax.random.key(0))._replace(
            edge_road=jnp.asarray(edge_road)[None], **fields
        )
    )


class TestLongestRoadGates:
    def test_road_build_gate_needs_five_roads(self) -> None:
        four = _edges_state(dict.fromkeys(_ROAD_P1[:4], 1))
        five = _edges_state(dict.fromkeys(_ROAD_P1, 1))
        assert not bool(awards.road_build_gate(four, jnp.int32(0)))
        assert bool(awards.road_build_gate(five, jnp.int32(0)))
        # Other players' roads don't count toward the builder's five.
        mixed = _edges_state(
            {**dict.fromkeys(_ROAD_P2, 2), **dict.fromkeys(_ROAD_P1[:4], 1)}
        )
        assert not bool(awards.road_build_gate(mixed, jnp.int32(0)))

    def test_settlement_break_gate_needs_two_same_opponent_edges(self) -> None:
        e0, e1, _e2 = _DEG3_EDGES
        v = jnp.int32(_DEG3_V)
        me = jnp.int32(0)  # owner code 1
        # Two edges of the same opponent: a trail could pass through -> True.
        assert bool(awards.settlement_break_gate(_edges_state({e0: 2, e1: 2}), v, me))
        # One opponent edge: the vertex is at most a trail endpoint -> False.
        assert not bool(awards.settlement_break_gate(_edges_state({e0: 2}), v, me))
        # Two edges of *different* opponents -> False.
        assert not bool(
            awards.settlement_break_gate(_edges_state({e0: 2, e1: 3}), v, me)
        )
        # Two of the builder's own edges -> False.
        assert not bool(
            awards.settlement_break_gate(_edges_state({e0: 1, e1: 1}), v, me)
        )

    def test_apply_action_settlement_break_reassigns_award(self) -> None:
        # Player 1 holds Longest Road with the 5-chain; player 0 drops a
        # (force-applied) settlement on its interior vertex, severing it into
        # 2 + 3 -> the award is recomputed and set aside.
        layout = _single(make_board(1, seed=0)[0])
        state = _edges_state(
            dict.fromkeys(_ROAD_P1, 2),
            longest_road_owner=jnp.asarray([1], jnp.uint8),
            longest_road_len=jnp.asarray([5], jnp.uint8),
        )
        out, _result = apply_action(
            layout,
            state,
            jnp.int32(ActionType.BUILD_SETTLEMENT),
            ActionParams(idx=jnp.int32(_P1_MID_V), target=jnp.int32(0)),
            jnp.bool_(True),
        )
        assert int(out.longest_road_owner) == NO_INDEX
        assert int(out.longest_road_len) == 0

    def test_apply_action_gated_build_keeps_stored_award(self) -> None:
        # Player 0 builds a 4th road (below the 5-road gate), so the recompute
        # is skipped and a deliberately stale stored award survives -- a full
        # recompute would have corrected it to player 1's 5-chain.
        layout = _single(make_board(1, seed=0)[0])
        free_edge = next(
            e for e in range(N_EDGES) if e not in _ROAD_P1 and e not in _ROAD_P2
        )
        state = _edges_state(
            {**dict.fromkeys(_ROAD_P1, 2), **dict.fromkeys(_ROAD_P2[:3], 1)},
            longest_road_owner=jnp.asarray([3], jnp.uint8),  # stale on purpose
            longest_road_len=jnp.asarray([9], jnp.uint8),
        )
        out, _result = apply_action(
            layout,
            state,
            jnp.int32(ActionType.BUILD_ROAD),
            ActionParams(idx=jnp.int32(free_edge), target=jnp.int32(0)),
            jnp.bool_(True),
        )
        assert int(out.longest_road_owner) == 3  # untouched: gate was off
        assert int(out.longest_road_len) == 9
