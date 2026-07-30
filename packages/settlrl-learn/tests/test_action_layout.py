"""Semantic checks for the flat-action -> board-structure map.

The module's own import-time asserts only check that ``SCATTER`` is a total,
onto, injective map into ``[0, BIG)``. These tests pin the *meaning* of that
map for one representative row of each class: which offset window it lands in,
and which element of that class it decodes to.
"""

from __future__ import annotations

import numpy as np
from settlrl_engine.env import ActionType
from settlrl_learn.nn.action_layout import (
    _E_OFF,
    _O_OFF,
    _T_OFF,
    _V_OFF,
    BIG,
    N_ECLASS,
    N_TCLASS,
    N_VCLASS,
    SCATTER,
    _is_other,
)
from settlrl_search.rows import flat_row


def test_scatter_class_offsets() -> None:
    # vertex: {setup-settlement=0, settlement=1, city=2} at vertex v -> V_OFF +
    # v*N_VCLASS + class, inside the vertex window.
    a = flat_row(ActionType.SETUP_SETTLEMENT, idx=3)
    assert SCATTER[a] == _V_OFF + 3 * N_VCLASS + 0
    assert _V_OFF <= SCATTER[a] < _E_OFF

    a = flat_row(ActionType.BUILD_CITY, idx=10)
    assert SCATTER[a] == _V_OFF + 10 * N_VCLASS + 2
    assert _V_OFF <= SCATTER[a] < _E_OFF

    # edge: {setup-road=0, road=1} at edge e -> E_OFF + e*N_ECLASS + class,
    # inside the edge window.
    a = flat_row(ActionType.SETUP_ROAD, idx=8)
    assert SCATTER[a] == _E_OFF + 8 * N_ECLASS + 0
    assert _E_OFF <= SCATTER[a] < _T_OFF

    a = flat_row(ActionType.BUILD_ROAD, idx=5)
    assert SCATTER[a] == _E_OFF + 5 * N_ECLASS + 1
    assert _E_OFF <= SCATTER[a] < _T_OFF

    # tile: {robber, knight} x {no-steal (target<0), steal (target>=0)} at tile
    # t -> T_OFF + t*N_TCLASS + tclass, inside the tile window. The victim
    # collapses: every steal target at a tile shares one "steal" slot.
    t = 7
    no_steal_robber = flat_row(ActionType.MOVE_ROBBER, idx=t, target=-1)
    steal_robber = flat_row(ActionType.MOVE_ROBBER, idx=t, target=0)
    no_steal_knight = flat_row(ActionType.PLAY_KNIGHT, idx=t, target=-1)
    steal_knight = flat_row(ActionType.PLAY_KNIGHT, idx=t, target=1)
    assert SCATTER[no_steal_robber] == _T_OFF + t * N_TCLASS + 0
    assert SCATTER[steal_robber] == _T_OFF + t * N_TCLASS + 1
    assert SCATTER[no_steal_knight] == _T_OFF + t * N_TCLASS + 2
    assert SCATTER[steal_knight] == _T_OFF + t * N_TCLASS + 3
    for a in (no_steal_robber, steal_robber, no_steal_knight, steal_knight):
        assert _T_OFF <= SCATTER[a] < _O_OFF
    # a second victim at the same tile collapses onto the same steal slot.
    assert (
        SCATTER[flat_row(ActionType.MOVE_ROBBER, idx=t, target=1)]
        == SCATTER[steal_robber]
    )

    # other (dense, non-spatial): decodes to its rank among "other" rows, in
    # row-table order -- distinct rows land at distinct, ordered slots.
    roll = flat_row(ActionType.ROLL_DICE)
    end = flat_row(ActionType.END_TURN)
    other_rows = np.flatnonzero(_is_other)
    assert SCATTER[roll] == _O_OFF + int(np.searchsorted(other_rows, roll))
    assert SCATTER[end] == _O_OFF + int(np.searchsorted(other_rows, end))
    assert SCATTER[roll] != SCATTER[end]
    assert _O_OFF <= SCATTER[roll] < BIG and _O_OFF <= SCATTER[end] < BIG
