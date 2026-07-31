"""Per-action tests for the vectorized engine.

Convention: one file per action, written as plain pytest functions. Each success
case is an ``expecttest.assert_expected_inline`` snapshot (a compact one-field-
per-line summary via ``fixtures.fmt``, and — for board-changing actions — an
ASCII board snapshot from the ``render`` fixture). Invalidation cases (wrong
phase, bad params, unaffordable, ...) assert ``INVALID`` and that state is
unchanged. Success boards come from the ``*_board`` pytest fixtures in
``conftest.py``; ``fixtures.py`` holds the low-level builders they compose.

All 15 actions are vectorized in ``action.py`` and covered here (one file per
action). DomesticTrade is intentionally deferred (never implemented). The trusted
single-game oracle is ``settlrl_game.reference`` (inside settlrl-game), bridged to
the engine by ``tests/conversion.py`` (the differential reference for
``tests/mechanics/test_rules.py`` and ``tests/test_reference_equivalence.py``).
``ActionResult`` lives in ``action.py``, and ``settlrl_engine.env`` exposes a
unified ``step`` / ``available`` over this action set (see ``tests/env/test_env.py``).
"""
