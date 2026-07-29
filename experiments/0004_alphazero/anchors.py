"""Frozen anchor checkpoints: committed nets with pinned provenance.

``load_anchor(name)`` rebuilds the net from ``anchors/<name>.json`` (the
architecture sidecar) and deserializes ``anchors/<name>.eqx`` into it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ANCHOR_DIR = Path(__file__).parent / "anchors"

# az0_gnn96x4's -58 Elo (JOURNAL.md's 2026-07-29 scale-reset entry) was fit
# playing the checkpoint's setup phase at GNNBackend's own defaults -- a frozen
# anchor must keep frozen semantics, so every caller pins these values rather
# than reading a run's cfg.net.setup_* (which varies per run/sweep). Shared by
# 0004_alphazero/run.py (arena_helpers.build_net_opponents) and
# 0001_bench_smoke/calibrate.py (az0_spec), which fit this checkpoint's Elo.
NET_OPPONENT_SETUP_DEPTH = 1
NET_OPPONENT_SETUP_TEMPERATURE = 2.0
NET_OPPONENT_SETUP_BEAM = 4


def load_anchor(name: str) -> tuple[Any, Any]:
    """The deserialized net and its ``GraphNetConfig``, from the committed
    ``anchors/`` artifact pair."""
    import equinox as eqx
    import jax
    from settlrl_learn.nn.graphnet import PRESETS
    from settlrl_learn.training import GNNBackend

    meta = json.loads((ANCHOR_DIR / f"{name}.json").read_text())
    netcfg = PRESETS[meta["preset"]]._replace(
        width=meta["width"], layers=meta["layers"], head_depth=meta["head_depth"]
    )
    template = GNNBackend(netcfg).init(jax.random.key(0))
    net = eqx.tree_deserialise_leaves(ANCHOR_DIR / f"{name}.eqx", template)
    return net, netcfg
