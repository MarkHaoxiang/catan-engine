"""Frozen anchor checkpoints: committed nets with pinned provenance.

``load_anchor(name)`` rebuilds the net from ``anchors/<name>.json`` (the
architecture sidecar) and deserializes ``anchors/<name>.eqx`` into it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ANCHOR_DIR = Path(__file__).parent / "anchors"


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
