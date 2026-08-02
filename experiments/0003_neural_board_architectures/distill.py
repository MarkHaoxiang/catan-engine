"""Frozen distillation dataset: the az2 anchor's own self-play, full-search
targets.

``generate`` rolls the anchor's net through the production self-play stack
(``settlrl_learn.training.loop``) and dumps every recorded key plus the raw
outcome ``value`` (z) and the root search value ``q`` -- kept separate, so the
guard applies the production value blend at training time -- cached under
``runs/_cache/0003``. Deliberate divergences from the production recipe:
uniform sims (no PCR mix) and a non-persistent generation batch -- both chosen
so every position is a full-search target.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from settlrl_learn.experiment import sibling_module
from settlrl_learn.training.config import (
    LearnConfig,
    SearchSettings,
    SelfPlayConfig,
    ValueBlendConfig,
)

_CACHE = Path(__file__).resolve().parents[2] / "runs" / "_cache" / "0003"
_ALPHAZERO_DIR = Path(__file__).resolve().parents[1] / "0004_alphazero"

_DISTILL_SCHEMA = 2
"""Suffix on every cache file; bump whenever the stored arrays' meaning changes
(fields, layout, or the set of knobs hashed into the key)."""


def _key(anchor: str, sims: int, batch: int, n_samples: int, seed: int) -> str:
    blob = json.dumps(
        {"anchor": anchor, "sims": sims, "batch": batch,
         "n_samples": n_samples, "seed": seed},
        sort_keys=True,
    )  # fmt: skip
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def learn_config(anchor: str, sims: int, batch: int, n_samples: int) -> LearnConfig:
    """The minimal ``LearnConfig`` the dataset is generated under: ``q``
    recording on (``value_blend.max > 0`` -- the loop's own predicate), the
    production opening-temperature schedule, PCR off (every position
    full-search, ``train_policy`` all 1), and the search semantics the anchor
    sidecar pins."""
    anchors = sibling_module(_ALPHAZERO_DIR, "anchors")
    meta = json.loads((anchors.ANCHOR_DIR / f"{anchor}.json").read_text())
    semantics = meta["search_semantics"]
    return LearnConfig(
        n_iterations=1,  # unused: generation calls run_selfplay directly
        search=SearchSettings(
            num_simulations=sims,
            # single sampled roll, matching the scale recipe the anchors
            # trained under. chance/dev/ordered come from the sidecar's
            # arena-scoped `search_semantics` block and coincide with the
            # training semantics (0004's conf/search/scale.yaml).
            expected_rolls=False,
            chance_nodes=semantics["chance_nodes"],
            dev_chance=semantics["dev_chance"],
            ordered=semantics["ordered"],
        ),
        selfplay=SelfPlayConfig(
            samples=n_samples,
            batch=batch,
            pcr_full_prob=1.0,
            # production opening-temperature schedule (conf/experiment/v2_hetero.yaml)
            temperature_moves=30,
        ),
        # only the > 0 predicate matters here (it switches q recording on);
        # the blend itself is applied at training time, never at dump time.
        value_blend=ValueBlendConfig(max=1.0),
    )


def generate(
    anchor: str, sims: int, batch: int, n_samples: int, seed: int
) -> dict[str, np.ndarray]:
    """Collect (or load from ``runs/_cache``) >= ``n_samples`` positions of the
    anchor's own self-play at ``sims`` full-search simulations per move.

    Keys: the GNN observation (``nodes``/``edges``/``glob``/``tiles``),
    ``policy`` (the search's improved policy over the flat action space),
    ``mask``, ``train_policy`` (all 1), ``q`` (root search value, [-1, 1]),
    ``value`` (raw outcome z), plus the generating featurization
    (``feature_version``/``incidence``/``with_tiles`` scalars) for load-time
    checks."""
    path = (
        _CACHE
        / f"distill-{_key(anchor, sims, batch, n_samples, seed)}-v{_DISTILL_SCHEMA}.npz"
    )
    if path.exists():
        with np.load(path) as d:
            return {k: d[k] for k in d.files}

    import functools

    import equinox as eqx
    from settlrl_learn.training import GNNBackend
    from settlrl_learn.training.loop import run_selfplay, selfplay_callables

    anchors = sibling_module(_ALPHAZERO_DIR, "anchors")
    net, netcfg = anchors.load_anchor(anchor)
    cfg = learn_config(anchor, sims, batch, n_samples)
    backend = GNNBackend(
        netcfg,
        setup_depth=anchors.NET_OPPONENT_SETUP_DEPTH,
        setup_temperature=anchors.NET_OPPONENT_SETUP_TEMPERATURE,
        setup_beam=anchors.NET_OPPONENT_SETUP_BEAM,
        chance_nodes=cfg.search.chance_nodes,
        dev_chance=cfg.search.dev_chance,
        ordered=cfg.search.ordered,
    )
    calls = selfplay_callables(backend, cfg, net)
    net_search = calls.make_net_search(cfg.search.num_simulations)
    search = functools.partial(net_search, eqx.partition(net, eqx.is_array)[0])
    samples, _stats, _carry = run_selfplay(calls, search, cfg, n_samples, seed)
    assert bool(np.all(samples["train_policy"] == 1.0))  # PCR off: full-search only
    data = {k: np.asarray(v) for k, v in samples.items()}
    data["feature_version"] = np.asarray(netcfg.feature_version)
    data["incidence"] = np.asarray(netcfg.incidence)
    # the generating backend featurizes tiles only for a hetero net
    data["with_tiles"] = np.asarray(netcfg.hetero)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **cast(dict[str, Any], data))
    return data
