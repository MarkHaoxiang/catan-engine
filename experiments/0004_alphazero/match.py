"""Checkpoint-vs-checkpoint match CLI: a paired, seat-swapped head-to-head
between two GNN net refs, at one shared search budget.

A lab tool, not a product: it reuses
``settlrl_learn.training.arena.arena_spec`` (the loop's own seat-swapped-pair
mechanics) and ``settlrl_agents.evaluate`` underneath it, no new match
machinery.

A net ref is either:

- an anchor name (``anchors/<ref>.json`` + ``.eqx``), loaded via
  ``anchors.load_anchor`` under its frozen setup semantics
  (``anchors.NET_OPPONENT_SETUP_*`` -- a frozen anchor must keep frozen
  semantics, matching ``arena_helpers.build_net_opponents``); or
- a run directory containing ``best.eqx`` (the loop's best-so-far checkpoint,
  ``run_gnn_experiment``) and ``manifest.json`` (``start_run``'s merged-config
  record) -- the architecture (``net.width``/``layers``/``preset``/``depth``),
  feature flags (``net.feature_version``/``incidence``), and setup-opener
  knobs (``net.setup_depth``/``setup_temperature``/``setup_beam``) are read
  back from ``manifest.json["config"]["net"]``, so a checkpoint's own recipe
  stays pinned to what it was actually trained with. Only ``net.kind="gnn"``
  runs are loadable (the mlp backend ships ``params.npz``, not an eqx tree).

Both sides then play under one shared arena-scale search budget
(``games``/``sims``/``considered``/``batch``, CLI-settable, defaulting to
``conf/arena/scale.yaml``) and the study's shared chance/ordering semantics
(``conf/search/scale.yaml``) -- not either side's own trained-with search
config -- per the checklist's doctrine: a match settles a lever, not a
training config.

    uv run python experiments/0004_alphazero/match.py <net_a> <net_b> \\
        [--games N] [--sims N] [--considered N] [--batch N] [--seed N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from settlrl_learn.nn.board_gnn import BoardGNN
    from settlrl_learn.nn.graphnet import GraphNetConfig

_ROOT = Path(__file__).resolve().parents[2]
_ALPHAZERO_DIR = Path(__file__).resolve().parent
_CONF_DIR = _ALPHAZERO_DIR / "conf"


class SearchSemantics(NamedTuple):
    """The arena-scale budget + the shared chance/ordering semantics both
    sides play under (not either side's own trained-with search config)."""

    games: int
    sims: int
    considered: int
    batch: int
    chance_nodes: bool
    dev_chance: bool
    ordered: bool


class SetupOpener(NamedTuple):
    """The fixed-policy setup-phase knobs a net plays with (``setup_policy``)."""

    depth: int
    temperature: float
    beam: int


class NetRef(NamedTuple):
    """One match side, loaded from an anchor name or a run directory."""

    net: BoardGNN
    netcfg: GraphNetConfig
    setup: SetupOpener


def _scale_defaults() -> SearchSemantics:
    """CLI budget defaults + the shared search semantics, read straight from
    ``conf/arena/scale.yaml`` / ``conf/search/scale.yaml`` (mirrors
    ``0001_bench_smoke/calibrate.py``'s ``search_semantics``) rather than
    duplicating the numbers."""
    from omegaconf import OmegaConf

    arena = OmegaConf.to_container(OmegaConf.load(_CONF_DIR / "arena" / "scale.yaml"))
    search = OmegaConf.to_container(OmegaConf.load(_CONF_DIR / "search" / "scale.yaml"))
    assert isinstance(arena, dict) and isinstance(search, dict)
    return SearchSemantics(
        games=int(arena["games"]),
        sims=int(arena["sims"]),
        considered=int(arena["considered"]),
        batch=int(arena["batch"]),
        chance_nodes=bool(search["chance_nodes"]),
        dev_chance=bool(search["dev_chance"]),
        ordered=bool(search["ordered"]),
    )


def load_net(ref: str) -> NetRef:
    """One match side's net, its ``GraphNetConfig``, and the setup-opener it
    plays with -- from an anchor name or a run directory (see the module
    docstring)."""
    if str(_ALPHAZERO_DIR) not in sys.path:
        sys.path.insert(0, str(_ALPHAZERO_DIR))  # same-dir sibling module
    from anchors import (
        ANCHOR_DIR,
        NET_OPPONENT_SETUP_BEAM,
        NET_OPPONENT_SETUP_DEPTH,
        NET_OPPONENT_SETUP_TEMPERATURE,
        load_anchor,
    )

    if (ANCHOR_DIR / f"{ref}.json").exists():
        net, netcfg = load_anchor(ref)
        setup = SetupOpener(
            NET_OPPONENT_SETUP_DEPTH,
            NET_OPPONENT_SETUP_TEMPERATURE,
            NET_OPPONENT_SETUP_BEAM,
        )
        return NetRef(net, netcfg, setup)

    run_dir = Path(ref)
    manifest_path = run_dir / "manifest.json"
    checkpoint = run_dir / "best.eqx"
    if not manifest_path.exists() or not checkpoint.exists():
        raise ValueError(
            f"{ref!r} is neither a known anchor ({ANCHOR_DIR}/{ref}.json) nor a "
            "run dir with manifest.json + best.eqx"
        )

    import equinox as eqx
    import jax
    from settlrl_learn.nn.graphnet import PRESETS
    from settlrl_learn.training import GNNBackend

    net_cfg = json.loads(manifest_path.read_text())["config"]["net"]
    if net_cfg["kind"] != "gnn":
        raise ValueError(
            f"{ref}: net.kind={net_cfg['kind']!r} -- match.py only loads gnn "
            "checkpoints (best.eqx); the mlp backend ships params.npz instead"
        )
    base = PRESETS.get(net_cfg["preset"], PRESETS["gn_global"])
    netcfg = base._replace(
        width=net_cfg["width"],
        layers=net_cfg["layers"],
        head_depth=net_cfg["depth"],
        feature_version=net_cfg["feature_version"],
        incidence=net_cfg["incidence"],
    )
    template = GNNBackend(netcfg).init(jax.random.key(0))
    net = eqx.tree_deserialise_leaves(checkpoint, template)
    setup = SetupOpener(
        net_cfg["setup_depth"], net_cfg["setup_temperature"], net_cfg["setup_beam"]
    )
    return NetRef(net, netcfg, setup)


def run_match(
    net_a_ref: str,
    net_b_ref: str,
    *,
    games: int,
    sims: int,
    considered: int,
    batch: int,
    seed: int,
) -> dict[str, Any]:
    """``net_a``'s wins/episodes/rate ± binomial SE over a paired, seat-swapped
    head-to-head against ``net_b`` (``arena_spec``'s own mechanics: same seed
    scheme for both seatings, real completed-game counts)."""
    from settlrl_agents import BeliefSpec
    from settlrl_learn.training import GNNBackend, arena_spec

    sem = _scale_defaults()
    a = load_net(net_a_ref)
    b = load_net(net_b_ref)
    backend_a = GNNBackend(
        a.netcfg,
        setup_depth=a.setup.depth,
        setup_temperature=a.setup.temperature,
        setup_beam=a.setup.beam,
        chance_nodes=sem.chance_nodes,
        dev_chance=sem.dev_chance,
        ordered=sem.ordered,
    )
    backend_b = GNNBackend(
        b.netcfg,
        setup_depth=b.setup.depth,
        setup_temperature=b.setup.temperature,
        setup_beam=b.setup.beam,
        chance_nodes=sem.chance_nodes,
        dev_chance=sem.dev_chance,
        ordered=sem.ordered,
    )

    def make_b_agent() -> Any:
        return backend_b.play_agent(
            b.net, num_simulations=sims, max_num_considered_actions=considered
        )

    opponent_b = BeliefSpec(make_b_agent, frozenset((2,)))
    result = arena_spec(
        backend_a,
        a.net,
        opponent=opponent_b,
        n_games=games,
        num_simulations=sims,
        max_num_considered_actions=considered,
        batch_size=batch,
        seed=seed,
    )
    p = result.winrate
    se = (p * (1 - p) / max(result.episodes, 1)) ** 0.5
    return {
        "net_a": net_a_ref,
        "net_b": net_b_ref,
        "games": games,
        "sims": sims,
        "considered": considered,
        "batch": batch,
        "seed": seed,
        "wins_a": result.wins,
        "episodes": result.episodes,
        "winrate_a": p,
        "se": se,
    }


def _slug(ref: str) -> str:
    return Path(ref).name or ref


def _default_out_path(net_a_ref: str, net_b_ref: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    return (
        _ROOT
        / "runs"
        / "0004_alphazero"
        / "matches"
        / f"{stamp}-{_slug(net_a_ref)}-vs-{_slug(net_b_ref)}.json"
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    defaults = _scale_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("net_a", help="anchor name or run dir")
    parser.add_argument("net_b", help="anchor name or run dir")
    parser.add_argument("--games", type=int, default=defaults.games)
    parser.add_argument("--sims", type=int, default=defaults.sims)
    parser.add_argument("--considered", type=int, default=defaults.considered)
    parser.add_argument("--batch", type=int, default=defaults.batch)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run_match(
        args.net_a,
        args.net_b,
        games=args.games,
        sims=args.sims,
        considered=args.considered,
        batch=args.batch,
        seed=args.seed,
    )
    print(
        f"{result['net_a']} vs {result['net_b']}: "
        f"{result['wins_a']:.0f}/{result['episodes']} "
        f"({result['winrate_a']:.1%} +/- {result['se']:.1%})"
    )
    out = args.out or _default_out_path(args.net_a, args.net_b)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    main()
