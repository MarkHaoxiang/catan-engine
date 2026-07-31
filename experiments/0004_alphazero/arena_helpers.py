"""Arena/gauntlet/bench glue for the AlphaZero experiment.

The frozen-checkpoint arena rungs (``build_net_opponents``), the end-of-run
verdict gauntlet (``run_final_gauntlet`` / ``gauntlet_verdict``), and the
throughput-bench mode (``BenchConfig`` / ``run_bench``).
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import jax
from pydantic import BaseModel, ConfigDict
from settlrl_agents import BeliefSpec
from settlrl_learn.experiment import Run
from settlrl_learn.training import ArenaConfig, Backend, GNNBackend, run_arena

if TYPE_CHECKING:
    # deferred (see `from __future__ import annotations`): avoids a runtime
    # circular import, since run.py imports this module's symbols.
    from run import AlphaZeroConfig


class BenchConfig(BaseModel):
    """Throughput-bench knobs (mode: bench): the frozen anchor + timing scheme."""

    model_config = ConfigDict(extra="forbid")

    anchor: str = "az0_gnn96x4"
    warmup: int = 1
    repeats: int = 3


def build_net_opponents(
    cfg: AlphaZeroConfig,
) -> dict[str, tuple[BeliefSpec, float, int]]:
    """The specs `learn` seats for ``cfg.arena.net_opponents``: each named anchor
    loaded from ``anchors/`` and played by its own GNN search at the arena's
    budget, under the calibration's frozen setup opener
    (``anchors.NET_OPPONENT_SETUP_*``) and this run's search semantics."""
    # same-dir sibling module (see run_bench's note on sys.path).
    from anchors import (
        NET_OPPONENT_SETUP_BEAM,
        NET_OPPONENT_SETUP_DEPTH,
        NET_OPPONENT_SETUP_TEMPERATURE,
        load_anchor,
    )

    s = cfg.search
    out: dict[str, tuple[BeliefSpec, float, int]] = {}
    for name, opp in cfg.arena.net_opponents.items():
        net, netcfg = load_anchor(name)
        backend = GNNBackend(
            netcfg, setup_depth=NET_OPPONENT_SETUP_DEPTH,
            setup_temperature=NET_OPPONENT_SETUP_TEMPERATURE,
            setup_beam=NET_OPPONENT_SETUP_BEAM,
            chance_nodes=s.chance_nodes, dev_chance=s.dev_chance, ordered=s.ordered,
        )  # fmt: skip
        agent = backend.play_agent(
            net,
            num_simulations=cfg.arena.sims,
            max_num_considered_actions=cfg.arena.considered,
        )
        out[name] = (
            BeliefSpec(lambda agent=agent: agent, frozenset((2,))),
            opp.elo,
            opp.every,
        )
    return out


def run_final_gauntlet(
    backend: Backend,
    net: Any,
    cfg: AlphaZeroConfig,
    net_opponents: dict[str, tuple[BeliefSpec, float, int]],
) -> dict[str, float]:
    """The end-of-run verdict gauntlet: every configured opponent (the registry
    ones plus ``net_opponents``) at ``cfg.final_games`` games each, with every
    per-round schedule neutralized so nothing is skipped (``opponent_every={}``,
    every net opponent's ``every`` forced to 1). With every schedule's period 1,
    ``round_index % 1 == 0`` unconditionally, so ``round_index`` plays no role;
    0 is passed for clarity. Seeded off ``cfg.seed + 99`` so the gauntlet's games
    stay disjoint from the in-loop training arenas.

    Drops jax's compilation caches first, so no compiled program a caller built
    before this call survives it."""
    # memory: the training loop's compiled programs (self-play at B=512, the
    # optimiser step, the in-loop arenas) stay loaded on the device via jax's jit
    # caches long after `learn` returned, and the gauntlet then compiles its own
    # (one per rung, plus two net-opponent searches). The clear costs a recompile
    # and frees them; the collect drops the training arrays reachable only
    # through cycles. Neither returns pool memory to the driver -- if a foreign
    # process is sharing the GPU, launch with
    # XLA_PYTHON_CLIENT_PREALLOCATE=false as well.
    jax.clear_caches()  # type: ignore[no-untyped-call]
    gc.collect()
    final_arena_cfg = ArenaConfig(
        **cfg.arena.model_dump(exclude={"net_opponents"})
    ).model_copy(update={"games": cfg.final_games, "opponent_every": {}})
    final_opponents = {
        name: (spec, elo, 1) for name, (spec, elo, _every) in net_opponents.items()
    }
    return run_arena(
        backend, net, final_arena_cfg, seed=cfg.seed + 99, round_index=0,
        net_opponents=final_opponents,
    )  # fmt: skip


def gauntlet_verdict(metrics: dict[str, float], gate_elo: float) -> str:
    """``pass`` iff the gauntlet's lower 2-sigma Elo bound clears ``gate_elo``.

    Raises ``ValueError`` if ``run_final_gauntlet`` produced no ``arena_elo``
    (every configured opponent was skipped, or none carried an anchor Elo) --
    a misconfigured gauntlet, not a legitimate zero score."""
    if "arena_elo" not in metrics:
        raise ValueError(
            "gauntlet produced no arena_elo -- check cfg.arena.opponents / "
            "arena.net_opponents are non-empty and every opponent has an "
            "anchor_elos entry (or is a net_opponent, which carries its own)"
        )
    return (
        "pass"
        if metrics["arena_elo"] - 2 * metrics["arena_elo_se"] >= gate_elo
        else "fail"
    )


def run_bench(run: Run, cfg: AlphaZeroConfig) -> None:
    """Pinned self-play throughput at a frozen net; verdict is always
    ``recorded`` -- the comparison is between two runs' result.json."""
    import jax

    # same-dir sibling module; sys.path already has this dir (script invocation
    # and the test conftest both put it there), matching 0002/0003's convention.
    from anchors import load_anchor
    from settlrl_learn.training import GNNBackend, bench_selfplay

    net, netcfg = load_anchor(cfg.bench.anchor)
    anchor_arch = (
        netcfg.width, netcfg.layers, netcfg.head_depth,
        netcfg.feature_version, netcfg.incidence,
    )  # fmt: skip
    preset_arch = (
        cfg.net.width, cfg.net.layers, cfg.net.depth,
        cfg.net.feature_version, cfg.net.incidence,
    )  # fmt: skip
    if anchor_arch != preset_arch:
        raise ValueError(
            f"anchor {cfg.bench.anchor!r} (width/layers/head_depth/feature_version/"
            f"incidence = {anchor_arch}) does not match preset cfg.net "
            f"({preset_arch}) -- the run manifest would misdescribe the "
            "measured workload"
        )
    s = cfg.search
    backend = GNNBackend(
        netcfg, setup_depth=cfg.net.setup_depth,
        setup_temperature=cfg.net.setup_temperature, setup_beam=cfg.net.setup_beam,
        chance_nodes=s.chance_nodes, dev_chance=s.dev_chance, ordered=s.ordered,
    )  # fmt: skip
    results = bench_selfplay(
        backend, net, cfg.to_learn_config(),
        warmup=cfg.bench.warmup, repeats=cfg.bench.repeats, seed=cfg.seed,
    )  # fmt: skip
    run.log(**results)
    run.finish("recorded", device=jax.devices()[0].device_kind, **results)
