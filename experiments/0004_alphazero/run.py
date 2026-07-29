"""AlphaZero self-play for 2-player Catan.

Hypothesis: a value+policy net trained by AlphaZero self-play (the search as its
own teacher) beats ``lookahead(heuristic)`` at 2p — the settlrl-learn Stage-1
gate. The loop itself lives in ``settlrl_learn.training``; this only composes it
with a config, per-iteration logging, and the gate verdict.

Config is composed by **hydra** from ``conf/`` (config groups + an ``experiment``
preset directory) and validated into the nested :class:`AlphaZeroConfig`
(pydantic). hydra's cwd takeover is disabled (``conf/config.yaml``'s ``hydra``
block) so ``start_run`` keeps owning the run dir + manifest.

    uv run python experiments/0004_alphazero/run.py [+experiment=<name>] [key=value ...]
    uv run python experiments/0004_alphazero/run.py -m +experiment=gnn,gnn_warm   # sweep
"""

from pathlib import Path
from typing import Any, Literal

import hydra
import jax
import wandb
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from settlrl_agents import BeliefSpec
from settlrl_learn import save_az_params
from settlrl_learn.experiment import Config, Run, start_run
from settlrl_learn.training import (
    ArenaConfig,
    Backend,
    EvalConfig,
    GNNBackend,
    LearnConfig,
    MLPBackend,
    OptimConfig,
    ReplayConfig,
    SearchSettings,
    SelfPlayConfig,
    TeacherConfig,
    ValueBlendConfig,
    learn,
    run_arena,
)


class _Sub(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NetConfig(_Sub):
    """The architecture + setup-opener knobs (experiment-side, not the loop's)."""

    kind: Literal["mlp", "gnn"] = "mlp"
    width: int = 64
    depth: int = 2  # trunk hidden layers (GNN: readout-head hidden layers)
    layers: int = 3  # GNN message-passing layers (ignored by mlp)
    preset: str = "gn_global"  # settlrl_learn.nn.graphnet.PRESETS key
    value_weight: float = 1.0  # mlp value-loss weight
    # the setup phase is played by a fixed policy (GNN path). setup_depth<=1 =
    # lookahead opener; >=2 = probabilistic-expectimax (>=3p / better value).
    setup_depth: int = 1
    setup_temperature: float = 2.0
    setup_beam: int = 4


class ArenaNetOpponent(_Sub):
    """A frozen checkpoint at the arena table: pinned Elo + play-every-N schedule
    for the ``anchors/`` artifact of the same name."""

    elo: float
    every: int = Field(default=1, ge=1)


class ArenaSettings(ArenaConfig):
    """The loop's arena knobs plus the experiment-only frozen-checkpoint rungs:
    the loop takes ready play specs (not config), so this only *names* them."""

    net_opponents: dict[str, ArenaNetOpponent] = Field(default_factory=dict)


class WandbConfig(_Sub):
    mode: Literal["online", "offline", "disabled"] = "online"
    project: str = "settlrl-0004-alphazero"
    # param-distribution histograms (gnn on_iter only) are far heavier than the
    # scalar log -- fire every `hist_every` iterations, not every one. 0 disables.
    hist_every: int = 10


class BenchConfig(_Sub):
    """Throughput-bench knobs (mode: bench): the frozen anchor + timing scheme."""

    anchor: str = "az0_gnn96x4"
    warmup: int = 1
    repeats: int = 3


class AlphaZeroConfig(Config):
    """The experiment schema: the loop's grouped config plus experiment-only
    sections (net architecture, wandb, the gate)."""

    seed: int = 0
    mode: Literal["train", "bench"] = "train"
    n_iterations: int = 20
    checkpoint_every: int = 5
    resume_from: str = ""  # prior run dir to continue bit-exactly (its runstate.eqx)
    gate_winrate: float = 0.55  # legacy gate vs lookahead; reported, no longer gates
    gate_elo: float = 35.0  # pass iff gauntlet's arena_elo - 2*arena_elo_se clears this
    final_games: int = 400  # games/rung in the end-of-run gauntlet (vs arena.games)
    net: NetConfig = Field(default_factory=NetConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    bench: BenchConfig = Field(default_factory=BenchConfig)
    search: SearchSettings = Field(default_factory=SearchSettings)
    selfplay: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    value_blend: ValueBlendConfig = Field(default_factory=ValueBlendConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    arena: ArenaSettings = Field(default_factory=ArenaSettings)

    def to_learn_config(self) -> LearnConfig:
        """Pack the loop groups into the net-agnostic ``LearnConfig``."""
        return LearnConfig(
            n_iterations=self.n_iterations, seed=self.seed,
            checkpoint_every=self.checkpoint_every,
            search=self.search, selfplay=self.selfplay, optim=self.optim,
            replay=self.replay, teacher=self.teacher, value_blend=self.value_blend,
            eval=self.eval,
            arena=ArenaConfig(**self.arena.model_dump(exclude={"net_opponents"})),
        )  # fmt: skip


# az0_gnn96x4's -58 Elo (JOURNAL.md's 2026-07-29 scale-reset entry) was fit
# playing the checkpoint's setup phase at GNNBackend's own defaults -- a frozen
# anchor must keep frozen semantics, so this pins those values rather than
# reading this run's cfg.net.setup_* (which varies per run/sweep).
NET_OPPONENT_SETUP_DEPTH = 1
NET_OPPONENT_SETUP_TEMPERATURE = 2.0
NET_OPPONENT_SETUP_BEAM = 4


def build_net_opponents(
    cfg: AlphaZeroConfig,
) -> dict[str, tuple[BeliefSpec, float, int]]:
    """The specs `learn` seats for ``cfg.arena.net_opponents``: each named anchor
    loaded from ``anchors/`` and played by its own GNN search at the arena's
    budget, under the calibration's frozen setup opener (``NET_OPPONENT_SETUP_*``)
    and this run's search semantics."""
    # same-dir sibling module (see run_bench's note on sys.path).
    from anchors import load_anchor

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
    0 is passed for clarity. Seeded off ``cfg.seed + 99`` -- the legacy final-arena
    base -- so the gauntlet's games stay disjoint from the in-loop training arenas."""
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


def run_gnn_experiment(run: Run, cfg: AlphaZeroConfig) -> None:
    """The board-GNN value+policy net (experiment 0003's recommendation) in the
    training loop, with the setup phase delegated to a fixed policy."""
    import equinox as eqx
    import numpy as np
    from settlrl_agents.value import heuristic_value
    from settlrl_learn.nn.board_gnn import BoardGNN
    from settlrl_learn.nn.graphnet import PRESETS

    s = cfg.search
    base = PRESETS.get(cfg.net.preset, PRESETS["gn_global"])
    netcfg = base._replace(
        width=cfg.net.width, layers=cfg.net.layers, head_depth=cfg.net.depth
    )
    backend = GNNBackend(
        netcfg, setup_depth=cfg.net.setup_depth,
        setup_temperature=cfg.net.setup_temperature, setup_beam=cfg.net.setup_beam,
        chance_nodes=s.chance_nodes, dev_chance=s.dev_chance, ordered=s.ordered,
    )  # fmt: skip
    resume = None
    if cfg.resume_from:
        prior = Path(cfg.resume_from) / "runstate.eqx"
        resume = prior if prior.exists() else None
    wb = wandb.init(
        project=cfg.wandb.project, mode=cfg.wandb.mode, config=cfg.dump(),
        reinit=True, dir=str(run.dir),
    )  # fmt: skip
    best = -1.0

    def on_iter(i: int, metrics: dict[str, float], model: BoardGNN) -> None:
        nonlocal best
        run.log(iteration=i, **metrics)  # scalars -> metrics.jsonl
        log: dict[str, object] = {"iteration": i, **metrics}
        # param distributions as wandb histograms (whole net + each head, where a
        # collapse shows first) -- heavier than the scalar log, so only every
        # `hist_every` iterations (0 disables).
        if cfg.wandb.hist_every and i % cfg.wandb.hist_every == 0:
            for name, tree in (
                ("params/all", model),
                ("params/policy", model.policy),
                ("params/value", model.value),
            ):
                arrs = [
                    np.asarray(x).ravel()
                    for x in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))
                ]
                if arrs:
                    log[name] = wandb.Histogram(np.concatenate(arrs))  # type: ignore[arg-type]
        wb.log(log, step=i)
        winrate = metrics.get("arena_winrate")
        if winrate is not None and winrate > best:
            best = winrate
            eqx.tree_serialise_leaves(run.dir / "best.eqx", model)

    net_opponents = build_net_opponents(cfg)
    try:
        model = learn(
            backend,
            cfg.to_learn_config(),
            teacher_value=heuristic_value if cfg.teacher.enabled else None,
            net_opponents=net_opponents,
            checkpoint_dir=run.dir,
            resume_from=resume,
            on_iter=on_iter,
            progress=True,
        )
    finally:
        wb.finish()

    metrics = run_final_gauntlet(backend, model, cfg, net_opponents)
    verdict = gauntlet_verdict(metrics, cfg.gate_elo)
    run.finish(
        verdict, best_arena_winrate=best,
        gate_elo=cfg.gate_elo, gate_winrate=cfg.gate_winrate, **metrics,
    )  # fmt: skip


def run_bench(run: Run, cfg: AlphaZeroConfig) -> None:
    """Pinned self-play throughput at a frozen net; verdict is always
    ``recorded`` -- the comparison is between two runs' result.json."""
    import jax

    # same-dir sibling module; sys.path already has this dir (script invocation
    # and the test conftest both put it there), matching 0002/0003's convention.
    from anchors import load_anchor
    from settlrl_learn.training import GNNBackend, bench_selfplay

    net, netcfg = load_anchor(cfg.bench.anchor)
    if (netcfg.width, netcfg.layers, netcfg.head_depth) != (
        cfg.net.width, cfg.net.layers, cfg.net.depth,
    ):  # fmt: skip
        raise ValueError(
            f"anchor {cfg.bench.anchor!r} (width={netcfg.width}, "
            f"layers={netcfg.layers}, head_depth={netcfg.head_depth}) does not "
            f"match preset cfg.net (width={cfg.net.width}, layers={cfg.net.layers}, "
            f"depth={cfg.net.depth}) -- the run manifest would misdescribe the "
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


def run_experiment(run: Run, cfg: AlphaZeroConfig) -> None:
    if cfg.mode == "bench":
        run_bench(run, cfg)
        return
    if cfg.net.kind == "gnn":
        run_gnn_experiment(run, cfg)
        return
    s = cfg.search
    backend = MLPBackend(
        (cfg.net.width,) * cfg.net.depth, value_weight=cfg.net.value_weight,
        chance_nodes=s.chance_nodes, dev_chance=s.dev_chance, ordered=s.ordered,
    )  # fmt: skip

    # Resume: restore the prior run's RunState and continue its wandb run so the
    # dashboard is one unbroken curve.
    resume_dir = Path(cfg.resume_from) if cfg.resume_from else None
    resume_from = None
    wandb_id = None
    if resume_dir is not None:
        runstate = resume_dir / "runstate.eqx"
        resume_from = runstate if runstate.exists() else None
        id_file = resume_dir / "wandb_id.txt"
        wandb_id = id_file.read_text().strip() if id_file.exists() else None

    wb = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=cfg.dump(),
        reinit=True,
        dir=str(run.dir),
        id=wandb_id,
        resume="allow" if wandb_id else None,
    )
    (run.dir / "wandb_id.txt").write_text(str(wb.id))  # so a later run can resume it

    best = -1.0  # best arena win rate seen -> best.npz (the shippable net)

    def on_iter(i: int, metrics: dict[str, float], net: object) -> None:
        nonlocal best
        run.log(iteration=i, **metrics)
        wb.log({"iteration": i, **metrics}, step=i)  # explicit step: resume-safe
        winrate = metrics.get("arena_winrate")
        if winrate is not None and winrate > best:
            best = winrate
            save_az_params(run.dir / "best.npz", net)  # type: ignore[arg-type]

    net_opponents = build_net_opponents(cfg)
    try:
        # learn writes the full-state checkpoint (run.dir/runstate.eqx) for
        # bit-exact resume; resume_from continues a prior run's checkpoint.
        final = learn(
            backend,
            cfg.to_learn_config(),
            net_opponents=net_opponents,
            checkpoint_dir=run.dir,
            resume_from=resume_from,
            on_iter=on_iter,
            progress=True,
        )
    finally:
        wb.finish()

    save_az_params(run.dir / "params.npz", final)  # final net
    metrics = run_final_gauntlet(backend, final, cfg, net_opponents)
    verdict = gauntlet_verdict(metrics, cfg.gate_elo)
    run.finish(
        verdict, best_arena_winrate=best,
        gate_elo=cfg.gate_elo, gate_winrate=cfg.gate_winrate, **metrics,
    )  # fmt: skip


def compose_config(overrides: list[str]) -> AlphaZeroConfig:
    """Hydra-compose ``conf/`` and validate into :class:`AlphaZeroConfig` -- the
    programmatic seam (smoke tests) that ``@hydra.main`` can't serve."""
    conf_dir = str(Path(__file__).parent / "conf")
    with hydra.initialize_config_dir(version_base=None, config_dir=conf_dir):
        dcfg = hydra.compose(config_name="config", overrides=overrides)
    return AlphaZeroConfig.model_validate(OmegaConf.to_container(dcfg, resolve=True))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = AlphaZeroConfig.model_validate(OmegaConf.to_container(dcfg, resolve=True))
    run_experiment(start_run(Path(__file__).parent, cfg.dump()), cfg)


if __name__ == "__main__":
    main()
