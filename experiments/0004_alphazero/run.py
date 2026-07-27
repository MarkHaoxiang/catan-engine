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
from typing import Literal

import hydra
import jax
import wandb
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from settlrl_learn import save_az_params
from settlrl_learn.experiment import Config, Run, start_run
from settlrl_learn.training import (
    ArenaConfig,
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
    arena,
    learn,
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


class WandbConfig(_Sub):
    mode: Literal["online", "offline", "disabled"] = "online"
    project: str = "settlrl-0004-alphazero"


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
    gate_winrate: float = 0.55  # pass iff the final net clears this vs lookahead
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
    arena: ArenaConfig = Field(default_factory=ArenaConfig)

    def to_learn_config(self) -> LearnConfig:
        """Pack the loop groups into the net-agnostic ``LearnConfig``."""
        return LearnConfig(
            n_iterations=self.n_iterations, seed=self.seed,
            checkpoint_every=self.checkpoint_every,
            search=self.search, selfplay=self.selfplay, optim=self.optim,
            replay=self.replay, teacher=self.teacher, value_blend=self.value_blend,
            eval=self.eval, arena=self.arena,
        )  # fmt: skip


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
        # collapse shows first).
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

    try:
        model = learn(
            backend,
            cfg.to_learn_config(),
            teacher_value=heuristic_value if cfg.teacher.enabled else None,
            checkpoint_dir=run.dir,
            resume_from=resume,
            on_iter=on_iter,
            progress=True,
        )
    finally:
        wb.finish()

    winrate = arena(
        backend, model, n_games=cfg.arena.games, num_simulations=cfg.arena.sims,
        batch_size=cfg.arena.batch, max_num_considered_actions=cfg.arena.considered,
        seed=cfg.seed + 99,
    ).winrate  # fmt: skip
    verdict = "pass" if winrate >= cfg.gate_winrate else "fail"
    run.finish(
        verdict, arena_winrate=winrate, best_arena_winrate=best, gate=cfg.gate_winrate
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

    try:
        # learn writes the full-state checkpoint (run.dir/runstate.eqx) for
        # bit-exact resume; resume_from continues a prior run's checkpoint.
        final = learn(
            backend,
            cfg.to_learn_config(),
            checkpoint_dir=run.dir,
            resume_from=resume_from,
            on_iter=on_iter,
            progress=True,
        )
    finally:
        wb.finish()

    save_az_params(run.dir / "params.npz", final)  # final net
    winrate = arena(
        backend, final, n_games=cfg.arena.games, num_simulations=cfg.arena.sims,
        batch_size=cfg.arena.batch, max_num_considered_actions=cfg.arena.considered,
        seed=cfg.seed + 99,
    ).winrate  # fmt: skip
    verdict = "pass" if winrate >= cfg.gate_winrate else "fail"
    run.finish(
        verdict, arena_winrate=winrate, best_arena_winrate=best, gate=cfg.gate_winrate
    )


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
