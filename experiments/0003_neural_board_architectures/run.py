"""Neural board architectures: which representation predicts board properties?

Hypothesis: on supervised board-prediction tasks, a structure-aware net over the
raw board graph (GNN) is competitive with — and ideally beats — an MLP over the
hand-tuned feature vector, and clearly beats a structure-blind net over the same
raw inputs (flat MLP / DeepSet). If so, the graph representation is the seam to
push for a learned value (settlrl-learn Stage 1).

Tasks (labels from greedy self-play, cached): ``heuristic`` regresses the
hand-tuned value (a *local* target); ``win`` predicts seat 0's game outcome (a
*global* target); ``road`` regresses seat 0's longest-road trail length (a
*structural* target the engineered vector cannot express); ``turns`` regresses
snapshots-to-game-end; ``multi`` trains one shared trunk with a head per target
(win + heur + road + turns). Each ``arch`` is trained and held-out-scored;
``arch=all`` sweeps the four baselines and ranks, ``arch=a,b,c`` sweeps a named
list (the GraphNet lever ablation).

    uv run python experiments/0003_neural_board_architectures/run.py [variant] [k=v ...]
"""

import sys
from pathlib import Path

from data import generate, split
from settlrl_learn.experiment import Config, Run, start_run
from settlrl_learn.nn.architectures import make_model
from train import TASK_FIELDS, select_metric, train

ARCHS = ("mlp_engineered", "mlp_flat", "deepset", "gnn")
# The GraphNet ablation: the engineered baseline + `gnn` + one preset per
# lever (settlrl_learn.nn.graphnet.PRESETS), so each row isolates one design choice.
ABLATION = (
    "mlp_engineered", "gnn", "gn_base", "gn_multi", "gn_norm",
    "gn_graphnorm", "gn_global", "gn_gat", "gn_jk", "gn_full",
)  # fmt: skip


class NeuralBoardArchitecturesConfig(Config):
    seed: int = 0
    task: str = "heuristic"  # heuristic | win
    arch: str = "all"  # all | one of ARCHS
    # data (greedy self-play, cached by these knobs under runs/_cache)
    agent: str = "greedy"
    players: int = 2
    n_samples: int = 20_000
    snapshot_every: int = 8
    collect_batch: int = 64
    val_frac: float = 0.2
    # model
    width: int = 64
    depth: int = 2
    layers: int = 3  # GNN message-passing layers
    # optimisation
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 2
    # logging
    wandb_project: str = "settlrl-0003-architectures"
    wandb_mode: str = "online"  # online | offline | disabled


VARIANTS: dict[str, dict[str, object]] = {
    "heuristic": {"task": "heuristic", "arch": "all"},
    "win": {"task": "win", "arch": "all"},
    "gnn_heuristic": {"task": "heuristic", "arch": "gnn"},
    "gnn_win": {"task": "win", "arch": "gnn"},
    # GraphNet lever ablation (one row per design choice) on each target.
    "ablate_heuristic": {"task": "heuristic", "arch": ",".join(ABLATION)},
    "ablate_win": {"task": "win", "arch": ",".join(ABLATION)},
    "ablate_road": {"task": "road", "arch": ",".join(ABLATION)},
    # Multi-task: one shared trunk, a head per target (win + heur + road + turns).
    "multi": {"task": "multi", "arch": "all"},
    "ablate_multi": {"task": "multi", "arch": ",".join(ABLATION)},
    "smoke": {
        "task": "heuristic",
        "arch": "all",
        "n_samples": 200,
        "snapshot_every": 16,
        "collect_batch": 8,
        "val_frac": 0.3,
        "width": 16,
        "depth": 1,
        "layers": 1,
        "epochs": 2,
        "batch_size": 64,
        "eval_every": 1,
        "wandb_mode": "disabled",
    },
}


def run_experiment(run: Run, cfg: NeuralBoardArchitecturesConfig) -> None:
    import jax

    data_cfg = {
        "agent": cfg.agent, "players": cfg.players, "n_samples": cfg.n_samples,
        "snapshot_every": cfg.snapshot_every, "batch_size": cfg.collect_batch,
        "seed": cfg.seed,
    }  # fmt: skip
    ds = generate(data_cfg)
    train_ds, val_ds = split(ds, cfg.val_frac, seed=cfg.seed)
    run.log(n_samples=int(ds.win.shape[0]), n_train=int(train_ds.win.shape[0]),
            win_rate=float(ds.win.mean()))  # fmt: skip

    archs = ARCHS if cfg.arch == "all" else tuple(cfg.arch.split(","))
    select = select_metric(cfg.task)  # e.g. "win_auc" / "road_r2" (primary head)
    out_dim = len(TASK_FIELDS[cfg.task])  # multi-task trains one head per target
    results: dict[str, dict[str, float]] = {}
    for arch in archs:
        model = make_model(
            arch, jax.random.key(cfg.seed),
            out_dim=out_dim, width=cfg.width, depth=cfg.depth, layers=cfg.layers,
        )  # fmt: skip
        sub = Run(run.dir / arch)
        sub.dir.mkdir(exist_ok=True)
        metrics = train(sub, {**cfg.dump(), "arch": arch}, model, train_ds, val_ds)
        results[arch] = metrics
        run.log(arch=arch, **metrics)
    run.save_json("results.json", results)

    # Verdict: a raw-board representation is competitive with the hand-tuned
    # baseline (within 0.02 of it on the selection metric), or — no baseline in
    # the run — the best model clears a sanity floor.
    floor = 0.55 if select.endswith("_auc") else 0.5
    key = f"best_{select}"
    learned = [a for a in archs if a != "mlp_engineered"]
    if "mlp_engineered" in results and learned:
        baseline = results["mlp_engineered"].get(key, float("-inf"))
        best_learned = max(results[a].get(key, float("-inf")) for a in learned)
        verdict = "pass" if best_learned >= baseline - 0.02 else "fail"
        run.finish(verdict, select=select, baseline=baseline, best_learned=best_learned,
                   **{a: results[a].get(key) for a in archs})  # fmt: skip
    else:
        score = max(results[a].get(key, float("-inf")) for a in archs)
        run.finish("pass" if score >= floor else "fail", select=select, score=score)


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "heuristic"
    if variant not in VARIANTS:
        raise SystemExit(f"usage: run.py [{'|'.join(VARIANTS)}] [key=value ...]")
    cfg = NeuralBoardArchitecturesConfig.resolve(
        VARIANTS[variant], overrides=sys.argv[2:]
    )
    run_experiment(start_run(Path(__file__).parent, cfg.dump()), cfg)


if __name__ == "__main__":
    main()
