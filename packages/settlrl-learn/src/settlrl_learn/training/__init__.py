"""The training loop: a net-agnostic self-play -> replay -> train -> arena loop
(:func:`learn`, :func:`arena`) over a :class:`Backend`. Two backends share it --
the flat engineered :class:`MLPBackend` and the board-graph :class:`GNNBackend`.

Training-side (equinox/optax/flashbax): not imported by the package root, so the
shipped plain-JAX play path stays dependency-light.
"""

from settlrl_learn.training.arena import ArenaResult, arena
from settlrl_learn.training.backend import (
    Backend,
    RunState,
    load_run_state,
    save_run_state,
)
from settlrl_learn.training.bench import bench_selfplay
from settlrl_learn.training.config import (
    ArenaConfig,
    EvalConfig,
    LearnConfig,
    OptimConfig,
    ReplayConfig,
    SearchSettings,
    SelfPlayConfig,
    TeacherConfig,
    ValueBlendConfig,
)
from settlrl_learn.training.elo import anchored_elo, expected_score
from settlrl_learn.training.gnn_backend import (
    GNNBackend,
    gnn_loss,
    make_net_agent,
    setup_policy,
)
from settlrl_learn.training.loop import (
    SelfPlayCallables,
    learn,
    run_selfplay,
    selfplay_callables,
)
from settlrl_learn.training.mlp_backend import MLPBackend, mlp_loss
from settlrl_learn.training.selfplay import SelfPlayStats, self_play
from settlrl_learn.training.steps import (
    evaluate,
    make_optimizer,
    prepare_targets,
    run_arena,
    train_epochs,
)

__all__ = [
    "ArenaConfig",
    "ArenaResult",
    "Backend",
    "EvalConfig",
    "GNNBackend",
    "LearnConfig",
    "MLPBackend",
    "OptimConfig",
    "ReplayConfig",
    "RunState",
    "SearchSettings",
    "SelfPlayCallables",
    "SelfPlayConfig",
    "SelfPlayStats",
    "TeacherConfig",
    "ValueBlendConfig",
    "anchored_elo",
    "arena",
    "bench_selfplay",
    "evaluate",
    "expected_score",
    "gnn_loss",
    "learn",
    "load_run_state",
    "make_net_agent",
    "make_optimizer",
    "mlp_loss",
    "prepare_targets",
    "run_arena",
    "run_selfplay",
    "save_run_state",
    "self_play",
    "selfplay_callables",
    "setup_policy",
    "train_epochs",
]
