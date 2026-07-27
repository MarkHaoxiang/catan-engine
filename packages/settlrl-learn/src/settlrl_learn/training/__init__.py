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
from settlrl_learn.training.loop import learn
from settlrl_learn.training.mlp_backend import MLPBackend, mlp_loss
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
    "SelfPlayConfig",
    "TeacherConfig",
    "ValueBlendConfig",
    "anchored_elo",
    "arena",
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
    "save_run_state",
    "setup_policy",
    "train_epochs",
]
