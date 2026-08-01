"""The net-specific side of the loop: the :class:`Backend` protocol and its two
implementations (flat-MLP, board-GNN)."""

from settlrl_learn.training.backends.base import (
    Backend,
    RunState,
    load_run_state,
    save_run_state,
)
from settlrl_learn.training.backends.gnn import (
    GNNBackend,
    gnn_loss,
    make_net_agent,
    setup_policy,
)
from settlrl_learn.training.backends.mlp import MLPBackend, mlp_loss

__all__ = [
    "Backend",
    "GNNBackend",
    "MLPBackend",
    "RunState",
    "gnn_loss",
    "load_run_state",
    "make_net_agent",
    "mlp_loss",
    "save_run_state",
    "setup_policy",
]
