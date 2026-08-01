"""Learned value and policy functions for settlrl-agents.

Networks plug into the agents through their existing seams — a value model is
a :class:`~settlrl_search.value.ValueFunction` and a policy model a
:class:`~settlrl_search.policy.PolicyPrior` — so every search agent
consumes them unchanged.
"""

from settlrl_learn.features import FEATURE_DIM, features
from settlrl_learn.nn.mlp import (
    AZParams,
    MLPParams,
    az_forward,
    init_az_params,
    init_mlp,
    init_prior_params,
    init_value_params,
    load_az_params,
    load_params,
    make_az,
    make_net_prior,
    make_net_value,
    mlp,
    save_az_params,
    save_params,
)
__all__ = [
    "FEATURE_DIM",
    "AZParams",
    "MLPParams",
    "az_forward",
    "features",
    "init_az_params",
    "init_mlp",
    "init_prior_params",
    "init_value_params",
    "load_az_params",
    "load_params",
    "make_az",
    "make_net_prior",
    "make_net_value",
    "mlp",
    "save_az_params",
    "save_params",
]
