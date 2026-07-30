"""Settlrl-playing agents over settlrl-engine's flat action space: pure-JAX
policies plus stateful plain-Python planners."""

from settlrl_engine.board.resources import N_PLAYERS
from settlrl_search import make_search
from settlrl_search.policy import (
    AgentSpec,
    BeliefPolicy,
    BeliefSpec,
    FlatAction,
    FlatMask,
    GameAgent,
    ObservationSpec,
    OpponentSpec,
    Policy,
    PolicyPrior,
    StatefulPolicy,
    StatefulSpec,
    ValuePrior,
)
from settlrl_search.sample import sample_world

from settlrl_agents.baselines import random_policy
from settlrl_agents.evaluate import EvalResult, evaluate
from settlrl_agents.greedy import greedy_policy
from settlrl_agents.planner import make_planner
from settlrl_agents.value import (
    TUNED_WEIGHTS,
    Value,
    ValueFunction,
    heuristic_value,
    make_heuristic,
    make_linear,
    tuned_value,
)

search_policy = make_search(heuristic_value)
"""The shipped search: re-determinizing SO-ISMCTS over :func:`heuristic_value`."""

lookahead_policy = make_search(heuristic_value, num_simulations=0, propose_rate=0.5)
"""One-step lookahead — the ``num_simulations=0`` special case of the search."""

_ANY_COUNT = frozenset(range(2, N_PLAYERS + 1))

POLICIES: dict[str, OpponentSpec] = {
    "random": ObservationSpec(lambda: random_policy, _ANY_COUNT),
    "greedy": ObservationSpec(lambda: greedy_policy, _ANY_COUNT),
    "planner": StatefulSpec(make_planner, _ANY_COUNT),
    # Lookahead is the search at zero simulations (root one-step sweep); the
    # only configuration that offers trades.
    "lookahead": BeliefSpec(
        make_search,
        _ANY_COUNT,
        defaults={"value": heuristic_value, "num_simulations": 0, "propose_rate": 0.5},
    ),
    # The re-determinizing SO-ISMCTS search (true per-simulation legality, mctx's
    # Gumbel-MuZero selection on a custom tree — see search/ismcts.py). The "mcts"
    # name is kept for bench/render/service string continuity.
    "mcts": BeliefSpec(
        make_search,
        _ANY_COUNT,
        defaults={"value": heuristic_value},
        # One tree and a small Gumbel budget: a cheap member of the same family
        # for the protocol tests.
        for_testing={
            "num_trees": 1,
            "num_simulations": 8,
            "max_num_considered_actions": 8,
        },
    ),
}
"""Every shipped agent by name (the family at its ``defaults``)."""

__all__ = [
    "POLICIES",
    "TUNED_WEIGHTS",
    "AgentSpec",
    "BeliefPolicy",
    "BeliefSpec",
    "EvalResult",
    "FlatAction",
    "FlatMask",
    "GameAgent",
    "ObservationSpec",
    "OpponentSpec",
    "Policy",
    "PolicyPrior",
    "StatefulPolicy",
    "StatefulSpec",
    "Value",
    "ValueFunction",
    "ValuePrior",
    "evaluate",
    "greedy_policy",
    "heuristic_value",
    "lookahead_policy",
    "make_heuristic",
    "make_linear",
    "make_planner",
    "make_search",
    "random_policy",
    "sample_world",
    "search_policy",
    "tuned_value",
]
