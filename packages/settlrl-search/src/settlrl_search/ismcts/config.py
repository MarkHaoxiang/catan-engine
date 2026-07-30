"""Search configuration: the user-facing :class:`SearchConfig`, the jit-static
``_Cfg`` the hot path closes over, the Sequential-Halving schedule, and the tree
node/count dtype helpers."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from settlrl_search.policy import PolicyPrior
from settlrl_search.value import ValueFunction

from ._types import _Table


class SearchConfig(BaseModel):
    """The user-facing scalar/bool search configuration, validated and frozen.

    ``chance_nodes`` supersedes ``expected_rolls`` (rolls resolve in-tree), so
    ``expected_rolls`` is forced False whenever ``chance_nodes`` is set.

    ``fused_leaf`` lets a :class:`~settlrl_search.policy.ValuePrior` prior serve
    the leaf value from the same forward as the leaf's prior — so it is the
    prior's value head, not the ``value`` argument, that scores leaves. Turn it
    off to score leaves with ``value`` even under such a prior (an *unpaired*
    ``value``/``prior``, e.g. a heuristic value under a net's policy head).
    """

    model_config = ConfigDict(frozen=True)

    num_simulations: int
    max_depth: int
    max_considered: int
    value_scale: float
    expected_rolls: bool
    chance_nodes: bool
    dev_chance: bool
    ordered: bool
    fused_leaf: bool

    @field_validator("num_simulations")
    @classmethod
    def _check_num_simulations(cls, v: int) -> int:
        if v < 0:
            raise ValueError("num_simulations must be >= 0")
        return v

    @field_validator("max_depth")
    @classmethod
    def _check_max_depth(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_depth must be >= 1")
        return v

    @field_validator("max_considered")
    @classmethod
    def _check_max_considered(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_considered must be >= 1")
        return v

    @field_validator("value_scale")
    @classmethod
    def _check_value_scale(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("value_scale must be > 0")
        return v

    @model_validator(mode="after")
    def _exclusive_rolls(self) -> "SearchConfig":
        # chance_nodes resolves rolls in-tree, so the leaf-only roll EV is off.
        object.__setattr__(
            self, "expected_rolls", self.expected_rolls and not self.chance_nodes
        )
        return self


class _Cfg(NamedTuple):
    """The static search configuration :func:`make_tree` captures and threads
    through the (module-level) phase functions. Plain pytree-free static — the hot
    path closes over it inside the jit (no pydantic)."""

    value: ValueFunction
    prior: PolicyPrior | None  # interior-node prior; tier table when None
    num_simulations: int
    max_depth: int
    max_considered: int
    value_scale: float
    expected_rolls: bool  # roll leaf = exact 11-roll expectation; else 1 sampled roll
    chance_nodes: bool  # explicit dice (+dev) chance nodes in the tree
    dev_chance: bool  # also make BUY_DEVELOPMENT_CARD a chance node (chance_nodes)
    ordered: bool  # apply the action-ordering lock-out to the in-tree legal set
    fused_leaf: bool  # a ValuePrior's value head scores the leaf (one forward)
    n_nodes: int  # num_simulations + 1
    table: _Table  # the Sequential-Halving considered-visits schedule


# --- Sequential Halving: the considered-visits schedule (static, baked) ---


def _considered_visits_seq(m: int, n: int) -> tuple[int, ...]:
    """Sequential Halving's visit schedule: length-``n`` list whose entry ``s`` is
    the visit count a candidate must hold to be selected at simulation ``s``."""
    if m <= 1:
        return tuple(range(n))
    log2max = math.ceil(math.log2(m))
    seq: list[int] = []
    visits = [0] * m
    num_considered = m
    while len(seq) < n:
        extra = max(1, int(n / (log2max * num_considered)))
        for _ in range(extra):
            seq.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return tuple(seq[:n])


def _considered_table(m: int, n: int) -> np.ndarray:
    """Row ``k`` is the schedule for ``k`` considered actions (shape
    ``[m + 1, n]``); indexed by ``min(m, num_legal)`` at search time."""
    return np.asarray([_considered_visits_seq(k, n) for k in range(m + 1)], np.int32)


# --- tree dtype helpers: small ints stay small (the values are exact) ---


def _node_dtype(n_nodes: int) -> np.dtype:
    """Node-id / child-index dtype: int16 unless the pool overflows it."""
    return np.dtype(np.int16) if n_nodes < 32768 else np.dtype(np.int32)


def _count_dtype(num_simulations: int) -> np.dtype:
    """Visit-count dtype: int16 unless the simulation budget overflows it."""
    return np.dtype(np.int16) if num_simulations < 32768 else np.dtype(np.int32)
