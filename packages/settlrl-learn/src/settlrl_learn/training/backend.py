"""The backend seam of the training loop and the run-state it checkpoints.

A :class:`Backend` bundles everything net-specific: how to build the net, adapt
it to the search seams and a play agent, turn board positions into replay items,
and run one optimiser step / eval. :mod:`settlrl_learn.training.loop` is otherwise
net-agnostic, so the flat-MLP (:class:`~settlrl_learn.training.mlp_backend.MLPBackend`)
and board-GNN (:class:`~settlrl_learn.training.gnn_backend.GNNBackend`) paths share
one loop.

:class:`RunState` is the whole mutable run state, eqx-serialised for **bit-exact
resume** -- eqx's leaf serialiser fits both an equinox module and a plain-JAX
pytree, and the per-iteration RNG is a pure function of the seed and iteration
index, so a resumed run continues bit-identically.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, Protocol, cast

import equinox as eqx
import numpy as np
import optax
from jaxtyping import Array, Float, Int
from settlrl_agents.value import ValueFunction
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, IntScalar
from settlrl_search.policy import BeliefPolicy, PolicyPrior

Metrics = dict[str, Float[Array, ""]]
StepFn = Callable[[Any, optax.OptState, Any], tuple[Any, optax.OptState, Metrics]]
"""One minibatch update: ``(net, opt_state, item) -> (net, opt_state, metrics)``."""


class Backend(Protocol):
    """The net-specific surface of the training loop. ``net`` and ``item`` are
    opaque to the loop (a plain-JAX pytree or an equinox module; a replay item
    pytree), so each backend owns their concrete types."""

    def init(self, key: Array) -> Any:
        """A fresh net from ``key``."""
        ...

    def seams(self, net: Any) -> tuple[ValueFunction, PolicyPrior]:
        """Adapt the net onto the search seams ``(value, prior)`` (``value_scale=2``)."""
        ...

    def setup_policy(self) -> BeliefPolicy | None:
        """The fixed policy that plays the setup phase (its positions are not
        recorded), or ``None`` to let the net play setup too."""
        ...

    def play_agent(
        self, net: Any, *, num_simulations: int, max_num_considered_actions: int
    ) -> BeliefPolicy:
        """The net as a play agent for the arena (the search, plus any setup
        delegation)."""
        ...

    def observe(
        self, layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> dict[str, Array]:
        """The net's observation of one position, as named arrays (the search's
        improved policy, the legality mask, and the outcome value are added by
        self-play)."""
        ...

    def to_item(self, samples: dict[str, np.ndarray]) -> Any:
        """Pack a self-play ``samples`` dict (observation keys + ``policy`` +
        ``mask`` + ``value``) into a batched replay item."""
        ...

    def empty_item(self) -> Any:
        """A single zero item, shaping the replay buffer and the resume template."""
        ...

    def init_opt(
        self, optimizer: optax.GradientTransformation, net: Any
    ) -> optax.OptState:
        """The optimiser state for ``net`` (an equinox model filters to its
        inexact arrays first)."""
        ...

    def make_step(self, optimizer: optax.GradientTransformation) -> StepFn:
        """A jitted minibatch update (policy CE + value logistic loss)."""
        ...

    def eval_metrics(self, net: Any, item: Any) -> Metrics:
        """Held-out diagnostics over an eval item batch (never trained on)."""
        ...


class RunState(NamedTuple):
    """One run's complete mutable state, eqx-serialised for bit-exact resume.

    ``selfplay_carry`` is last so it forms the file's trailing section, which is
    what lets :func:`load_run_state` read a checkpoint written before the field
    existed.
    """

    net: Any  # AZParams | BoardGNN
    opt_state: optax.OptState
    buffer_state: Any  # flashbax buffer state pytree
    iteration: Int[Array, ""]  # iterations completed
    best: Float[Array, ""]  # best arena win rate so far
    selfplay_carry: Any  # settlrl_learn.training.selfplay.PaddedCarry


_PRE_CARRY_FIELDS = RunState._fields.index("selfplay_carry")


def save_run_state(path: str | Path, state: RunState) -> None:
    """Serialises ``state`` to ``path`` atomically: writes to a sibling
    ``<path>.tmp`` first, then ``os.replace``s it into place (same filesystem).
    The validated operating procedure for a long run is kill-and-resume, and a
    multi-GiB in-place write killed mid-flight would otherwise truncate the
    only checkpoint; a stale ``.tmp`` from an earlier kill is silently
    overwritten by the next successful write."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    eqx.tree_serialise_leaves(tmp, state)
    os.replace(tmp, path)


def load_run_state(path: str | Path, template: RunState) -> RunState:
    """``path``'s run state, shaped by ``template``.

    A pre-``selfplay_carry`` checkpoint (the file ends where that section would
    start) loads with the template's carry, and a ``None`` template carry skips
    the section entirely (a run that never reads the pool). A carry that does not
    fit the template raises ``ValueError``; every other mismatch raises out of
    eqx's leaf checks.
    """
    with Path(path).open("rb") as f:
        net, opt_state, buffer_state, iteration, best = cast(
            "tuple[Any, ...]",
            eqx.tree_deserialise_leaves(f, template[:_PRE_CARRY_FIELDS]),
        )
        carry = template.selfplay_carry
        if carry is not None and f.read(1) != b"":
            f.seek(-1, 1)
            try:
                carry = eqx.tree_deserialise_leaves(f, carry)
            except RuntimeError as e:
                raise ValueError(
                    "the checkpointed self-play carry does not fit this run: it "
                    "was written under a different selfplay.persistent, "
                    "selfplay.batch, selfplay.max_game_len or value_blend. "
                    "Resume with the settings that wrote it, or start a fresh "
                    f"run. ({e})"
                ) from e
    return RunState(net, opt_state, buffer_state, iteration, best, carry)
