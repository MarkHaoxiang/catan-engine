"""The seat interfaces: decision functions over one game's view.

Three protocols, split by what a seat consumes — none sees anything the
player wouldn't: :class:`Policy` reads the acting player's partial
observation; :class:`BeliefPolicy` reads the engine's honest
:class:`~settlrl_engine.belief.BeliefView` (model-based agents rebuild a
concrete world with ``sample_world`` and search there); :class:`GameAgent`
reads the same partial observation but is a *stateful per-game object*,
driven step-by-step in Python (plans and memories persist across its own
moves; not traceable, so it never enters a jit/vmap). All are valid at any
player count.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
from jaxtyping import Array, Bool, Float, Int
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, KeyScalar, Player
from settlrl_engine.env import N_FLAT, Observation

from settlrl_search.value import Value

FlatMask = Bool[Array, f"flat={N_FLAT}"]
"""Legality of every concrete flat action for the acting player (one game)."""

FlatAction = Int[Array, ""]
"""A chosen flat action index in ``[0, N_FLAT)``."""

HostFlatMask = Bool[np.ndarray, f"flat={N_FLAT}"]
"""``FlatMask`` fetched to the host (the stateful agents' form)."""

HostObservation = Mapping[str, np.ndarray]
"""An :class:`~settlrl_engine.env.Observation` fetched to the host: the same
keys with single-game numpy leaves (one ``jax.device_get``, no per-field
device syncs in agent logic)."""


@runtime_checkable
class Policy(Protocol):
    """A single-game decision function, pure and ``jit`` / ``vmap`` compatible.

    ``key`` is a JAX PRNG key, ``obs`` the acting player's partial observation
    (one game, no batch axis), ``mask`` the flat legality of that player's
    moves. Returns the chosen flat action index; decode it with
    :func:`settlrl_engine.env.flat_to_action`. When ``mask`` has no legal move
    the returned index is arbitrary (the engine rejects it as ``INVALID``).
    """

    def __call__(
        self, key: KeyScalar, obs: Observation, mask: FlatMask
    ) -> FlatAction: ...


@runtime_checkable
class PolicyPrior(Protocol):
    """Flat-action prior logits for one game's position, pure and ``jit`` /
    ``vmap`` compatible.

    Returns unmasked logits over the flat actions from ``player``'s point of
    view; consumers apply legality masking. The seam for learned policy
    heads: the search agents accept one in place of their built-in priors.
    """

    def __call__(
        self, layout: BoardLayout, state: BoardState, player: Player
    ) -> Float[Array, f"flat={N_FLAT}"]: ...


@runtime_checkable
class ValuePrior(PolicyPrior, Protocol):
    """A :class:`PolicyPrior` that can also return the state's value.

    ``with_value`` must return exactly ``(paired ValueFunction(...),
    self(...))`` — it is the same pair, from one evaluation. The search calls
    it wherever it needs both for one state (the leaf), so a shared-trunk net
    runs its forward once instead of twice; a prior without it is called
    alongside the value function as before.
    """

    def with_value(
        self, layout: BoardLayout, state: BoardState, player: Player
    ) -> tuple[Value, Float[Array, f"flat={N_FLAT}"]]: ...


@runtime_checkable
class BeliefPolicy(Protocol):
    """A single-game decision function over the player's honest world model.

    ``layout`` is one game's board layout, ``view`` everything ``player``
    knows about it (a :class:`~settlrl_engine.belief.BeliefView`), ``mask`` the
    flat legality of the player's moves. Same return and no-legal-move
    conventions as :class:`Policy`.
    """

    def __call__(
        self,
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: Player,
        mask: FlatMask,
    ) -> FlatAction: ...


@runtime_checkable
class GameAgent(Protocol):
    """One seat of one game, driven step-by-step in Python.

    ``obs`` is the agent's partial observation (one game, host numpy),
    ``mask`` the flat legality of its moves. Returns the chosen flat action
    index, legal whenever any legal move exists (same no-legal-move
    convention as :class:`Policy`). Calls arrive in game order, so the agent
    may keep state across them; it is never shared between games.
    """

    def act(self, obs: HostObservation, mask: HostFlatMask) -> int: ...


@runtime_checkable
class StatefulPolicy(Protocol):
    """Builds a fresh :class:`GameAgent` for one game.

    ``seed`` makes the agent's tie-breaking deterministic; drivers derive a
    distinct seed per (game, seat).
    """

    def __call__(self, seed: int) -> GameAgent: ...


P = TypeVar("P", Policy, BeliefPolicy, StatefulPolicy)
# `for_tests` returns the spec's own class; a bound TypeVar instead of Self
# because the tests' beartype hook can't check PEP 673 on hook-decorated
# methods (and only resolves unsubscripted forward-ref bounds).
S = TypeVar("S", bound="AgentSpec")


@dataclasses.dataclass(frozen=True)
class AgentSpec(Generic[P]):
    """A registry entry: a policy *family* plus the parameters to build it at.

    ``make(**defaults)`` builds the shipped agent (cached as :attr:`policy`;
    parameterless families pass an empty ``defaults``). ``for_testing`` holds
    overrides applied on top of ``defaults`` for a cheaper member of the same
    family (see :attr:`for_tests`). The subclass is the protocol tag —
    :class:`ObservationSpec` families build a :class:`Policy`,
    :class:`BeliefSpec` a :class:`BeliefPolicy`, :class:`StatefulSpec` a
    :class:`StatefulPolicy` — so consumers dispatch with ``isinstance``.
    ``n_players`` holds the player counts the agent may be seated at.
    """

    make: Callable[..., P]
    n_players: frozenset[int]
    defaults: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    for_testing: Mapping[str, Any] | None = None

    @functools.cached_property
    def policy(self) -> P:
        """The shipped agent: the family built at ``defaults``."""
        return self.make(**self.defaults)

    @property
    def for_tests(self: S) -> S:
        """The same family at its cheap test parameters (itself when none).

        The protocol properties (legality, determinism, completing games)
        are parameter-independent, so tests exercise this member instead of
        the full-size shipped one.
        """
        if self.for_testing is None:
            return self
        return dataclasses.replace(
            self, defaults={**self.defaults, **self.for_testing}, for_testing=None
        )


class ObservationSpec(AgentSpec[Policy]):
    """A family of observation-driven seats."""


class BeliefSpec(AgentSpec[BeliefPolicy]):
    """A family of belief-driven seats."""


class StatefulSpec(AgentSpec[StatefulPolicy]):
    """A family of stateful per-game seats (``policy`` is the agent factory).

    Not traceable: drivers seat these through a per-step Python loop
    (``evaluate`` switches to one automatically), never a fused scan.
    """


OpponentSpec = ObservationSpec | BeliefSpec | StatefulSpec
"""Any seatable opponent: a ``POLICIES`` entry or a spec built elsewhere (e.g.
an experiment wrapping a frozen checkpoint's play agent)."""
