"""Plain-JAX MLPs and their adapters onto the settlrl-agents seams.

Parameters are an ordinary pytree (a tuple of ``(weights, bias)`` layers), so
a trained model needs nothing beyond jax to run — settlrl-agents can consume an
exported artifact without any training dependencies.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from settlrl_agents.value import Value, ValueFunction
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, IntScalar, KeyScalar
from settlrl_engine.env import N_FLAT
from settlrl_search.policy import PolicyPrior, ValuePrior

from settlrl_learn.features import FEATURE_DIM, features

MLPParams = tuple[tuple[Float[Array, "fan_out fan_in"], Float[Array, "fan_out"]], ...]
"""One ``(weights, bias)`` pair per layer, input to output."""


def init_mlp(key: KeyScalar, sizes: Sequence[int], scale: float = 1.0) -> MLPParams:
    """He-initialised layers of ``sizes`` (input dim first, output dim last);
    ``scale`` multiplies the output layer (small values start near-uniform /
    near-zero)."""
    layers = []
    for i, (fan_in, fan_out) in enumerate(itertools.pairwise(sizes)):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (fan_out, fan_in)) * jnp.sqrt(2.0 / fan_in)
        if i == len(sizes) - 2:
            w = w * scale
        layers.append((w, jnp.zeros((fan_out,))))
    return tuple(layers)


def mlp(params: MLPParams, x: Float[Array, "fan_in"]) -> Float[Array, "fan_out"]:
    """Forward pass: ReLU hidden layers, linear output."""
    for w, b in params[:-1]:
        x = jax.nn.relu(w @ x + b)
    w, b = params[-1]
    return w @ x + b


def make_net_value(params: MLPParams) -> ValueFunction:
    """A :class:`ValueFunction` over :func:`features` (scalar head)."""

    def value(layout: BoardLayout, state: BoardState, player: IntScalar) -> Value:
        return mlp(params, features(layout, state, player))[0]

    return value


def make_net_prior(params: MLPParams) -> PolicyPrior:
    """A :class:`PolicyPrior` over :func:`features` (``N_FLAT`` logits head;
    consumers apply legality masking)."""

    def prior(
        layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> Float[Array, f"flat={N_FLAT}"]:
        return mlp(params, features(layout, state, player))

    return prior


def init_value_params(key: KeyScalar, hidden: Sequence[int] = (64, 64)) -> MLPParams:
    """Stand-in value model: ``features -> hidden -> 1``."""
    return init_mlp(key, (FEATURE_DIM, *hidden, 1), scale=0.01)


def init_prior_params(key: KeyScalar, hidden: Sequence[int] = (64,)) -> MLPParams:
    """Stand-in policy model: ``features -> hidden -> N_FLAT`` logits."""
    return init_mlp(key, (FEATURE_DIM, *hidden, N_FLAT), scale=0.01)


# --- AlphaZero net: one trunk, a value head and a policy head ---

_Head = tuple[Float[Array, "out h"], Float[Array, "out"]]


class AZParams(NamedTuple):
    """A shared trunk feeding a scalar value head and an ``N_FLAT`` policy head.

    ``value`` is a **win-probability logit** for the player the features are
    framed for; map it to the search's [-1, 1] leaf with ``value_scale=2``
    (``tanh(logit / 2) = 2*P(win) - 1``). ``policy`` is unnormalised logits;
    consumers legality-mask.
    """

    trunk: MLPParams  # features -> hidden, ReLU on every layer
    value: _Head  # hidden -> 1
    policy: _Head  # hidden -> N_FLAT


def az_forward(
    params: AZParams, x: Float[Array, "fan_in"]
) -> tuple[Value, Float[Array, f"flat={N_FLAT}"]]:
    """``(value_logit, policy_logits)`` from one shared trunk pass."""
    h = x
    for w, b in params.trunk:
        h = jax.nn.relu(w @ h + b)
    (wv, bv), (wp, bp) = params.value, params.policy
    return (wv @ h + bv)[0], wp @ h + bp


def init_az_params(key: KeyScalar, hidden: Sequence[int] = (64, 64)) -> AZParams:
    """A fresh AZ net: ``features -> hidden`` trunk, then value and policy heads
    initialised small (value starts near a 0.5 win logit, policy near uniform)."""
    k_trunk, k_v, k_p = jax.random.split(key, 3)
    trunk = init_mlp(k_trunk, (FEATURE_DIM, *hidden))
    h = hidden[-1]
    wv = jax.random.normal(k_v, (1, h)) * jnp.sqrt(2.0 / h) * 0.01
    wp = jax.random.normal(k_p, (N_FLAT, h)) * jnp.sqrt(2.0 / h) * 0.01
    return AZParams(trunk, (wv, jnp.zeros((1,))), (wp, jnp.zeros((N_FLAT,))))


@dataclasses.dataclass(frozen=True)
class _AZPrior:
    """The AZ net's policy seam as a :class:`~settlrl_search.policy.ValuePrior`,
    so the search takes both heads off one trunk pass where it needs both."""

    params: AZParams

    def __call__(
        self, layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> Float[Array, f"flat={N_FLAT}"]:
        return az_forward(self.params, features(layout, state, player))[1]

    def with_value(
        self, layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> tuple[Value, Float[Array, f"flat={N_FLAT}"]]:
        return az_forward(self.params, features(layout, state, player))


def make_az(params: AZParams) -> tuple[ValueFunction, ValuePrior]:
    """Adapt one AZ net onto the search seams as ``(value, prior)``; both run
    the shared trunk, so build the search with ``value_scale=2`` (see
    :class:`AZParams`)."""

    def value(layout: BoardLayout, state: BoardState, player: IntScalar) -> Value:
        return az_forward(params, features(layout, state, player))[0]

    return value, _AZPrior(params)


def save_az_params(path: str | Path, params: AZParams) -> None:
    """Write an :class:`AZParams` as an ``.npz`` artifact (load with
    :func:`load_az_params`)."""
    arrays: dict[str, np.ndarray] = {}
    for i, (w, b) in enumerate(params.trunk):
        arrays[f"trunk_w{i}"], arrays[f"trunk_b{i}"] = np.asarray(w), np.asarray(b)
    arrays["value_w"], arrays["value_b"] = (
        np.asarray(params.value[0]),
        np.asarray(params.value[1]),
    )
    arrays["policy_w"], arrays["policy_b"] = (
        np.asarray(params.policy[0]),
        np.asarray(params.policy[1]),
    )
    np.savez(path, **arrays)  # type: ignore[arg-type]  # typeshed kwargs quirk


def load_az_params(path: str | Path) -> AZParams:
    """Read a :func:`save_az_params` artifact back into device arrays."""
    with np.load(path) as doc:
        n = sum(1 for k in doc.files if k.startswith("trunk_w"))
        trunk = tuple(
            (jnp.asarray(doc[f"trunk_w{i}"]), jnp.asarray(doc[f"trunk_b{i}"]))
            for i in range(n)
        )
        return AZParams(
            trunk,
            (jnp.asarray(doc["value_w"]), jnp.asarray(doc["value_b"])),
            (jnp.asarray(doc["policy_w"]), jnp.asarray(doc["policy_b"])),
        )


def save_params(path: str | Path, params: MLPParams) -> None:
    """Write ``params`` as an ``.npz`` artifact (load with :func:`load_params`)."""
    arrays = {}
    for i, (w, b) in enumerate(params):
        arrays[f"w{i}"] = np.asarray(w)
        arrays[f"b{i}"] = np.asarray(b)
    np.savez(path, **arrays)  # type: ignore[arg-type]  # typeshed kwargs quirk


def load_params(path: str | Path) -> MLPParams:
    """Read an :func:`save_params` artifact back into device arrays."""
    with np.load(path) as doc:
        n = sum(1 for k in doc.files if k.startswith("w"))
        return tuple(
            (jnp.asarray(doc[f"w{i}"]), jnp.asarray(doc[f"b{i}"])) for i in range(n)
        )
