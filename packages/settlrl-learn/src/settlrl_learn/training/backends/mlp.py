"""The flat-engineered-MLP backend: an :class:`~settlrl_learn.nn.mlp.AZParams`
net over the engineered feature vector, with an unmasked policy CE + value
logistic loss. The net plays the setup phase itself (no delegation).

A training-side module (optax): not imported by the package root.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float
from settlrl_agents.value import ValueFunction
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, IntScalar
from settlrl_engine.env import N_FLAT
from settlrl_search import make_search
from settlrl_search.policy import BeliefPolicy, PolicyPrior

from settlrl_learn.features import FEATURE_DIM, features
from settlrl_learn.nn.mlp import AZParams, az_forward, init_az_params, make_az
from settlrl_learn.training.backends.base import Metrics, StepFn
from settlrl_learn.training.selfplay import Samples


class MLPItem(NamedTuple):
    """One replay item: the engineered features, the search's improved-policy
    target, the game-outcome value label, and the ``train_policy`` flag (1 =
    train the policy head here; 0 = value-only playout-cap position). A leading
    sample axis when batched."""

    features: Float[Array, "*b feat"]
    policy: Float[Array, "*b flat"]
    value: Float[Array, "*b"]
    train_policy: Bool[Array, "*b"]


def mlp_loss(
    params: AZParams, item: MLPItem, value_weight: float
) -> tuple[Float[Array, ""], Metrics]:
    """Policy cross-entropy (against the search target) + ``value_weight`` x the
    value logistic loss (win/loss). The policy CE is averaged over
    ``train_policy`` = 1 positions only (value trains on all); all 1 is the plain
    mean."""
    vs, logits = jax.vmap(az_forward, in_axes=(None, 0))(params, item.features)
    logp = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.sum(item.policy * logp, axis=-1)  # per position
    tp = jnp.asarray(item.train_policy, jnp.float32)  # stored bool, exact 0/1
    policy_loss = jnp.sum(tp * ce) / jnp.maximum(jnp.sum(tp), 1.0)
    value_loss = jnp.mean(jax.nn.softplus(vs) - item.value * vs)
    total = policy_loss + value_weight * value_loss
    return total, {"policy_loss": policy_loss, "value_loss": value_loss}


class MLPBackend:
    """A :class:`~settlrl_learn.training.backends.base.Backend` over an ``AZParams`` net."""

    def __init__(
        self,
        hidden: Sequence[int],
        *,
        value_weight: float = 1.0,
        chance_nodes: bool = False,
        dev_chance: bool = True,
        ordered: bool = False,
    ) -> None:
        self.hidden = tuple(hidden)
        self.value_weight = value_weight
        self.chance_nodes = chance_nodes
        self.dev_chance = dev_chance
        self.ordered = ordered

    def init(self, key: Array) -> AZParams:
        return init_az_params(key, self.hidden)

    def seams(self, net: AZParams) -> tuple[ValueFunction, PolicyPrior]:
        return make_az(net)

    def setup_policy(self) -> BeliefPolicy | None:
        return None

    def play_agent(
        self, net: AZParams, *, num_simulations: int, max_num_considered_actions: int
    ) -> BeliefPolicy:
        value_fn, prior_fn = make_az(net)
        return make_search(
            value_fn, prior=prior_fn, value_scale=2.0,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            chance_nodes=self.chance_nodes, dev_chance=self.dev_chance,
            ordered=self.ordered,
        )  # fmt: skip

    def observe(
        self, layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> dict[str, Array]:
        return {"features": features(layout, state, player)}

    def to_item(self, samples: Samples) -> MLPItem:
        return MLPItem(
            jnp.asarray(samples["features"], jnp.float32),
            jnp.asarray(samples["policy"], jnp.float32),
            jnp.asarray(samples["value"], jnp.float32),
            jnp.asarray(samples["train_policy"], jnp.bool_),
        )

    def empty_item(self) -> MLPItem:
        return MLPItem(
            jnp.zeros((FEATURE_DIM,), jnp.float32),
            jnp.zeros((N_FLAT,), jnp.float32),
            jnp.float32(0.0),
            jnp.bool_(True),
        )

    def init_opt(
        self, optimizer: optax.GradientTransformation, net: AZParams
    ) -> optax.OptState:
        return optimizer.init(net)

    def make_step(self, optimizer: optax.GradientTransformation) -> StepFn:
        grad_fn = jax.value_and_grad(mlp_loss, has_aux=True)
        value_weight = self.value_weight

        @jax.jit
        def step(
            net: AZParams, opt_state: optax.OptState, item: MLPItem
        ) -> tuple[AZParams, optax.OptState, Metrics]:
            (total, aux), grads = grad_fn(net, item, value_weight)
            updates, opt_state = optimizer.update(grads, opt_state, net)
            net = cast(AZParams, optax.apply_updates(net, updates))
            return net, opt_state, {"loss": total, **aux}

        return step

    def eval_metrics(self, net: AZParams, item: MLPItem) -> Metrics:
        return cast(Metrics, _eval(net, item))


@jax.jit
def _eval(net: AZParams, item: MLPItem) -> Metrics:
    vs, logits = jax.vmap(az_forward, in_axes=(None, 0))(net, item.features)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return {
        "val_policy_loss": -jnp.mean(jnp.sum(item.policy * logp, axis=-1)),
        "val_value_loss": jnp.mean(jax.nn.softplus(vs) - item.value * vs),
        "val_value_acc": jnp.mean((vs > 0).astype(jnp.float32) == item.value),
    }
