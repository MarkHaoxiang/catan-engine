"""Consistency invariants of world sampling (``sample_world``).

A sample rebuilds a playable ``BoardState`` from a ``BeliefView`` without
disturbing anything public: the public fields are copied through, and hand
sizes, dev counts, per-type totals, and the observer's own rows must all match
the public record.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from settlrl_engine.belief import BeliefView, PublicState
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_search.sample import sample_world


def _played_view(n_players: int, n_steps: int, seed: int = 0) -> BeliefView:
    """A mid-game view: lane 0, observer 0, after random play."""
    env = BatchedSettlrlEnv(
        batch_size=4, seed=seed, n_players=n_players, track_beliefs=True
    )
    key = jax.random.key(seed)
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
    return cast(BeliefView, jax.tree.map(lambda x: x[0], env.belief_view(0)))


@pytest.mark.parametrize("n_players", [2, 4])
def test_sample_matches_the_public_record(n_players: int) -> None:
    view = _played_view(n_players, n_steps=300)
    pb = view.belief
    me = jnp.int32(0)
    keys = set()
    for seed in range(4):
        world = sample_world(jax.random.key(seed), view, me)
        # The public fields are copied through untouched.
        for name in PublicState._fields:
            assert bool(jnp.all(getattr(world, name) == getattr(view.public, name)))
        res = world.player_resources.astype(jnp.int32)
        # Public resource counts: sizes per player, totals per type.
        assert bool(jnp.all(res.sum(axis=1) == pb.hand_size))
        assert bool(jnp.all(res.sum(axis=0) == pb.res_total))
        # The proven floor is respected; own row is exact (lo is pinned there).
        assert bool(jnp.all(pb.res_lo.astype(jnp.int32) <= res))
        assert bool(jnp.all(res[0] == pb.res_lo[0]))
        # Dev cards: public counts, own hand, and pool conservation.
        dev = world.dev_hand.astype(jnp.int32)
        assert bool(jnp.all(dev.sum(axis=1) == pb.dev_count))
        assert bool(jnp.all(dev[0] == view.own_dev))
        assert bool(
            jnp.all(
                world.dev_deck.astype(jnp.int32) + dev.sum(axis=0)
                == view.unseen_dev.astype(jnp.int32) + view.own_dev.astype(jnp.int32)
            )
        )
        keys.add(tuple(int(x) for x in jax.random.key_data(world.key).ravel()))
    # Each sample carries its own fresh PRNG key.
    assert len(keys) == 4


def test_sample_varies_with_the_key() -> None:
    # Grant the opponent three (hidden) dev cards: the deal must have freedom.
    view = _played_view(2, n_steps=0)
    view = view._replace(
        belief=view.belief._replace(dev_count=view.belief.dev_count.at[1].set(3))
    )
    me = jnp.int32(0)
    hands = {
        tuple(int(c) for c in sample_world(jax.random.key(s), view, me).dev_hand[1])
        for s in range(12)
    }
    assert len(hands) > 1  # the posterior is actually being sampled


def test_two_player_resources_are_pinned() -> None:
    # 2p beliefs are exact on resources, so every sample agrees on them.
    view = _played_view(2, n_steps=300)
    me = jnp.int32(0)
    worlds = [
        sample_world(jax.random.key(s), view, me).player_resources for s in range(3)
    ]
    for w in worlds[1:]:
        assert bool(jnp.all(w == worlds[0]))


def test_sample_relaxes_infeasible_upper_bounds() -> None:
    # Construct a BeliefView whose per-type upper bounds are jointly infeasible
    # for opponent 1 (sum(res_hi[1]) < hand_size[1]) by collapsing its hi to a
    # single type. The proportional-headroom deal must relax the hi cap (the
    # surrogate at sample.py:~94) rather than crash or under-fill the hand.
    view = _played_view(4, n_steps=120, seed=2)
    pb = view.belief
    p = 1  # the opponent whose bounds we make infeasible
    need = int(pb.hand_size[p])
    assert need >= 2  # a meaningful infeasibility (one type can't hold the hand)
    # Free its lo (so lo <= hi holds) and cap its hi at one card of one type:
    # sum(hi) == 1 < need, so every per-type headroom hits zero mid-deal.
    res_lo = pb.res_lo.at[p].set(jnp.zeros_like(pb.res_lo[p]))
    res_hi = pb.res_hi.at[p].set(jnp.zeros_like(pb.res_hi[p]).at[0].set(1))
    view = view._replace(belief=pb._replace(res_lo=res_lo, res_hi=res_hi))
    assert int(view.belief.res_hi[p].sum()) < need  # genuinely infeasible

    me = jnp.int32(0)
    for s in range(4):
        world = sample_world(jax.random.key(s), view, me)
        res = world.player_resources.astype(jnp.int32)
        assert bool(jnp.all(res >= 0))  # no negative counts
        # Public record still honoured despite the relaxation.
        assert bool(jnp.all(res.sum(axis=1) == view.belief.hand_size))
        assert bool(jnp.all(res.sum(axis=0) == view.belief.res_total))
        # The proven floor is still respected for the relaxed opponent.
        assert bool(jnp.all(res[p] >= view.belief.res_lo[p].astype(jnp.int32)))


def test_sample_dev_hands_pinned_at_2p() -> None:
    # Mirror of the resource pin for dev cards: when nothing about the opponent's
    # dev cards is uncertain (it holds none), the deal is deterministic -- the
    # opponent's dev_hand and dev_deck are identical across keys.
    view = _played_view(2, n_steps=300)
    pb = view.belief
    view = view._replace(belief=pb._replace(dev_count=pb.dev_count.at[1].set(0)))
    me = jnp.int32(0)
    worlds = [sample_world(jax.random.key(s), view, me) for s in range(3)]
    for w in worlds[1:]:
        assert bool(jnp.all(w.dev_hand == worlds[0].dev_hand))
        assert bool(jnp.all(w.dev_deck == worlds[0].dev_deck))
    # And it really is the empty deal: opponent holds nothing, deck = unseen pool.
    assert bool(jnp.all(worlds[0].dev_hand[1] == 0))
