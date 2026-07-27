"""Supervised dataset from self-play: positions -> {win, heuristic, road} labels.

Roll out games with a scripted agent, snapshot positions from seat 0's
perspective, and label each with three targets:

- ``win`` — did seat 0 win that game (a value/win-probability target);
- ``heur`` — :func:`settlrl_agents.value.heuristic_value` at the position (a
  cheap regression target: can a net reproduce the hand-tuned value from the
  board, and which representation does it most easily?);
- ``road`` — seat 0's true longest-road trail length (a *structural* target the
  engineered vector cannot express: it carries only a road *count*, not the
  connectivity/opponent-break DFS, so this is the clean GNN-vs-engineered test);
- ``turns`` — snapshots remaining until the game ends (a *global* tempo target;
  free from the per-lane buffer length at flush time).

Positions are featurized on the *true* board (honest: hidden fields are real),
so no belief sampling is needed. Episode ids group correlated rows for a
leak-free split. The result is cached by config under ``runs/_cache`` so every
architecture trains on identical data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from settlrl_agents import POLICIES
from settlrl_agents.evaluate import _picker
from settlrl_agents.value import heuristic_value
from settlrl_engine.env import BatchedSettlrlEnv, flat_to_action
from settlrl_engine.mechanics.longest_road import longest_road_length
from settlrl_learn.features import features as engineered_features
from settlrl_learn.nn.graph import Sample, board_sample
from settlrl_search.policy import StatefulSpec

_CACHE = Path(__file__).resolve().parents[2] / "runs" / "_cache" / "0003"


class Dataset(NamedTuple):
    samples: Sample  # batched over a leading sample axis
    win: np.ndarray  # (n,) 0/1
    heur: np.ndarray  # (n,) heuristic value at the position
    road: np.ndarray  # (n,) seat-0 longest-road trail length
    turns: np.ndarray  # (n,) snapshots remaining until the game ends
    episode: np.ndarray  # (n,) game id, for a grouped split


def _key(cfg: dict) -> str:
    keys = ("agent", "players", "n_samples", "snapshot_every", "batch_size", "seed")
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _collect(cfg: dict) -> Dataset:
    agent = POLICIES[cfg["agent"]]
    assert not isinstance(agent, StatefulSpec), "collector drives pure agents"
    players = cfg["players"]
    bs = cfg["batch_size"]
    env = BatchedSettlrlEnv(
        batch_size=bs, seed=cfg["seed"], reward="sparse",
        n_players=players, track_beliefs=True,
    )  # fmt: skip
    pickers = [jax.jit(_picker(agent, players, i)) for i in range(players)]
    feat = jax.jit(
        jax.vmap(
            lambda lo, st: board_sample(
                lo, st, jnp.int32(0), features=engineered_features
            )
        )
    )
    heur = jax.jit(jax.vmap(lambda lo, st: heuristic_value(lo, st, jnp.int32(0))))
    road = jax.jit(
        jax.vmap(
            lambda st: longest_road_length(st.edge_road, st.vertex_owner, jnp.int32(0))
        )
    )

    buffers: list[list[tuple[Sample, float, float]]] = [[] for _ in range(bs)]
    rows: list[tuple[Sample, float, float]] = []
    wins: list[int] = []
    turns: list[int] = []
    episodes: list[int] = []
    key = jax.random.key(cfg["seed"])
    n_ep = 0
    step = 0
    while len(rows) < cfg["n_samples"]:
        layout, state = env.board
        if step % cfg["snapshot_every"] == 0:
            s = jax.device_get(feat(layout, state))
            h = np.asarray(heur(layout, state))
            r = np.asarray(road(state))
            for lane in range(bs):
                buffers[lane].append(
                    (
                        jax.tree.map(lambda x, lane=lane: x[lane], s),
                        float(h[lane]),
                        float(r[lane]),
                    )
                )
        key, k = jax.random.split(key)
        seat_keys = jax.random.split(k, players)
        sel = np.asarray(env.agent_selection)
        mask = env.flat_mask()
        flat = np.zeros((bs,), np.int32)
        for i in range(players):
            picks = np.asarray(
                pickers[i](seat_keys[i], layout, state, env.beliefs, mask)
            )
            flat[sel == i] = picks[sel == i]
        env.step(*flat_to_action(jnp.asarray(flat)))
        rewards = np.asarray(env.rewards)
        for lane in np.flatnonzero(np.asarray(env.terminations).any(axis=1)).tolist():
            won = int(rewards[lane, 0] > 0)
            buf = buffers[lane]
            for j, (sample, hv, rv) in enumerate(buf):
                rows.append((sample, hv, rv))
                wins.append(won)
                turns.append(len(buf) - 1 - j)  # snapshots from here to game end
                episodes.append(n_ep)
            n_ep += 1
            buffers[lane] = []
        step += 1

    samples = jax.tree.map(lambda *xs: np.stack(xs), *(r[0] for r in rows))
    return Dataset(
        samples=samples,
        win=np.asarray(wins, np.float32),
        heur=np.asarray([r[1] for r in rows], np.float32),
        road=np.asarray([r[2] for r in rows], np.float32),
        turns=np.asarray(turns, np.float32),
        episode=np.asarray(episodes, np.int64),
    )


def generate(cfg: dict) -> Dataset:
    """Collect (or load from ``runs/_cache``) the supervised dataset for ``cfg``."""
    path = _CACHE / f"{_key(cfg)}-v5.npz"  # -v5: per-hex tile features
    if path.exists():
        with np.load(path) as d:
            samples = Sample(
                nodes=d["nodes"],
                edges=d["edges"],
                glob=d["glob"],
                tiles=d["tiles"],
                extra=d["engineered"],
            )
            return Dataset(
                samples, d["win"], d["heur"], d["road"], d["turns"], d["episode"]
            )
    ds = _collect(cfg)
    assert ds.samples.extra is not None  # _collect computes the engineered vector
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        nodes=ds.samples.nodes, edges=ds.samples.edges, glob=ds.samples.glob,
        tiles=ds.samples.tiles,
        engineered=ds.samples.extra, win=ds.win, heur=ds.heur, road=ds.road,
        turns=ds.turns, episode=ds.episode,
    )  # fmt: skip
    return ds


def split(ds: Dataset, val_frac: float, seed: int = 0) -> tuple[Dataset, Dataset]:
    """Train/val split grouped by episode (rows within a game are correlated)."""
    games = np.unique(ds.episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_frac))
    val_games = set(games[:n_val].tolist())
    is_val = np.array([e in val_games for e in ds.episode])

    def take(mask: np.ndarray) -> Dataset:
        return Dataset(
            jax.tree.map(lambda x: x[mask], ds.samples),
            ds.win[mask], ds.heur[mask], ds.road[mask], ds.turns[mask],
            ds.episode[mask],
        )  # fmt: skip

    return take(~is_val), take(is_val)
