# settlrl

A fast, batched implementation of Settlrl, a hex-tile trading board game, built
for reinforcement learning and large-scale simulation.

## Packages

| Package | Description |
|---|---|
| [`settlrl-engine`](reference/engine.md) | The game engine — board generation, rules, and actions, vectorised in JAX. |
| [`settlrl-game`](reference/game.md) | The shared game model: the plain-Python reference rules (the differential-test oracle for `settlrl-engine`) plus the serialization / replay layer the app and bot service build on. Engine-free. |
| [`settlrl-search`](reference/search.md) | The re-determinizing search over the engine — seat protocols, world sampling, and the SO-ISMCTS tree (`mcts`) plus one-step lookahead. |
| [`settlrl-agents`](reference/agents.md) | Agents that play the game — heuristics (greedy) and the search agents built on `settlrl-search` — plus a CLI for matches and benchmarks, and the bot service that serves their moves over HTTP. |
| [`settlrl-learn`](reference/learn.md) | Learned value and policy functions that plug into the agents. |
| [`settlrl-app`](reference/app.md) | The web game server: REST + SSE API, async game runtime, storage, auth, and the browser frontend. |

## This site

- **Getting started** walks through installing the workspace and running each
  package.
- **Reference** is generated from the source — docstrings, signatures, and type
  annotations — so it tracks the code. Narrative and design rationale that the
  code cannot express live in the per-package `CLAUDE.md` files.
