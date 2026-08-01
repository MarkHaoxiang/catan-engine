# settlrl-learn

Learned value and policy functions for [settlrl-agents](../settlrl-agents).

Models plug into the agents through their existing seams: a value model is a
`ValueFunction` and a policy model a `PolicyPrior`, so the search agents consume
them unchanged.

## Shipped models (plain-JAX)

`import settlrl_learn` pulls only JAX, so a trained model ships without any
training libraries.

- `features(layout, state, player)` — the position as a flat vector from one
  seat: the player's own block, the max and mean over opponents' blocks, and a
  global block. The width (`FEATURE_DIM`) is the same at every player count, and
  features are only ever computed on concrete (sampled) worlds.
- `AZParams` — a shared-trunk value+policy net (the value head is a
  win-probability logit). `init_az_params` builds one, `make_az` adapts it onto
  the search's value + prior seams, and `save_az_params` / `load_az_params` move
  it through `.npz` artifacts.
- `make_net_value` / `make_net_prior` adapt single-head MLP params onto the
  seams; `init_value_params` / `init_prior_params` build untrained stand-ins;
  `save_params` / `load_params` for `.npz`.

## Training side

Not imported by the package root (equinox / optax / flashbax stay off the
shipped path); `experiments/` composes these.

- `settlrl_learn.nn` — network definitions: the plain-JAX MLP (`mlp`), the
  equinox board-graph nets (`graph` / `graphnet` / `board_gnn` /
  `architectures`), and the flat-action board-structure map
  (`action_layout`).
- `settlrl_learn.training` — the self-play → replay → train → arena loop
  (`learn`) behind a `Backend` seam, with a flat-MLP and a board-GNN backend.
- `settlrl_learn.evaluation` — head-to-head gating of frozen nets: the arena
  driver and the anchored-Elo MLE.
- `settlrl_learn.experiment` — the lab harness (run bookkeeping + the
  pydantic / OmegaConf config base) for `experiments/`.

Experiment 0004 composes the training loop; experiment 0003, the architectures.

Micro-benchmarks for the training loop's units live in `tests/benchmark/` —
see [its README](tests/benchmark/README.md) for how to run them.
