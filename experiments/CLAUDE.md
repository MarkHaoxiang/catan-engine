# experiments — internal notes

The lab-notebook contract is in README.md. The unit here is the experiment
*framework*: a numbered directory holding `run.py` and its named configs —
`VARIANTS` deltas selected by argv, or hydra config groups + `experiment/`
presets (0004; the harness section below covers the split) — its own helper
modules, and a report that accumulates one section per concluded variant.
Don't scaffold a new number for a question an existing framework can express
as a config — add a variant or preset instead; git history is the framework's
changelog, the report its conclusions.

## Shared harness — `settlrl_learn.experiment`

No shared libraries live under `experiments/` (only per-framework scripts +
`new.py`). The reusable harness is the `settlrl_learn.experiment` subpackage:

- `start_run` (run dir + manifest pinning git commit / uncommitted-diff digest
  / merged config / environment; repo root derived from the framework dir it's
  handed, so it's location-independent), `Run.log` (metrics.jsonl),
  `Run.save_json`, `Run.finish` (result.json + the printed verdict). `Run` takes
  any `dir`, so a smoke test points it at `tmp_path` and skips the git/manifest
  work.
- `Config`: the pydantic base + `resolve(base, variant, overrides)` (OmegaConf
  merge of schema-defaults ◁ variant-delta ◁ CLI dotlist, then validate). The
  pydra seam: pydantic is always the validation boundary; `extra="forbid"`, so a
  typo'd knob fails loudly. Heavier frameworks validate at the boundary and pass
  `cfg.dump()` (a plain dict) inward so their internals stay dict-threaded.
  - **Composition: `resolve` (dict variants) vs hydra config groups.** 0002 and
    0003 use `resolve` over an in-`run.py` `VARIANTS` dict. **0004 is the hydra pilot**:
    its `conf/` holds config groups + an `experiment/` preset dir (its analogue
    of `VARIANTS`), composed by `@hydra.main` and validated into the nested
    `AlphaZeroConfig`. hydra's cwd takeover is disabled in `conf/config.yaml` (`hydra.job.chdir: false`,
    `output_subdir: null`, run dir pointed into the gitignored `runs/`), so
    `start_run` keeps owning the run dir + manifest. `run.compose_config(overrides)`
    is the programmatic seam (smoke tests) that `@hydra.main` can't serve.
  - When other processes share the GPU, launch 0004 training runs with
    `XLA_PYTHON_CLIENT_PREALLOCATE=false` — `run_final_gauntlet` clears jax's
    jit caches before playing (a5bda5b), but a preallocated pool never returns
    memory to the driver.

The harness lives in `settlrl-learn`, not `settlrl-agents`, so `import settlrl_agents`
does not pull `pydantic`/`omegaconf`. A framework's *same-dir* helpers (e.g.
`value_fitting`, `data`, `models`) still live beside its `run.py`.

## Testing (`tests/`, mypy)

The whole suite (`uv run pytest experiments/tests`, no `-m` filter) must fit
~2-3 minutes cold — that budget is the design constraint, not an aspiration.
It holds because `tests/test_smoke.py` is mostly config-composition checks:
every named variant/preset for a framework resolves and validates
(`Config.resolve` / `compose_config`), which is hydra+pydantic only, no JAX,
so it costs milliseconds per variant and still catches a typo'd knob or a
renamed seam. The actual training loop, backends, and search already have
end-to-end coverage at tiny budgets in the owning packages
(`settlrl-learn`'s `test_training.py` for both the mlp and gnn backends
including bit-exact resume, `settlrl-search`'s `test_ismcts.py` for
chance/ordered search) — a framework's *unique* surface here is only the
composition layer (config groups, `run.py` wiring, the bench gate, a recorded
verdict), so each framework keeps its real end-to-end runs to only what
proves a distinct piece of that layer, never one per variant. 0004 keeps
three: the mlp `smoke` (the ordinary `run_experiment` path — arena, gate,
verdict), `bench_throughput_smoke` (the separate `mode=bench` wiring the mlp
path never exercises), and `test_0004_anchor_loads_and_forwards` (not a
`run_experiment` at all — a deserialization-integrity check that a saved
anchor checkpoint loads and forwards; `test_0004_builds_net_opponent_specs`
rides the same load to check the frozen-checkpoint arena rung composes into a
seatable spec, still without playing a game). (`conftest.load_run` imports a `run.py`
by path; the digit-prefixed dirs aren't packages.) A smoke asserts only a
recorded verdict, never strength.

Keep any real end-to-end run's budgets trivial in **every** group — 0004's
`smoke` once left `arena` at production defaults while the rest was trivial,
and the arena alone cost 895s of a 907s iteration. The whole suite (every
framework, unfiltered) measures 91-129s cold; JAX's compilation cache persists
across runs (`~/.cache/jax-settlrl` by default, wired in
`experiments/tests/conftest.py`), so a warm pre-commit run is cheaper still.
That's why only the mlp `smoke` — the heaviest of the three real 0004 runs —
still carries `@pytest.mark.slow` (pre-commit runs `-m "not slow"`; CI runs
the suite unfiltered): it trims the pre-commit loop, not the measured budget,
which already holds without it. A pure compose/resolve check never needs the
marker, and a new real run shouldn't reach for it unless it's dramatically
heavier than today's. `mypy_experiments.sh` checks each framework dir
separately (the `run.py` modules would collide on one invocation) plus
`new.py` and the tests; the shared harness is checked by the learn package
mypy. New frameworks: add a variants-resolve check, at most one tiny
end-to-end case, and a `test_<nnnn>_*` name so the mypy loop picks the dir up
automatically.

## `0002_linear_value_fitting/` — linear fits over the engineered features

`value_fitting.py` optimizes weights over
`settlrl_agents.internal.feature_engineering.BoardFeatures`, deploys them
through `value.make_linear` into one-step lookahead, and gates against the
hand-tuned weights (pass iff the lower 2-sigma bound clears 50%). Config
knobs: `features` (list of `BoardFeatures` names), `target`
(`predict` — outcome fits, {logistic, sign-constrained NNLS} × {all, early
positions}, ranked by match probes; `maximise` — cross-entropy search with
the measured seat-swapped win rate as the objective, common random numbers
within a generation), `opponent` (a `POLICIES` name, or `"self"` for the
self-play ladder: each round's opponent is the current champion and a
challenger replaces it only by winning the acceptance match), `rounds`, and
the budgets (collection, CEM, probe/bench/gate games).

Lessons baked into the design:

- **Select by matches, never fit metrics**: held-out AUC was flat
  (0.831–0.843) across candidates whose match probes spanned 52.8–78.0%.
- **Prediction is not control**: unconstrained outcome regression
  redistributes correlated credit (production fit at +0.008, the discard
  penalty fit *positive*); NNLS pins the signs, early-position fits force
  economy to carry signal — both exist as candidates for this reason.
- **Fixed-opponent optimization breeds specialists**: both targets beat or
  matched the hand weights against their objective opponent and lost ~43%
  head-to-head against the hand-tuned lookahead — hence the self-play
  variant.
- **Group the held-out split by episode** — rows within a game are
  correlated, a row-level split leaks.
- Each distinct weight vector is a fresh value closure: `evaluate` retraces
  its scan per candidate (~seconds), which is most of a maximise
  generation's overhead — budget `eval_games` accordingly.

## `0003_neural_board_architectures/` — representation × architecture sweep

Supervised benchmark for *which net learns the board best*, the seam toward a
learned value (settlrl-learn Stage 1). `data.py` rolls out greedy self-play and
caches seat-0 positions (true board) under `runs/_cache`, labelled with both the
hand-tuned `heuristic_value` and the eventual win. The featurization
(`settlrl_learn.nn.graph`: board → a fixed-topology graph, 54 vertices / 72 edges
with senders/receivers as module constants, plus the engineered vector) and the
architectures (`settlrl_learn.nn.architectures`: `mlp_engineered` baseline,
`mlp_flat` structure-blind, `deepset` set, `gnn` jraph `GraphNetwork` + readout)
live in settlrl-learn (their symmetry contracts are tested there); this
framework only composes them. `feature_version` / `incidence` select the
versioned `settlrl_learn.nn.graph` feature set: threaded into `board_sample`
during collection (part of the data cache key, whose filename suffix is
`data._CACHE_SCHEMA`) and into `make_model`, which sizes every architecture
from `graph.dims`. `seeds` trains per-arch replicates varying only the
model-init key and minibatch-shuffle seed (data collection and split stay on
`seed`); `results.json` records per-seed values plus `<metric>_mean` /
`<metric>_spread`, and the verdict reads the mean. `train.py` is optax adamw +
wandb (`mode` configurable; `disabled` in tests) + best-val equinox
checkpointing, standardizing inputs on the train split.

Stack additions (dev group): `equinox`, `optax`, `jraph`, `wandb`. Run on GPU
with `XLA_PYTHON_CLIENT_PREALLOCATE=false` to coexist with other GPU work.

First finding (report.md): a GNN over the *raw* board nearly matches the
engineered-feature MLP (heuristic R² 0.978 vs 0.996; win AUC 0.825 vs 0.834),
while a flat MLP on the same raw inputs is ≈chance — structure is what makes raw
board features usable.

The `road` target (seat-0 longest-road trail length) and the `ablate_*`
variants drive the GraphNet lever ablation over `settlrl_learn.nn.graphnet.PRESETS`
(`gn_hetero` included; report.md): GNNs ≫ engineered on the structural target
(R² 0.99 vs 0.83), attention is the wrong bias for counting tasks, and
`gn_global` (sum-MPNN + global node + multi readout + LayerNorm) won the 0003
supervised ablation; the 0004 four-arm study (2026-07-30) then showed
`gn_hetero` beats it by ~3σ in the full AZ loop, and `gn_hetero` is the
adopted trunk. `hetero_v2` is the architecture-guard head-to-head: `gn_global`
vs `gn_hetero` on the multi task at `feature_version=2`, three seed
replicates.

The `distill` task (`guard` variant) is the architecture-decision guard:
`distill.py` generates a frozen dataset from the az2 anchor's own self-play
through the production stack (`selfplay_callables`/`run_selfplay`; search
semantics from the anchor sidecar, the production opening-temperature
schedule) holding the GNN observation, the search's improved policy, `mask`,
the root search value `q` and the raw outcome `z` — kept separate in the
dump, cached under `runs/_cache/0003` keyed by
`(anchor, sims, batch, n_samples, seed)` + `_DISTILL_SCHEMA`. Deliberate
divergences from the production recipe: uniform sims (no PCR mix) and a
non-persistent generation batch — both chosen so every position is a
full-search target. `distill_train.py` trains the *production* net
(`GNNBackend` over a GraphNet preset — the `arch` list must name presets)
with the backend's own train step and optimizer
(lr/weight_decay/grad_clip/batch read from 0004's `optim/scale.yaml`),
blending the value target at training time via `steps.prepare_targets` with
alpha read from 0004's `value_blend/scale.yaml`; val metrics are masked
policy KL (the best-checkpoint criterion), top-1 agreement, and P(win) MSE
vs the blended target and vs raw `z`. Train and val are two independently
generated datasets (generation seeds `seed` and `seed+1000`), so the split
is leak-free by construction. Verdict (`run.distill_verdict`): with the
incumbent (`distill_incumbent`, default `gn_hetero`) in the arch list, a
challenger passes iff its worst-seed `best_policy_kl` is strictly below the
incumbent's best-seed value (a zero-overlap win, lower = better) — every
challenger passing → `"pass"`, none → `"fail"`, otherwise `"mixed"`, with
per-challenger `beats_incumbent` booleans in `results.json`; incumbent
absent → `"recorded"`.
