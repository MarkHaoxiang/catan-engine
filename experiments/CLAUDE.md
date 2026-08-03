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
  work. **`start_run`'s diff digest covers tracked files only** — an untracked
  or in-flight-edited framework is unpinnable, so commit a framework before
  running it and leave its source alone while a run is in flight.
- `sibling_module(framework_dir, name)`: import another framework's same-dir
  helper (a script dir is not a package, so its directory has to reach
  `sys.path` first). Used for the cross-framework anchor import by 0003's
  `distill.py` and 0005's `duel.py`.
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

The `distill` task (the `guard`/`guard_dnorm` variants, one challenger arch
list each) is the architecture-decision guard:
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

## `0005_search_guard/` — the cheap screen on a search-behavior change

The search-side analogue of 0003's guard: a duel between two search
configurations over one *frozen* net (`0004_alphazero/anchors.py`, imported
cross-framework through `settlrl_learn.experiment.sibling_module`), so search
behavior is the only difference between the sides. `duel.py` is the measurement
layer (`Arm`, `MatchResult`, `search_settings`, `duel`, `seconds_per_move`,
`elo_delta`, `games_for_elo`); `run.py` composes the three matches and the
verdict. The single variant is `chance`; the incumbent is the production
default, checked at run start against the anchor sidecar's `search_semantics`
(a future anchor trained under different flags fails loudly instead of
silently redefining the contrast).

Why the shape:

- **It is a screen, not a gate**, and the cost asymmetry runs the other way: a
  false negative shelves an idea for good, a false positive costs one training
  A/B. So `run.guard_verdict` reads the head-to-head's 2-sigma interval around
  `1 / n_players` and returns three outcomes — `promising` (whole interval
  above the line, earns the A/B), `rejected` (whole interval below),
  `inconclusive` (spanning it, or no decided game: the measurement failed, not
  the idea). It never demands an edge larger than the 35-Elo ship gate it
  screens for, which the old pass rule did (+46 Elo at n=228, 31% power against
  a true +35).
- **`games` is sized from that threshold**: `duel.games_for_elo(35)` is ~390
  decided games and the default 800 buys ±2 sigma of ±3.5 points (≈±24 Elo),
  about a GPU-hour per head-to-head — still ~30x cheaper than the A/B it
  screens.
- **A head-to-head alone hides a shared regression**, so both arms also play
  `lookahead`, the Elo-0 reference, and `run.reference_gap` reports the
  difference. Reported, never gating: at `reference_games=80` per arm its
  2-sigma band is ~90 Elo, inert as a rule but still the number that shows a
  reader the head-to-head and the outside arm pointing opposite ways. The two
  arms start their reference matches from the same seed, so they meet it from
  the same *initial* boards, but the matches diverge at the first differing
  decision and auto-reset then regenerates boards at different steps (101 vs
  106 decided games on one seed) — the rates are independent binomials, not a
  paired comparison.
- **Equal simulations is not equal wall-clock**, so every arm is priced
  (`seconds_per_move`) and `wall_clock_matched` re-runs the head-to-head with
  the challenger's `num_simulations` scaled by the measured ratio. That ratio
  is load-bearing and a single window is not stable enough for it (two probes
  of the same arms disagreed by 13% and 37%), hence the median over
  `timing_repeats` windows. A short timing window measures the same rate as a
  long one because the price is phase-independent: the net agent selects
  between its search and the setup opener with a `where`, so both run on every
  step of every lane.
- **Budget model** (per seating): no lane finishes before a whole game, so
  wall-clock ≈ `game_length × (batch + games_per_seating) × the per-move cost
  of both seats`. Lanes past what a seating harvests buy nothing, which is why
  `batch` sits under the games per seating rather than at the arena's 128.
  Per-move cost, az2 at 128 sims / 16 considered (RTX 5090, batch
  32/64/128/256): incumbent 10.96 / 8.52 / 8.28 / 9.24 ms, `chance_nodes`
  5.34 / 2.98 / 1.74 / 1.17 ms, `ordered` 8.04 ms at 128 — all measured at
  `expected_rolls=True`, i.e. not the contrast the guard runs (below).

Two play-time semantics that shape what the guard can measure:

- An arm is a whole validated `SearchSettings`, wider than
  `make_net_agent`'s keyword surface (which fixes `value_scale` and takes no
  `max_depth`/`fused_leaf`), so `duel._net_agent` composes `gnn_seams` +
  `make_search` + the setup opener itself instead of routing through
  `GNNBackend.play_agent`, and `duel.search_settings` raises if validation
  rewrites any flag an arm asked for. It has to: `chance_nodes` *supersedes*
  `expected_rolls` (forced off by `SearchConfig`), so an arm pair that let that
  through would differ in two leaf semantics at 984 vs 141 MFLOP/search, not in
  the one the variant names.
- `ordered` **cannot be screened here** and has no variant. At play time the
  ordering overlay never reaches the root: `settlrl_agents.evaluate` builds its
  env without `track_ordering`, and the overlay lives in `BatchedSettlrlEnv.step`,
  not in the fused `_rollout_core` a rollout runs. An arm would search a
  tree-pruned model of an unrestricted environment — inconsistent with its own
  env and strictly handicapped, which is not the flag's contract
  (`ismcts/loop.py`). The prerequisite is an engine change: `track_ordering`
  threaded through the fused rollout path. (At 2 players there is also no
  domestic trade, so the lock-out would only order builds and buys.)

`n_players` runs the duel as a seat rotation and the verdict against
`1 / n_players`, but the committed anchors are 2p-trained and **stall at 3
players** — see `report.md`'s Scope for the measurement. A 3p duel needs a
3p-capable net before it measures anything.

`duel.duel` is 0002's `seat_rotated` in another framework: the library's own
`arena_spec` is 2 players only and hands both sides the same simulation count,
neither of which fits a duel whose arms may differ in budget.

The `smoke` variant is a manual CLI recipe
(`run.py smoke`), not a test: even at `sims=0` an in-suite `run_experiment`
costs ~120 s, which the 2-3 minute suite budget does not have room for (0004's
`gnn_smoke` preset is kept the same way — composed in the suite, run by hand).
What the suite does cover is the composition and the pure logic: variants
resolve to two different searches, the three-way verdict rule at its 2-sigma
boundaries, the default `games` against `games_for_elo`, and the seat rotation
with `evaluate` stubbed.
