# experiments — internal notes

The lab-notebook contract is in README.md. The unit here is the experiment
*framework*: a numbered directory holding `run.py` (named variants selected
by argv), its own helper modules, and a report that accumulates one section
per concluded variant. Don't scaffold a new number for a question an
existing framework can express as a config — extend its `VARIANTS` instead;
git history is the framework's changelog, the report its conclusions.

## Shared harness — `settlrl_agents.experiment`

No shared libraries live under `experiments/` (only per-framework scripts +
`new.py`). The reusable harness is the `settlrl_agents.experiment` subpackage:

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
  - **Composition: `resolve` (dict variants) vs hydra config groups.** 0001–0003
    use `resolve` over an in-`run.py` `VARIANTS` dict. **0004 is the hydra pilot**:
    its `conf/` holds config groups + an `experiment/` preset dir (the former
    `VARIANTS`), composed by `@hydra.main` and validated into the nested
    `AlphaZeroConfig`. hydra's cwd takeover (the reason it was first kept at
    arm's length) is disabled in `conf/config.yaml` (`hydra.job.chdir: false`,
    `output_subdir: null`, run dir pointed into the gitignored `runs/`), so
    `start_run` keeps owning the run dir + manifest. `run.compose_config(overrides)`
    is the programmatic seam (smoke tests) that `@hydra.main` can't serve.

The subpackage is not imported by the agents runtime, so `import settlrl_agents`
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
anchor checkpoint loads and forwards). (`conftest.load_run` imports a `run.py`
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
`new.py` and the tests; the shared harness is checked by the agents package
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
live in settlrl-learn now (their symmetry contracts are tested there); this
framework only composes them. `train.py` is optax adamw + wandb (`mode`
configurable; `disabled` in tests) + best-val equinox checkpointing,
standardizing inputs on the train split.

Stack additions (dev group): `equinox`, `optax`, `jraph`, `wandb`. Run on GPU
with `XLA_PYTHON_CLIENT_PREALLOCATE=false` to coexist with other GPU work.

First finding (report.md): a GNN over the *raw* board nearly matches the
engineered-feature MLP (heuristic R² 0.978 vs 0.996; win AUC 0.825 vs 0.834),
while a flat MLP on the same raw inputs is ≈chance — structure is what makes raw
board features usable. Not yet promoted to a shipped value; close the win gap,
then gate `lookahead(gnn)` through `settlrl-agents bench`.

The `road` target (seat-0 longest-road trail length) and the `ablate_*`
variants drive the GraphNet lever ablation over `settlrl_learn.nn.graphnet.PRESETS`
(report.md): GNNs ≫ engineered on the structural target (R² 0.99 vs 0.83),
attention is the wrong bias for counting tasks, and `gn_global` (sum-MPNN +
global node + multi readout + LayerNorm) is the recommended net.
