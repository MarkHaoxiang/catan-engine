# experiments

The lab notebook: one numbered directory per experiment *framework* — a
class of related experiments sharing machinery — committed; everything a run
*produces* lands in the git-ignored `runs/` at the repo root.

## Contract

- `NNNN_slug/` — one framework. `run.py` exposes its named configs — either as
  `VARIANTS` deltas (`run.py [variant]`) or, for hydra-based frameworks, as
  `conf/` config groups + an `experiment/` preset dir (`run.py +experiment=<name>`,
  the 0004 pilot); helpers specific to the framework live beside it in the same
  directory. A framework accumulates configs and runs over time — the
  *conclusions* in its report are what stays immutable, and `runs/` collects many
  logs per framework.
- Each run is deterministic given its config (seeds included); it writes only
  under its run directory, which `start_run` creates with a manifest
  (git commit + uncommitted-diff digest, the merged config, environment).
- `NNNN_slug/report.md` — hypothesis → setup → results → decision, one
  section per concluded variant, citing the `runs/` directories the numbers
  came from.
- `JOURNAL.md` — append-only index: one verdict line per concluded finding.
- Strength claims gate through `settlrl-agents bench` (`--json` emits the
  machine-readable verdict) or an in-run match with the threshold asserted in
  code — the script decides pass/fail, not a reading of it.

## Configuration

A framework's config is a typed `settlrl_learn.experiment.Config` (pydantic)
schema — the defaults live in the schema, named *variants* are deltas onto them,
and `key=value` arguments on the command line override either (OmegaConf
dotlist: `maximise.iterations=1`). `Config.resolve(base, variant, overrides)`
merges and validates; the *validated* config is what the run manifest pins. The
shared harness (`Run` / `start_run` / `Config`) lives in
`settlrl_learn.experiment`, not under `experiments/` — these directories hold
only per-framework scripts and `new.py`.

```
uv run python experiments/new.py "<title>"                       # scaffold a framework
uv run python experiments/NNNN_slug/run.py [variant] [k=v...]     # resolve-based framework
uv run python experiments/0004_alphazero/run.py +experiment=<name> [k=v...]  # hydra-based
```

`0001_bench_smoke` is the minimal worked example; `0002_linear_value_fitting`
and `0003_neural_board_architectures` are multi-variant frameworks;
`0004_alphazero` composes its config with hydra (`conf/` groups + `experiment/`
presets; `-m` for sweeps) — working recipes like `scale2`, the `v2_*` study
arms, and the `bench_throughput` throughput probe.

## Checks

Every framework is type-checked (`mypy_experiments.sh`, on pre-commit and CI).
The test suite is composition checks: `uv run pytest experiments/tests` —
details in `CLAUDE.md`.
