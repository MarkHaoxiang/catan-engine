# Measurement Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the measurement infrastructure the throughput-optimization effort will be judged against: a pinned self-play throughput benchmark (headline `samples_per_s`) plus honest strength-measurement statistics (real game counts into the Elo MLE, a Fisher standard error, a committed frozen-checkpoint anchor).

**Architecture:** Strength side: `settlrl_learn.training.elo` gains a closed-form SE; `arena`/`run_arena` feed *real* `(wins, episodes)` into the MLE instead of nominal counts. Throughput side: `self_play` reports step/discard stats, a new `training/bench.py::bench_selfplay` times warmed self-play at a fixed net, and experiment 0004 gains a `mode=bench` preset (`bench_throughput`) that loads a committed frozen checkpoint (`az0`, the June 23 overnight run's `best.eqx`) as the pinned workload. A micro pytest-benchmark suite lands in settlrl-learn. Finally a GPU baseline run is recorded in JOURNAL.md.

**Tech Stack:** JAX/Equinox, pydantic+hydra (exp 0004 conf groups), pytest + pytest-benchmark, uv workspace.

## Global Constraints

- **Git safety (parallel sessions):** NEVER run `git reset --hard`, `git checkout -- .`, or `git clean` — other sessions may hold uncommitted work. Stage and commit only the paths you touched. Do not commit or modify `.claude/settings.json`.
- **Commit per task, on `main`, hooks on.** Pre-commit runs ruff check/format, mypy over every package, and the engine test suite. Do not use `--no-verify`. End commit messages with exactly:

  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UrdWHP1eGe4c9GRCkCk3M5
  ```
- **mypy must stay green** for every touched package: `uv run --package settlrl-learn mypy packages/settlrl-learn/src packages/settlrl-learn/tests` (same pattern per package), and `./mypy_experiments.sh` when `experiments/` changes.
- **jaxtyping annotations are enforced at test time** (conftest hooks) — annotations must be exact, and shared aliases reused, not redefined.
- **Docstrings state the caller contract only** — no implementation detail, perf notes, or motivation (those go in the package `CLAUDE.md`). Update `packages/settlrl-learn/CLAUDE.md` in the same task that changes behavior it describes.
- **Tests: small budgets.** Tiny batch/sims/samples in every unit test; no shape-echo or tautology tests. Benchmarks are `benchmark`-marked and deselected from default runs.
- **Bit-exact resume is a hard invariant.** `RunState` serialization and the per-iteration RNG derivation must not change. The resume tests in `packages/settlrl-learn/tests/test_training.py` are the gate — run them after any `loop.py`/`selfplay.py` change.
- **GPU:** an RTX 5090 is available; run CUDA benchmark variants directly (`-k cuda`), skip CPU benchmark sweeps. Check `nvidia-smi` before long GPU runs in case a parallel session holds the device.
- The three files currently modified in the working tree (`.claude/settings.json` + two formatter-only diffs) belong to another workstream — leave them alone; a CI-fix session may commit them concurrently.

---

### Task 1: Fisher standard error for anchored Elo

**Files:**
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/elo.py`
- Test: `packages/settlrl-learn/tests/test_elo.py`

**Interfaces:**
- Consumes: existing `anchored_elo(anchors)` and `expected_score(rating, opponent)` in the same module.
- Produces: `anchored_elo_se(anchors: Iterable[tuple[float, float, int]], *, rating: float | None = None) -> float` — Task 2 calls it from `steps.run_arena`.

- [ ] **Step 1: Read `packages/settlrl-learn/tests/test_elo.py`** to match its test style, then write failing tests:

```python
def test_anchored_elo_se_single_even_anchor() -> None:
    # 20/40 vs a 0-Elo anchor: MLE rating 0, SE = (400/ln10)/sqrt(40*0.25) ~= 54.93
    assert anchored_elo_se([(0.0, 20.0, 40)]) == pytest.approx(54.93, abs=0.05)


def test_anchored_elo_se_saturated_anchor_adds_little() -> None:
    base = anchored_elo_se([(0.0, 20.0, 40)])
    with_saturated = anchored_elo_se([(0.0, 20.0, 40), (-800.0, 39.5, 40)])
    with_even = anchored_elo_se([(0.0, 20.0, 40), (0.0, 20.0, 40)])
    # another near-parity anchor cuts the SE ~sqrt(2)x; a saturated one barely moves it
    assert with_even < with_saturated < base


def test_anchored_elo_se_empty_is_nan() -> None:
    assert math.isnan(anchored_elo_se([]))
```

- [ ] **Step 2: Run the tests, verify they fail** (`anchored_elo_se` undefined):
  `uv run --package settlrl-learn pytest packages/settlrl-learn/tests/test_elo.py -v`

- [ ] **Step 3: Implement** in `elo.py` (add `import math` at the top):

```python
def anchored_elo_se(
    anchors: Iterable[tuple[float, float, int]], *, rating: float | None = None
) -> float:
    """Standard error of :func:`anchored_elo` from the Fisher information at the
    MLE: ``(400/ln 10) / sqrt(sum_a games_a * p_a * (1 - p_a))``, ``p_a`` the
    fitted win probability vs anchor ``a``. ``rating`` overrides the MLE fit.
    Returns ``nan`` if no anchor has games."""
    data = [(elo, w, g) for elo, w, g in anchors if g > 0]
    if not data:
        return float("nan")
    r = anchored_elo(data) if rating is None else rating
    info = sum(
        g * expected_score(r, elo) * (1.0 - expected_score(r, elo))
        for elo, _, g in data
    )
    return (400.0 / math.log(10.0)) / math.sqrt(info) if info > 0 else float("inf")
```

- [ ] **Step 4: Run the tests, verify they pass.** Also run mypy for settlrl-learn.

- [ ] **Step 5: Commit** — `git add packages/settlrl-learn/src/settlrl_learn/training/elo.py packages/settlrl-learn/tests/test_elo.py`, message `learn: Fisher SE for anchored Elo (anchored_elo_se)`.

---

### Task 2: Real game counts into the Elo MLE + `arena_elo_se` metric

The arena currently feeds *nominal* counts into the MLE: `steps.run_arena` passes `(anchor_elo, wr * cfg.games, cfg.games)` while `evaluate` overshoots `n_episodes` (its win-count sync happens between scan windows), so the true episode counts differ from `cfg.games`. Fix by returning real counts from `arena`.

**Files:**
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/arena.py`
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/steps.py` (`run_arena`, lines ~84–105)
- Modify: `experiments/0004_alphazero/run.py` (two final-arena callsites, lines ~165–169 and ~234–238)
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/__init__.py` (export `ArenaResult` if `arena` is exported there)
- Modify: `packages/settlrl-learn/CLAUDE.md` (the `training/arena.py` bullet: real counts + SE)
- Test: `packages/settlrl-learn/tests/test_training.py` (adapt existing arena tests; add the SE-metric assertion)

**Interfaces:**
- Consumes: `anchored_elo_se` from Task 1.
- Produces: `class ArenaResult(NamedTuple)` with `wins: float`, `episodes: int`, and a `winrate` property; `arena(...) -> ArenaResult`; `run_arena` metrics gain `"arena_elo_se"`.

- [ ] **Step 1: Write/adapt failing tests.** Find the existing `run_arena`/`arena` coverage in `test_training.py` and extend: with a stubbed/monkeypatched `arena` returning `ArenaResult(wins=30.0, episodes=50)` for a `lookahead` opponent with anchor 0, assert `metrics["arena_elo"]` equals `anchored_elo([(0.0, 30.0, 50)])` (i.e. real counts, not `wr * cfg.games`) and `metrics["arena_elo_se"]` equals `anchored_elo_se([(0.0, 30.0, 50)])`. Keep budgets tiny for any live-arena test already present.

- [ ] **Step 2: Run, verify failure.**

- [ ] **Step 3: Implement.** In `arena.py`:

```python
class ArenaResult(NamedTuple):
    """Seat-swapped match outcome: net wins over actual completed episodes."""

    wins: float
    episodes: int

    @property
    def winrate(self) -> float:
        return self.wins / max(self.episodes, 1)
```

and the `arena` tail becomes:

```python
    r1 = evaluate([net_spec, base], n_episodes=half, batch_size=batch_size, seed=seed)
    r2 = evaluate(
        [base, net_spec], n_episodes=half, batch_size=batch_size, seed=seed + 1
    )
    return ArenaResult(float(r1.wins[0] + r2.wins[1]), int(r1.episodes + r2.episodes))
```

In `steps.py::run_arena`, use `res = arena(...)`, record `res.winrate` in the metrics, append `(cfg.anchor_elos[opp], res.wins, res.episodes)` to `elo_inputs`, and after `metrics["arena_elo"]` add `metrics["arena_elo_se"] = anchored_elo_se(elo_inputs)` (import it beside `anchored_elo`). In `run.py`, both callsites become `winrate = arena(...).winrate`.

- [ ] **Step 4: Run the settlrl-learn suite + mypy** (learn package and `./mypy_experiments.sh` since `run.py` changed). The docstring of `run_arena` should mention the new metric key.

- [ ] **Step 5: Update `packages/settlrl-learn/CLAUDE.md`** — in the `training/arena.py` bullet, note that real `(wins, episodes)` feed the MLE (the evaluate overshoot means episodes ≠ nominal games) and that `arena_elo_se` (Fisher, `elo.py`) is reported beside `arena_elo`.

- [ ] **Step 6: Commit** — message `learn: arena returns real (wins, episodes); Elo MLE uses them + arena_elo_se`.

---

### Task 3: Commit the frozen `az0` anchor checkpoint + loader

The June 23 overnight run's best net becomes a committed artifact serving two future roles: the pinned throughput-benchmark workload (Task 5) and, later, a mid-rung gauntlet opponent. Source: `runs/0004_alphazero/2026-06-23T121706Z/best.eqx` (2.4 MB, a `BoardGNN` serialized with `eqx.tree_serialise_leaves`; net shape `gn_global` preset, `width=96, layers=4, head_depth=2` — from `conf/net/gnn.yaml` depth 2 + the `gnn_overnight` overrides).

**Files:**
- Create: `experiments/0004_alphazero/anchors/az0_gnn96x4.eqx` (byte-copy of the source `best.eqx`)
- Create: `experiments/0004_alphazero/anchors/az0_gnn96x4.json` (sidecar)
- Create: `experiments/0004_alphazero/anchors.py` (loader helper beside `run.py`)
- Test: `experiments/tests/test_smoke.py` (or a sibling test file if the conftest pattern fits better — follow `conftest.load_run`'s by-path import convention)

**Interfaces:**
- Produces: `load_anchor(name: str) -> tuple[Any, GraphNetConfig]` returning the deserialized `BoardGNN` and the net config built from the sidecar — Task 5's bench mode consumes it.

- [ ] **Step 1: Create the sidecar** `az0_gnn96x4.json`:

```json
{
  "preset": "gn_global",
  "width": 96,
  "layers": 4,
  "head_depth": 2,
  "source_run": "runs/0004_alphazero/2026-06-23T121706Z",
  "source_argv": "+experiment=gnn_overnight n_iterations=300 arena.every=10",
  "best_arena_winrate_vs_lookahead": 0.361
}
```

Verify the shape claims against the source run's `manifest.json` (merged config) before writing — if the manifest disagrees, the manifest wins.

- [ ] **Step 2: Copy the artifact** — `cp runs/0004_alphazero/2026-06-23T121706Z/best.eqx experiments/0004_alphazero/anchors/az0_gnn96x4.eqx`. Confirm `git check-ignore` does NOT match the new path (only `runs/` is ignored) and that `git add` stages the 2.4 MB binary.

- [ ] **Step 3: Write a failing test** — loading the anchor returns a net whose value output on a real position is finite. Cheap validation pattern (no search, no self-play): build a tiny env, take the backend observation, run the net forward. Read `packages/settlrl-learn/src/settlrl_learn/training/gnn_backend.py` first for the exact `GNNBackend` construction and how `init`/`seams` work; the test should be shaped like:

```python
def test_0004_anchor_loads_and_forwards() -> None:
    anchors = load_run("0004_alphazero", module="anchors")  # match conftest's import-by-path helper
    net, netcfg = anchors.load_anchor("az0_gnn96x4")
    # value seam on one fresh-board observation must be finite
    ...
```

(Adapt the import mechanics to what `experiments/tests/conftest.py` actually provides — it imports framework modules by path since the digit-prefixed dirs aren't packages. If `load_run` only loads `run.py`, extend the conftest helper minimally or import `anchors.py` by path directly in the test.)

- [ ] **Step 4: Implement `anchors.py`:**

```python
"""Frozen anchor checkpoints: committed nets with pinned provenance.

``load_anchor(name)`` rebuilds the net from ``anchors/<name>.json`` (the
architecture sidecar) and deserializes ``anchors/<name>.eqx`` into it.
"""

import json
from pathlib import Path
from typing import Any

ANCHOR_DIR = Path(__file__).parent / "anchors"


def load_anchor(name: str) -> tuple[Any, Any]:
    """The deserialized net and its ``GraphNetConfig``, from the committed
    ``anchors/`` artifact pair."""
    import equinox as eqx
    import jax
    from settlrl_learn.nn.graphnet import PRESETS
    from settlrl_learn.training import GNNBackend

    meta = json.loads((ANCHOR_DIR / f"{name}.json").read_text())
    netcfg = PRESETS[meta["preset"]]._replace(
        width=meta["width"], layers=meta["layers"], head_depth=meta["head_depth"]
    )
    template = GNNBackend(netcfg).init(jax.random.key(0))
    net = eqx.tree_deserialise_leaves(ANCHOR_DIR / f"{name}.eqx", template)
    return net, netcfg
```

Verify `GNNBackend(netcfg)` constructs with defaults for the setup knobs (check its `__init__` signature in `gnn_backend.py`); adjust to the real signature if it requires more.

- [ ] **Step 5: Run the test, mypy (`./mypy_experiments.sh`).** If JIT compile makes the test slow, mark it `slow` (CI-only) per the experiments testing convention.

- [ ] **Step 6: Commit** — message `exp 0004: commit the az0 frozen anchor (overnight best.eqx) + loader`.

---

### Task 4: Self-play stats + `bench_selfplay` in settlrl-learn

Two changes: (1) `self_play` reports what it did — env steps taken and positions discarded at exit — so throughput is measurable and the known ~72% iteration-boundary discard becomes a logged metric; (2) a `training/bench.py` module times warmed self-play at a fixed net. To avoid duplicating the subtle jit/vmap callable construction, `loop.py`'s callable-building block is extracted into a shared helper.

**Files:**
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/selfplay.py` (return stats)
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/loop.py` (extract callable construction; consume stats into metrics)
- Create: `packages/settlrl-learn/src/settlrl_learn/training/bench.py`
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/__init__.py` (export `bench_selfplay`, `SelfPlayStats`)
- Modify: `packages/settlrl-learn/CLAUDE.md` (selfplay stats + bench module bullets)
- Test: `packages/settlrl-learn/tests/test_training.py` (adapt `self_play` callers; add bench test)

**Interfaces:**
- Consumes: existing `self_play`, `Backend`, `LearnConfig`.
- Produces:
  - `class SelfPlayStats(NamedTuple): env_steps: int; recorded: int; discarded: int` (`discarded` = pending positions dropped at exit: unfinished games + over-cap trims).
  - `self_play(...) -> tuple[Samples, SelfPlayStats]` (contract change; the loop and all tests updated).
  - `selfplay_callables(backend, cfg: LearnConfig)` in `loop.py` — returns the once-built jitted callables (`view_of`, `observe_of`, `setup_search`, and a `make_net_search(num_simulations)` factory) exactly as `learn` builds them today; `learn` now calls it (pure code motion, no RNG/jit semantic change).
  - `bench_selfplay(backend, net, cfg: LearnConfig, *, warmup: int = 1, repeats: int = 3, seed: int = 0) -> dict[str, float]` in `bench.py` — keys: `samples_per_s`, `moves_per_s`, `sims_per_s`, `samples`, `env_steps`, `discarded`, `t_median_s`, and `t_0..t_{repeats-1}`. Definitions: per repeat, `moves = env_steps * cfg.selfplay.batch`; `sims_per_s = moves_per_s * cfg.search.num_simulations`; the reported per-second numbers are medians across repeats. One untimed warmup call at `seed + 1000` pays JIT compile; all timed repeats run at `seed` (identical workload). Uses `time.perf_counter` around fully-blocked calls (`self_play` returns host numpy, so no extra sync needed).

- [ ] **Step 1: Write failing tests.**
  - `self_play` (tiny mlp-backend config, batch 2, a few samples) returns `(samples, stats)` with `stats.env_steps > 0`, `stats.recorded == samples["value"].shape[0]`, `stats.discarded >= 0` — adapt every existing `self_play` call in the test file to unpack the tuple.
  - `bench_selfplay` at the same tiny config with `repeats=2, warmup=1`: result has all documented keys, `samples_per_s > 0`, `samples > 0`.

- [ ] **Step 2: Run, verify failure.**

- [ ] **Step 3: Implement `selfplay.py`.** Count `env_steps` in the existing host loop; at exit compute `discarded` as the total pending positions never flushed (sum over lanes) plus any over-`max_game_len` trims if those are counted there; return `(out_samples, SelfPlayStats(env_steps, recorded, discarded))`. Do NOT change any RNG draw, the recording logic, or the order of operations.

- [ ] **Step 4: Implement the `loop.py` extraction.** Move lines ~121–147 (the `view_of`/`observe_of`/`setup_search` builds and `_make_net_search`) into `selfplay_callables(backend, cfg)` plus the returned factory, preserving construction order exactly; `learn` consumes it. Thread the new stats into the iteration metrics: `metrics["selfplay_steps"] = float(stats.env_steps)`, `metrics["selfplay_discarded"] = float(stats.discarded)`. Note: the teacher-search build stays in `learn` (it needs `teacher_value`).

- [ ] **Step 5: Implement `bench.py`** using `selfplay_callables` + `functools.partial(net_search, net_arrays)` mirroring `learn`'s per-iteration wiring (PCR off: assert or document that `cfg.selfplay.pcr_full_prob == 1.0` for honest `sims_per_s`).

- [ ] **Step 6: Run the FULL settlrl-learn suite** — especially the bit-exact resume tests — plus mypy. The extraction must not move any test.

- [ ] **Step 7: Update `packages/settlrl-learn/CLAUDE.md`** — `self_play` bullet gains the stats contract (and that `selfplay_discarded` measures the iteration-boundary waste); new bullet for `training/bench.py` (what it measures, the warmup/repeat scheme, medians).

- [ ] **Step 8: Commit** — message `learn: self-play stats (steps/discarded) + bench_selfplay throughput probe`.

---

### Task 5: Experiment 0004 `bench_throughput` preset

**Files:**
- Modify: `experiments/0004_alphazero/run.py` (`mode` field, `BenchConfig`, `run_bench` branch)
- Modify: `experiments/0004_alphazero/conf/config.yaml` (add `mode: train` + `bench:` defaults)
- Create: `experiments/0004_alphazero/conf/experiment/bench_throughput.yaml`
- Test: `experiments/tests/test_smoke.py` (bench smoke at trivial budgets)
- Modify: `experiments/0004_alphazero/report.md` — only if it documents the variant list; do not write results here.

**Interfaces:**
- Consumes: `load_anchor` (Task 3), `bench_selfplay` (Task 4).
- Produces: `uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput` → a run dir whose `result.json` has verdict `recorded` and the throughput metrics; Task 6/7 and all future perf PRs consume this.

- [ ] **Step 1: Write the failing smoke test** following the existing smoke pattern (`compose_config` + `run_experiment` with a `Run` pointed at `tmp_path`): overrides `["+experiment=bench_throughput", "selfplay.samples=4", "selfplay.batch=2", "search.num_simulations=2", "bench.repeats=1", "bench.warmup=0"]`; assert the recorded verdict and that `samples_per_s` appears in the result. Mark `slow` if compile time dominates (per the experiments convention).

- [ ] **Step 2: Run, verify failure** (unknown `mode`/`bench` keys).

- [ ] **Step 3: Implement.** In `run.py`:

```python
class BenchConfig(_Sub):
    """Throughput-bench knobs (mode: bench): the frozen anchor + timing scheme."""

    anchor: str = "az0_gnn96x4"
    warmup: int = 1
    repeats: int = 3
```

`AlphaZeroConfig` gains `mode: Literal["train", "bench"] = "train"` and `bench: BenchConfig = Field(default_factory=BenchConfig)`. `run_experiment` branches first on `cfg.mode == "bench"` → `run_bench(run, cfg)`:

```python
def run_bench(run: Run, cfg: AlphaZeroConfig) -> None:
    """Pinned self-play throughput at a frozen net; verdict is always
    ``recorded`` -- the comparison is between two runs' result.json."""
    import jax
    from anchors import load_anchor  # by-path sibling import; match run.py's helper-import style
    from settlrl_learn.training import GNNBackend, bench_selfplay

    net, netcfg = load_anchor(cfg.bench.anchor)
    s = cfg.search
    backend = GNNBackend(
        netcfg, setup_depth=cfg.net.setup_depth,
        setup_temperature=cfg.net.setup_temperature, setup_beam=cfg.net.setup_beam,
        chance_nodes=s.chance_nodes, dev_chance=s.dev_chance, ordered=s.ordered,
    )  # fmt: skip
    results = bench_selfplay(
        backend, net, cfg.to_learn_config(),
        warmup=cfg.bench.warmup, repeats=cfg.bench.repeats, seed=cfg.seed,
    )  # fmt: skip
    run.log(**results)
    run.finish(
        "recorded", device=jax.devices()[0].device_kind, **results
    )
```

(Resolve the `anchors` import the way the framework imports its same-dir helpers — check how other frameworks/`conftest.load_run` do it; hydra runs `run.py` as `__main__` from the repo root, so a plain `from anchors import ...` may need the `sys.path`/importlib pattern already used elsewhere. Match the existing convention, don't invent one.)

`conf/experiment/bench_throughput.yaml` — the pinned config is deliberately the `gnn_overnight` training shape:

```yaml
# @package _global_
# Pinned self-play throughput benchmark: the overnight training shape at a
# frozen net (anchors/az0). Compare result.json across commits; never a gate.
defaults:
  - override /net: gnn
net:
  width: 96
  layers: 4
mode: bench
selfplay:
  samples: 16384
  batch: 256
search:
  num_simulations: 64
  max_considered: 16
  expected_rolls: false
  chance_nodes: false
wandb:
  mode: disabled
```

`conf/config.yaml` gains top-level `mode: train` and the `bench:` block with the `BenchConfig` defaults.

- [ ] **Step 4: Run the smoke test + `./mypy_experiments.sh`.**

- [ ] **Step 5: Commit** — message `exp 0004: bench_throughput mode (pinned self-play throughput at the az0 anchor)`.

---

### Task 6: settlrl-learn micro-benchmark suite

Micro-level timings of the units the optimization work will edit. Mirror the agents suite's conventions exactly (`tests/benchmark/`, `benchmark` marker, `_DEVICES` cpu/cuda sweep with cuda skip, JIT warm + `np.asarray` sync inside the timed lambda) — read `packages/settlrl-agents/tests/benchmark/test_agents_benchmark.py` and the agents `pyproject.toml` benchmark wiring first and copy the pattern.

**Files:**
- Create: `packages/settlrl-learn/tests/benchmark/test_training_benchmark.py`
- Create: `packages/settlrl-learn/tests/benchmark/README.md` (short: what's measured, how to run — structure only)
- Modify: `packages/settlrl-learn/pyproject.toml` (pytest-benchmark dev dep + marker/addopts wiring, mirroring agents)
- Modify: `run_benchmarks.sh` (add `settlrl-learn` to the package loop)

**Interfaces:**
- Consumes: `selfplay_callables` / `bench_selfplay` (Task 4), `GNNBackend`, `board_sample`/`BoardGNN` via the backend, random-init nets (the committed anchor lives in `experiments/`, out of reach of package tests — random init is fine for kernel timing; note it in the README).

Benchmarks (keep budgets small; GNN preset `gn_global`, width 96, layers 4 to match the pinned config):
1. `test_net_forward` — one jitted+vmapped `BoardGNN` forward at B=256, cpu/cuda.
2. `test_search_step` — one warmed dispatch of the vmapped net search (`selfplay_callables`' `make_net_search(64)`) on a mid-game batch, B ∈ {64, 256}, cpu/cuda. This is `search_step_ms`, the unit the parallel-descent work will move; `search_step / (64 * net_fwd)` is the batching-headroom ratio.
3. `test_selfplay_window` — `bench_selfplay` at a reduced budget (`samples=256, batch=64, repeats=1, warmup=1`), reporting via the benchmark fixture.
4. `test_optimizer_step` — one warmed `backend.make_step` dispatch on a broadcast zero batch at `batch_size=1024`.

- [ ] **Step 1: Write the suite** (benchmarks aren't TDD — correctness is "it runs and reports"; keep each body minimal).
- [ ] **Step 2: Run it on the GPU**: `uv run --package settlrl-learn pytest packages/settlrl-learn/tests/benchmark -m benchmark --benchmark-only --no-cov -n 0 -k cuda` — all four must produce numbers. Fix until green.
- [ ] **Step 3: Add the `run_benchmarks.sh` loop entry**; verify the script still runs end-to-end with `-k cuda`.
- [ ] **Step 4: mypy settlrl-learn** (benchmark tests included in the tests path).
- [ ] **Step 5: Commit** — message `learn: micro benchmark suite (net fwd / search step / selfplay window / optim step)`.

---

### Task 7: GPU baseline run + doctrine

**Files:**
- Modify: `experiments/JOURNAL.md` (one baseline line)
- Modify: `CLAUDE.md` (repo root — one sentence in the Experiments section)
- No code changes.

- [ ] **Step 1: Check the GPU is free** (`nvidia-smi` — no compute processes), then run the real benchmark:
  `uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput`
  Expect roughly 5–10 minutes (warmup + 3 timed self-play iterations; the overnight run's t_selfplay was ~84–94 s at this config). Sanity: `samples_per_s` should land within ~2x of 175 (the overnight run's implied rate); if it is wildly off, STOP and report — the harness is mismeasuring, do not record.

- [ ] **Step 2: Append the JOURNAL.md line** (match the existing entry style; cite the run dir and commit):

```
- 0004 bench_throughput baseline (2026-07-27). Pinned self-play throughput at
  the frozen az0 net (overnight shape: B=256, 64 sims): <X> samples/s
  (<Y> moves/s, <Z> sims/s), <D>% of searched positions discarded at the
  iteration boundary — the optimization track's before number (run <dir>).
```

- [ ] **Step 3: Add the doctrine line to the repo-root `CLAUDE.md`** Experiments section, alongside the strength-claims sentence: throughput claims gate through experiment 0004's `bench_throughput` preset (pinned config + frozen anchor) — quote `result.json` before/after at the same config.

- [ ] **Step 4: Commit** — message `exp 0004: record the bench_throughput baseline + throughput-claim doctrine`.

---

## Execution notes

- Tasks 1→2 are ordered; Task 3 is independent of 1–2; Task 4 is independent of 1–3; Task 5 needs 3+4; Task 6 needs 4; Task 7 needs 5 (and ideally 6). Execute sequentially in numeric order — SDD dispatches one implementer at a time anyway.
- A CI-fix session may be pushing to `main` concurrently — `git pull --rebase` before each task's commit if the push is rejected.
