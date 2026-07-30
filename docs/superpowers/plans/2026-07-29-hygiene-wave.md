# Hygiene Wave Implementation Plan (cleanup, structure, small fixes)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pay down the structural debt the two feature waves accumulated: `selfplay.py`/`loop.py` grew multiple responsibilities, `learn()` re-traces its jit closures every call (a measured 68s startup spike + the test-suite floor), exp 0004's `run.py` has become a grab-bag, and a handful of cross-wave loose ends (wandb-id persistence, mirrored constants, stale cruft) are unowned.

**Architecture:** Three tasks, STRICTLY SEQUENTIAL (one implementer at a time — the parallel-commit races are documented): (1) settlrl-learn structure + the closure cache (the only behavior-adjacent change, gated by the bit-exact suite); (2) exp 0004 decomposition + small fixes; (3) repo-wide sweep (cruft, dead code, doc tightening).

## Global Constraints

- NEVER `git reset --hard` / `git checkout -- .` / `git clean`; explicit-pathspec commits; `.claude/settings.json` untouched; hooks on (no --no-verify); `git pull --rebase` on rejected push; FOREGROUND commands only.
- **Bit-exact resume + flag-off equivalence remain hard invariants.** Any `training/` change runs the FULL settlrl-learn suite. Pure code motion must be verifiably pure (same construction order, no RNG changes).
- mypy green per touched package + `./mypy_experiments.sh`; ruff clean. Docstrings contract-only; CLAUDE.md updated in the same task; docs concise, structure-only in READMEs.
- Commit messages end with:

  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UrdWHP1eGe4c9GRCkCk3M5
  ```

---

### Task 1: settlrl-learn structure + the closure cache

**Files:**
- Create: `packages/settlrl-learn/src/settlrl_learn/training/carry.py` (move `SelfPlayCarry`, `PaddedCarry`, `to_padded`/`from_padded`/`empty_padded`/`carry_template`/`recorded_spec` + their private helpers out of `selfplay.py`/`loop.py`; `selfplay.py` keeps `self_play` + sampling; no public-name changes — re-export from the old locations if anything external imports them, check `training/__init__.py` and tests)
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/loop.py` — **cache `selfplay_callables` on the backend/config identity across `learn()` calls** (module-level or backend-attached memo keyed by `(id(backend), the search-affecting config subset)`; the goal: a second `learn()` call with the same backend + same shapes reuses the jitted callables instead of re-tracing). Design constraint: the cache must not capture net *arrays* (they're traced args) and must not break bit-exactness — the callables are pure functions of static shape/config, so reuse is semantically free; prove it with the resume tests plus one new test (two sequential `learn()` calls, second measurably reuses — assert via a trace-counter hook or `._cache_size`-style introspection, not wall-clock).
- Modify: `packages/settlrl-learn/CLAUDE.md` (module map + the re-trace note in the test-hygiene paragraph becomes "fixed by the callable cache").
- Test: `packages/settlrl-learn/tests/test_training.py` — SPLIT this ~1300-line file by area into `test_selfplay.py`, `test_carry.py`, `test_arena_elo.py`, `test_bench.py`, keeping `test_training.py` for the learn-loop/resume family. Pure moves, no assertion changes; update the CI/pre-commit mypy file lists (`.github/workflows/ci.yml`, `.pre-commit-config.yaml`) that pin test files by name.

Steps: baseline suite green → carry.py extraction (pure motion; full suite) → closure cache (TDD with the reuse test; full suite incl. resume; measure the startup-spike improvement with a quick two-call timing in the report) → test split (collect counts identical before/after) → docs → commit (`learn: carry module + cached selfplay callables + test split`).

### Task 2: exp 0004 decomposition + small fixes

**Files:**
- Create: `experiments/0004_alphazero/arena_helpers.py` (or similarly named — move `build_net_opponents`, `run_final_gauntlet`, `gauntlet_verdict`, `BenchConfig`/bench glue out of `run.py`; `run.py` keeps the config schema + the two run paths + main; same-dir import per the established convention)
- Modify: `experiments/0004_alphazero/run.py` — gnn path writes `wandb_id.txt` and honors it on resume (mirror the mlp path's existing pattern).
- Fix the mirrored az0 setup constants: move `NET_OPPONENT_SETUP_*` into `experiments/0004_alphazero/anchors.py` (uniquely-named module, importable from 0001's calibrate.py via the sys.path convention — VERIFY that works at runtime for `experiments/0001_bench_smoke/run.py calibrate`, not just under pytest; if runtime import fails, keep the mirror but add a test pinning the two constant sets equal).
- Test: composition smokes stay green; add the constants-equality test if the mirror survives.

Steps: baseline → moves (pure) → wandb-id fix (+ a tiny test if cheap) → constants dedup → `./mypy_experiments.sh` + experiments suite → commit (`exp 0004: decompose run.py + wandb resume id + shared az0 constants`).

### Task 3: repo-wide sweep

- Delete `packages/settlrl-learn/tests/test_training.py.bak` (June cruft).
- Move the `OpponentSpec` union to its natural home (`settlrl_search.policy` beside the spec classes), re-export from `settlrl_learn.training` for compatibility, collapse the restatements in `settlrl_agents/__init__.py`/`cli.py`/`evaluate.py` onto it (type-alias only — zero runtime change; full agents+search+learn mypy).
- Grep for dead code left by the waves (old `Spec` aliases, unused imports, `_SELFPLAY_N_PLAYERS`-style constants that are now dead, commented-out blocks) — delete what's provably unreferenced.
- CLAUDE.md tightening pass for `packages/settlrl-learn/CLAUDE.md` ONLY (it has grown the most): collapse duplicated rationale, keep every load-bearing fact, target ~25% shorter without losing content (this is prose judgment — prefer cutting repetition over cutting facts).
- Commit (`chore: hygiene sweep — cruft, OpponentSpec home, dead code, CLAUDE.md tightening`).

## Execution notes

Sequential; each task gets a scoped review (Task 1 on the strong model — the closure cache is behavior-adjacent). No parallel committers.
