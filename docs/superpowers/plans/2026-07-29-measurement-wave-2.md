# Measurement Wave 2 Implementation Plan (gauntlet + test hygiene)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the strength-measurement design (frozen-checkpoint gauntlet rung, calibrated anchors, an Elo-based gate replacing the single-opponent 0.55 win rate) and pay down the test-suite runtime debt in settlrl-learn — so the upcoming `scale2` long runs are judged on a gauntlet that can actually resolve progress.

**Architecture:** Arena opponents generalize from `POLICIES` names to also accept frozen checkpoints (loaded via the 0004 anchor machinery, played through `backend.play_agent`). A one-off calibration round-robin among the fixed rungs replaces the internally-inconsistent guessed anchors (random is ~−1075, not −800). The experiment gate becomes `arena_elo − 2·arena_elo_se ≥ gate_elo` over a heavyweight final gauntlet. Separately, settlrl-learn's tests get the persistent XLA compile cache that made the experiments suite cheap, plus a budget/dedupe pass.

**Tech Stack:** JAX/Equinox, the existing `anchored_elo`/`anchored_elo_se` MLE machinery, hydra conf groups, pytest.

## Global Constraints

- **Git safety:** NEVER `git reset --hard` / `git checkout -- .` / `git clean`. Commit ONLY files you touched, with explicit pathspecs (`git commit -- <paths>`), and verify `git show --stat` afterward. Do not touch `.claude/settings.json`. Run long commands in the FOREGROUND with adequate timeouts — never background-and-wait on a notification.
- **Commit per task on `main`, hooks on** (no `--no-verify`), `git pull --rebase` on rejected push. End commit messages with exactly:

  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UrdWHP1eGe4c9GRCkCk3M5
  ```
- **mypy green** for touched packages (`uv run --package settlrl-learn mypy packages/settlrl-learn/src packages/settlrl-learn/tests`; `./mypy_experiments.sh` for experiments changes). jaxtyping exact; shared aliases.
- **Docstrings contract-only**; rationale in the per-package `CLAUDE.md`, updated in the same task.
- **Tests minimal**; the experiments suite stays ≤2–3 min total; never weaken a load-bearing assertion — cut budgets, dedupe, cache compiles.
- **Anchors must stay frozen within a run**; changing anchor Elos rescales history — the calibration task therefore lands as ONE atomic config change with a JOURNAL note (the scale reset), and nothing else may adjust anchors afterward.
- **GPU:** check `nvidia-smi` before long runs. Strength numbers are measured (never asserted from reasoning); every match count and win rate quoted in JOURNAL comes from a run dir.
- Bit-exact resume and flag-off equivalence remain hard invariants for any `training/` change.

---

### Task 1: settlrl-learn test hygiene — persistent XLA cache + budget pass

**Files:**
- Modify: `packages/settlrl-learn/tests/conftest.py`
- Modify: `packages/settlrl-learn/tests/test_training.py` (budgets/dedupe only)
- Possibly modify: `packages/settlrl-learn/pyproject.toml` (if cache env wiring lives there)

The learn suite (~72 tests) is dominated by XLA compiles: every resume/carry test runs miniature training loops that recompile the same tiny programs per process. The experiments suite solved this with a persistent compile cache (`~/.cache/jax-settlrl`, see `experiments/tests/conftest.py`) — port that pattern.

- [ ] **Step 1:** Baseline: `uv run --package settlrl-learn pytest packages/settlrl-learn/tests -q --durations=15` (record total + tail; run twice to see warm-process vs cold).
- [ ] **Step 2:** Port the persistent-cache conftest wiring from `experiments/tests/conftest.py` (same cache dir — shared compiles across suites are a feature). Verify a second run is substantially faster and REMAINS CORRECT (the cache must not break bit-exact resume tests — they compare within-process results, so caching is safe, but verify green).
- [ ] **Step 3:** Budget/dedupe pass, cuts only where a property survives: overlapping resume tests (flag-off, persistent, flip×2, old-checkpoint, zero-sample — each exists for a distinct property; do NOT merge away distinctions, but shrink any that runs more iterations/samples than its property needs — most need n_iterations 2-4, not more); duplicate carry round-trips; parametrizations that re-compile without new coverage. Consider `xdist_group` pinning if module fixtures recompute across workers (the engine suite's proven trick).
- [ ] **Step 4:** After: rerun with durations; target ≥50% cold-cut and a warm suite ≤ ~90s. Full suite green + mypy.
- [ ] **Step 5:** Commit (`learn: test hygiene — persistent XLA cache + budget pass — <before>s -> <after>s`).

---

### Task 2: Checkpoint opponents in the arena (the az0 rung)

**Files:**
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/config.py` (`ArenaConfig.net_opponents: dict[str, tuple[float, int]]` — name → (anchor_elo, every); or a small pydantic sub-model — pick the cleanest that keeps `extra="forbid"`)
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/arena.py` (accept a pre-built opponent `BeliefSpec`, not just a `POLICIES` name)
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/steps.py` (`run_arena` plays net opponents; their Elo joins `elo_inputs`)
- Modify: `experiments/0004_alphazero/run.py` + `experiments/0004_alphazero/anchors.py` (build the az0 opponent spec from `load_anchor` + `GNNBackend.play_agent`, hand it to the loop — the LIBRARY stays anchor-file-agnostic: the experiment composes the loaded net into a spec and passes ready specs in; design the seam accordingly, e.g. `run_arena(..., net_opponents: dict[str, tuple[BeliefSpec, float, int]])` threaded from `learn(...)` via a new optional argument)
- Modify: `experiments/0004_alphazero/conf/arena/scale.yaml` (az0 rung, provisional Elo −100, every: 1)
- Modify: `packages/settlrl-learn/CLAUDE.md`
- Test: `packages/settlrl-learn/tests/test_training.py`

Interfaces: keep `arena()`'s seat-swap/ArenaResult contract; a net opponent is just another `BeliefSpec` at the table. The az0 provisional anchor is −100 (from its measured 0.361 vs lookahead ⇒ ≈−99); Task 4 calibrates properly. `opponent_every` semantics apply to net opponents too.

- [ ] **Step 1: Failing tests** — stubbed: `run_arena` with one net opponent produces `arena_vs_<name>` and its `(elo, wins, episodes)` joins the MLE; `opponent_every` skipping applies.
- [ ] **Step 2:** Implement library side (arena/steps/config + the `learn` pass-through).
- [ ] **Step 3:** Experiment side: `run.py` builds the az0 spec when the conf names it (sims/considered = arena settings; the same setup-delegation the net player uses). A tiny composition smoke only if free (suite budget!).
- [ ] **Step 4:** Full learn suite + both mypy gates; CLAUDE.md arena bullet.
- [ ] **Step 5:** Commit (`learn+exp0004: frozen-checkpoint arena opponents (az0 mid-rung)`).

---

### Task 3: Elo gate for the experiment verdict

**Files:**
- Modify: `experiments/0004_alphazero/run.py` (final verdict: heavyweight gauntlet → `arena_elo − 2·arena_elo_se ≥ gate_elo`)
- Modify: `experiments/0004_alphazero/conf/config.yaml` (+`gate_elo: 35.0`; keep `gate_winrate` reported-not-gating for one release — the result.json carries both)
- Test: composition-level only.

- [ ] **Step 1:** Implement: the end-of-run arena becomes a gauntlet over the configured opponents (incl. az0) at `games: 400` per informative rung (config override at the final call, not a new config group — a `final_games` field on the experiment schema), producing `arena_elo`/`arena_elo_se`; verdict `pass iff elo − 2·se ≥ gate_elo`. `result.json` records elo, se, per-rung rates, and the legacy winrate.
- [ ] **Step 2:** `./mypy_experiments.sh` + the experiments suite (stays in budget — the final gauntlet only runs in real runs, not smokes; smokes keep tiny games).
- [ ] **Step 3:** Commit (`exp 0004: Elo gate (elo − 2·se ≥ 35) over the final gauntlet`).

---

### Task 4: Anchor calibration — the one-off scale reset (GPU)

**Files:**
- Modify: `experiments/0001_bench_smoke/run.py` (a `calibrate` variant: round-robin among {random, greedy, lookahead, mcts} + az0 via `settlrl_agents` evaluate/bench machinery, joint MLE fit)
- Create: same-dir helper if needed (`calibrate.py` beside run.py)
- Modify: `experiments/0004_alphazero/conf/arena/scale.yaml` + `default.yaml` (the calibrated `anchor_elos` — ONE atomic change)
- Modify: `experiments/JOURNAL.md` (the scale-reset entry)

Budgets: heuristic-vs-heuristic pairs are cheap (n=600+/pair); pairs involving mcts or az0 pay search cost (n=300–400/pair). The joint fit: hold `lookahead = 0` fixed, coordinate-ascent the logistic MLE over the others (~30 lines; reuse `expected_score`). Report SEs per rung (Fisher, same formula).

- [ ] **Step 1:** Implement the variant + fit (unit-test the fit on synthetic win matrices — exact recovery of known Elos within tolerance).
- [ ] **Step 2:** GPU run (foreground, chunked into per-pair invocations if any single command would exceed a 10-min timeout; a couple of hours total is acceptable). Record the matrix + fitted Elos + SEs in the run dir.
- [ ] **Step 3:** Sanity: lookahead ≈ 0 by construction; greedy expected ≈ −330; random ≈ −1000±; az0 between greedy and lookahead. If the fit is wildly outside expectations (random > −600 or az0 > 0), STOP — report BLOCKED with the matrix.
- [ ] **Step 4:** The atomic config change + JOURNAL entry (old→new anchors, n per pair, the caveat that all historical arena_elo values shift scale).
- [ ] **Step 5:** Commit (`exp 0001+0004: calibrated anchor Elos (one-off scale reset)`).

---

### Task 5: Backlog polish (one commit)

**Files:** as listed per item.

- [ ] `experiments/0004_alphazero/conf/net/gnn_hetero.yaml` (+ a `gnn_hetero` experiment variant delta over the current best recipe) — `gn_hetero` finally reachable by name. Compose-validate.
- [ ] `packages/settlrl-learn/src/settlrl_learn/training/loop.py`: `recorded_spec` helper collapsing `carry_template`'s derived-key list with selfplay's `_DERIVED_KEYS` (Task-4-wave deferred minor).
- [ ] `packages/settlrl-learn/src/settlrl_learn/training/selfplay.py`: the anneal's argmax branch calls `_sample_moves(..., temperature=0.0)` instead of duplicating the expression (Task-5-wave minor).
- [ ] `packages/settlrl-learn/src/settlrl_learn/training/config.py`: fold `opponent_every`'s attribute-docstring into the class docstring (style consistency).
- [ ] Full learn suite + mypy + `./mypy_experiments.sh`; commit (`polish: wave-2 backlog (gn_hetero conf, recorded_spec, sampling dedup)`).

---

## Execution notes

- Order: 1 → 2 → 3 → 5 → 4 (the calibration GPU run last — it consumes the az0 rung from Task 2 and its config landing is atomic; Tasks 3/5 don't depend on it since the gate code is anchor-agnostic).
- SEQUENTIAL implementers only; commits serialized (the parallel-commit races are documented — never run two committing agents at once on this tree).
- This wave does NOT start a scale2 long training run — that's a science decision for Mark in the morning, on top of the calibrated gauntlet.
