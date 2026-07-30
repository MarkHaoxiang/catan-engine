# Throughput Wave 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise self-play training throughput from the recorded baseline (193 samples/s, 72.8% discard — run `2026-07-28T065337Z`) by the first three ranked levers: bigger batch (measured, ~1.6–2.3×), the persistent self-play lane pool (~3.5×, kills the iteration-boundary discard), and a temperature anneal — every claim gated through the `bench_throughput` doctrine (quote `result.json` before/after).

**Architecture:** Measurement first (a batch sweep needs zero code). Then the lane pool: `self_play` gains an opt-in *carry* — env, per-lane pending buffers, and RNG survive across calls — and `learn` threads it through iterations; the carry joins the serialized `RunState` in a fixed-shape padded form so **bit-exact resume stays a hard invariant**. Arena signal-per-second improves by config (games ≈ batch, rarer random rung). A closing GPU validation run + before/after bench lines land the wave's verdict in JOURNAL.md.

**Tech Stack:** JAX/Equinox (eqx leaf serialization needs fixed-shape templates — the reason for padded carry), flashbax replay, hydra config groups (exp 0004), pytest.

## Global Constraints

- **Git safety (parallel sessions):** NEVER `git reset --hard`, `git checkout -- .`, or `git clean`. Stage and commit only the paths you touched. Do not commit or modify `.claude/settings.json`. A parallel agent may be editing docs/config files (guidelines alignment) — `git pull --rebase` before each commit.
- **Commit per task, on `main`, hooks on** (no `--no-verify`). End commit messages with exactly:

  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UrdWHP1eGe4c9GRCkCk3M5
  ```
- **Bit-exact resume is a HARD invariant.** With the lane pool OFF, existing behavior must be bit-identical (flag-off equivalence). With it ON, an interrupted+resumed run must reproduce the uninterrupted run bit-exactly — which is exactly why the carry must serialize. The resume tests in `packages/settlrl-learn/tests/test_training.py` gate every task touching `selfplay.py`/`loop.py`/`backend.py`; run the FULL settlrl-learn suite before committing those tasks.
- **Throughput claims gate through `bench_throughput`** (repo CLAUDE.md doctrine): quote `result.json` at identical configs. The frozen baseline for the pinned config is 193.07 samples/s (`runs/0004_alphazero/2026-07-28T065337Z`).
- **mypy green** for every touched package (`uv run --package settlrl-learn mypy packages/settlrl-learn/src packages/settlrl-learn/tests`; `./mypy_experiments.sh` when `experiments/` changes). jaxtyping annotations exact; reuse shared aliases.
- **Docstrings: caller contract only.** Perf/motivation prose goes in `packages/settlrl-learn/CLAUDE.md`, updated in the same task that changes behavior it describes.
- **Tests: minimal budgets** (batch 2, few samples/sims); no shape-echo/tautology tests. The experiments suite must stay ≤2–3 minutes total.
- **GPU:** RTX 5090 on this box; check `nvidia-smi` for competing compute processes before long runs. Measurement tasks run the GPU directly.
- Opt-in discipline: new features ship behind config flags defaulting to current behavior (`persistent: false`, `temperature_moves: 0` = off), matching the repo's PCR/chance_nodes pattern.

---

### Task 1: Batch-size sweep (measurement only, no code)

**Files:**
- Modify: `experiments/JOURNAL.md` (one sweep-verdict line)

The design analysis measured per-lane search cost dropping 1123→713µs from B=256→512 on a synthetic replica; this task measures the real thing with the shipped bench.

- [ ] **Step 1:** `nvidia-smi` — GPU free (light desktop use OK). Then run three benches (each ~6–10 min):
  - `uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput` (B=256 re-run — confirms the frozen baseline reproduces, ±5%)
  - `... +experiment=bench_throughput selfplay.batch=512`
  - `... +experiment=bench_throughput selfplay.batch=1024`
  Record `samples_per_s`, `moves_per_s`, `t_median_s`, and peak GPU memory (`nvidia-smi --query-gpu=memory.used -l 5` in the background, or read after) for each.
- [ ] **Step 2:** Decide the winner: highest samples_per_s with headroom (<80% of 32GB) — expected order 1024 > 512 > 256; if 1024 OOMs or regresses, 512. If B=256 fails to reproduce the baseline within ±10%, STOP and report BLOCKED (the ruler is broken; do not proceed on a bad baseline).
- [ ] **Step 3:** Append one JOURNAL line in house style: the three samples/s numbers, the winner, run dirs. This is a *measurement* entry, not yet an adopted config (Task 6 adopts).
- [ ] **Step 4:** Commit (`exp 0004: batch sweep — bench_throughput at 256/512/1024`).

---

### Task 2: Arena signal-per-second (config + one small knob)

**Files:**
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/config.py` (`ArenaConfig`)
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/steps.py` (`run_arena`)
- Modify: `experiments/0004_alphazero/conf/arena/scale.yaml`
- Modify: `packages/settlrl-learn/CLAUDE.md` (arena bullet)
- Test: `packages/settlrl-learn/tests/test_training.py`

**Interfaces:**
- Produces: `ArenaConfig.opponent_every: dict[str, int] = {}` — opponent name → play it only every Nth arena round (absent/1 = every round). `run_arena` gains a `round_index: int` parameter (the loop passes the arena-round count, i.e. how many arena invocations have happened, derived from `(i + 1) // cfg.arena.every`); an opponent with `opponent_every[opp] = N` is skipped unless `round_index % N == 0`. Skipped opponents contribute nothing to that round's metrics or Elo inputs (the MLE handles a missing anchor fine; `arena_elo_se` reflects the round's actual information).

Rationale (measured): `arena_vs_random` has been 1.0 from iteration ~9 in every run — it burns ~half the arena wall for zero information. Games-per-eval rises to ≈ the eval batch (the evaluator's real episode count, which we now feed honestly into the MLE).

- [ ] **Step 1: Failing test** — `run_arena` with `opponent_every={"random": 5}`: at `round_index=1..4` the metrics lack `arena_vs_random` and the Elo inputs contain only the lookahead anchor; at `round_index=5` random is played. Stub `arena` as in the existing diagnostic test.
- [ ] **Step 2:** Implement (config field with `extra="forbid"` intact; thread `round_index` from `loop.py`'s arena call — compute it where the arena gate fires). Default `{}` keeps today's behavior bit-for-bit.
- [ ] **Step 3:** `conf/arena/scale.yaml`: `games: 128` (≈ arena batch — the evaluator overshoots to ~batch anyway; nominal games should match what's actually played), `opponent_every: {random: 5}`.
- [ ] **Step 4:** Full learn suite + mypy; CLAUDE.md arena bullet gains one sentence (random rung sampled every Nth round; games sized to the eval batch).
- [ ] **Step 5:** Commit (`learn: arena opponent_every scheduling + games sized to the eval window`).

---

### Task 3: Persistent lane pool A — the self-play carry

**Files:**
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/selfplay.py`
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/config.py` (`SelfPlayConfig.persistent: bool = False`)
- Modify: `packages/settlrl-learn/CLAUDE.md`
- Test: `packages/settlrl-learn/tests/test_training.py`

**Interfaces (pinned design — deviations need controller sign-off):**
- `class SelfPlayCarry` — a pytree (NamedTuple of arrays + host lists is fine at this layer; Task 4 owns the fixed-shape serialized form) holding: the env (or its full array state, whichever `BatchedSettlrlEnv` supports re-entering exactly — read `env/batched.py` first; if the env object itself can simply be held and re-stepped, hold it), the per-lane `pending` buffers, the RNG `key`, and any loop counters `self_play` needs to continue mid-game.
- `run_selfplay`/`self_play` signature grows `carry: SelfPlayCarry | None = None` and returns `(Samples, SelfPlayStats, SelfPlayCarry | None)`. `carry=None` + `persistent=False` → exactly today's behavior (fresh env, discard at exit, returned carry `None`). `persistent=True` → first call builds fresh state but RETURNS the carry instead of discarding; subsequent calls resume from it; the collection loop's exit condition becomes "≥ n_samples flushed" (games in flight stay in the carry, `discarded` counts only `max_game_len` trims).
- RNG: in persistent mode the key lives in the carry (split-and-carry each step); the `seed` argument seeds only the *initial* carry. Document this in the docstring — it changes the "per-iteration RNG is pure in (seed, i)" property into "pure in (seed, carried state)", which is exactly why Task 4 must serialize the carry.

- [ ] **Step 1: Failing tests** (tiny mlp config, batch 2):
  - *Flag-off equivalence:* `persistent=False` output (samples + stats) is bit-identical to the pre-change function on the same seed (golden: capture expected values from the current code BEFORE editing, hardcode them or compute via the old path in the test setup).
  - *Continuity:* two persistent calls of `n_samples=N` each, carry threaded, produce combined samples equal to one `2N` call's on the same initial seed (same env stepping order ⇒ identical positions; assert on the concatenated `value`/`policy` arrays).
  - *Discard collapse:* in persistent mode, `stats.discarded` counts only trims (assert 0 for a config whose games can't hit the cap).
- [ ] **Step 2:** Implement. The existing loop body is the hard part — restructure minimally: extract the per-step body if needed but do NOT reorder any RNG draw or recording operation on the `persistent=False` path.
- [ ] **Step 3:** FULL learn suite (resume tests especially — they run the flag-off path) + mypy.
- [ ] **Step 4:** CLAUDE.md: `self_play` bullet gains the carry contract + the RNG-property change.
- [ ] **Step 5:** Commit (`learn: opt-in persistent self-play carry (lane pool part A)`).

---

### Task 4: Persistent lane pool B — RunState serialization + loop integration

**Files:**
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/backend.py` (`RunState`)
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/loop.py` (`learn` threads the carry; metrics)
- Modify: `packages/settlrl-learn/CLAUDE.md`
- Test: `packages/settlrl-learn/tests/test_training.py`

**Interfaces (pinned design):**
- The serialized form is FIXED-SHAPE (eqx template requirement): pending buffers pad to `(batch, max_game_len)` per recorded field plus a per-lane `pending_len` int array; the env's array state is already fixed-shape; the key is an array. A `SelfPlayCarry <-> padded pytree` conversion pair lives beside the carry (`to_padded(carry, max_game_len)` / `from_padded(...)`). Checkpoint-size note: at scale (B=256, cap 800, GNN obs + 662-wide policy) the pad adds roughly 1–2 GB to `runstate.eqx`. That is accepted for now. IF the implementation lands >3 GB of pad or serialization takes >30s, STOP and report BLOCKED — the fallback (relaxing bit-exact resume for the carry) is the controller's/user's decision, not yours.
- `RunState` grows `selfplay_carry: Any` — the padded pytree, or a same-structure zero pytree when `persistent=False` (the template must have one fixed structure regardless of flag, or old checkpoints break: verify `load_run_state` of a PRE-change checkpoint still works — if the NamedTuple growth breaks old templates, provide a migration read path or a versioned loader; test this explicitly with a checkpoint written by the old code).
- `learn`: when `cfg.selfplay.persistent`, hold the carry across iterations, pass/receive it through `run_selfplay`, fold it into every checkpoint write, and restore it on resume. Metrics unchanged in names (`selfplay_discarded` now naturally ~0).

- [ ] **Step 1: Failing tests:**
  - *Bit-exact resume WITH persistence:* run N iterations persistent, checkpoint at k<N, resume from k, assert final net leaves bit-identical to the uninterrupted run (mirror the existing resume test's structure).
  - *Old-checkpoint compatibility:* a `runstate.eqx` written by pre-change code (generate it in-test by serializing the old-shape NamedTuple, or check in a tiny fixture) still loads with `persistent=False`.
  - *Round-trip:* `from_padded(to_padded(carry)) == carry` on a mid-game carry.
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** FULL learn suite + mypy; measure and report (in your task report) the actual pad size and serialization time at the scale config's shapes (a quick script is fine — scratchpad, not the repo).
- [ ] **Step 4:** CLAUDE.md: RunState bullet + the checkpoint-size note.
- [ ] **Step 5:** Commit (`learn: carry joins RunState — persistent self-play resumes bit-exactly (lane pool part B)`).

---

### Task 5: Opening-temperature anneal

**Files:**
- Modify: `packages/settlrl-learn/src/settlrl_learn/training/selfplay.py`, `config.py` (`SelfPlayConfig.temperature_moves: int = 0`)
- Modify: `packages/settlrl-learn/CLAUDE.md`
- Test: `packages/settlrl-learn/tests/test_training.py`

`temperature_moves = K > 0`: a lane samples at `temperature` for its first K *recorded moves of the current game*, then argmax (temperature 0) for the rest; K=0 keeps today's flat temperature (bit-exact flag-off). Per-lane move counts already effectively exist (pending lengths track recorded positions per lane — verify; else add a per-lane counter that resets with the lane). Works in both persistent and fresh modes.

- [ ] **Step 1: Failing test** — flag-off equivalence (K=0 bit-identical), and a behavioral test: with K=1 and a fixed seed, moves after the first recorded move per game are argmax (assert via a crafted small case or by checking determinism across two different sampling keys past move K).
- [ ] **Step 2:** Implement (the anneal decision is per-lane, so it's a `jnp.where` on the vmapped sampling path — keep `_sample_moves` pure).
- [ ] **Step 3:** Full suite + mypy; CLAUDE.md selfplay bullet.
- [ ] **Step 4:** Commit (`learn: opening-temperature anneal (temperature_moves)`).

---

### Task 6: Adopt + validate — the wave's verdict

**Files:**
- Create: `experiments/0004_alphazero/conf/experiment/scale2.yaml` (the adopted training preset)
- Modify: `experiments/JOURNAL.md`, `experiments/0004_alphazero/report.md` (a short adopted-config section)
- No library code.

- [ ] **Step 0 (added after Task 1's finding):** Task 1 measured batch scaling REGRESSING without persistence (193.8/166.4/120.2 samples/s at 256/512/1024) because discard rises with batch (72.7%→91.0%) — the batch lever is gated behind the lane pool. Re-run the sweep WITH persistence: `+experiment=bench_throughput selfplay.persistent=true selfplay.batch={256,512,1024}`; the batch winner is chosen from THESE numbers (expect the per-lane search-cost advantage to surface once discard ≈ 0).
- [ ] **Step 1:** Write `scale2.yaml` — the `gnn_overnight`/`small` shape with the wave's levers on: `selfplay.batch` = Step 0's winner, `selfplay.persistent: true`, `temperature_moves: 30` (opening ≈ setup+first builds; note it as a starting value, not tuned), PCR on (`pcr_full_prob: 0.25`, `pcr_fast_sims: 16`, `search.num_simulations: 128` — machinery already shipped, config-only), arena from Task 2's scale.yaml. Compose-validate it (`compose_config`).
- [ ] **Step 2:** Before/after bench, both at the SAME pinned search/net shape:
  - `+experiment=bench_throughput selfplay.batch=<winner> selfplay.persistent=true` vs the frozen 193 baseline (and vs Task 1's batch-only number — isolating the lane pool's contribution). Bench mode with PCR stays off (the bench guard requires it); sims stay 64 — PCR's effect is a training-quality lever, not a bench-comparable one; say so in the JOURNAL line.
- [ ] **Step 3:** Validation training run (~30–45 min GPU): `+experiment=scale2 n_iterations=8 arena.every=4 wandb.mode=disabled` (or the minimal override set that keeps it short). Verify from metrics.jsonl: `selfplay_discarded` ≈ 0 from iteration 2 on; `samples`/`t_selfplay` consistent with the bench number; loss/val metrics sane (no NaNs, entropy not collapsed); checkpoint+resume smoke — kill after iter ~4's checkpoint, resume, confirm it continues without error (bit-exactness is already unit-tested; this is the integration smoke).
- [ ] **Step 4:** JOURNAL verdict line: baseline 193 → batch-only X → +lane-pool Y samples/s (Nx compounded), discard 72.8% → ~0, run dirs. `report.md` gains the adopted `scale2` paragraph.
- [ ] **Step 5:** Commit (`exp 0004: scale2 adopted config + wave-1 throughput verdict`).

---

## Execution notes

- Order: 1 → 2 → 3 → 4 → 5 → 6. Task 1 is pure measurement and can run while Task 2 is implemented if convenient, but SDD runs sequentially — fine.
- Task 3 and 4 are the risk concentration: both carry pinned design decisions and explicit BLOCKED tripwires (env re-entry shape, checkpoint growth, old-checkpoint compat). An implementer hitting any tripwire stops rather than improvising.
- Parallel-descent search work (K-way SH blocks, virtual loss) is explicitly OUT of this wave — it lands only after a scale2-based long run re-tests the plateau.
