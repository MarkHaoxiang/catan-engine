# GPU-day checklist (exp 0004)

Perf/strength items that need real GPU time, promoted out of the
featurization-v2 plan (`docs/superpowers/plans/2026-07-30-featurization-v2.md`,
Task 5) once the live training run releases the card. Each item names the
exact command(s) and the decision it feeds — run in order, stop and record if
a gate fails, don't chain items past a failed decision without updating this
file.

**Closed, not on this list**: the fused-leaf seam-fix confirmation
(`search_step_ms` / `bench_throughput` before-vs-after) already ran and
resolved GPU-neutral — `experiments/JOURNAL.md` (the seam-fix GPU-neutral
entry). Don't re-run
it; a regression there would show up as a `bench_throughput` gate failure on
any other item below and should be investigated as a regression, not
re-litigated as this checklist item.

## 1. Featurizer-DFS GPU pricing

The Longest-Road DFS (`settlrl_engine`'s `awards.road_build_gate`, gated at
≥5 own roads) was priced at 4-8% of a net forward pass **on CPU**
(`packages/settlrl-learn/CLAUDE.md`). That number doesn't close the question
of whether v2's decision to read the award holder's *stored* length instead
of recomputing per-player (a structural choice, not a cost dodge) is even
relevant on the hardware that matters.

```bash
./run_benchmarks.sh -k cuda
```

Compare `settlrl-engine`'s env-step benchmark (includes the DFS whenever a
step crosses the 5-road gate) against `settlrl-learn`'s net-forward benchmark
(`tests/benchmark/test_training_benchmark.py`), both GPU. **Decision:** if the
GPU ratio is still single-digit percent of a forward, the CPU number was
representative and nothing changes; if it's materially higher (GPU forwards
are cheaper than CPU ones, so the *ratio* could easily be worse even with the
DFS itself unchanged), the v2 stored-length design earns a line in
`packages/settlrl-learn/CLAUDE.md` citing the GPU number instead of the CPU
one, and the DFS becomes a candidate for its own optimization pass.

## 2. Layout-transpose HLO check

Deferred at throughput-wave-1 behind GPU verification (`docs/superpowers/plans/
2026-07-30-featurization-v2.md`'s "Explicitly deferred" list). Reuse the dot-census
pattern `packages/settlrl-learn/tests/test_leaf_seam.py` already has for
counting trunk forwards, but grep for `transpose`/`copy` (layout-changing ops)
instead of `dot`:

```python
compiled = jax.jit(fn).lower(*args).compile()
text = compiled.as_text()
# grep for r'transpose\(' / r'copy\(' with op_name, same regex shape as
# test_leaf_seam.py's dot census
```

Run against `gnn_seams`'s forward on a GPU backend (`JAX_PLATFORMS=cuda` or
just run where CUDA is the default device). **Decision:** if XLA already
folds the transposes into adjacent ops on GPU (the same "CPU-specific"
resolution the seam fix hit — GPU CSE/layout-assignment is more aggressive
than CPU's), close this the same way: adopt-if-cheap-anyway, no fix. If a
real transpose survives in the hot path, it becomes a scoped fix, not part of
this checklist.

## 3. bf16 probe

```bash
uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput
```
as the f32 baseline, then the same command against a bf16-cast copy of the
net's inexact-array leaves (`jax.tree.map` over `eqx.is_inexact_array`,
params only — keep the environment/search state at their native dtypes).
Check `result.json`'s `samples_per_s` and scan for NaN in a short
`n_iterations` train smoke at bf16.

**Decision:** adopt bf16 for the four-arm study's training path only if (a)
throughput clears the same 10% sanity gate the seam-fix confirmation used,
and (b) no NaN/precision regression shows up in a short train smoke. A win
under 10% is noise per that established gate, not a decision either way.

## 4. Width sweep {96, 144, 192}, re-priced at the confirmed ~15% NN share

```bash
uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput net.width=96
uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput net.width=144
uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput net.width=192
```

The plan draft priced this sweep against a **predicted** post-seam-fix NN
share of ~26% of a sim ("not 15%"). That prediction is now known to be wrong
— the seam-fix GPU confirmation bench measured +0.6% (noise, under the 10%
gate), so the NN share never moved off its pre-fix reading of ~15%. Re-run
the sweep's arithmetic against the **confirmed ~15%** share, not the
~26% the plan drafted before the confirmation bench existed.

**Decision:** pick a width for the four-arm study's shared recipe (currently
96 in every `v2_*` preset) only if a wider net's `samples_per_s` cost, at the
15%-share pricing, still leaves total iteration wall-clock within the
`scale2`-derived budget the study presets inherit. Otherwise 96 stays.

## 5. Four-arm study: paired vs-v2_base head-to-heads (optional follow-up)

All four arms (`v2_base`, `v2_incidence`, `v2_deep`, `v2_hetero`) trained
and were judged by independent 400-game final gauntlets against the shared
anchor set (`experiments/JOURNAL.md`, 2026-07-30 entry: `v2_hetero` won,
+76 ± 10 over `v2_base`, sole gate pass). That gauntlet verdict stands;
`v2_incidence`'s config comment records the interpretation note for its
underperformance (the slot-rank input geometry, not a verdict on per-tile
identity).

What remains optional: **paired** per-arm head-to-heads directly against
`v2_base`'s checkpoint — same seeds both sides, seat-swapped
(`packages/settlrl-learn/CLAUDE.md`'s paired-seed doctrine) — as a
variance-cut confirmation of the gauntlet deltas. The tooling exists:
`experiments/0004_alphazero/match.py` runs checkpoint-vs-checkpoint matches
(loading via `anchors.py::load_anchor`, both sides under the same search
budget). Run it only if a gauntlet delta needs a tighter error bar; record
any result in `experiments/0004_alphazero/report.md` per repo doctrine
(strength claims gate through a recorded match).

## 6. 512-vs-1024 batch re-measurement (the sample-yield wobble note)

```bash
uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput selfplay.persistent=true selfplay.batch=512
uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput selfplay.persistent=true selfplay.batch=1024
```

The original persistent-batch sweep (`experiments/JOURNAL.md`, 2026-07-28)
picked 512 over 1024 (922.53 vs 834.76 samples/s). `training/bench.py`'s own
docstring notes that under `persistent`, repeats are **sequential
continuations of one pool**, not repetitions of an identical workload — so a
single sweep's samples/s carries pool-phase noise (the "sample-yield wobble")
that a one-shot before/after comparison doesn't average out. Re-run at more
`repeats` than the original sweep used, both PCR and the seam fix now on by
default in `scale2`/`v2_base` (neither was active for the original sweep).

**Decision:** confirm 512 remains the winner under today's full recipe
(seam-fix + PCR both live); if 1024 closes the gap or overtakes, it becomes a
candidate batch bump for `v2_base` and every `v2_*` preset, not adopted
silently.

## 7. Arena-cadence retune — measured, rotation landed

Measured (2026-07-31 hetero run, `metrics.jsonl` `t_arena` over wall-clock):
the in-loop arena was **24.9%** of the run's wall — **mean 870 s/round**, 16
rounds — well past scale2_long's 9.8% budget and its earlier 16.1% reading
(`runs/0004_alphazero/2026-07-29T122854Z`, first 1719 iterations; that run
carried only the az0 rung — the hetero run's three net rungs at ~280 s/match
each, all firing every round, are the gap).

**Landed fix (2026-08-01):** the frozen net rungs rotate instead of stacking —
`ArenaNetOpponent.phase` (predicate `(round_index + phase) % every == 0`), with
`conf/arena/scale.yaml` at az0/az1/az2 `every: 3`, phases 0/1/2, so ~1 net rung
fires per round (lookahead every round, random every 5). Expected ~550-650
s/round vs the 870 s pre-rotation mean. The in-loop Elo pools 3 anchors/round
instead of 5 (SE ~20 → ~27); the final gauntlet neutralizes every schedule, so
the gate is untouched. If the realized share still overshoots the 9.8% budget,
halving `games` (64 → 32) is the next-cheapest lever; record any further retune
and the before/after realized share in `experiments/JOURNAL.md`.
