# GPU-day checklist (exp 0004)

Perf/strength items that need real GPU time, promoted out of the
featurization-v2 plan (`docs/superpowers/plans/2026-07-30-featurization-v2.md`,
Task 5) once the live training run releases the card. Each item names the
exact command(s) and the decision it feeds — run in order, stop and record if
a gate fails, don't chain items past a failed decision without updating this
file.

**Closed, not on this list**: the fused-leaf seam-fix confirmation
(`search_step_ms` / `bench_throughput` before-vs-after) already ran and
resolved GPU-neutral — `.superpowers/sdd/gnn-optimization-notes.md`
("2026-07-30 — GPU confirmation") and `experiments/JOURNAL.md`. Don't re-run
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

## 5. Four-arm study head-to-heads (400 games paired, per arm vs v2_base)

Train each arm to a checkpoint:

```bash
uv run python experiments/0004_alphazero/run.py +experiment=v2_base
uv run python experiments/0004_alphazero/run.py +experiment=v2_incidence
uv run python experiments/0004_alphazero/run.py +experiment=v2_deep
uv run python experiments/0004_alphazero/run.py +experiment=v2_hetero
```

Then, per arm (`v2_incidence`, `v2_deep`, `v2_hetero`), a **paired**
400-game head-to-head against `v2_base`'s checkpoint: same seed for both
sides (the repo's established paired-seed doctrine —
`packages/settlrl-learn/CLAUDE.md`'s arena-seed-fixed-across-iterations
rationale), seat-swapped. Load each side via the anchor machinery
(`experiments/0004_alphazero/anchors.py::load_anchor`) and pit them through
`arena_spec`/`build_net_opponents` rather than either side's own search
config alone, so both play under the same search budget.

**Decision:** this is the study's actual verdict mechanism — it settles
whether each lever (per-tile identity, +2 layers, hex message-passing) beats
`v2_base` outright. `v2_incidence`'s config comment already records the
interpretation note if it underperforms (the slot-rank input geometry, not a
verdict on per-tile identity). Record win rates + Elo deltas + n=400 in
`experiments/0004_alphazero/report.md`, one section per arm, before drawing
any strength conclusion (repo doctrine: strength claims gate through a
recorded match, never a reading of it).

## 6. 512-vs-1024 batch re-measurement (the sample-yield wobble note)

```bash
uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput selfplay.persistent=true selfplay.batch=512
uv run python experiments/0004_alphazero/run.py +experiment=bench_throughput selfplay.persistent=true selfplay.batch=1024
```

The original persistent-batch sweep (`experiments/JOURNAL.md`, 2026-07-28)
picked 512 over 1024 (896.60 vs 834.76 samples/s). `training/bench.py`'s own
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

## 7. Arena-cadence retune check (16.1% measured vs 9.8% budget)

`experiments/0004_alphazero/conf/experiment/scale2_long.yaml`'s comment
derives a 9.8% arena wall-share budget (`arena.games=64`, `arena.every=150`)
from the pre-run cost model (`295s / (150 * 20s/iter) ~= 9.8%`). The live
run's realized wall-share measured 16.1% instead — pull the actual per-round
arena duration and per-iteration wall-clock from the run's `metrics.jsonl` /
wandb history and recompute the realized share directly (no new script
needed, just the arithmetic against logged timestamps) rather than trusting
the pre-run estimate.

**Decision:** if 16.1% is confirmed (not an artifact of a cold-start
iteration or a one-off slow round), retune `arena.games`/`arena.every` in
`scale2_long.yaml` (and the `v2_*` study presets, which share the `scale`
arena group) down to actually hit the 9.8% budget — halving `games` again
(64 → 32) is the cheapest lever before touching `every`. Record the retuned
values and the before/after realized share in `experiments/JOURNAL.md`.
