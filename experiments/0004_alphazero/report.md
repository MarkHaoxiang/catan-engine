# 0004 — AlphaZero (2-player)

Status: open (loop built; proof-of-concept only, no strength run)

## Hypothesis

A value+policy net trained by AlphaZero self-play — the re-determinizing search
as its own teacher — beats `lookahead(heuristic)` at 2p, lifting the leaf the
search ladder is stuck against (the settlrl-learn Stage-1 gate).

## Setup

`run.py [+experiment=<name>]` (hydra `conf/` groups + `experiment/` presets;
no flag = the `default` MLP config). The loop lives in `settlrl_learn.training`
(composable); `run.py` only composes it with a config, per-iteration logging,
and the gate verdict. Each iteration:

1. **self-play** — net-guided re-determinizing search (`value_scale=2`); record
   each move's true-board features, the search's improved policy (target), and
   the eventual win/loss (value);
2. **buffer** — a flashbax item buffer (recent positions);
3. **train** — optax adamw on policy cross-entropy + value logistic;
4. **arena** (periodic) — seat-swapped win rate vs `lookahead(heuristic)`.

2-player only: belief is near-exact, so the multiplayer paranoid-frame problem
never arises. Stack: mctx (search), optax, flashbax, the settlrl-learn net.

## Results

Smoke (1 iteration, 8 samples, 4 sims) runs the whole loop end-to-end and
records a verdict. Component PoCs: self-play yields valid policy/value targets
(395 samples, policy rows sum to 1, balanced wins); training drops the loss
7.17 → 1.74 over 100 steps. No strength run yet — PoC scope.

### Throughput (2026-07-28)

`bench_selfplay` (`+experiment=bench_throughput`) isolates self-play, the
loop's dominant cost. The frozen anchor (`az0_gnn96x4`, B=256, 64 sims)
measured 193.07 samples/s with 72.8% of searched positions discarded at
the iteration boundary (runs/0004_alphazero/2026-07-28T065337Z); a
batch-only sweep regressed (166.4 @ 512, 120.2 @ 1024) as discard climbed
to 91% — the batch lever was gated behind that waste.
`selfplay.persistent` (a self-play pool carried across iterations, so only
finished games are discarded — trims, not everything in flight) removes it:
persistent@256 482.42 samples/s (discard 0.0%), persistent@512 **922.53
samples/s (4.78x the baseline)**, persistent@1024 834.76 samples/s — B=512
is the sweep winner (runs/0004_alphazero/2026-07-28T230720Z,
.../2026-07-28T231144Z, .../2026-07-28T231545Z).

`conf/experiment/scale2.yaml` adopts this: `small`'s shape (gnn 96x4,
teacher warm-start, Canopy q-blend) plus `selfplay.batch=512`,
`selfplay.persistent=true`, `temperature_moves=30` (an untuned starting
value), and playout-cap randomization on (`pcr_full_prob=0.25`,
`pcr_fast_sims=16`, `search.num_simulations=128` for the full steps). An
8-iteration validation run confirms `selfplay_discarded=0.0` at every
iteration, losses drop with no NaNs, and policy entropy stays alive
(runs/0004_alphazero/2026-07-28T233122Z); a checkpoint+resume smoke
(kill after iteration 4, resume) continues without error
(.../2026-07-28T234523Z → .../2026-07-28T235055Z).

## Decision

Loop adopted; `scale2.yaml` is the throughput-wave production preset
(4.78x self-play throughput over the frozen baseline). Not yet run at
scale — next is a real `scale2` run against the gate. Parallel-descent
search work (K-way SH blocks, virtual loss) waits until that run re-tests
the plateau.
