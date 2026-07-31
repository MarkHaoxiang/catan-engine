# 0004 — AlphaZero (2-player)

Status: open (Stage-1 gate cleared twice — scale2_long at 2500 iterations,
`v2_hetero` at 300; long `v2_hetero` run COMPLETE 2026-07-31 — final
400-game gauntlet `arena_elo` 186.759 ± 10.877, 0.7403 vs lookahead,
0.6014 vs az1, `az2_hetero96x4` minted; see JOURNAL)

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
never arises. Stack: settlrl-search's SO-ISMCTS tree (search), optax,
flashbax, the settlrl-learn net.

## Results

Smoke (1 iteration, 8 samples, 4 sims) runs the whole loop end-to-end and
records a verdict. Component PoCs: self-play yields valid policy/value targets
(395 samples, policy rows sum to 1, balanced wins); training drops the loss
7.17 → 1.74 over 100 steps. At PoC time no strength run had happened yet —
the strength results below came later.

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
(4.78x self-play throughput over the frozen baseline). The recipe has since
run at scale and cleared the Stage-1 gate twice — see the scale2_long and
four-arm study sections below.

### scale2_long — first Stage-1 gate clear (2026-07-30)

The scale2 recipe run to 2500 iterations (v1 features), bit-exactly resumed
at iteration 1710 across `runs/0004_alphazero/2026-07-29T122854Z` and
`.../2026-07-30T094945Z`. Final 400-game gauntlet: `arena_elo` 109.857 ±
12.867 SE (lower bound +84.1 vs. the +35 gate), 0.6264 vs. lookahead (gate
0.55), 0.7530 vs. the frozen `az0_gnn96x4` anchor. In-loop `arena_elo`
climbed with no sign of saturating: −211.8 @ 149 → −75.4 @ 299 → +19.1 @ 599
→ +76.5 @ 1349 → +130.9 @ 2099 → +109.857 (final). Falsifies the earlier
"plateau" reading from the throughput-wave sweep — the short runs there read
a still-cold net, not a ceiling. Anchor minted: `az1_gnn96x4` (elo 109.86 ±
12.87, `conf/arena/scale.yaml`).

### Four-arm architecture study — the hypergraph wins (2026-07-30)

**Question**: on top of the scale2 recipe, does moving to v2 board
featurization plus one architecture lever beat the plain v2 control at v1's
matched 300-iteration budget? Three levers, each isolation-tested against a
`v2_base` reference (v1's recipe + v2 featurization only — Task 1/2's global
repairs [own hand/dev composition, `free_roads`, `longest_road_len`,
`pending_discard`] + a `[max, sum, std]`/LayerNorm readout, still on the
`gn_global` trunk):

| arm | delta over `v2_base` | arena_elo | SE | vs az1 | verdict | run dir |
|---|---|---|---|---|---|---|
| `v2_base` | — (reference) | 33.914 | 9.864 | 39.7% | fail | 2026-07-30T142232Z + 2026-07-30T161824Z (gauntlet, post-OOM-fix) |
| `v2_incidence` | + per-vertex incident-hex identity features, no new message passing | 40.862 | 9.670 | 38.3% | fail | 2026-07-30T165116Z |
| `v2_deep` | + 2 more GNN layers (4→6), reach/capacity only | 35.728 | 9.820 | 40.0% | fail | 2026-07-30T222151Z |
| `v2_hetero` | `gn_global` → `gn_hetero` (hexes as first-class nodes, vertex↔hex message passing each layer) | **76.171** | 9.815 | **46.5%** | **pass** | 2026-07-30T192420Z |

Gate: `arena_elo` ≥ +35 and `arena_winrate` ≥ 0.55 vs. lookahead, 400-game
final gauntlet. `v2_base`'s training run (2026-07-30T142232Z) completed all
300 iterations, then died `CUDA_ERROR_OUT_OF_MEMORY` while the final gauntlet
tried to compile against training's still-resident jit caches; a5bda5b fixed
`run_final_gauntlet` to clear jax's compilation caches (and gc) first, and
the gauntlet was re-run standalone via `resume_from` at the fix commit
(2026-07-30T161824Z) — that second run dir is where `v2_base`'s numbers
above come from.

**Reading**: `v2_base`, `v2_incidence`, and `v2_deep` are statistically
indistinguishable from each other (33.9–40.9 Elo, heavily overlapping ±9.7–9.9
SE bands) — neither more reach (`v2_deep`) nor exposing per-tile identity as
input features (`v2_incidence`) moves the needle over the plain control.
`v2_hetero` clears every control by ~3σ and is the only arm to clear the
gate. The win is attributable to the hex-node star-expansion **structure**
itself — message passing that treats a hex as a node peer to vertices, not
just an unordered incident-feature bag — rather than to reach (`v2_deep`
shows more layers alone don't help) or to raw information content
(`v2_incidence` shows the same information without the structural pathway
doesn't help either).

**Context**: v2 featurization is worth roughly +110 Elo over v1 at matched
300-iteration compute — `v2_base`'s final gauntlet (+33.9) vs. v1's
in-loop reading at the same iteration count (scale2_long @ iter 299, −75.4).

**Caveats / follow-ups**:

- The long `v2_hetero` run (2500 iterations, matching scale2_long's budget)
  completed across two segments (`runs/0004_alphazero/2026-07-31T010419Z` +
  `.../2026-07-31T073531Z`, bit-exact resume at iteration 980). Final
  400-game gauntlet: `arena_elo` +186.76 ± 10.88; `az2_hetero96x4` minted.
  Ladder: az0 −58 / az1 +109.86 / az2 +186.76.
- HNHN degree normalization (the alpha/beta parameterization on the
  vertex↔hex aggregation) is still unimplemented — `v2_hetero` ships with a
  plain segment-sum aggregation. This is a follow-up, not part of this
  study's result.
- The structural win strengthens the prior for further hex/roll-event node
  experiments, but those stay behind an adversarial panel (this study is one
  isolation test at one seed/budget, not a robustness sweep).
- `v2_incidence`'s slot-rank input geometry note stands: slots are a
  lexicographic rank of a hex's own attributes (the only D3-equivariant
  option available), not a canonical geometric direction — if a future
  incidence-style arm underperforms again, that geometry is the leading
  suspect before concluding per-tile identity itself doesn't help.
