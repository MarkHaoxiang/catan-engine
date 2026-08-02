# Experiment journal

One line per concluded experiment — number, verdict, the load-bearing fact.
Full evidence lives in each experiment's `report.md`; raw outputs under
`runs/` (git-ignored, regenerable from the manifest's commit + config).

- 0002_linear_value_fitting — framework adopted, weights not: fixed-opponent
  optimization breeds specialists (both targets beat or match their objective
  opponent yet lose ~43% head-to-head vs the hand-tuned lookahead); held-out
  AUC flat while match probes span 53–78% — select by matches, never fit
  metrics.
- 0002_linear_value_fitting/self_play — pass at 2p: champion-ladder CEM beats
  the hand-tuned weights 56.1% head-to-head (n=310, lower 2σ 50.5%) — evolving
  the opponent fixes the specialist failure. The edge does not transfer to 4p
  (27.0% ± 2.8% vs three hand lookaheads, chance 25%; the 4p-arena champion
  reads 30.4% ± 3.2%, lower 2σ 24.0%) — the 4p tuned slot stays hand-tuned.
- 0003_neural_board_architectures — pass (first sweep, 12k positions): a jraph
  GNN over the *raw* board nearly matches the engineered-feature MLP
  (heuristic R² 0.978 vs 0.996, win AUC 0.825 vs 0.834) while a
  structure-blind flat MLP on the same inputs is ≈chance (R² 0.54, AUC 0.52)
  — structure is what makes raw board features usable.
- 0003 GraphNet lever ablation (2026-06-18, 20k positions): GNNs recover the
  structural `road` target the engineered vector lacks (R² 0.99 vs 0.83);
  attention is the wrong bias for counting/structural tasks (`gn_gat` 0.86 on
  `road` vs sum-MPNN 0.99); GraphNorm and jumping-knowledge don't pay on a
  54-node graph. Winner: `gn_global` (sum-MPNN + virtual global node +
  multi-aggregator readout + LayerNorm).
- 0003 multi-task + road probe (2026-06-18): shared-trunk multi-task shows
  negative transfer onto the structural `road` head (gn_global 0.98→0.82) —
  a single AZ value+policy trunk wants sum-MPNN + loss-balancing. Road probe:
  the GNN is not edge-counting but is depth-limited (1 layer fails R² 0.58,
  depth 2 → 0.92, then plateaus).
- 0004 bench_throughput baseline (2026-07-28) — pinned self-play throughput at
  the frozen az0 net (B=256, 64 sims): **193 samples/s** (741 moves/s, 72.8%
  of searched positions discarded at the iteration boundary) — the gn_global
  track's before number. Run runs/0004_alphazero/2026-07-28T065337Z,
  commit 2811210.
- 0004 persistent self-play adopted (2026-07-28): the lane pool removes the
  iteration-boundary discard entirely. 193.07 samples/s (B=256
  non-persistent, 72.8% discarded) → persistent@256 482.42 (2.50x, discard
  0.0%) → **persistent@512 922.53 (4.78x, sweep winner)**; persistent@1024
  regresses (834.76). Non-persistent B-scaling had regressed instead
  (193.8 → 166.4 → 120.2 samples/s at 256/512/1024, discard climbing to 91%)
  — a discard artifact, not a batch ceiling. Adopted in
  `conf/experiment/scale2.yaml`: `selfplay.batch=512`, `persistent=true`,
  `temperature_moves=30`, playout-cap randomization (`pcr_full_prob=0.25`,
  `pcr_fast_sims=16`, sims=128 full) — PCR/anneal are training-lever
  adoptions, not bench-measured (bench mode keeps PCR off, sims=64).
- 0004 fused-leaf seam fix (2026-07-30) — GPU-throughput-neutral: 194.91 vs
  193.81 samples/s (+0.6%, noise). GPU XLA already common-subexpression-
  eliminates the duplicate leaf forward; the CPU dot-census win that
  motivated the fix was CPU-specific. The fix stays for the structural
  single-forward-by-construction guarantee, pinned by
  `settlrl-learn/tests/test_leaf_seam.py`.
- 0001_bench_smoke/calibrate — pass, one-off anchor-scale reset (2026-07-29):
  joint Elo MLE (`calibrate.py::joint_fit`, coordinate-ascent Bradley-Terry)
  over the full round-robin {random, greedy, lookahead, mcts, az0_gnn96x4},
  n=4238, lookahead pinned at 0. Fitted: random **−1115 ± 71**, greedy
  −231 ± 12, mcts +38 ± 12, az0_gnn96x4 **−58 ± 12** (Fisher SE); adopted
  into `conf/arena/*.yaml` (random −800 → −1115, az0 −100 → −58). Valid only
  for `conf/arena/scale.yaml` + `conf/search/scale.yaml` semantics plus the
  pinned az0 setup-opener constants; pre-calibration `arena_elo` readings are
  not face-comparable. Run runs/0001_bench_smoke/2026-07-29T034807Z.
- 0004 scale2_long — **pass, first Stage-1 gate clear (2026-07-30)**: the
  scale2 recipe (gn_global 96x4, v1 features) at 2500 iterations, bit-exact
  resume across runs/0004_alphazero/2026-07-29T122854Z +
  .../2026-07-30T094945Z. Final 400-game gauntlet: `arena_elo`
  **109.857 ± 12.867** SE (lower bound +84.1 vs the +35 gate), 0.6264 vs
  lookahead, 0.7530 vs az0, 1.0 vs random. Falsifies the earlier plateau
  reading — the net climbs ~340 Elo from its iter-149 trough with no
  saturation; under-scaling, not a leaf/value ceiling, was the cause. Anchor
  minted: `az1_gnn96x4` (feature_version=1, elo 109.86 ± 12.87).
- 0004 four-arm architecture study — hetero passes, sole gate clear
  (2026-07-30) at matched 300 iterations: `v2_hetero` (hexes promoted to
  first-class nodes, vertex↔hex message passing) **76.171 ± 9.815**, pass —
  the controls `v2_base` / `v2_incidence` / `v2_deep` fail indistinguishably
  (33.9–40.9, overlapping ±9.7–9.9 SE) — the win is the hex-node
  star-expansion structure itself, not reach (deep) and not information
  content (incidence). v2 featurization alone is worth ~+110 Elo over v1 at
  matched compute.
- 0004 v2_hetero long run — **second gate pass, az2 minted (2026-07-31)**:
  full 2500-iteration budget, bit-exact resume across
  runs/0004_alphazero/2026-07-31T010419Z + .../2026-07-31T073531Z. Final
  400-game gauntlet: `arena_elo` **186.759 ± 10.877** SE (lower bound
  +165.0), 0.7403 vs lookahead, 0.6014 vs az1, 0.8180 vs az0. The hypergraph
  advantage compounds with compute — v2_hetero reaches v1's final +110 level
  by iteration ~449. Anchor minted: `az2_hetero96x4` — the ladder is
  az0 −58 / az1 +109.86 / az2 +186.76.
- 0003 guard — az2-distillation architecture guard calibrated (2026-08-01).
  Doctrine: architecture decisions gate through the supervised guard (train
  the production net on frozen az2 self-play targets), never short RL runs —
  300-iteration RL screens are retired; a guard pass earns a full
  2500-iteration run judged by the gauntlet gate. Calibration (3 seeds/arch,
  50k train / 10k val positions at sims=64): `gn_hetero` beats `gn_global`
  with zero seed overlap — best val policy KL **0.0851 vs 0.1089**, top-1
  agree 91.8% vs 90.4%, value MSE(z) 0.103 vs 0.126 — reproducing the
  RL-loop ground truth in ~20 GPU-min vs ~8 GPU-h. Caveat: the teacher is
  itself `gn_hetero`. Run
  runs/0003_neural_board_architectures/2026-08-01T094430Z.
- 0004 bench_throughput_hetero baseline minted (2026-08-01) — pinned
  self-play throughput for the adopted hetero recipe (az2_hetero96x4 anchor,
  persistent pool, B=512, 128 sims, no PCR): **675.36 samples/s** (discard 0,
  RTX 5090) — the hetero track's before number. Run
  runs/0004_alphazero/2026-08-01T154603Z. The gn_global pin (az0, B=256/64
  sims) stays valid for its own recipe.
- 0004 setup-row-restricted opener adopted (2026-08-01, d11b50e): the setup
  policy sweeps the 126 setup rows instead of all 662 (bit-identical moves
  and RNG stream, lockstep contract test): pinned hetero gauge 675.36 →
  **770.94 samples/s (+14.2%)** (runs/0004_alphazero/2026-08-01T154603Z →
  .../2026-08-01T174013Z).
- 0004 backup scatter-add rejected (2026-08-01): batching the ISMCTS
  `_backup` scatters into one `.at[].add` is bit-exact on CPU but not CUDA —
  the scatter lowers through `atomicAdd(float)`, which flushes subnormal f32
  updates to +0.0 (divergence at |v| < 1.1754944e-38) — and the GPU
  search-step win was noise (326.3 → 324.0 ms B=64). Bit-exactness bar +
  neutral win → not landed.
- learn checkpoint slimming, format rev (2026-08-01): replay
  `mask`/`train_policy` stored bool (losses byte-exact on CPU, pinned test)
  and `selfplay.checkpoint_pad` bounds the persistent-carry pad (scale
  presets 512; measured pending tail median 88 / p99 272 / max 398 at B=512;
  over-bound lanes log `checkpoint_truncated_lanes`/`_rows`). Save bench at
  the production shape: **6.24 GB / 1.95 s → 4.43 GB / 1.18 s** per write
  (~0.40 GB from the bool dtypes, ~1.41 GB from the 800→512 pad). Pre-rev
  checkpoints fail to load — no migration.
- 0004 selfplay batch 1024 adopted (2026-08-01): pinned hetero gauge
  **770.94 (B=512) → 885.27 samples/s (B=1024), +14.8%** (run noise ~±1.7%,
  the win is well clear). Flips the 2026-07-28 "512 is the winner" verdict —
  that was a different recipe (gn_global v1, 64 sims, pre-setup-opener), and
  the pre-persistent 1024 loss was the discard artifact. Adopted in the
  seven production presets (gn_global trunks by extrapolation); the frozen
  bench pins keep their own shapes. Throughput-only: a game now spans ~2×
  the iterations, so flushed policy targets are up to ~2× staler; carry pad
  doubles (~6.5 GB/write expected — watch `checkpoint_truncated_lanes` on
  the first B=1024 final checkpoint, pad 640 restores headroom).
- Root-legality reuse adopted unconditionally (2026-08-01): the search takes
  the caller's env root mask as every determinization's root legal set and
  peels the descent's depth-0 step — no flag, no per-world root sweep.
  Bit-identity verified pre-deletion (8,400-root probe with 0 differing
  bits; move-for-move CPU bit-identity over real games, 2p + 3p). GPU
  search-step micro: 324.6→316.5 ms B=64, 930.3→921.1 ms B=256 (~1–2.5%).
- blocked_linear landed (2026-08-01): `GraphNetConfig.blocked_linear`
  weight-blocks the message/update MLPs' first Linears (one matmul per
  unique input row, gathered). GPU micros (scratchpad harness, NOT the
  pinned gate): hetero-v2 forward B=512 4.6→2.8 ms (−38%), search step
  B=512/sims=128 968→854 ms (−12%). Same function up to float summation
  order (float64 identity residual 7.1e-15), not bit-exact.
- blocked linears adopted unconditionally, flag deleted (2026-08-01): the
  reassociation-identity proof means the old form is dead code, so
  weight-blocked first Linears are the only formulation. Pinned hetero gauge
  (B=512 ruler): 770.94 → **911.72 samples/s (+18.3%)**
  (runs/0004_alphazero/2026-08-01T174013Z → .../2026-08-01T205749Z; run
  noise ~±1.7%). The blocked matmuls slice the same Linear weights, so every
  checkpoint/anchor loads unchanged; forward bits shift at float32 rounding
  level, so a resume across the boundary diverges. Proof pinned by
  `tests/test_message_linears.py` against a naive gather-concat-transform
  reference (float64 identity ≤ 1e-12, parameter identity).
- 0003 guard_dnorm — fail (2026-08-01): HNHN-style degree normalization of
  the hetero incidence aggregates (`gn_hetero_dnorm`) vs `gn_hetero` on the
  frozen az2 distill targets, 3 seeds each: challenger worst best_policy_kl
  0.09054 vs incumbent best 0.08097, top-1 tie, value MSE uniformly worse →
  `distill_verdict` fail, no full-budget run. Consistent with per-node
  LayerNorm already absorbing degree-dependent scale; the lever stays a
  `GraphNetConfig` ablation knob (default off, off-path byte-identical).
- Action-axis narrowing rejected (2026-08-02, CPU profile). At 2p main-loop
  search 228/662 flat rows are always illegal (126 setup + 102 domestic
  trade), but slicing the tree's `act` axis does not pay. Tree select/backup
  do scale with width (W^1.25-1.35 at B>=64; 662->434 cuts them 35-43%) —
  the old `tree.py` TODO's premise was wrong — yet they are only 5-10% of a
  search behind the net forwards (~71%), so the ceiling is ~2-4% CPU and
  <=2-3% GPU by analogy with the root-sweep removal (1-2.5%), at or below
  the gauge's noise floor, against a wide index-surface change (`act` is
  also the chance-outcome axis; policy targets would need scatter-back).
  The 100 trade rows are separately worthless: dropping them from the flat
  mask saves <=0.11 us/lane of 3.51 (3%, in-noise) — the `n > 2` gate folds
  to a constant. Open lead from the same run: select+expand+backup together
  cost 2-7x the sum of the isolated ops, excess proportional to n_nodes,
  with a cliff at the width where the batched tree leaves L3 (not an emitted
  copy — HLO census clean). GPU repeat of that mechanism is the next step;
  if it is CPU-only cache pressure, both leads close.
- 0004 v2_hetero 5000-iter — gate pass, but scaling did NOT convert
  (2026-08-02, run 2026-08-01T215549Z, 16.6 h). Double the iteration budget
  on the post-wave recipe (B=1024, ~4x az2's self-play volume in samples):
  final gauntlet **arena_elo 196.10 +/- 9.11** (gate `elo - 2*se >= 35`
  passes at 177.9), yet **head-to-head vs az2 is 0.485** — a tie — and the
  delta to az2's own reading (186.76 +/- 10.88) is +9.3 Elo at 0.66 sigma.
  vs az1 0.623, vs az0 0.827, vs random 1.0. The in-loop curve plateaus from
  ~iteration 2400 (196-250, no trend, 18 rounds) — i.e. az2's stopping point
  was already the plateau. **The under-scaling hypothesis is falsified at
  this scale for "more of the same": 2x iterations / ~4x samples buys
  nothing measurable.** No anchor minted (a rung that ties az2 adds no
  information to the ladder). Confound noted: the recipe also changed
  (B=512->1024, blocked linears), so this is not a pure iteration sweep —
  but both changes are throughput-only or function-preserving, which makes
  the null stronger, not weaker. Next levers are qualitative (search
  structure, targets), not volume. Side note: `checkpoint_truncated_lanes`
  fired 2 and 3 at the first two checkpoints then 0 for all 18 after — the
  pad-512 tail only binds during early weak play, when games run long.
- 0005_search_guard — framework adopted; `chance_nodes` excluded at the ship
  gate (2026-08-02, run 2026-08-02T193349Z, ~51 GPU-min). The guard duels two
  search configurations over a frozen anchor (az2, 128 sims / 16 considered,
  seat-rotated paired seeds) and screens with a 2-sigma interval around the
  no-edge line: promising / rejected / inconclusive. First measurement:
  `chance_nodes` vs the production default, one flag apart, **821 decided
  games: 49.8% +/- 1.75%, Elo -1.3 +/- 12.1** — verdict `inconclusive` by the
  rule, but the upper bound (+23 Elo) **excludes a gain at or above the
  35-Elo ship gate**, so it does not earn the ~32 GPU-h training A/B. Cost is
  identical at production semantics (4.077 vs 4.083 ms/move) — the earlier
  "chance is 5x cheaper" reading was an `expected_rolls` confound, since
  `GNNBackend.play_agent` does not thread it and `chance_nodes` forces it off
  (that play-time/train-time mismatch is a production bug, still open).
  `ordered` is NOT evaluable here: at play time the ordering overlay never
  reaches the root (`evaluate` builds the env without `track_ordering`, and
  the overlay lives outside the fused rollout core), so an arm would measure
  a tree pruned inconsistently with its own env; threading it is an engine
  change and the prerequisite. Caveat on all guard readings: `evaluate`
  discards games still in flight at the budget (~14% here, systematically
  the longest), and this gate is play-time only — better search also makes
  better training targets, which no duel can see.
