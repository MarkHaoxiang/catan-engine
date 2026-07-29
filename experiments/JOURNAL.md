# Experiment journal

One line per concluded experiment — number, verdict, the load-bearing fact.
Full evidence lives in each experiment's `report.md`; raw outputs under
`runs/` (git-ignored, regenerable from the manifest's commit + config).

- 0001_bench_smoke — pass: greedy beats random 83.1% ± 4.7% (n=65, 2p);
  infrastructure worked example.
- 0002_linear_value_fitting — framework adopted, weights not (both targets):
  predict reaches hand level vs greedy (75.2% vs 77.8%), maximise beats it
  (80.8%), yet both lose head-to-head (~43%) — fixed-opponent optimization
  breeds specialists. Held-out AUC flat while match probes span 53–78%:
  select by matches, never fit metrics.
- 0002_linear_value_fitting/self_play — pass: 3-round champion-ladder CEM
  beat the hand-tuned weights 56.1% head-to-head (n=310, lower 2σ 50.5%) and
  80.9% vs greedy (hand: 77.8%) — fixed-opponent specialists fixed by making
  the opponent evolve; adoption of the weights deferred (leaf cascade).
- 0002_linear_value_fitting/self_play at 4p — the 2p edge does not transfer:
  27.0% ± 2.8% vs three hand lookaheads (chance 25%), 64.7% vs greedy tables
  (hand: 68.1%); champion reproduced bit-identically from config (framework
  determinism verified). Adoption now 2p-conditional or needs a mixed-count
  arena.
- 0002_linear_value_fitting/self_play_4p — near miss: the 4p-arena champion
  reads 30.4% ± 3.2% vs three hand lookaheads (chance 25%, lower 2σ 24.0%,
  n=230) and 69.3% vs greedy tables; 4p tuned slot stays hand-tuned pending
  an n≈600 confirmation.
- 0003_neural_board_architectures — pass (framework + first sweep, 12k
  positions): a jraph GNN over the *raw* board nearly matches the hand-tuned
  feature MLP — heuristic R² 0.978 vs 0.996, win AUC 0.825 vs 0.834 — while a
  structure-blind flat MLP on the same inputs is ≈chance (R² 0.54, AUC 0.52)
  and DeepSet sits between. Structure is what makes raw board features usable;
  a learnable leaf is within reach (settlrl-learn Stage 1 seam). Not yet
  promoted: close the win gap, then gate lookahead(gnn) through bench.
- 0003_neural_board_architectures — GraphNet lever ablation (2026-06-18, 20k
  positions, +`road` structural target = seat-0 longest-road trail length). On
  `road` the GNNs hit R² 0.99 vs the engineered MLP's 0.83 (replicated seed 1:
  eng 0.835, plain-MPNN 0.986) — the graph recovers a connectivity quantity the
  hand-tuned vector lacks. **Attention is the wrong bias for counting/structural
  tasks**: `gn_gat` collapses to 0.86 on `road` (sum-MPNN 0.99) while leading on
  the global `win` target (0.77) — the target's locality picks the architecture.
  GraphNorm and jumping-knowledge don't pay on a 54-node graph. Every preset is
  board-symmetry + player-relabel invariant (no absolute PE). Recommended net:
  `gn_global` (sum-MPNN + virtual global node + multi-aggregator readout +
  LayerNorm; no attention/GraphNorm/JK) — the robust all-rounder for the AZ
  value+policy net.
- 0003 multi-task + road-reasoning probe (2026-06-18). Multi-task (shared trunk,
  head per target win+heur+road+turns): negative transfer onto the structural
  `road` head (gn_global 0.98→0.82) while easy heads hold; plain `gn_base` is the
  most robust trunk and its win head *improves* (0.738→0.766) from the structural
  auxiliary — a single AZ value+policy trunk wants sum-MPNN + loss-balancing.
  Road probe (`road_probe.py`, controlled paths/branchy/broken mix): the GNN is
  not edge-counting (on |count-longest|≥2 cases it predicts true longest 7.2 not
  count 11.1) but is depth-limited as expected — 1 layer fails (R² 0.58), depth 2
  →0.92, then plateaus; greedy road 0.99 was an easy near-simple-path artefact.
- 0004 bench_throughput baseline (2026-07-28). Pinned self-play throughput at
  the frozen az0 net (overnight shape: B=256, 64 sims): 193 samples/s
  (741 moves/s, 47438 sims/s), 72.8% of searched positions discarded at the
  iteration boundary — the optimization track's before number (run
  runs/0004_alphazero/2026-07-28T065337Z, commit 2811210).
- 0004 bench_throughput batch sweep (2026-07-28, measurement only, no config
  adopted). B=256 re-run reproduces the frozen baseline (193.81 vs 193.07
  samples/s, +0.4%, ruler holds). B=512 and B=1024 both *regress*, opposite
  the design-phase synthetic-replica prediction of a 1.6x per-lane win:
  193.81 (256) > 166.39 (512) > 120.15 (1024) samples/s, with discard
  fraction climbing 72.7%→84.5%→91.0% — larger batch means more
  lockstep-truncation waste, not less. No OOM at any size; peak GPU memory
  is ~flat at 25.3-25.4GiB (78% of 32GB) across all three, which is JAX's
  preallocated pool, not a per-batch signal. Winner: B=256, the current
  default — batch-size increase does not pay off at this anchor (64 sims).
  Runs: runs/0004_alphazero/2026-07-28T103722Z (256),
  runs/0004_alphazero/2026-07-28T104353Z (512),
  runs/0004_alphazero/2026-07-28T105136Z (1024).
- 0004 throughput wave verdict: persistent self-play adopted (2026-07-28).
  `selfplay.persistent` (the lane pool) reruns the batch sweep with the
  discard term actually gone, and it flips the earlier verdict: baseline
  193.07 samples/s (B=256, non-persistent, 72.8% discarded) →
  persistent@256 482.42 samples/s (2.50x, the lane pool's isolated
  contribution — same batch, discard 72.8%→0.0%) →
  persistent@512 **922.53 samples/s (4.78x compounded)**, discard 0.0%
  — the batch lever now pays off once the iteration-boundary waste is gone,
  matching the design-phase prediction the non-persistent sweep falsified.
  persistent@1024 regresses off that peak (834.76 samples/s, still
  discard 0.0%) — B=512 is the sweep winner. Adopted into
  `conf/experiment/scale2.yaml` (small's shape: gnn 96x4, teacher
  warm-start, Canopy q-blend): `selfplay.batch=512`, `persistent=true`,
  `temperature_moves=30` (untuned starting value), playout-cap
  randomization on (`pcr_full_prob=0.25`, `pcr_fast_sims=16`,
  `search.num_simulations=128` for the full steps) — PCR/anneal are
  training-lever adoptions, not bench-measured (bench mode keeps PCR off,
  sims=64, per Step 2). An 8-iteration validation run
  (`+experiment=scale2 n_iterations=8 arena.every=4 wandb.mode=disabled`,
  runs/0004_alphazero/2026-07-28T233122Z) confirms `selfplay_discarded`
  is 0.0 at every iteration (not just from iter 2 on), losses drop
  2.29→~1.2 with no NaNs, policy entropy stays alive (0.46-0.52, no
  collapse), and per-iteration self-play throughput (steady-state
  ~850-1775 samples/s across a live, still-cold net) is order-of-magnitude
  consistent with the frozen-net bench number. Checkpoint+resume smoke
  (`checkpoint_every=4`, killed after iteration 4's checkpoint —
  runs/0004_alphazero/2026-07-28T234523Z — then `resume_from`'d into
  runs/0004_alphazero/2026-07-28T235055Z) resumes cleanly at iteration 4/8
  and completes iterations 5-8 with no errors, matching losses. Sweep
  runs: runs/0004_alphazero/2026-07-28T230720Z (persistent@256),
  runs/0004_alphazero/2026-07-28T231144Z (persistent@512, winner),
  runs/0004_alphazero/2026-07-28T231545Z (persistent@1024). Parallel-descent
  search work (K-way SH blocks, virtual loss) stays out of scope until a
  scale2-based long run re-tests the plateau.
- 0001_bench_smoke/calibrate — pass, one-off anchor-scale reset (2026-07-29):
  a joint Elo MLE (coordinate-ascent Bradley-Terry, `calibrate.py::joint_fit`)
  over the complete round-robin {random, greedy, lookahead, mcts, az0_gnn96x4}
  (10 unordered pairs, seat-swapped 2p, n=4238 games total; lookahead pinned
  at 0). Fitted: random −1115 ± 71, greedy −231 ± 12, mcts +38 ± 12,
  az0_gnn96x4 −58 ± 12 (Fisher SE). All four sanity gates held (lookahead = 0
  by construction; greedy in [−480, −180]; random < −600; az0 in (−400, 0)).
  Old → new: `arena.anchor_elos.random` −800.0 → −1115.0
  (`conf/arena/default.yaml` + `conf/arena/scale.yaml`); az0_gnn96x4's
  `net_opponents` elo (provisional, from its 0.361 winrate vs lookahead
  alone) −100.0 → −58.0 (`conf/arena/scale.yaml`). **Historical-shift
  caveat**: every past `arena_elo` reading (every 0004 run to date) was on
  the old −800 scale and is not comparable at face value to a run using
  these anchors — the *relative* ordering of checkpoints within one run is
  unaffected, only the absolute number's meaning shifts. **Semantics scope**:
  the calibration is valid only for arena/search settings matching
  `conf/arena/scale.yaml` (sims=24, considered=16) + `conf/search/scale.yaml`
  (chance_nodes=false, dev_chance=true, ordered=false) — recorded verbatim in
  the run's `result.json` (`search_semantics`) alongside the fitted ratings
  and matrix — plus az0_gnn96x4's setup opener (`setup_depth=1`,
  `setup_temperature=2.0`, `setup_beam=4`, GNNBackend's own defaults): pinned
  as constants (`0004_alphazero/run.py::NET_OPPONENT_SETUP_*`,
  `0001_bench_smoke/calibrate.py::AZ0_SETUP_*`) rather than read off a run's
  `cfg.net.setup_*`, so a frozen anchor keeps frozen semantics regardless of
  what any given run configures for its own net. A differently-configured
  arena (different sims/considered, chance_nodes/
  dev_chance/ordered flipped, or a different az0 setup opener) needs its own
  calibration; these anchors should not be reused there. Run:
  runs/0001_bench_smoke/2026-07-29T034807Z.
