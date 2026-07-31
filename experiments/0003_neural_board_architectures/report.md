# 0003 — neural board architectures

Status: open (framework live; first sweep concluded)

## Hypothesis

On supervised board-prediction tasks, a structure-aware net over the *raw* board
graph (GNN) is competitive with an MLP over the hand-tuned feature vector, and
clearly beats a structure-blind net over the same raw inputs. If raw-board
learning reaches hand-engineered level, the graph representation is the seam to
push for a learned value (settlrl-learn Stage 1).

## Setup

`uv run python experiments/0003_neural_board_architectures/run.py [variant] [k=v]`
— config schema and variants at the top of `run.py`.

Data: greedy 2p self-play, seat-0 positions on the *true* board, cached under
`runs/_cache`. Two labels per position: `heur` (the hand-tuned
`heuristic_value`) and `win` (seat 0's game outcome). Held-out split grouped by
episode. Four architectures, all reading the same position, same width 64:

- `mlp_engineered` — MLP over the 118-d hand-tuned feature vector (baseline);
- `mlp_flat` — MLP over the flattened graph (node features in fixed vertex
  order + globals), structure-blind;
- `deepset` — permutation-invariant mean-pool over per-vertex features +
  globals (set; no edges);
- `gnn` — jraph `GraphNetwork` message passing over the board graph (54
  vertices, 72 edges, both directions) + global readout.

Stack: equinox (models), optax adamw (training), jraph (message passing),
wandb (logging, `mode` configurable), equinox checkpointing of the best-val
model. Node/edge/global features standardized on the train split.

## Results

12,000 positions, 40 epochs, width 64, seed 0 (RTX 5090).

Heuristic regression — held-out R² (`runs/.../2026-06-17T003236Z`):

| arch | R² |
| --- | --- |
| mlp_engineered | **0.996** |
| gnn | 0.978 |
| deepset | 0.951 |
| mlp_flat | 0.538 |

Win prediction — held-out AUC (`runs/.../2026-06-17T003343Z`):

| arch | AUC |
| --- | --- |
| mlp_engineered | **0.834** |
| gnn | 0.825 |
| deepset | 0.789 |
| mlp_flat | 0.519 |

The two tasks agree:

1. **Structure is what makes raw board features usable.** The flat MLP — same
   raw inputs, fixed vertex order — is near-useless (R² 0.54; AUC 0.52 ≈
   chance). Permutation-invariant pooling (DeepSet) and message passing (GNN)
   recover most of the signal from identical inputs.
2. **Edges/topology add a little over a pure set.** GNN > DeepSet on both tasks
   (R² 0.978 vs 0.951; AUC 0.825 vs 0.789).
3. **The GNN nearly matches hand engineering from the raw board** — within
   0.018 R² and 0.009 AUC of the engineered-feature MLP, with no hand-crafted
   features beyond per-vertex production/ports/ownership. The engineered MLP's
   near-perfect heuristic R² is expected (it is fed exactly the heuristic's own
   inputs) and is the ceiling for that task, not a fair strength signal.

## Decision

Framework adopted; architecture not yet promoted to a shipped value. The
hypothesis holds — a GNN over the raw board is competitive with hand
engineering and dominates structure-blind baselines — so the graph
representation is the right seam for settlrl-learn Stage 1. Before promoting:
close the last gap to (and ideally past) the engineered baseline on `win` with
more data / capacity / tuning, then gate a `lookahead(gnn_value)` through
`settlrl-agents bench` per the staged plan. The leaf-is-the-ceiling finding in
`settlrl-agents`'s search notes is the reason this matters: search only starts
paying once the leaf improves, and this shows a learnable leaf is within reach.

Open levers: deeper/wider GNN, more samples, lookahead-agent data (greedy
positions may be off the strong-play manifold), and node-level targets (per-spot
value) the global readout currently discards.

## GraphNet lever ablation (2026-06-18)

A configurable `GraphNet` (`settlrl_learn.nn.graphnet`) turns the major design
choices into config knobs; `ablate_{heuristic,win,road}` sweep one preset per
lever (`gn_*`) against the engineered baseline. Added a third, *structural*
target — `road`, seat 0's true longest-road trail length — which the engineered
vector cannot express (it carries a road *count*, not the connectivity DFS).
20k greedy 2p positions, width 64, 3 layers, 60 epochs.

Held-out scores (R² for heuristic/road, AUC for win; **bold** = best per column):

| arch (lever vs `gn_base`) | road R² | win AUC | heur R² |
| --- | --- | --- | --- |
| `mlp_engineered` (baseline) | 0.83 | **0.82** | **1.00** |
| `gn_base` (plain MPNN, mean readout) | **0.99** | 0.74 | 0.98 |
| `gn_multi` (+ mean‖max‖sum readout) | 0.98 | 0.75 | 0.98 |
| `gn_norm` (+ LayerNorm) | 0.98 | 0.74 | 0.97 |
| `gn_graphnorm` (GraphNorm) | 0.99 | 0.72 | 0.97 |
| `gn_global` (+ virtual global node) | 0.98 | 0.76 | 0.97 |
| `gn_gat` (GATv2 attention) | 0.86 | **0.77** | 0.97 |
| `gn_jk` (+ jumping knowledge) | 0.98 | 0.75 | 0.98 |
| `gn_full` (gat + jk) | 0.87 | 0.74 | 0.97 |

Findings:

1. **The graph representation earns its keep on structure.** On `road` the GNNs
   hit R² 0.99 vs the engineered MLP's 0.83 — a ~0.15 gap that replicates at a
   second seed (eng 0.835, `gn_base` 0.986, `gn_gat` 0.880). This is the
   raison d'être for the board graph: it recovers a connectivity quantity the
   hand-tuned vector fundamentally lacks.
2. **Attention is the wrong inductive bias for counting/structural board
   tasks.** `gn_gat` collapses to 0.86 on `road` (vs 0.99 for plain MPNN) while
   *leading* on `win` (0.77) — softmax averaging dilutes the count/path signal
   that sum-aggregation message passing preserves. Longest road is a sum along a
   path, not a weighted average.
3. **The target's locality picks the architecture.** `heuristic` is a *local*
   sum of per-vertex production → plain MPNN reproduces it best (gat/global add
   nothing). `win` is *global* → the virtual global node and attention help
   (the two biggest gains over `gn_base`). `road` is *structural* → sum-MPNN
   dominates and attention hurts.
4. **GraphNorm and jumping-knowledge don't pay** on this small fixed graph (54
   nodes, ~5 diameter): GraphNorm is neutral-to-worse, JK neutral-to-worse, both
   adding parameters and optimisation difficulty for no gain — over-engineering
   for a graph this size.
5. **No absolute positional encoding.** Every preset is invariant under the
   board symmetry group and the player relabeling (enforced in
   `settlrl-learn/tests/test_architectures.py`) — a rotated board is the same
   game, so the signal must come from features, not vertex indices. The
   invariance survives every lever (attention, GraphNorm, global node, JK).

### Multi-task (one shared trunk, a head per target)

`ablate_multi` trains one trunk with four heads (win + heur + road + a new
`turns`-to-end head) — the supervised rehearsal for the AZ value+policy net, and
a test of whether one trunk serves local/global/structural targets at once.
Held-out per head (vs the single-task scores above):

| arch | win | road | heur | turns |
| --- | --- | --- | --- | --- |
| `mlp_engineered` | 0.81 | 0.81 | 0.98 | 0.52 |
| `gn_base` | 0.77 | 0.94 | 0.96 | 0.36 |
| `gn_global` | 0.75 | 0.82 | 0.94 | 0.40 |
| `gn_gat` | 0.74 | 0.76 | 0.94 | 0.42 |

- **Negative transfer onto the structural head.** Sharing the trunk degrades
  `road` (gn_base 0.99→0.94, gn_global 0.98→0.82, gn_gat 0.86→0.76) while the
  easy heads (heur, win) barely move — the structural signal competes for trunk
  capacity and loses. The higher-capacity global/attention variants degrade
  *more*, not less: their extra capacity is pulled toward the easy heads.
- **Plain sum-MPNN is the most robust shared trunk.** `gn_base` loses the least
  `road` and is the only arch whose `win` *improves* under multi-task
  (0.738→0.766) — the structural auxiliary genuinely helps its value head.
- `turns`-to-end is the one target where the engineered MLP clearly beats every
  GNN (0.52 vs ≤0.47): a global tempo signal the hand-tuned race/VP terms carry
  and the raw board does not.

Caveat: equal loss weights, single seed — the negative transfer is partly a
loss-balancing artefact (the easy heads dominate the gradient). Down-weighting
heur/turns, or separate trunks, should protect `road`. The lesson for AZ stands:
a single value+policy trunk wants plain sum-MPNN, with loss-balancing to keep
the structural signal the policy needs.

### Does the GNN reason about longest road, or just count edges? (`road_probe.py`)

Longest road is a longest-trail problem (NP-hard in general, classically hard for
a depth-limited GNN), so R² 0.99 on the greedy `road` target was suspicious. The
probe builds a *controlled* mix — near-simple paths, branchy bushes, and
opponent-broken trails, edge counts 3..15 — and trains `gn_base` at several
depths. The discriminating cases are where **edge-count ≠ longest-road** (45% of
the set; there the mean count is 11.1 but the true longest road is 7.2): a net
that merely counts edges would predict ~11.

| depth | R² all | R² hard | pred(hard) | true(hard) | count(hard) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.81 | 0.58 | 8.0 | 7.2 | 11.1 |
| 2 | 0.97 | 0.92 | 7.0 | 7.2 | 11.1 |
| 3 | 0.98 | 0.94 | 7.4 | 7.2 | 11.1 |
| 5 | 0.98 | 0.94 | 7.3 | 7.2 | 11.1 |

- **It is not edge-counting.** On the hard subset the predictions track the true
  longest road (7.0–7.4), not the edge count (11.1) — real path/break reasoning.
- **But it is depth-limited, as the BFS intuition predicts.** A 1-layer GNN
  fails on the hard cases (R² 0.58, predictions drifting toward the count at 8.0):
  one hop cannot reason about a trail, so it falls back to counting. The jump is
  at **depth 2** (0.58→0.92), then it plateaus — the receptive field has to span
  the trail, and depth 2–3 suffices for this board's ~5 diameter (a larger board
  would demand more depth).
- The greedy `road` 0.99 was thus partly an easy-distribution artefact (greedy
  roads are near-simple paths of mean length 3.4, where count≈longest and even a
  shallow net wins); on a genuinely hard mix the capability is real but needs
  depth ≥ 2.

**Architecture recommendation (toward the AlphaZero value+policy net):**
`gn_global` — sum-aggregation MPNN + a virtual global node + a
count-preserving multi-aggregator readout + LayerNorm, **no attention, no
GraphNorm, no JK**. It is the robust all-rounder: best-or-tied on the global
(`win`) target where the value head lives, and still ≫ engineered on structural
`road` (0.98). Attention is rejected despite its `win` edge because a value+policy
net must also read structure, where attention is catastrophic. Single seed on
`win`/`heuristic` (the ~0.01–0.02 spreads there are within noise); the `road`
and attention effects are large and replicated.

*Superseded 2026-07-30/31*: the 0004 four-arm study and the az2 gate pass
showed `gn_hetero` beats `gn_global` in the full AZ loop; `gn_hetero` is the
adopted trunk (see JOURNAL).
