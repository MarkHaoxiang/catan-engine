# Featurization v2 Implementation Plan (APPROVED by Mark 2026-07-30 — "go ahead with featurization")

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task, ONLY after Mark approves it. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the probe-demonstrated featurization gaps that cap the value head — the net cannot see its own hand composition, dev-card composition, `free_roads`, `longest_road_len`, or `pending_discard`, and its readout carries 96 collinear dims — as a versioned, flag-gated featurization (v1 stays bit-exact default) feeding the four-arm architecture study.

**Evidence base:** the 2026-07-30 correctness audit (probe-verified byte-identical featurizations for materially different states; policy KL 0.004 = converged, value_loss flat = the binding constraint) + the HGNN research fan-out (incidence features as the structure-free control; HNHN normalization for hetero; +2-layers as the reach control).

## Global Constraints

- Standard session constraints (no git reset/clean; explicit pathspecs; hooks on; foreground; CPU-only until the GPU frees; `.claude/settings.json` untouched; standard commit trailers).
- **v1 bit-exactness**: `board_sample(version=1)` (default) byte-identical to today — the flag-off goldens and the az0 anchor keep working untouched. v2 is opt-in via `GraphNetConfig`/experiment config; new goldens minted for v2 in the same task that lands it.
- The az0 anchor sidecar gains explicit dims so `load_anchor` builds v1-shaped templates regardless of the code's current default.
- Every claim of "the net can now distinguish X" gets the SAME byte-level probe that demonstrated the blindness, inverted (assert the featurizations now differ).

---

### Task 1: Versioned `board_sample` + the global-feature repairs

`graph.py`: `board_sample(..., version: int = 1)`; v2 `_global_features` appends (all player-relative): `player_resources[p]` (5), `dev_hand[p]` (5), opponent hand-size vector as today, `free_roads` (own), `longest_road_len` (own + max-opponent), `pending_discard` flag. Scaling: divide count features consistently (resources /5?, document choices); fix the pips/5 inconsistency between `_node_features` and `_tile_features` in v2 only. DESERT: already fixed in the small-fixes commit (verify). Tests: inverted blindness probes (hand-composition pairs now differ; dev pairs differ; v1 unchanged byte-identical), dims assertions via `_dims()`, symmetry suite green for v2 (player-relative additions preserve relabel invariance — the tests prove it).

### Task 2: Readout + normalization v2

`graphnet.py`: v2 readout `[max, sum, std]` (drop the collinear mean — evidence: mean ≡ sum/54 on fixed N, trained contribution std 0.081 vs 3.377); LayerNorm on the readout context (`ctx`) before the heads (evidence: init value logit ~80% bias, unnormalized norms 373/10/7/34). Both v2-gated in `GraphNetConfig`. Tests: init-scale probe (value logit spread across positions materially > bias share), v1 untouched.

### Task 3: Incidence-features arm (the structure-free control)

v2 option `incidence: bool`: per-vertex concatenation of its ≤3 adjacent hexes' (resource one-hot 6 incl. desert, pips/5, robber, number one-hot 11) — UN-summed, fixed order (canonical hex index; padded for coast vertices) — restoring per-tile identity and number identity without new message passing (the research-endorsed cheap arm; also the pips-collapse fix: 6-vs-8 distinguishable via number one-hot). NODE_DIM grows accordingly. Equivariance: the fixed per-vertex hex ORDER must be symmetry-consistent — verify against `_symmetry.py`'s permutations (if a canonical order breaks D3 equivariance, use a symmetric encoding: sum + max + sorted-by-pips, and document the tradeoff). This is the subtle step — the symmetry suite is the gate.

### Task 4: root_q target repair — RESOLVED, DROPPED (2026-07-30)

The probe ran with the seam fix: root_q sits ~2–3pp of win-prob below the Gumbel mixed value and the gap shrinks with sims — the expected Sequential-Halving exploration effect, not a structural bias, and not the 0.38-vs-0.53 explanation (the hand-blindness fixed by Tasks 1–3 is the leading suspect for that). Numbers recorded in `.superpowers/sdd/gnn-optimization-notes.md` and `packages/settlrl-search/CLAUDE.md`. No code change.

### Task 5: Study configs + goldens + GPU-day checklist

`conf/experiment/` variants for the four-arm study, all on v2 features: `v2_base`, `v2_incidence`, `v2_deep` (+2 layers), `v2_hetero` (+HNHN degree-norm — port per the research's α/β parameterization). New v2 flag-off goldens. The GPU-day checklist file (scratchpad → docs/): seam-fix confirm (search_step_ms), layout-transpose check, bf16 probe, width sweep {96,144,192} re-priced at the post-seam-fix NN share (~26% of a sim, not 15% — the panel's w144 arithmetic needs redoing), price the featurizer DFS on GPU — the CPU number (4-8% of a forward) does not close it, then the four-arm head-to-heads at 400 games paired.

---

## Explicitly deferred / rejected

- Roll-event nodes: behind the four-arm study + panel (unchanged).
- bf16, layout-transpose fix: GPU-day verification first (perf examiner's checklist).
- Trade-field features (3-4p only): with the eventual multi-player push.
- expected_rolls late-entry-globals architecture: only if expected_rolls is ever wanted back.
