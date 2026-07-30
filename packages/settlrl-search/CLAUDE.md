# settlrl-search — internal notes

The re-determinizing search subsystem and the engine-only seam primitives it
sits on, extracted from settlrl-agents so the learn/agents layers share it
without a cycle. Layering: `settlrl-engine -> settlrl-search -> settlrl-agents
-> settlrl-learn`. No agent registry or strength tuning lives here — those stay
in settlrl-agents; this package is the *search* and its substrate.

Module map:

- `ismcts/` — the custom fixed-capacity SO-ISMCTS tree (`make_tree`,
  `SearchConfig`); split into `config.py` (the pydantic `SearchConfig`, the
  jit-static `_Cfg`, the Sequential-Halving schedule, the tree dtype helpers),
  `tree.py` (the `_Tree` store + select/expand/backup), `descent.py` (the
  determinize/descend/evaluate walk + engine seam), and `loop.py` (`_run` +
  `make_tree`).
- `__init__.py` — the `make_search` wrapper assembling root prior / lookahead /
  `num_trees` over the tree, plus the trade machinery.
- `_common.py` — shared prior/dice constants (`_TIER_LOGITS`, `_ROLL_P`,
  `_NO_PROPOSE`) and the `PolicyWeights` / `PolicyWeightsValue` types.
- `expectimax.py` — the setup-phase search (`make_setup_search`).
- `policy.py` — the seat protocols and `AgentSpec` registry machinery.
- `sample.py` — `sample_world` determinization.
- `rows.py` — the flat-action decode (was `settlrl_agents.internal.rows`).
- `value.py` — the `Value` / `ValueFunction` seam types.
- `priors.py` — `TIER_SCORES`, the action-priority prior shared with greedy.

## policy.py / sample.py / rows.py

- `rows.py` — the flat action table decoded once (device `ROW_TYPE` /
  `ROW_PARAMS` for the vmapped sweeps; host `ROW_IDX` / `ROW_TARGET` /
  `ROWS_OF_TYPE` / `flat_row` for the planner). Every agent imports from
  here — there used to be one decode per module, and they can silently
  diverge.
- `policy.py` — the seat protocols and `AgentSpec`: a policy *family*
  (`make` + `defaults`, with `policy` the cached shipped build) plus optional
  `for_testing` parameter overrides — `spec.for_tests` is the cheap family
  member the protocol tests run (the tested properties are
  parameter-independent). `AgentSpec` is generic over its protocol and the
  subclass is the tag (`ObservationSpec` / `BeliefSpec` / `StatefulSpec`), so
  consumers dispatch with `isinstance` and `spec.policy` is precisely typed —
  no casts. A `StatefulSpec`'s `policy` is a *factory* (`seed -> GameAgent`):
  the agent object holds per-game state, so drivers build one per (game,
  seat) and replace it when the lane auto-resets. `GameAgent.act` takes
  *host* data (`HostObservation` / `HostFlatMask`, numpy): handing device
  arrays to host-side logic cost ~10 ms per decision in thirty per-field
  syncs; one `jax.device_get` of the batched observation costs ~0.1 ms/lane.
  The generic cannot type `defaults` itself: `make(**mapping)` is uncheckable
  (ParamSpec doesn't apply to dynamic unpacking). `PolicyPrior` is the
  learned-policy-head seam: `make_search` takes one in place of its built-in
  priors (root sweep + tier table), applying legality masking itself.
  `ValuePrior` extends it with `with_value` — the shared-trunk nets' seam, which
  also serves the leaf value (see the leaf-evaluation note below); both are
  re-exported from the package root. Policies are masked-argmax style: with no legal
  move the returned index is arbitrary and the engine rejects it as
  `INVALID` (the lane stalls until auto-reset), matching
  `BatchedSettlrlEnv.random_actions`.
- `sample.py` — `sample_world` fills every hidden field with a posterior
  sample. Guaranteed (tested in `tests/test_sample.py`): public fields
  untouched; hand sizes, dev counts, per-type totals, and the observer's own
  rows all match the public record. The resource deal's
  proportional-headroom weighting is a *surrogate* for the exact posterior,
  not the posterior (`hi` is relaxed if jointly infeasible). The deal's
  `while_loop` stops at the owed count rather than the worst-case 95: same
  sampling law (each draw is a fresh key), 13x on `sample_world`, 7x on a
  lookahead move at B=1 (RTX 5090; the sequential chain was launch-bound).
  The closing
  `BoardState(...)` is built by explicit keyword on purpose: a new
  `BoardState` field fails to compile here until classified public or hidden.

## value.py / priors.py

- `value.py` — the `Value` / `ValueFunction` seam types: a value scores a
  board for one player (higher is better), the search evaluates a leaf
  through it. The shipped heuristics that fill the seam (`heuristic_value` /
  `make_linear` / `tuned_value`) live in settlrl-agents' `value.py`, which
  re-exports these types.
- `priors.py` — `TIER_SCORES`, the static per-action-type priority tier. It
  is both greedy's per-row tier score (the dominant term of its argmax, in
  settlrl-agents) and the search's interior-node prior (`_TIER_LOGITS` in
  `_common.py`, scaled there). Tiers are spaced so no per-target bonus
  (|bonus| < 50) can cross between them; types sharing a tier are never both
  the argmax. The one exception is deliberate: a *productive* MARITIME_TRADE
  row carries a +150 gate in greedy's bonus channel (not here), so priors are
  unchanged.

## ismcts/ / __init__.py / _common.py

`make_search` / `make_search_weights` (`__init__.py`) are the
re-determinizing **Single-Observer ISMCTS**: a custom fixed-capacity tree
(`ismcts/`, `make_tree`) that determinizes a fresh `sample_world` per
simulation and descends the live engine, filtering legality per simulation — the
half mctx's fixed action axis could not express (Cowling 2012; the Canopy custom
tree). Selection is mctx's Gumbel-MuZero ported onto it (no `mctx` dependency).
`make_search` argmaxes the improved policy; `make_search_weights` returns the
distribution (the AlphaZero policy target — experiment 0004);
`make_search_weights_value` returns `(distribution, root value)` — the searched
root value (searcher frame, 2·P(win)−1) is the AZ value-blend `q` target, the
visit-weighted mean of the root edges (`_run`). Shared prior/dice
constants live in `_common.py`, the trade/lookahead/`num_trees` wrapper in
`__init__.py`, the tree in `ismcts/`. It replaced a former
`mcts`/`smcts`/`ismcts`/`lookahead` quartet (2026-06-17) then the `mctx` engine
behind it (2026-06-19, 742b94b). ~5–6 ms/move (B=1 CPU; was 7.4 with mctx).

**One leaf evaluation per simulation.** A leaf needs both a value and an
expansion prior, and XLA does **not** merge two calls of the same shared-trunk
net on the same state (bisected 2026-07-30: trivial featurizations merge, the
real `board_sample` does not — a dot census found two complete trunks in the
simulation `while` body). So a net hands the search one `ValuePrior` seam
(`policy.py`) whose `with_value` returns both from a single forward;
`descent._value_and_logits` takes it when the prior offers one and otherwise
calls the two seams as before (the heuristic-leaf + tier-prior path is
unchanged). **It is then the prior's value head, not the `value` argument, that
scores leaves** — right for a *paired* seam, wrong for an unpaired
`value`/`prior` (a heuristic value under a net's policy head, measured Δq up to
0.15), so `fused_leaf=False` (`make_tree` / `make_search*` / `SearchConfig`)
forces the two-seam path back. Measured on a `gn_global` 96×4 net, 4 sims: 211.7 → 141.4 MFLOP per
compiled search (2.99 → 2.00 forwards), 18 → 9 in-loop node dots; with
`expected_rolls`, 983.9 → 913.5 MFLOP (the roll EV's 11 value calls stay).
Outputs are bit-identical — the same graph, emitted once — and settlrl-learn's
`tests/test_leaf_seam.py` censuses the in-loop dots against a deliberately split
seam (observed 5 vs 10 at width 8 / 2 layers; the *ratio* is the assertion) and
pins that both emit the same weights, so the duplication cannot come back
silently. `tests/test_ismcts.py` covers the dispatch itself with a fake
`ValuePrior` whose value head is a scaled leaf: fused and unfused differ when the
seams disagree, and are bit-identical when they agree.

**The leaf is the ceiling.** The binding constraint is the stationary heuristic
leaf, not search machinery: win rate vs lookahead does *not* climb with sims (64
*loses*), so the lever is the leaf (experiment 0003 / settlrl-learn) and prior
agents all tied at ~parity — the cleanest one won. The merges, each a falsified
strength lever:
- `smcts`'s explicit dice/dev chance nodes (49.3% h2h, ~2× wall-clock; dev node
  50.5% n=210; 64→128 sims 53.3%→49.5%). Roll-EV leaves + per-simulation
  resampling subsume them — *with the stationary heuristic leaf*. **Re-added as
  an opt-in flag 2026-06-22** (`chance_nodes`, `dev_chance` in `make_tree` /
  `make_search[_weights[_value]]`): the descent is now a decision/chance state
  machine — a stochastic action (roll always; dev-buy under `dev_chance`) defers
  to a chance node (afterstate) that samples nature at its true probability
  (`_ROLL_P` / deck composition) and applies the engine's forced-outcome seam
  (`apply_action` with a forced `idx`), so the search plans *past* a roll. It
  supersedes `expected_rolls` (mutually exclusive). Default OFF (flag-off is
  bit-identical, the 13 baseline contracts hold; 4 chance contracts added). The
  bet: a *learned* value (settlrl-learn q-blend, exp 0004) may convert what the
  stationary leaf couldn't — pending a gated arena A/B.
- **Action-ordering lock-out** (opt-in `ordered` flag, 2026-06-22): the descent
  ANDs `settlrl_engine.ordering.ordering_mask` into the in-tree legal set and
  threads the per-turn `category` (reset on turn change), so the search explores
  only the canonical order of a turn's builds/buys/trades (transposition cut). The
  engine owns the rule; the search consumes it. Root mask comes from the env
  (`track_ordering`); the search continues the lock-out deeper from category 0
  (max-so-far keeps it consistent with the env category). Default OFF; 3 ordered
  contracts added. Also a gated A/B lever.
- `mcts` (frozen-world): per-simulation determinization is its principled
  superset, *parity not a win* at 3p (0.352 ± 0.031, n=244; 64 worlds 0.307 —
  more doesn't help; ~ties at 2p where the belief is ~exact).
- the **mctx engine**: its fixed action axis no-oped illegal path-actions; the
  custom tree reaches the same strength with true per-sim legality and faster.

**Strength: ~0.55–0.58 vs lookahead** (2p seat-swapped, n≥220 GPU; h2h vs old
mctx 0.49–0.52 across 16/32/64 sims). Contracts in `tests/test_ismcts.py`.

Design choices, each fixing a measured ply-2 bias:
- **Root prior = the raw one-step value sweep** (`values / prior_scale`, ±20
  spread), not a tanh+tier compression (which flattened it to near-uniform,
  0.184 vs lookahead — the decisive port bug, 928f370 / 742b94b; uniform made
  Gumbel's candidates a random subset, 6%). Interior prior = greedy's tier table
  (`TIER_SCORES` in `priors.py`, scaled into `_TIER_LOGITS`; uniform +
  deterministic argmax expanded the lowest-index action); a learned `prior`
  replaces both.
- **Two-sided paranoid frame** (searcher vs the table): every node holds the
  searcher's value signed into the mover's side — the true max^n reduction. At 2p
  provably identical to flipping on every mover change (632/640 same picks); the
  every-mover-flip rule negates the searcher's own next turn and went below chance
  at 4p (20% vs 3× lookahead; the side frame took the same seeds to 32.3%, n=161,
  chance 25%; 62.2% vs 2× lookahead n=90).
- **Chance**: per-simulation resampling, plus the immediate roll's exact 11-roll
  expectation (`ROLL_DICE` leaves).
- completed-Q does **not** min-max rescale (it amplified any Q ranking to ~8 nats
  regardless of noise).

The "search subtracts value" bug (34–43% vs lookahead, flat across
sims/candidates/scale): at ~32 sims trees are ~2 plies and full-depth selection
flipped ~9% of decisions, 92% losing 1-ply value (END_TURN → BUY_DEV/TRADE, the
optimizer's curse over noisy follow-ups). The fixes above took flips 12% → 7%
and the search 37% → **57%** vs lookahead (n=200). Defaults are the local
optimum (sims 64 loses, considered peaks at 16, prior_scale 5 loses, value_scale
12/38 tie-or-lose to 20). Diagnose decision-level, not by ~20-game matches (SE
±11%); 4p evals need matched seeds or n ≥ 240.

`num_simulations=0` is the **lookahead** special case: no tree, the masked argmax
of the root one-step sweep over `num_trees` sampled worlds. It is also the *only*
configuration that offers trades — `propose_rate` > 0 (default 0 for the search,
0.5 for the `lookahead` registry entry) scores proposals by their *accepted*
outcome under a partner model (the partner's seat must prefer accepting) minus
`trade_penalty` (0.25), gated so a mispredicted partner can't re-offer forever.
Offers are root-only: under the paranoid frame the in-tree responder prices every
offer as rejected, so proposals are dropped from the in-tree prior (`_NO_PROPOSE`)
and the search *answers* trades but never offers one. Lookahead offering measured
37.8% pooled (n=373) at 3p.

## expectimax.py

`make_setup_search` is the compile-efficient beam expectimax over the setup
phase — deep setup expectimax that ties lookahead at depth 6 (heuristic leaf
~additive; see [[gnn-alphazero-0004]]). The GNN backend uses it as the fixed
setup opener.
