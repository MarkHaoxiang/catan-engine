# settlrl-agents — internal notes

Settlrl agents over `settlrl-engine`'s public flat-action seam: pure-JAX
policies plus stateful plain-Python planners
(`planner/`). The search subsystem and its substrate (seat protocols, world
sampling, flat decode, value/prior seams) live in the `settlrl-search` package
(layering `settlrl-engine -> settlrl-search -> settlrl-agents`); agents depends
on it.

**No agent assumes full observability.** Model-based agents consume the
engine's honest `BeliefView` (see the engine's `belief.py` notes); hidden
state is unrepresentable there, so the only road back to a playable position
is `sample_world`. There is no 2p/4p module split: with two players the
tracked belief is exact on resources (tested in the engine), so "2p is
perfect-info" is a property of the data, not an API boundary — the same
agents run at 2–4 players with beliefs of varying sharpness.

## Layout

The API layer is the top-level modules — `value.py` (the heuristic's *weights*,
re-exporting the `Value`/`ValueFunction` types from `settlrl_search.value`),
`evaluate.py`, the registry in `__init__.py` (which defines `search_policy` /
`lookahead_policy` locally over `make_search` imported from `settlrl_search`),
`cli.py` — plus the agents (`baselines.py`, `greedy.py`, `planner/`).
`internal/` holds only
`feature_engineering.py` (the weight-free hand-engineered features —
`board_features` for the value terms, `target_build` / `maritime_ratio` for
greedy's trade sense). Weights always live with an agent or in `value.py`;
features never carry them.

The lab harness (`Run`/`start_run` bookkeeping + the pydantic/OmegaConf `Config`
base) lives in `settlrl_learn.experiment` — a training-side concern, which
keeps `settlrl-agents` (the play/serve library) free of
`pydantic`/`omegaconf`.

`service/` is the **one-bot SDK + service**. A service hosts a single `Bot`
(`sdk.py`: subclass it, implement `act(view) -> MoveModel`; the framework tracks
each game in flight and hands the bot a `GameView` from the acting seat). `app.py`
(`create_app(bot)`) serves it over the shared wire protocol
(`settlrl_game.botproto`): `GET /info` (the bot's `BotInfo`) and `POST /act`,
which applies the moves the bot has not seen yet — the incremental, structured
`MoveModel` tail after `base`, with a `409 {resync, have}` handshake when its
tracked game is behind — then returns the chosen move. `bots.py` is the four
bundled engine bots (`EngineBot` over a `POLICIES` policy): it bridges the
tracked reference position to an engine board (`bridge.py`) and translates the
chosen engine flat back to a `MoveModel`. The CLI `settlrl-bot-service --bot KIND`
serves one. It's the `settlrl-app` game server's only source of bot moves
(delegated over HTTP), kept here because it *is* the agents running. Behind the
`[service]` optional extra (`fastapi` + `settlrl-game`) and never imported by
`__init__`, so `import settlrl_agents` stays free of `fastapi`
(`sdk.py`/`app.py` themselves are JAX-free; only `bots.py` pulls the engine). The
`bridge.py` dtype contract is exact: engine
`BoardLayout`/`BoardState` arrays are `uint8` (jaxtyping-enforced under the test
hook — render's conftest never checked engine types, so an int32 slipped by).

## API layer and agents

- `value.py` — heuristic strength function; value = own strength − best
  opponent's. On a *sampled* world the "hidden" fields it reads are
  belief-consistent samples, so it stays honest. Tuning evidence (2p
  seat-swapped CLI matches, 200–700 games): the expansion + progress terms
  took lookahead-vs-greedy from 34% to 86.5% (without the
  best-buildable-spot term, lookahead never expanded); the hand-diversity
  term is worth ~55% head-to-head over without; `w_spot` 1.0 vs 0.5 and a
  port-count term measured neutral. June 11 overnight sweep: the wheat/ore
  production premium (`w_wheat_ore=0.25`; 0.4 overshoots, 44.5% vs 0.25) plus
  the closing-urgency term (`w_race=0.8`; 1.2 measured equal) beat the prior
  weights 57.3% (n=600) on the lookahead rung and 57.5% (n=200) on the mcts
  rung, no greedy regression (90.5%); a production-matched 2:1-port synergy
  term measured negative (47.3%) and stays at `w_port=0`. June 12: the
  production-scarcity hand term (`w_scarce`, a sqrt-hand copy weighted by
  1/(1+production)) prices cards the player cannot produce, so port/domestic
  conversions toward them read as gains — 57.3% (n=205) over `w_scarce=0` on
  the lookahead rung at the adopted 1.0 (0.5 measured 52.2%; 1.5 lost to 1.0
  head-to-head 43.1%). June 12 night feature round (gated
  lookahead-vs-lookahead ablations, n=200 probes → n=400 confirms): adopted
  `w_numbers=0.3` (distinct dice numbers collected on — income smoothness)
  + `w_kheld=0.8` (held knights toward the army race), 56.1% ± 2.4% (n=415)
  over without; the full triple with `w_spots=0.6` measured *worse* (53.2%
  — spots overlaps the best-spot term) and a summed-completeness `w_fill`
  was outright negative (43.1% at 1.0: it rewards hoarding toward several
  builds, i.e. discard exposure) — both knobs kept at 0. Ladder re-checked
  on the new defaults: relative rungs unchanged (every top agent shares the
  leaf), absolute level up. `make_linear(weights)` deploys any named-
  coefficient fit over `BoardFeatures` as a ValueFunction — the classical-
  fit seam (experiments/0002: outcome-fit logistic reaches hand-tuned level
  vs greedy but loses head-to-head 42.7%; held-out AUC is flat across
  candidates whose match probes span 53–78%, so candidate selection must be
  by match, never by fit metric; `second_spot` / `reach` / `army_lead`
  joined the features at weight 0). Value-as-win-prob
  calibration (183k self-play positions): P(win) = σ(0.053·v), phase-stable —
  but the calibrated `value_scale≈38` *lost* to the sharper hand-picked 20 in
  mcts (44.5%, n=200): honest calibration is not the best search temperature.
- `greedy.py` — the scripted agent: its tier table and bonus coefficients
  over the obs-side features. A static per-row tier score plus small
  observation bonuses. Invariant: tier gaps (≥ 100) exceed every bonus range
  (|bonus| < 50), so bonuses only reorder within a tier; types sharing a tier
  are phase-disjoint, dominated, or deliberately bonus-decided. Two sanctioned
  exceptions, both trade-shaped (June 12, new greedy beat the old 79.0%
  n=305 2p, 67.5% vs chance 33% at 3p; vs random 85%→98.7%): a *productive*
  MARITIME row carries a +150 gate lifting it over END_TURN — productive
  means the bought card is needed for the target build (city if a settlement
  stands, else settlement if a spot is buildable now, else dev card) and the
  sale comes entirely out of surplus, so conversions can't ping-pong; and
  ACCEPT vs REJECT is bonus-decided — accept iff paid purely from surplus
  and (a need advances or it consolidates toward scarcity). The discard
  prefers surplus before most-held. Still deliberately simple: never offers
  a trade (an obs-only policy has no rejected-offer memory), ignores whose
  production the robber blocks. Greedy's tier table is `TIER_SCORES`, which
  lives in `settlrl_search.priors` (greedy imports it) and is also the search's
  interior prior (`_TIER_LOGITS`) — the maritime gate lives in the bonus
  channel, so priors are unchanged.
- `evaluate.py` — fused driver over the engine's `rollout(actor=...)` seam:
  every seat's vmapped agent picks in every lane each step inside the scan and
  the acting seat's pick is kept — n_seats policy evals per step, fine for
  ≤ 4 seats. Steps run in `_SYNC_WINDOW`-sized scans; the win count syncs only
  between windows, so `n_episodes` may overshoot by up to a window of lanes.
  Measured June 11 (B=32, RTX 5090): 1.7× over the per-step loop on
  lookahead-vs-greedy, ~1.3× steady-state on mcts matches (42 vs ~57 ms/step).
  Caveat: the scan retraces per `evaluate` call (the actor closure is fresh
  each time) — ~12 s per call for mcts-sized bodies, amortised over 200-game
  matches, noticeable on ≤ 20-game probes. `compile_evaluate(make_agents, ...)`
  is the memoisable escape: it returns a `run(params, ...)` bit-identical to
  `evaluate(make_agents(params), ...)` on the fused path, but builds the seats
  *inside* its jit from the traced `params` pytree (via the same
  `_picker`/`_actor` and the engine's private `_rollout_core`, imported here so
  the two paths cannot drift) — so new arrays of the same structure reuse the
  compiled scan. Callers own the caching (settlrl-learn's arena memoises the
  returned callables); stateful seats are rejected.
  A `StatefulSpec` seat switches `evaluate` to `_evaluate_stepwise`: the same
  seating/budget semantics through a per-step Python loop (stateful seats act
  lane by lane on host-fetched observations; pure seats keep their `_picker`,
  but jitted — eagerly the vmapped greedy alone is 46 ms/step vs 0.1 jitted).
  ~18 ms/step at B=16 on GPU, dominated by the per-step env dispatch; win
  counts sync every step, so the `n_episodes` overshoot is at most a batch.

## search/

The re-determinizing SO-ISMCTS search (`make_search` / `make_search_weights` /
`make_search_weights_value`, the tree, the lookahead `num_simulations=0` case,
the chance-node and action-ordering flags, the trade machinery) lives in the
`settlrl-search` package — see its CLAUDE.md. `__init__.py` here imports
`make_search` from it and defines the `search_policy` / `lookahead_policy`
registry families locally.

## planner/

The stateful decision-tree class: per-game plain-Python agents whose
*strategy* is code (plans, saving, award races, trade memory) and whose
*tactics* consult a one-step lookahead (hybrid sanctioned June 12; it was
value-free before). `pov.py` is the host-side toolkit — one `Pov` per
decision wrapping the host-fetched observation, the static board graph
re-stated as numpy/python tables (`VERTEX_*`, `EDGE_ENDPOINTS`,
`TILE_CORNERS`), and the flat table's host decode (`flat_row`,
`ROWS_OF_TYPE`, re-exported from `settlrl_search.rows`). `tree.py` is the framework
(`Node` / `Selector` / `Plan` / `Blackboard`); `goals.py` the goal economics
(`plan_candidates` / `choose_plan` and their scoring weights); `tactic.py`
the lookahead seam; `agent.py` the tree's nodes and the shipped `planner`
family. The tactic's batched seams (`_after_many`, `_best_replies`) take
fixed-size row blocks padded with a repeated legal row — fixed shapes mean
one jit trace instead of one per block size.

`tactic.py`: `reconstruct` rebuilds a single-game engine board from the
observation — public fields exactly; hidden ones neutrally (opponent hands
spread evenly, their dev cards as knights since only the count enters the
opponent strength term, the dev deck scaled to the unseen remainder,
`free_roads` inferred from an unaffordable-but-legal BUILD_ROAD). The env's
own mask is passed through as the availability `apply_action` trusts, so
reconstruction gaps can't make an illegal action look applied. `Tactic`
caches one 662-wide successor-value sweep per decision (`values`), and
`combo_best` runs the own-turn second ply: apply an enabler, re-sweep
legality (`flat_available_for`), and value the follow-ups — the structural
edge over `lookahead`, which is one-ply and provably weak exactly there
(the June 11 END_TURN→BUY_DEV/TRADE flip diagnosis). `best_paranoid` is the
opponent reply: my candidates within 1.0 of the top are tie-broken by the
opponents' best answer — apply my row, fabricate their MAIN turn on the
result, max over their *grounded* options (builds/bank trades; their dev
plays are masked out because the reconstructed dev hand is fiction).
Measured neutral on the lookahead rung; kept as the principled tie-break
(robber/knight targets, surplus builds). Expectimax over the engine's two
chance seams: `roll_expectation` (exact 11-outcome forced rolls — decides
the pre-roll knight: play iff the relocation raises the expectation of our
own pending roll by > 0.3) and a deck-weighted exact dev-draw expectation
replacing the sweep's single-sample BUY_DEV value. Both measured neutral —
the same lesson as smcts: with the shared stationary heuristic leaf, exact
chance handling doesn't convert into strength; the leaf is the ceiling.

Design invariants:

- **Legality only ever comes from the mask.** Leaves pick among legal rows
  (or build a row with `flat_row` and check it); no rule is re-implemented,
  so engine rule changes can't silently desync the agent. If the whole tree
  declines, a fixed-priority fallback picks some legal row (PROPOSE_TRADE
  deliberately absent there — an unmanaged offer could re-propose forever).
- **Plan steps are declarative targets, not queued actions.** Every tick the
  plan reports its first step missing from the board and the agent re-checks
  the rest; a step gone impossible (spot taken, path edge claimed) triggers a
  replan. That re-validation is what makes state safe across opponents'
  moves *and* auto-reset (a stale plan against a fresh board just invalidates).
- **Memory covers what the engine forgets.** The engine keeps no record of
  rejected trade offers, so the blackboard does: a proposal is remembered
  with the hand that made it, marked rejected if the next MAIN tick shows no
  trace of it, and never re-offered that turn (`max_proposals_per_turn` caps
  the rest). This is the stall-guard the completing-games test exercises.
- **Goals are scored in economic time, switched on clear margins.** A goal's
  score is quality-weighted pips (`_RES_WEIGHT`: wheat/ore premium) minus
  0.35 × the bottleneck rounds production needs to afford it, plus
  closing-urgency near 10 VP (a goal that wins outright dominates at
  25/builds). Settlement goals carry the spot-race model (+3.4 vs lookahead,
  the largest opponent-integration win): an opponent road already touching
  the spot beats any multi-road path of ours (−2·len−1), one edge away is a
  reason to hurry (+0.5), and each path edge adjacent to their network risks
  being cut (−0.4).
  Candidates include the two award races: a Longest Road grab when within two
  trail extensions of taking the card (`my_longest_trail` DFS), and dev buys
  boosted while Largest Army is live. A healthy plan is re-scored against
  fresh rivals each turn and replaced only on a > 2.5 margin (≫ the 0.3
  tie-break noise, so no thrash); `_PLAN_PATIENCE` (8) remains only as the
  starvation backstop for invisibly-starved goals (no dev-deck size or
  `free_roads` in the observation).
- Turn shape after the plan step: `OpportunisticBuild` (any build/dev buy
  fundable from pure surplus, argmaxed by successor value against END_TURN's
  as the do-nothing baseline), `EnablerCombo` (a maritime trade or YOP whose
  *pair* value with an immediate build beats ending the turn; the follow-up
  is committed via `Blackboard.forced_row` and played next tick),
  `Acquire` (YOP/monopoly toward need — monopoly also fires on a ≥ 4-card
  mass grab — maritime from surplus, capped 1:1 proposals), `SpendDown`
  (ending a turn above seven cards banks the excess in a dev card,
  reservations or not), `DenialKnight` (value-timed end-of-turn knight).
  Setup spots, robber/knight targets, discards, and trade responses all pick
  by successor value (setup keeps scripted new-resource/expansion bonuses the
  one-ply value can't see).
- Gotchas: A dev-buy step's "realized"
  check is a baseline comparison on the public dev count (the hand count
  drops again when cards are played, but the plan completes on the next tick,
  before that can happen). The jaxtyping hook enforces `pov.py`/`agent.py`
  annotations at test time — `sum()` of an empty generator is `int`, not
  `float`.

Strength (seat-swapped, June 12, hybrid + opponent model): 85.0% vs
greedy (n=200), **~55% vs lookahead** (54.7% and 55.7% on n=300 runs) and
**49.6% vs mcts pooled** (546/1100 over the final configs) — tied with mcts
at the top, clearly over lookahead. Gate planner tweaks at n ≥ 200 — n=100
probes swung ±6 points around the n=300 truth. The one measured trap:
letting plain maritime sales dip into plan-reserved cards (47% → 29% vs
lookahead, reverted; combos may, because the pair value prices the whole
exchange). Move latency: the tactic sweep adds ~1 jit dispatch per decision
on top of the ~0.1 ms/lane scripted tick; matches run ~1.5-2x slower than
the pure-scripted planner.

## cli.py

`compare` is a seat-swapped head-to-head: two `n_episodes` evaluate runs
(`seed`, `seed + 1`) with the agents' seats exchanged. `bench` is the
experiment harness: agents are `build_spec` strings (a registry name, or
JSON `{"kind", "params", "value"}` building a configured family member /
reweighted heuristic), seat-swapped at 2 players and seat-rotated at 3-4
(one evaluate per seating, `seed + position`); output includes the binomial
SE and the per-seating split. Tournaments etc. slot in as new subparsers.

## Registry and tests

`__init__.py` exports the `POLICIES` registry — the single list of shipped
agents, consumed by both the protocol tests and settlrl-app's bot seam
(which dispatches on the spec's class and filters seat counts).

Tests are protocol-level only (`tests/test_policies.py`), parametrized over
every agent in `POLICIES` at its `for_testing` parameters: legality through
self-play, seed reproducibility, and episode-budgeted rollouts that must
complete games (`_self_play` drives stateful specs lane by lane, mirroring
`_evaluate_stepwise`). No per-agent internal-logic
tests — a new agent just registers in `POLICIES`. settlrl-app's bot seam
skips `StatefulSpec` families (its `bot_act` is per-move and stateless; a
stateful seat there needs a per-session agent cache that doesn't exist yet).
`sample_world` lives in `settlrl-search` with its unit tests
(`tests/test_sample.py`) — infrastructure, not a policy, so it has no
protocol-level tests here.
`tests/conftest.py` installs the jaxtyping/beartype hook for all
`settlrl_agents` modules.

`tests/benchmark/` holds the pytest-benchmark suite (move latency,
`sample_world`, the fused self-play window), parametrized over `POLICIES` at
*shipped* defaults — it measures what ships, unlike the tests' cheap
`for_testing` members. `benchmark`-marked and deselected from the default
run; the repo-root `run_benchmarks.sh` re-selects it (engine + agents). The
self-play window reuses `evaluate`'s private `_picker`/`_actor` so one actor
identity persists across rounds — calling `evaluate()` per round would
retrace its fresh closure every time.
