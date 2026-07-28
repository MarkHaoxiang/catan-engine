# settlrl-learn — internal notes

Training-side package: depends on engine + agents, never the reverse.
Anything an agent needs at play time (the plain-JAX `mlp` forward, params as
an ordinary pytree, `.npz` artifacts) is deliberately dependency-free so a
trained model can ship without training libraries.

`experiment/` is the lab harness for `experiments/` (`Run`/`start_run`
bookkeeping + the pydantic/OmegaConf `Config` base) — moved here from
settlrl-agents (it is a training-side concern; this keeps the play/serve
library free of `pydantic`/`omegaconf`). *Not* imported by `__init__`, so
`import settlrl_learn` stays free of those; `pydantic`/`omegaconf` are learn
deps only because this subpackage uses them.

- `features.py` — engineered blocks mirror the heuristic's terms (production,
  expansion, ports, awards): we know they carry signal, and a model that
  cannot beat the heuristic *from the heuristic's own inputs* is not worth
  shipping. `FEATURE_DIM` is computed at import via `jax.eval_shape` on a
  2-player template (the own/max/mean aggregation makes the width
  player-count invariant, so 2p suffices).
- `train.py` — full-batch SGD only (the toy fitter); the real loop is the
  AlphaZero modules below. The value head is a win-probability logit: the
  searches read leaves as `tanh(v / value_scale) = 2P(win) − 1`, so logistic
  targets line up with the June 11 calibration finding (P(win) =
  σ(0.053·v_heuristic)). The AZ net's logit maps in with `value_scale=2`
  (`tanh(logit/2) = 2P−1`).
- **Networks live under `nn/`** (training-side, *not* imported by the package
  root — a subprocess guard test, `tests/test_import_light.py`, asserts
  `import settlrl_learn` pulls no equinox/jraph/flashbax/optax/orbax). The
  shipped MLP is the exception, reached by the root via an equinox-free
  `nn/__init__`:
  - `nn/mlp.py::AZParams` — the shared-trunk value+policy net (`make_az` adapts
    it onto the search's `value`/`prior` seams). Plain-JAX, root-importable.
  - `nn/graph.py` — the board-as-graph featurization (`board_sample` → a
    `Sample` of per-node/-edge/global features + per-hex `tiles` + the engineered
    vector, fixed topology as module constants). `tiles` (the per-hex node
    features) and the vertex↔hex incidence (`VT_V`/`VT_T`) feed the *heterogeneous*
    trunk; `board_sample(with_tiles=False)` skips them (a constant-zero `tiles`)
    for a non-hetero net, keeping its graph free of tile ops.
  - `nn/architectures.py` — the equinox architectures over it (`mlp_engineered`
    / `mlp_flat` / `deepset` / `gnn`, via `make_model`); experiment 0003
    composes them. `deepset`/`gnn` are invariant under the board's symmetry
    group and player relabeling (readout pools over nodes; ownership is read
    relatively); `mlp_flat` is not, by design. Enforced in
    `tests/test_architectures.py` against the generators in `tests/_symmetry.py`
    — the automorphism group is order 6 (D3), not the bare graph's D6, because
    the harbors are only 3-fold symmetric.
  - `nn/graphnet.py::GraphTrunk` — the shared message-passing trunk (encoders +
    layers → per-node embeddings + global + pooled readout); `GraphNet` (single
    head) and `BoardGNN` build their heads on it. `GraphNetConfig.hetero` (preset
    `gn_hetero`) adds **hexes as first-class nodes** (19, their own encoder) with
    vertex↔hex message passing per layer (vertex→hex and hex→vertex aggregates +
    a hex update), *keeping* the vertex↔vertex road-edge MPNN + global node; the
    hex pool joins the readout, and the per-tile policy head reads hex embeddings.
    Equivariant by construction (message passing over the static incidence; the
    symmetry tests already permute tiles). Off by default and **bit-identical**
    to before (the non-hetero init is preserved by only consuming the extra RNG
    keys when hetero).
  - `nn/action_layout.py` — the static map from the flat 662 action space to its
    board structure (per-vertex / -edge / -tile vs. dense "other") + `SCATTER` to
    place a factored head's compact logits back into the flat vector. The
    robber/knight *victim* collapses to no-steal/steal (opponent-relative
    features can't individuate victims, so a per-victim logit could not be
    relabel-invariant).
  - `nn/board_gnn.py::BoardGNN` — the value+policy net (`GraphTrunk` over
    `board_sample`; the `gnn_seams` search adapter lives here). Value and policy
    heads **split right after the trunk**; the policy is **structure-factored**
    (a shared per-vertex / per-edge / per-tile head for spatial actions, a dense
    head for the rest, plus a per-type class-balance bias).
    `tests/test_architectures.py` enforces value invariance, policy
    *equivariance* under board symmetry (action at v ↦ action at σv,
    `action_permutation`), and player-relabel invariance.

- **The training loop** (`training/`, training-side, *not* imported by the
  package root): one net-agnostic self-play → replay → train → arena loop behind
  a `Backend` seam, so the flat-MLP and board-GNN paths share it. Experiment
  0004 composes it (`net=mlp|gnn`).
  - `training/config.py` — the grouped, validated knob surface (`LearnConfig`
    and its sub-configs: `SelfPlayConfig` / `OptimConfig` / `ReplayConfig` /
    `TeacherConfig` / `ValueBlendConfig` / `EvalConfig` / `ArenaConfig`, plus
    `SearchSettings` — a subclass of settlrl-search's pydantic `SearchConfig`
    that adds training defaults). `learn` takes one `LearnConfig`; each group is
    `extra="forbid"` so a typo'd knob fails loudly. `SearchSettings.value_scale`
    is the *net* leaf's logit scale (2); the heuristic teacher search keeps the
    factory default (its own calibration).
  - `training/steps.py` — the per-iteration body as pure, separately-testable
    units (`prepare_targets` = held-out split + value-blend; `train_epochs` =
    the inner minibatch loop; `evaluate`; `run_arena`). The loop derives every
    RNG key from `seed` + iteration index and threads it in, so the steps stay
    pure and bit-exact resume is preserved.
  - `training/backend.py` — the `Backend` protocol (the net-specific surface:
    `init` / `seams` / `play_agent` / `setup_policy` / `observe` / `to_item` /
    `empty_item` / `init_opt` / `make_step` / `eval_metrics`) and `RunState`
    (net + optimiser moments + replay buffer + iteration + best + the self-play
    carry). `RunState` is **eqx-serialised** (`save_run_state`) — eqx's leaf
    serialiser fits both an equinox module and a plain-JAX pytree, so it
    replaced orbax for *both* backends.
    `selfplay_carry` is the padded self-play pool (below) and is deliberately the
    **last** field, so a checkpoint written before it existed is exactly the file
    minus its trailing section: `load_run_state` reads the older fields, and if
    the file ends there keeps the template's (empty) carry. Resuming into a
    differently-configured run is covered in three distinct ways, because the
    shapes do not see everything: `selfplay.persistent` **off** skips the carry
    section unread (a `None` template carry — the run has no reader for it, so a
    persistent checkpoint's pad costs nothing and checks nothing); a mismatched
    section that *is* read — `persistent` turned **on** over a pool-less
    checkpoint, or a changed `selfplay.batch` / `max_game_len` / `value_blend` —
    trips eqx's per-leaf shape/dtype gate, which `load_run_state` re-raises as a
    `ValueError` naming those knobs; and `search.ordered`, which no shape can
    see, rides as its own leaf checked by `from_padded`.
    Checkpoint size: the pad is fixed-shape, so a *persistent* run pays it in
    full at every write — **1.82 GiB** at B=256 / `max_game_len` 800 / GNN obs +
    662-wide policy, ~0.5 s to build and ~0.5 s to write (measured 2026-07-28,
    independent of how full the pool actually is). It is transient: the loop
    frees each `to_padded` result after the write and drops the zero template
    once a persistent run is under way (every checkpoint then pads the live
    carry), so steady-state host RAM is the live pool alone (~0.6 GiB at those
    shapes). A non-persistent run pads to zero rows and pays 266 KiB (the env
    arrays). If that write cost ever bites, the lever is a separate pad bound
    below `max_game_len`, not the fixed shape.
  - `training/selfplay.py::self_play` — batched n-player self-play, the search
    (net's or a fixed teacher's) guiding the re-determinizing moves and improved
    policy. The backend's `observe` records the *true* board (net learns the
    belief-averaged value); values are the acting seat's eventual win/loss.
    `setup_fn` (when given) plays the setup phase with a fixed policy and those
    positions are *not* recorded (the GNN path; the MLP path passes `None` and
    the net plays setup too). **Playout-cap randomization** (KataGo): with a
    `fast_search` and `full_prob` < 1, each *step* (not per-move — the
    vmap-lockstep constraint) is full (deep `search`) with prob `full_prob` else
    fast (cheap `fast_search`); every position records its outcome value, but the
    `train_policy` flag is 1 only on full-search positions, so the policy loss
    trains on deep targets only (value on all). `full_prob` = 1 disables it.
    **Opening-temperature anneal** (`temperature_moves` > 0): a lane samples at
    `temperature` for its first that many *recorded* moves of its current game,
    then argmax for the rest — counted off the live pending length
    (`len(pending[lane])`), which is already the per-lane recorded-move count
    (resets on flush; only undercounts once a lane is trimmed past
    `max_game_len`, by which point the count is already far past any sane
    `temperature_moves`), so no carry field was added for it. 0 (the default)
    keeps `temperature` flat and draws no extra RNG.
    Returns `(Samples, SelfPlayStats, SelfPlayCarry | None)`: `env_steps`
    (batched env steps, each advancing all lanes), `recorded`, and `discarded` —
    positions generated but never returned, i.e. the pending buffers of games
    still unfinished when the call hit its sample target, plus `max_game_len`
    trims. The loop logs the latter as `selfplay_discarded`: the
    *iteration-boundary waste* (games are cut mid-flight every iteration, and
    only finished games yield samples) — measured at 72.8% of searched positions
    at B=256 and 91% at B=1024, which is what gates the batch lever (exp 0004,
    2026-07-28).
    **The persistent carry** (`selfplay.persistent`, opt-in) is the fix: the
    call returns a live `SelfPlayCarry` — the env object itself (a stateful
    wrapper over batched arrays with auto-reset, so it can simply be held and
    re-stepped), the per-lane pending buffers, the RNG key, the per-key sample
    shape+dtype `spec`, and a `surplus` counter — and passing it back resumes
    the games in flight. `surplus` is the samples handed out past the cumulative
    request (a finished game flushes whole, so a call overshoots); crediting it
    to the next call is what makes a sequence of persistent calls of `n` produce
    *exactly* what one call of their total would, positions and RNG stream
    included (asserted in `tests/test_training.py`) — provided no call exhausts
    its own `max_steps`, which is a per-call budget. A resumed call whose request
    the surplus already covers takes **zero** env steps and returns empty arrays
    built from `spec` — hence the carried dtypes, since `mask` is bool and a
    float32 empty would silently promote a concatenated stream.
    `discarded` then counts only trims.
    Flag off, everything is bit-identical to before (a frozen digest golden
    captured pre-change guards the RNG stream and recording order). The cost:
    a persistent call's output is pure in (`seed`, carried state) rather than in
    `seed` alone — `seed` seeds only the first call — which is why the carry
    reaches the checkpoint (`to_padded` / `from_padded`, beside the carry).
    The padded form is fixed-shape because that is what an eqx deserialisation
    template is: every recorded key pads to `(batch, max_game_len, …)` in host
    numpy with a per-lane `pending_len`, and the env — a held *object*, not a
    pytree — contributes `PaddedEnv`, its complete array state with PRNG keys as
    raw uint32 (eqx cannot serialise typed key arrays). `from_padded` rebuilds an
    equivalent env by re-constructing one and overwriting that state; the ctor's
    `seed` is not live state (only `reset` reads it). A test asserts `PaddedEnv`
    still names every array attribute the env holds, so an engine-side addition
    breaks loudly instead of silently not being carried.
  - `training/loop.py::learn(backend, cfg: LearnConfig, *, teacher_value=…,
    checkpoint_dir=…, resume_from=…, …)` — the orchestrator over the `steps`
    units. Per-iteration RNG is a pure function of `cfg.seed` and the iteration
    index, so `resume_from` (a `runstate.eqx`) continues bit-identically (tested
    in `tests/` resume checks for both backends, and with `selfplay.persistent`
    on — where the pool in flight, not just the seed, decides what comes next).
    Under `persistent` the loop holds the carry across iterations and folds it
    into every checkpoint; a *zero-sample* iteration is then ordinary rather than
    degenerate (the carried surplus already covered the request, so the call took
    no env step), so it skips only the data steps — eval, replay add, optimiser —
    and still counts, checkpoints and proceeds. `cfg.optim.reuse` caps
    updates/iter at the AZ sample-reuse factor (the value-overfit fix); every
    `cfg.eval.every` iters the first `cfg.eval.samples` of that iter's fresh batch
    are scored (`eval_metrics` -> the `val_*` metrics) under the *pre-train* net,
    before the batch trains -- a valid held-out-in-time signal that wastes no data
    (the whole batch still trains); `teacher_value` (with `cfg.teacher.iters` > 0)
    warm-starts from a fixed strong search at `cfg.teacher.sims` (the cold-start
    fix). `cfg.value_blend.max` > 0 trains value on Canopy's `(1−α)z + α·q`
    (game outcome blended with the searched root `q` from
    `make_search_weights_value`, α ramped 0→max over `cfg.value_blend.ramp`
    iters) — the dice-variance fix; only the training slice is blended, the eval
    slice keeps raw `z` (see the Canopy reference below). `cfg.search.chance_nodes`
    /`dev_chance` thread the search's explicit chance-node mode through self-play
    (the backends carry the same flags for the arena `play_agent`), so the search
    plans past rolls at train and play time. `cfg.search.ordered` turns on the
    action-ordering lock-out (`settlrl_engine.ordering`): self-play's env runs
    `track_ordering` and the search threads the lock-out deeper; the backends
    carry it for the arena agent too.
    `loop.selfplay_callables(backend, cfg, net)` builds the once-built
    jitted+vmapped self-play callables (`view_of` / `observe_of` / `setup_search`
    + a `make_net_search(num_simulations)` factory closing over the net's *static*
    part); `learn` and `bench_selfplay` share it so the wiring cannot drift.
  - `training/arena.py::arena` — the net's `ArenaResult(wins, episodes)` vs. a
    `POLICIES` opponent, seat-swapped at 2p (`lookahead` = the Stage-1 gate;
    `random` = the lower-bound sanity check); the play agent comes from
    `backend.play_agent`. `episodes` is the real completed-game count, not the
    requested `n_games` (`evaluate`'s win-count sync happens between scan
    windows, so it overshoots) — real `(wins, episodes)`, not `winrate * n_games`,
    feed the Elo MLE. `steps.run_arena` plays each `cfg.arena.opponents` entry and
    reports `arena_winrate` / `arena_vs_<opp>` **plus `arena_elo`** — the MLE Elo
    (`training/elo.py::anchored_elo`) on the fixed `cfg.arena.anchor_elos` scale
    (heuristic pinned at 0 = the gate; random well below) — **and `arena_elo_se`**,
    its standard error (`anchored_elo_se`, Fisher information at the MLE).
    `cfg.arena.opponent_every` (opponent -> N) skips an opponent on rounds where
    `run_arena`'s `round_index` (the loop's count of arena invocations) isn't a
    multiple of N, saving wall-clock on anchors that no longer carry information
    (e.g. `random`, which pins at 1.0 winrate early). The
    loop holds the arena **seed fixed across iterations** (no `+i`), so every
    checkpoint faces the same games and the curve is paired (the dice/board luck
    differences out)
    — the chosen variance cut, matching canopy/lc0's paired-seed tournaments
    over a checkpoint round-robin (a within-pool round-robin drifts when the pool
    changes; the anchored gauntlet stays comparable across runs). Anchors must
    stay frozen for a run. The per-iter `val_*` / `policy_*` / `value_*` health
    metrics (from `Backend.eval_metrics`) are the cheap high-frequency proxies
    between arena rounds.
    The optimiser is `steps.make_optimizer(cfg.optim)` — adamw, optionally
    preceded by `clip_by_global_norm` (`cfg.optim.grad_clip`, default 1.0; 0
    disables). The clip is stateless, so an unclipped checkpoint must be resumed
    with `grad_clip=0` (its opt-state has no clip layer).
  - `training/bench.py::bench_selfplay(backend, net, cfg, *, warmup, repeats,
    seed)` — the self-play throughput probe (the loop's dominant cost, isolated:
    no optimiser/replay/arena). `warmup` untimed calls at `seed + 1000` pay the
    XLA compile, then every timed repeat runs the *same* workload at `seed`, so
    the spread across `t_0..t_n` is measurement noise. Reports medians:
    `samples_per_s`, `moves_per_s` (a move is one lane-step, `env_steps * batch`),
    `sims_per_s` (`moves_per_s * search.num_simulations`). Rejects
    `pcr_full_prob` < 1 — playout-cap randomization breaks the sims-per-move
    accounting — and rejects `selfplay.persistent`: the repeats deliberately do
    not thread a carry, so the workload would stay fresh-env while `discarded`
    reported a persistent run's.
  - `tests/benchmark/` holds the pytest-benchmark suite (net forward, one
    vmapped search step, one self-play window, one optimiser step) on a
    random-init `BoardGNN` — `benchmark`-marked and deselected from the
    default run; the repo-root `run_benchmarks.sh` re-selects it. Each
    benchmark's compile-paying warm-up runs outside the timed region (the
    self-play one via `benchmark.pedantic`'s `setup`), so the headline
    min/mean/median is steady-state throughput.
  - `training/mlp_backend.py::MLPBackend` — the `AZParams` net over the
    engineered feature vector; **unmasked** policy CE + value-logistic loss,
    optax adamw, the net plays setup itself.
  - `training/gnn_backend.py::GNNBackend` — the `BoardGNN` net over the board
    graph; **masked** policy CE (softmax over the legal set only) + value loss,
    eqx-filtered optax step. `setup_policy` (a fixed `lookahead`/expectimax
    opener) plays the setup phase in both self-play and the arena;
    `make_net_agent` composes setup + the net's search; `gnn_loss` is the masked
    loss (its finiteness is contract-tested).
  - Both losses average the policy CE over the item's `train_policy` = 1
    positions only (value-only playout-cap positions are skipped; value trains on
    all); with `train_policy` all 1 it is the plain mean (bit-exact-preserving).

The gates (June 11 plan; value-tuning evidence in settlrl-agents/CLAUDE.md,
search/leaf evidence in settlrl-search/CLAUDE.md): Stage 1 ships a
value only if `lookahead(net)` beats `lookahead(heuristic)` at ≥2σ, n≥400
(`settlrl-agents bench`); Stage 2 reruns the sims ladder — depth pays nowhere
with the stationary heuristic leaf, and that falsification is the reason this
package exists; Stage 3 (policy head, self-play iteration) only after.

## Reference: Canopy (`cullback/canopy`)

A Rust AlphaZero framework whose flagship example is a 1v1 Catan agent
(`nexus-v3`, claimed "strongest public 1v1 Catan agent" — unbenchmarked against
ours). It is the point past our leaf-is-the-ceiling gate: learned policy + WDL
value head, self-play, Gumbel improved-policy interior selection + PUCT/Dirichlet
root (800 sims), explicit chance nodes for dice and dev draws, and Single-Observer
ISMCTS that filters per-simulation legality in a custom tree (our search now does
this too — `settlrl_search.ismcts`, which retired the mctx engine). It is
1v1 only, so it never meets the 3-4p
paranoid-frame / opponent-model problem, and it *disables determinization during
self-play* (the net learns the Bayesian-average-over-hands policy; determinize
only at play time).

Techniques aimed at Catan's dice variance (the variance-starved-depth problem):

- **Value-target blending** `(1−α)·z + α·q` (outcome blended with the searched
  root q) — **done**: `learn`'s `value_blend_max`, q from
  `make_search_weights_value`.
- **Explicit chance nodes** for dice + dev draws — **done** (opt-in
  `chance_nodes`/`dev_chance`; details in settlrl-search/CLAUDE.md). Canopy also
  forces a canonical **action ordering** to cut transpositions — **done** (opt-in
  `ordered`, `settlrl_engine.ordering`).
- **EMA auxiliary value heads** at horizons (e.g. `[4, 10, 30]`), trained on
  `ema = α·Q[t] + (1−α)·ema`, sharing the trunk — *not yet*.
- **Playout-cap randomization** (KataGo): most moves a small search, a fraction
  the full budget; only full-search positions contribute policy targets —
  **done** (opt-in `selfplay.pcr_full_prob` < 1 + `pcr_fast_sims`; per-*step*
  full/fast rather than per-move, the vmap-lockstep constraint; the policy CE is
  masked to full-search positions via the item's `train_policy` flag). Pairs with
  a larger `search.num_simulations` for the full steps — the affordable way to add
  the search depth the policy diagnostic wants.

Repo + METHODS.md + examples/catan/OPTIMIZATIONS.md; see [[canopy-reference]].
