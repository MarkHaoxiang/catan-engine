# settlrl-learn — internal notes

Training-side package: depends on engine + agents, never the reverse.
Anything an agent needs at play time (the plain-JAX `mlp` forward, params as
an ordinary pytree, `.npz` artifacts) is deliberately dependency-free so a
trained model can ship without training libraries.

`experiment/` is the lab harness for `experiments/` (`Run`/`start_run`
bookkeeping + the pydantic/OmegaConf `Config` base) — moved here from
settlrl-agents, a training-side concern. *Not* imported by `__init__`, so
`import settlrl_learn` (and the play/serve library) stays free of
`pydantic`/`omegaconf`, which are learn deps only because this subpackage
uses them.

- `features.py` — engineered blocks mirror the heuristic's terms (production,
  expansion, ports, awards): we know they carry signal, and a model that
  cannot beat the heuristic *from the heuristic's own inputs* is not worth
  shipping. `FEATURE_DIM` is computed at import via `jax.eval_shape` on a
  2-player template (the own/max/mean aggregation makes the width
  player-count invariant, so 2p suffices).
- **Networks live under `nn/`** (training-side, *not* imported by the package
  root — a subprocess guard test, `tests/test_import_light.py`, asserts
  `import settlrl_learn` pulls no equinox/jraph/flashbax/optax/orbax). The
  shipped MLP is the exception, reached by the root via an equinox-free
  `nn/__init__`:
  - `nn/mlp.py::AZParams` — the shared-trunk value+policy net (`make_az` adapts
    it onto the search's `value`/`prior` seams). Plain-JAX, root-importable.
    The value head is a win-probability logit: the searches read leaves as
    `tanh(v / value_scale) = 2P(win) − 1`, so logistic targets line up with the
    June 11 calibration finding (P(win) = σ(0.053·v_heuristic)), and the logit
    maps in with `value_scale=2` (`tanh(logit/2) = 2P−1`).
    Both adapters (`make_az`, `gnn_seams`) return their prior as a
    `settlrl_search.policy.ValuePrior` — a prior that also serves the value off
    the *same* forward. The search evaluates a leaf through it, so one trunk pass
    covers both heads by construction: on CPU, XLA does not merge two separate
    calls; on GPU, XLA already CSEs the duplicate forward, so the fix is
    GPU-throughput-neutral, confirmed by an in-loop bench (194.91 vs 193.81
    samples/s, +0.6%, noise) — its value is structural, not speed (evidence and
    CPU-only numbers in settlrl-search/CLAUDE.md). It is then the *prior's* value head
    that scores leaves, so pairing one with a different `value` needs
    `cfg.search.fused_leaf=False`; the loop's own pairings are all paired (the
    heuristic teacher search passes no prior at all).
  - `nn/graph.py` — the board-as-graph featurization (`board_sample` → a
    `Sample` of per-node/-edge/global features + per-hex `tiles` + the engineered
    vector, fixed topology as module constants). `tiles` (the per-hex node
    features) and the vertex↔hex incidence (`VT_V`/`VT_T`) feed the *heterogeneous*
    trunk; `board_sample(with_tiles=False)` skips them (a constant-zero `tiles`)
    for a non-hetero net, keeping its graph free of tile ops.
    **Versioned** (`board_sample(version=)`, `FEATURE_VERSIONS`, widths from
    `dims(version, incidence)`; `GraphNetConfig.feature_version` is what every consumer
    threads, and a frozen anchor pins its own in its sidecar). Version 1 is the
    original, kept byte-exact (digest-pinned in `tests/test_architectures.py`) —
    the az0 anchor and every v1 checkpoint read exactly those bytes.
    **Version 2** closes the blindness the 2026-07-30 correctness audit
    demonstrated byte-for-byte: v1's globals carry hand/dev *sizes* only, so
    "2 wheat + 3 ore" vs "5 wool" (and 2 knights vs 2 monopolies) featurized
    identically — the net could not see what it can afford or play. v2 appends
    the perspective player's per-resource hand, per-type dev hand, `free_roads`
    (masked by "my turn" — the engine stores it per *game*), the Longest Road
    length split own / opponent, and the pending discard; and it puts the
    per-vertex production term on the same `pips/5` scale as the per-hex features
    (v1 left node pips raw). Every count is divided by a "large but ordinary"
    value for that quantity (the divisors carry their reasoning at the constant)
    because nothing normalises `glob`. The Longest Road entries read the *award
    holder's* stored length rather than calling `longest_road_length` per player.
    The reason is **structural, not cost** (the DFS measures 4–8% of a forward,
    CPU B=128): `awards.road_build_gate` runs the DFS only at ≥5 own roads, so the
    engine never computes a sub-threshold length and there is no per-player vector
    to reuse — producing one would mean loosening the engine's own perf gate.
    Two blind spots follow, both by design: a trail below the 5-road award
    threshold reads 0 (road *counts* are in v1's per-player block), and so does a
    rulebook tie — a 2+ way tie among non-holders leaves the award unheld
    (`awards._reassign_award`), which zeroes the stored length, so both slots read
    0 even when the tied trails are 7 roads long.
    GPU pricing of the featurizer DFS is on the Task 5 checklist.
    **Incidence** (`board_sample(incidence=True)`, `GraphNetConfig.incidence`; a
    v2 *option*, not a version — the study's `v2_incidence` arm) appends to each
    vertex's node features its ≤3 incident hexes **un-summed**: resource one-hot
    (6 wide, the desert its own column), pips/5, robber flag, number one-hot
    (11, for numbers 2–12). +57 node columns (3 slots × 19), nothing else moved
    — the block is *appended*, so the v2 prefix and every other array keep their
    bytes, and no architecture changes (the node encoder just gets wider). It is
    the fix for the collapse that `tile_pips` (`6 − |7 − n|`) imposes on every
    other featurization (6≡8, 5≡9, 4≡10, 3≡11, 2≡12): number identity and
    same-number income correlation were unrepresentable, demonstrated by
    swapping a 6-tile's number with an 8-tile's — a materially different board
    that featurizes **byte-identically** under v1 and v2 (`tiles` block
    included), and differs only in the incidence number one-hot with it on.
    **Slot order is by the hexes' own attributes, not by hex index, and that is
    forced by equivariance.** A fixed canonical-hex-index order is *not*
    D3-equivariant: for symmetry #1 vertex 0's hexes `[0, 3, 4]` map to
    `[16, 12, 13]`, while σ(0) = 40 lists its own in index order as
    `[12, 13, 16]` — slot *k* of *v* and of σ(*v*) hold different hexes (128 of
    324 vertex-symmetry pairs mismatch). What a symmetry *does* preserve is the
    **multiset** of incident-hex attributes (it carries resource / number /
    robber along with the hex, permuting only positions), so sorting each
    vertex's slots by `_tile_sort_key` — injective on `(resource, number,
    robber)` and a function of those attributes alone — makes the slot sequence
    a function of the multiset, hence equivariant. Ties are the subtlety and are
    harmless *because the key is the whole payload*: two slots tie only when
    their feature rows are identical, so which one lands first cannot change the
    output. Sorting by pips (the plan's first candidate) would *not* do — it
    ties roughly an order of magnitude more slots between hexes with
    *different* payloads (board-dependent counts, not reproducible constants),
    where order would then fall back to index; the full-payload key's ties are
    all genuinely interchangeable. The tradeoff bought: no geometric slot
    identity — the net cannot tell which incident hex sits in which direction
    (unrecoverable in any equivariant featurization, since there is no canonical
    frame), and the "slot" index it reads is a lexicographic rank. Coast
    vertices (1 or 2 hexes) pad with an **all-zero slot placed last**; no
    presence flag, because a real hex always sets exactly one resource column
    (desert included), so all-zero is unambiguously "no hex".
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
    `feature_version>=2` (Task 2 of the featurization-v2 plan) changes two
    things, both gated on the same flag as the wider globals (one gate, since
    the four-arm study's `v2_base` needs all three together): the `"multi"`
    readout pools `[max, sum, std]` instead of `[max, sum, mean]` — on this
    fixed 54-node graph `mean` is a scalar multiple of `sum` (collinear;
    correctness-audit evidence: a trained net's mean-block contribution had
    std 0.081 vs the sum-block's 3.377) — and `GraphTrunk.ctx` (the pooled
    readout ++ global node `g`, the shared input to the value head and the
    policy's `other`/`type_bias` heads) is LayerNorm'd before those heads
    consume it. v1's raw `ctx` sits ~25x over the L2 norm its random-init
    heads are scaled for (sum-pooling over 54 nodes inflates magnitude
    regardless of signal; the audit's own measurement: block norms ~373/10/7
    vs `g`~34) — LayerNorm bounds it to `sqrt(dim)` by construction. Per-node
    embeddings (`h`, `h_t`) feed the spatial heads directly and are
    deliberately *not* normalized here — that would perturb the spatial
    heads' scale relationships, out of scope. v1 is byte-exact (`ctx_norm`
    is `None`, and `_pool`'s v1 branch is untouched).
    `GraphNetConfig.blocked_linear` (default off; no run adopts it yet)
    weight-blocks each message/update MLP's first Linear: one matmul per
    unique input row (vertex / tile / the single global row), gathered onto
    the edge set, instead of gather-then-transform per edge (perf evidence:
    the 2026-08-01 blocked_linear line in experiments/JOURNAL.md). Same
    parameters (the blocked matmuls slice the existing Linear's weight
    columns — a checkpoint loads under either flag) and the same function up
    to float summation order, but NOT bit-exact, so a resume across a flag
    flip diverges; it ships as an opt-in architecture revision for new runs.
    `tests/test_blocked_linear.py` pins parameter identity, float32
    closeness, the float64 reassociation identity (asserted 1e-12, observed
    ~7e-15), and the symmetry contracts flag-on.
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
  - `training/backends/base.py` — the `Backend` protocol (the net-specific surface:
    `init` / `seams` / `play_agent` / `setup_policy` / `observe` / `to_item` /
    `empty_item` / `init_opt` / `make_step` / `eval_metrics`) and `RunState`
    (net + optimiser moments + replay buffer + iteration + best + the self-play
    carry). `RunState` is **eqx-serialised** (`save_run_state`): eqx's leaf
    serialiser fits both an equinox module and a plain-JAX pytree, replacing
    orbax for both backends. `selfplay_carry` is the padded self-play pool
    (below) and deliberately the **last** field, so a checkpoint written before
    it existed is exactly the file minus its trailing section — `load_run_state`
    reads the older fields and keeps the template's empty carry if the file ends
    there. Resuming into a differently-configured run is covered three ways,
    since shapes alone don't see everything: `selfplay.persistent` **off** skips
    the carry section unread (no cost, no check); a mismatched section that *is*
    read — `persistent` turned **on** over a pool-less checkpoint, or a changed
    `selfplay.batch` / `max_game_len` / `checkpoint_pad` / `value_blend` —
    trips eqx's per-leaf shape/dtype gate, re-raised by `load_run_state` as a
    `ValueError` naming those knobs; `search.ordered`, invisible to any shape,
    rides as its own leaf checked by `from_padded`.
    **Checkpoint format (rev 2026-08-01):** replay items store `mask` and
    `train_policy` as **bool**; the losses cast to float32 at use — same
    values in, and loss/grads verified byte-exact on CPU
    (`test_bool_item_dtypes_are_loss_and_grad_bit_exact_with_float32`). A
    pre-rev `runstate.eqx` fails at load with eqx's per-leaf dtype error
    naming the leaf — by design, no migration; the pre-carry fields raise
    eqx's error directly (only carry mismatches get the `ValueError` rewrap).
    Checkpoint size: the pad is fixed-shape, so a *persistent* run pays it at
    every write, independent of pool fullness. `selfplay.checkpoint_pad`
    bounds the pad below `max_game_len`: at save time a lane whose pending
    exceeds the bound keeps only its most recent rows (the game's tail), so a
    resumed run **diverges from the uninterrupted one on any truncated lane**
    — `to_padded` counts what the bound dropped and the loop logs it as
    `checkpoint_truncated_lanes`/`_rows` on checkpoint iterations. Only the
    default `None` (full pad) is lossless by construction. The scale-class
    presets (scale2, scale2_long, v2_*) set 512, clearing the measured
    production pending distribution (v2_hetero final checkpoint, B=512:
    median 88 / p90 189 / p99 272 / max 398 rows) with headroom — at 256 the
    observed 7/512 over-bound lanes would put P(no lane truncated) around
    0.1% per save, i.e. practically every resume diverging. That distribution
    is audited on a final (trained-net) checkpoint; early-training games run
    longer, so an early-run kill-and-resume may truncate more lanes — each
    truncated lane still keeps its most recent `checkpoint_pad` rows, so the
    cost stays bounded: divergence plus the dropped samples, never corruption.
    A persistent B=512 / 200k-item-buffer checkpoint writes **~4.4 GB in
    ~1.2 s** (measured 2026-08-01, CPU, NVMe; before/after split in the exp
    0004 JOURNAL line); the production presets run B=1024, roughly doubling
    the carry share (~6.5 GB expected — re-audit the pending tail there).
    Transient: the loop frees each `to_padded` result after writing and drops
    the zero template once persistent, so steady-state host RAM is just the
    live pool; non-persistent pads to zero rows, 266 KiB. `save_run_state`
    writes to a sibling `.tmp` and `os.replace`s it into place — atomic, so a
    kill mid-write (the validated long-run procedure) leaves the previous
    checkpoint intact.
  - `training/selfplay/play.py::self_play` — batched n-player self-play, the search
    (net's or a fixed teacher's) guiding the re-determinizing moves and improved
    policy. The backend's `observe` records the *true* board (net learns the
    belief-averaged value); values are the acting seat's eventual win/loss.
    `setup_fn` (when given) plays the setup phase with a fixed policy and those
    positions are *not* recorded (the GNN path; the MLP path passes `None` and
    the net plays setup too). **Playout-cap randomization** (KataGo): with a
    `fast_search` and `full_prob` < 1, each *step* (not per-move — the
    vmap-lockstep constraint) is full (deep `search`) with prob `full_prob` else
    fast (cheap `fast_search`); every position records its outcome value, but
    `train_policy` is 1 only on full-search positions, so the policy loss trains
    on deep targets only (value on all). `full_prob` = 1 disables it.
    **Opening-temperature anneal** (`temperature_moves` > 0): a lane samples at
    `temperature` for its first that many *recorded* moves of its current game,
    then argmax for the rest — counted off the live pending length
    (`len(pending[lane])`, the per-lane recorded-move count, reset on flush; it
    only undercounts once a lane is trimmed past `max_game_len`, well past any
    sane `temperature_moves`), so no carry field was needed for it. 0 (default)
    keeps `temperature` flat and draws no extra RNG.
    Returns `(Samples, SelfPlayStats, SelfPlayCarry | None)`: `env_steps`
    (batched env steps, each advancing all lanes), `recorded`, and `discarded` —
    positions generated but never returned: the pending buffers of games still
    unfinished when the call hit its sample target, plus `max_game_len` trims.
    The loop logs the latter as `selfplay_discarded`, the *iteration-boundary
    waste* (games cut mid-flight every iteration; only finished games yield
    samples) — measured at 72.8% of searched positions at B=256 and 91% at
    B=1024, gating the batch lever (exp 0004, 2026-07-28).
    **The persistent carry** (`selfplay.persistent`, opt-in) is the fix: the call
    returns a live `SelfPlayCarry` — the env object itself (a stateful wrapper
    over batched arrays with auto-reset, so it can simply be held and
    re-stepped), the per-lane pending buffers, the RNG key, the per-key sample
    shape+dtype `spec`, and a `surplus` counter — and passing it back resumes
    the games in flight. `surplus` is the samples handed out past the
    cumulative request (a finished game flushes whole, so a call overshoots);
    crediting it to the next call makes a sequence of persistent calls of `n`
    produce *exactly* what one call of their total would, positions and RNG
    stream included (asserted in `tests/test_selfplay.py`), provided no call
    exhausts its own per-call `max_steps` budget. A resumed call the surplus
    already covers takes **zero** env steps and returns empty arrays built from
    `spec` — hence the carried dtypes, since `mask` is bool and a float32 empty
    would silently promote a concatenated stream. `discarded` then counts only
    trims. Flag off, everything is bit-identical to before (a frozen digest
    golden captured pre-change guards the RNG stream and recording order). The
    cost: a persistent call's output is pure in (`seed`, carried state) rather
    than `seed` alone — `seed` seeds only the first call — which is why the
    carry reaches the checkpoint (`training/selfplay/carry.py`, below).
  - `training/selfplay/carry.py` — the pool types and the projection that checkpoints
    them: `SelfPlayCarry` (live, above), `PaddedCarry`/`PaddedEnv`, the
    `to_padded`/`from_padded` pair and the `empty_padded`/`carry_template` zero
    template, plus `recorded_spec` (the single source of truth for the derived
    recorded keys, so a call site's spec and the padding code's notion of
    "derived" cannot drift) and `make_env` — self-play's *only* env construction
    site, since a carried pool is restorable only into an identically-built env.
    The padded form is fixed-shape, as an eqx deserialisation template must be:
    every recorded key pads to `(batch, pad, …)` in host numpy with a per-lane
    `pending_len`, where `pad` is `max_game_len` bounded by
    `selfplay.checkpoint_pad` (truncation semantics under `backend.py` above);
    the env — a held *object*, not a pytree —
    contributes `PaddedEnv`, its complete array state with PRNG keys as raw
    uint32 (eqx cannot serialise typed key arrays). `from_padded` rebuilds an
    equivalent env by re-constructing one and overwriting that state (the
    ctor's `seed` isn't live state — only `reset` reads it). A test asserts
    `PaddedEnv` still names every array attribute the env holds, so an
    engine-side addition breaks loudly instead of silently not being carried.
  - `training/loop.py::learn(backend, cfg: LearnConfig, *, teacher_value=…,
    checkpoint_dir=…, resume_from=…, …)` — the orchestrator over the `steps`
    units. Per-iteration RNG is a pure function of `cfg.seed` and the iteration
    index, so `resume_from` (a `runstate.eqx`) continues bit-identically (tested
    for both backends, and with `selfplay.persistent` on, where the pool in
    flight — not just the seed — decides what comes next). Under `persistent`
    the loop holds the carry across iterations and folds it into every
    checkpoint; a *zero-sample* iteration is then ordinary rather than
    degenerate (the carried surplus already covered the request, so the call
    took no env step) — it skips only the data steps (eval, replay add,
    optimiser) and still counts, checkpoints and proceeds. `cfg.optim.reuse`
    caps updates/iter at the AZ sample-reuse factor (the value-overfit fix);
    every `cfg.eval.every` iters the first `cfg.eval.samples` of that iter's
    fresh batch are scored (`eval_metrics` → `val_*`) under the *pre-train* net,
    before the batch trains — a held-out-in-time signal that wastes no data (the
    whole batch still trains); `teacher_value` (with `cfg.teacher.iters` > 0)
    warm-starts from a fixed strong search at `cfg.teacher.sims` (the cold-start
    fix). `cfg.value_blend.max` > 0 trains value on Canopy's `(1−α)z + α·q`
    (outcome blended with the searched root `q` from
    `make_search_weights_value`, α ramped 0→max over `cfg.value_blend.ramp`
    iters) — the dice-variance fix; only the training slice is blended, the eval
    slice keeps raw `z` (see the Canopy reference below). `cfg.search.chance_nodes`
    /`dev_chance` (explicit dice/dev chance nodes) and `cfg.search.ordered`
    (the `settlrl_engine.ordering` action lock-out, via self-play's
    `track_ordering`) thread through self-play and the arena `play_agent`
    alike, so both plan past rolls and respect the lock-out at train and play
    time.
    `loop.selfplay_callables(backend, cfg, net)` builds the once-built
    jitted+vmapped self-play callables (`view_of` / `observe_of` / `setup_search`
    + a `make_net_search(num_simulations)` factory closing over the net's *static*
    part); `learn` and `bench_selfplay` share it so the wiring cannot drift. It is
    **memoised** (`loop._CALLABLES_CACHE`, keyed on the backend's identity and the
    whole search config + the value-blend factory choice, guarded by the net's
    static -- a mismatch on the single per-key entry rebuilds and overwrites it):
    a fresh closure is a jit cache miss, so an uncached second `learn` in one
    process re-traced every self-play jit — a measured ~68 s startup spike at
    scale (8.49 s → 1.54 s per warm `learn`, even on the tiny test config,
    against a cache-clear control). It does **not** move the test-suite floor
    (65 s → 83 s over the split: xdist workers are separate processes, most
    tests build a fresh backend, and the XLA disk cache already absorbs the
    compiles). Reuse is semantically free (the callables are pure in that key;
    the net's arrays are a traced argument, never closed over), and
    `test_selfplay_callables_*` pins it — a warm-hit `learn` must reproduce the
    cold-built one leaf-for-leaf. The key omits `selfplay.batch` and the seat
    count (they ride the traced arguments' shapes, which jax keys its own cache
    on) and the setup knobs (which live on the backend); its static check
    compares *treedef and non-array fields*, not array shapes (`AZParams`
    statics are all-`None`, so two MLP widths compare equal) — it catches a
    structurally different net reaching an entry, while a same-shape
    different-width net is already separated by its backend's identity.
- **Evaluation** (`evaluation/`, head-to-head gating of frozen nets; no
  runtime dependency on `training` — `steps.run_arena` composes it):
  - `evaluation/arena.py::arena` — the net's `ArenaResult(wins, episodes)` vs. a
    `POLICIES` opponent, seat-swapped at 2p (`lookahead` = the Stage-1 gate;
    `random` = the lower-bound sanity check); the play agent comes from
    `backend.play_agent`. The seat-swap/seed/episode logic lives once, in
    `arena_spec`, which takes a **pre-built** opponent spec or a `NetOpponent`
    (a frozen net + its backend); `arena` is the name-based wrapper that
    resolves `POLICIES`. Each seating's evaluate callable is **memoised**
    (`arena._EVAL_CACHE`, keyed on backend identity + opponent identity + seat
    order + sims/considered/batch; mirrors `loop._CALLABLES_CACHE`'s mechanics
    — id-keyed, referents held alive in the value, a net-static guard) over
    `settlrl_agents.evaluate.compile_evaluate`, with the current net's *and* a
    `NetOpponent`'s arrays as traced arguments — so an arena round retraces
    nothing after its first firing (previously ~2 fused-scan recompiles per
    opponent per round every round, the fresh-closure retrace: production
    round at the scale config measured 548.5 s cold → 363.0 s cached,
    2026-08-01, vs 521.2 s/round pre-change where every round paid the
    compiles). Bit-identity with the plain `evaluate` composition is pinned
    by `test_arena_spec_matches_the_plain_evaluate_composition`; a
    `StatefulSpec` opponent falls back to the uncached stepwise driver
    (nothing fused to cache). `episodes` is the real
    completed-game count, not the requested `n_games` (`evaluate`'s win-count sync
    happens between scan windows, so it overshoots) — real `(wins, episodes)`, not
    `winrate * n_games`, feed the Elo MLE. `steps.run_arena` plays each
    `cfg.arena.opponents` entry and reports `arena_winrate` / `arena_vs_<opp>`
    **plus `arena_elo`** — the MLE Elo (`evaluation/elo.py::anchored_elo`) on the
    fixed `cfg.arena.anchor_elos` scale (heuristic pinned at 0 = the gate; random
    well below) — **and `arena_elo_se`**, its standard error (`anchored_elo_se`,
    Fisher information at the MLE).
    `cfg.arena.opponent_every` (opponent → N) skips an opponent on rounds where
    `run_arena`'s `round_index` isn't a multiple of N, saving wall-clock on
    anchors that no longer carry information (e.g. `random`, which pins at 1.0
    winrate early). **Frozen checkpoints join the gauntlet** through
    `learn(..., net_opponents={name: (opponent, elo, every, phase)})` →
    `run_arena`: ready specs or `NetOpponent`s, so the library never learns
    about checkpoint files or architectures — the experiment composes them
    (0004's `anchors.load_anchor` + a pinned-setup `GNNBackend` wrapped as
    `NetOpponent`, keeping the rung's arrays traced-arg rather than
    closure-baked; the frozen ladder is three rungs — az0 −58 /
    az1 +109.86 / az2 +186.76, all in `experiments/0004_alphazero/anchors/`,
    seated via `net_opponents` in `conf/arena/scale.yaml`. az0's Elo is
    calibrated by a joint round-robin MLE —
    `experiments/0004_alphazero/calibrate.py`, JOURNAL 2026-07-29; az1/az2's
    Elos are their own final-gauntlet readings). They're scheduled by their own
    `(every, phase)` — play iff `(round_index + phase) % every == 0`, so
    same-`every` rungs with staggered phases rotate one-per-round instead of all
    firing together (scale.yaml runs the three rungs at every=3, phases 0/1/2;
    arena measured 24.9% of wall / 870 s/round with all three firing every
    round, 2026-07-31 hetero run) — reported as
    `arena_vs_<name>`, and join the same Elo MLE; their seeds start at `seed +
    steps.NET_OPPONENT_SEED_BASE` (50k — room for five registry opponents) and
    follow dict enumeration order (`phase` schedules, never re-seeds), so
    adding one leaves the registry opponents' games bit-identical — a mid-rung
    must not perturb the curve it refines. The loop holds the arena **seed
    fixed across iterations** (no `+i`), so every checkpoint faces the same
    games and the curve is paired (the dice/board luck differences out) — the
    chosen variance cut, matching canopy/lc0's paired-seed tournaments over a
    checkpoint round-robin (a within-pool round-robin drifts when the pool
    changes). Anchors must stay frozen for a run. The per-iter `val_*` /
    `policy_*` / `value_*` health metrics (`Backend.eval_metrics`) are the
    cheap high-frequency proxies between arena rounds.
    The optimiser is `steps.make_optimizer(cfg.optim)` — adamw, optionally
    preceded by `clip_by_global_norm` (`cfg.optim.grad_clip`, default 1.0; 0
    disables). Stateless, so an unclipped checkpoint must resume with
    `grad_clip=0` (its opt-state has no clip layer).
  - `training/bench.py::bench_selfplay(backend, net, cfg, *, warmup, repeats,
    seed)` — the self-play throughput probe (the loop's dominant cost, isolated:
    no optimiser/replay/arena). Under `selfplay.persistent` off (the default),
    `warmup` untimed calls at `seed + 1000` pay the XLA compile, then every
    timed repeat rebuilds a fresh env and runs the *same* workload at `seed`,
    so the spread across `t_0..t_n` is measurement noise. Under `persistent`,
    the warmup call(s) instead *create* the carry (compile + pool ramp-up) and
    every timed repeat *threads* it — continuing the games in flight rather
    than discarding them, so `discarded` stays honest (trims only) — meaning
    repeats are sequential continuations of one pool, not repetitions of an
    identical workload; the reported `samples`/`env_steps`/`discarded` are the
    last repeat's, and the timing headline stays the across-repeat median
    (steady-state flush rate). Reports medians: `samples_per_s`, `moves_per_s`
    (a move is one lane-step, `env_steps * batch`), `sims_per_s`
    (`moves_per_s * search.num_simulations`). Rejects `pcr_full_prob` < 1 —
    playout-cap randomization breaks the sims-per-move accounting.
  - `tests/benchmark/` holds the pytest-benchmark suite (net forward, one
    vmapped search step, one self-play window, one optimiser step) on a
    random-init `BoardGNN` — `benchmark`-marked and deselected from the
    default run; the repo-root `run_benchmarks.sh` re-selects it. Each
    benchmark's compile-paying warm-up runs outside the timed region (the
    self-play one via `benchmark.pedantic`'s `setup`), so the headline
    min/mean/median is steady-state throughput.
  - `training/backends/mlp.py::MLPBackend` — the `AZParams` net over the
    engineered feature vector; **unmasked** policy CE + value-logistic loss,
    optax adamw, the net plays setup itself.
  - `training/backends/gnn.py::GNNBackend` — the `BoardGNN` net over the board
    graph; **masked** policy CE (softmax over the legal set only) + value loss,
    eqx-filtered optax step. `setup_policy` (at `setup_depth <= 1` the
    setup-row-restricted `make_setup_lookahead` opener — under vmap-lockstep
    self-play it runs on every lane every step, so its sweep width is hot;
    `>= 2` the beam expectimax) plays the setup phase in both self-play and
    the arena;
    `make_net_agent` composes setup + the net's search; `gnn_loss` is the masked
    loss (its finiteness is contract-tested).
  - Both losses average the policy CE over the item's `train_policy` = 1
    positions only (value-only playout-cap positions are skipped; value trains on
    all); with `train_policy` all 1 it is the plain mean (bit-exact-preserving).

The promotion gate (value-tuning evidence in settlrl-agents/CLAUDE.md,
search/leaf evidence in settlrl-search/CLAUDE.md): a run ships via the
final-gauntlet Elo gate — `arena_elo − 2·se ≥ 35`
(experiments/0004_alphazero's `arena_helpers.gauntlet_verdict`) — cleared
twice (az1, az2 — JOURNAL 2026-07-30/31).

## Reference: Canopy (`cullback/canopy`)

A Rust AlphaZero framework whose flagship example is a 1v1 Catan agent
(`nexus-v3`, claimed "strongest public 1v1 Catan agent" — unbenchmarked against
ours). It sits past our leaf-is-the-ceiling gate: learned policy + WDL value
head, self-play, Gumbel improved-policy interior selection + PUCT/Dirichlet
root (800 sims), explicit chance nodes for dice and dev draws, and
Single-Observer ISMCTS filtering per-simulation legality in a custom tree
(ours does this too — `settlrl_search.ismcts`). 1v1 only, so it never meets the 3-4p paranoid-frame / opponent-model
problem, and it *disables determinization during self-play* (the net learns
the Bayesian-average-over-hands policy; determinize only at play time).

Techniques aimed at Catan's dice variance (the variance-starved-depth problem):

- **Value-target blending** `(1−α)·z + α·q` — **done** (`cfg.value_blend.max`,
  mechanics under `training/loop.py` above).
- **Explicit chance nodes** for dice + dev draws — **done**
  (`cfg.search.chance_nodes`/`dev_chance`, above; details in
  settlrl-search/CLAUDE.md). Canopy also forces a canonical **action ordering**
  to cut transpositions — **done** (`cfg.search.ordered`, above).
- **Playout-cap randomization** (KataGo): most moves a small search, a fraction
  the full budget; only full-search positions contribute policy targets —
  **done** (`selfplay.pcr_full_prob`/`pcr_fast_sims`, mechanics under
  `training/selfplay/play.py` above). Pairs with a larger `search.num_simulations`
  for the full steps — the affordable way to add the search depth the policy
  diagnostic wants.

Repo + METHODS.md + examples/catan/OPTIMIZATIONS.md; see [[canopy-reference]].
