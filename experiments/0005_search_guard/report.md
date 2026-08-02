# 0005 — search guard

Status: open (framework live; serves as the search-decision screen). No variant
concluded.

## Hypothesis

A search-behavior change worth a full training A/B (two ~16 h runs) first shows
up as a play-strength gain at a *frozen* net. If so, a duel between two search
configurations over one checkpoint decides in about a GPU-hour whether the
expensive comparison is worth starting — the search-side analogue of 0003's
architecture guard.

The two built search flags (`chance_nodes`/`dev_chance` and `ordered`) sit
default-off because their only judgment path was that expensive run. Their
standing rejections were measured under the stationary heuristic leaf; this
re-asks under the condition that changed, a learned value.

Caveat carried into every verdict below: a play-time null does not exclude
training-time value, because a better search also makes better *targets*.

## Setup

```
uv run python experiments/0005_search_guard/run.py chance [wall_clock_matched=true]
```

Both arms play the frozen `az2_hetero96x4` anchor (0004's gauntlet-passing
checkpoint) under its pinned setup opener, at the production self-play search
scope — 128 simulations, 16 considered actions, `expected_rolls: false`. The
incumbent is the production default, verified at run start against the anchor
sidecar's own `search_semantics`; a variant moves one flag.

Three matches per variant: the challenger against the incumbent (seat-rotated),
and each arm against `lookahead` — the Elo-0 reference, which catches a
challenger that wins the head-to-head while losing more to the outside.

`run.guard_verdict` is a *screen*, so it reports three outcomes off the
head-to-head's 2-sigma interval around the no-edge line: `promising` (interval
entirely above — the change earns the training A/B), `rejected` (entirely
below), `inconclusive` (spanning it). The default 800 decided games resolve a
win rate to about ±3.5 points (±24 Elo), inside the 35-Elo threshold the A/B
itself is judged by; the reference-arm gap is reported, not gated (at 80 games
per arm its 2-sigma band is ~90 Elo).

Each arm's wall-clock per searched move is measured on an idle GPU, so an
equal-simulation result is not mistaken for an equal-wall-clock one.

## Scope

Three limits on anything this framework measures.

**`ordered` is out of scope.** At play time the ordering overlay never reaches
the root: `settlrl_agents.evaluate` builds its env without `track_ordering`, and
the overlay lives in `BatchedSettlrlEnv.step`, not in the fused `_rollout_core`
that a rollout runs. An `ordered` arm would therefore search a tree-pruned model
of an environment that enforces no ordering — inconsistent with its own env and
strictly handicapped, which is not what the flag does in self-play. The
prerequisite is an engine change: `track_ordering` threaded through the fused
rollout path. Until then the guard has no `ordered` variant.

**The duel measures a short-game subsample.** `evaluate` counts the games that
finish inside its budget and discards those still running when it trips — and
the discarded ones are systematically the longest, since a lane that has not
finished is a lane whose game is long. The bias shrinks with the request (at
most a batch of in-flight lanes per seating: ~14% of started games at
`games=800, batch=64`, against ~36% at the 100-per-seating budget), but it never
goes away, and it runs against a variance-reduction flag like `chance_nodes`,
whose value should be largest in exactly the long, high-variance games the
sample drops.

**3 players needs a 3p-capable net.** `n_players` runs the duel as a seat
rotation and the verdict against `1 / n_players`, but the committed anchors are
2p-trained and stall at three seats: 8 lanes × 500 steps at 32 simulations
finish 18 games at 2p and 1 at 3p (at 0 simulations, none at all, while three
`lookahead` seats finish 25). This is what blocks the ordering question at the
player count where domestic trade exists.
