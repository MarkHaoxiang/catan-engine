# Evaluation: anchored Elo + paired-seed arena

Date: 2026-06-23
Status: implemented

## Problem

The arena logged two raw win-rates (`arena_winrate` vs `lookahead`,
`arena_vs_random`) at n≈40 with a per-iteration seed, so the strength curve was
unreadably noisy (SE≈0.07) and not a single comparable number.

## Research (how the field evaluates)

Survey of AlphaGo Zero / AlphaZero / MuZero / Gumbel-MuZero / mctx / KataGo /
lc0 / OpenSpiel / canopy (see the conversation log). Conclusions used here:

- **Latest-net, no gating is the modern norm** (AZ dropped AGZ's 55% gate;
  MuZero/lc0/canopy followed). Only KataGo still gates, optionally. → our
  latest-net self-play is correct; no change.
- **Comparable strength = a fixed anchored gauntlet + anchored Elo**, not a
  round-robin among checkpoints (which is only a drifting within-pool scale).
  AZ/MuZero anchor to a fixed external baseline; lc0 uses `ordo -a 0 -A sf`.
- **Paired/common seeds + side-swap is the biggest variance cut** for a
  high-variance game (canopy's tournament pre-generates a seed list + swaps
  sides; lc0 `-repeat`).
- **openskill** fits the multi-player free-for-all arena (3–4p), not the 2p
  fixed gauntlet; for 2p, plain anchored logistic Elo is cleaner. Pitfall:
  changing the anchor set silently shifts every historical number — freeze it.

## Implemented

- `training/elo.py` — `expected_score` (the 400-point logistic) and
  `anchored_elo(anchors)` = the MLE Elo from `(anchor_elo, wins, games)` tuples,
  solved by bisection (expected total score is monotone in R). Wins are
  continuity-corrected to `[0.5, games-0.5]` so a saturated 0%/100% anchor can't
  drive R to ±∞. Pure, unit-tested (`tests/test_elo.py`: parity→anchor,
  gate 0.55→+35, monotone, saturated-finite, symmetric).
- `ArenaConfig.anchor_elos: dict[str,float]` (default `{lookahead: 0, random:
  -1115.0}` — recalibrated 2026-07-29; see `conf/arena/default.yaml`) — the
  frozen Elo scale; `lookahead`(heuristic) pinned at 0 so
  `arena_elo` reads directly as the net's margin over the Stage-1 gate.
- `steps.run_arena` now also returns `arena_elo`, computed from the per-anchor
  win-rates via `anchored_elo`.
- `loop.learn` holds the **arena seed fixed across iterations** (`cfg.seed +
  20_000`, no `+i`), so every checkpoint faces the same games → the curve is
  paired and the dice/board luck differences out. (Arena is off in the resume
  tests, so bit-exactness is unaffected.)
- `arena_elo` flows to wandb automatically (run.py's `on_iter` logs all metrics).

The per-iteration `val_*` / `policy_*` / `value_*` metrics from
`Backend.eval_metrics` already serve as the cheap high-frequency proxies between
arena rounds (the canopy-style net-vs-search agreement / value-calibration
signal), so no new proxy plumbing this round.

- **Frozen self-play checkpoint anchors** — implemented since: `net_opponents`
  lets the arena take frozen checkpoints as anchors, and the ladder now carries
  rungs az0/az1/az2, each snapshotted at a fixed rating.

## Deferred

- **openskill ladder for the 3–4p mixed-count arena** (separate from this 2p
  gauntlet number).
- Canopy's `mean|q − V_net|` "search correction" proxy (needs recorded `q` +
  an extra net-value forward; backend-specific).
