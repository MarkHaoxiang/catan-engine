# 0001 — bench smoke

Status: concluded (pass)

## Hypothesis

The scripted greedy agent beats uniform-random play decisively (2-player,
seat-swapped). Known true from the strength ladder — the experiment exists to
prove the contract end to end: manifest-pinned run, streamed metrics, a saved
bench verdict, a gate asserted in code.

## Setup

`uv run python experiments/0001_bench_smoke/run.py` — config at the top of
run.py: greedy vs random, 60 games, seats swapped halfway, seed 0; gate
pass iff (rate − 2·se) ≥ 0.70.

## Results

From `runs/0001_bench_smoke/2026-06-12T154132Z` (RTX 5090, jax 0.10.1):
greedy won 54/65 (83.1%, se 4.7%), lower 2σ bound 73.8% — gate passed.
Seat-balanced: 27/32 seated first, 27/33 seated second.

## Decision

Infrastructure adopted. The pattern for every following experiment:
`settlrl_learn.experiment.start_run` for the manifest, `metrics.jsonl` for anything stepwise,
`settlrl_agents.cli.bench` (saved as `bench.json`) for strength claims, and the
verdict computed by the script.

## calibrate — anchor-scale reset (measurement wave 2, task 4)

Status: concluded (pass)

A joint Elo MLE (`calibrate.py::joint_fit`, coordinate-ascent Bradley-Terry)
over the complete round-robin {random, greedy, lookahead, mcts, az0_gnn96x4},
lookahead pinned at 0. 10 unordered pairs, seat-swapped 2p, n=4238 games total
(heuristic pairs n≈600-620, pairs touching mcts/az0 n≈300-370); search
semantics for the az0 pairs pinned at `conf/arena/scale.yaml` +
`conf/search/scale.yaml`'s settings (sims=24, considered=16,
chance_nodes=false, dev_chance=true, ordered=false), plus az0_gnn96x4's setup
opener (`setup_depth=1`, `setup_temperature=2.0`, `setup_beam=4` — GNNBackend's
own defaults, pinned as constants in both `calibrate.py::AZ0_SETUP_*` and
`0004_alphazero/run.py::NET_OPPONENT_SETUP_*` rather than read off a run's
`cfg.net.setup_*`).

Fitted (Fisher SE): random −1115.2 ± 71.1, greedy −230.9 ± 11.7, lookahead
0.0 (fixed), mcts +38.2 ± 11.7, az0_gnn96x4 −57.9 ± 12.0. All four sanity
gates held. Run: `runs/0001_bench_smoke/2026-07-29T034807Z`.

Adopted into `experiments/0004_alphazero/conf/arena/{default,scale}.yaml`
(`anchor_elos.random` −800.0 → −1115.0) and `conf/arena/scale.yaml`'s
`net_opponents.az0_gnn96x4.elo` (−100.0 → −58.0). Full detail (old→new,
caveats, semantics scope) in `../JOURNAL.md`.
