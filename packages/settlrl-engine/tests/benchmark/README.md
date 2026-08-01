# Benchmarks

Throughput benchmarks for the RL environments under random legal play, built on
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/). Three rollouts are
measured:

- **`test_batched_env_random_rollout[N-Pp]`** — a batch of games stepped in
  lockstep through `BatchedSettlrlEnv` (the vectorised surface), one random legal
  action per lane per step. Swept over batch sizes `N` ∈ {1, 8, 64, 512}
  (grouped per player count in the output) to show how the per-step cost
  amortises across the batch.
- **`test_batched_env_rollout_scan[N-Pp]`** — the same random play through the
  fused `rollout` (one `lax.scan` dispatch per window), without the per-step
  host round trips; same batch-size sweep.
- **`test_aec_env_random_rollout[Pp]`** — a single game played turn-at-a-time
  through the PettingZoo `SettlrlAECEnv`.

All are swept over player counts `P` ∈ {2, 4} and over devices: `cpu` always,
plus `cuda` when an NVIDIA GPU is usable (see below) — otherwise the CUDA
variants are skipped. Each variant pins its device explicitly, so the CPU
numbers stay CPU numbers even on a machine where the GPU is JAX's default
backend.

All pick only *legal* moves (screened by the engine's own action mask), so every
action type is exercised, including the forced discard / move-robber after a 7.

These tests carry the `benchmark` marker and are **deselected from the default
`pytest` run** (`-m 'not benchmark'` in `addopts`), since they dominate
wall-clock and aren't correctness tests. The wrapper below re-selects them.

## Running

From the repo root (`uv sync` first if you haven't), use the wrapper script:

```bash
./run_benchmarks.sh
```

It runs only the benchmarks with coverage off (coverage would instrument the hot
loop and distort timings). Extra arguments pass straight through to
pytest-benchmark (see below). JIT is warmed up before each timed region, so the
numbers are steady-state throughput, not first-call latency.

## Useful options

```bash
# Save this run as a named baseline ...
./run_benchmarks.sh --benchmark-save=baseline

# ... then compare a later run against it (fails if it regresses past 5%).
./run_benchmarks.sh --benchmark-compare --benchmark-compare-fail=mean:5%

# Run a single benchmark.
./run_benchmarks.sh -k batched

# Only the GPU variants.
./run_benchmarks.sh -k cuda
```

See `pytest-benchmark --help` for the full set of `--benchmark-*` flags.

## Running on GPU

On Linux the workspace installs the CUDA jaxlib by default (the `cuda`
dependency group; the wheels bundle the CUDA runtime, so only the NVIDIA
driver is needed), and `./run_benchmarks.sh` picks up the GPU automatically,
running both the `cpu` and `cuda` variants. Without a usable GPU the `cuda`
variants report as skipped. Standalone installs of `settlrl-engine` outside the
workspace get the same wheels via the `cuda` extra.

## Profiling

When a benchmark looks slow and you want to know *why* (not just *how fast*),
use the profiler at `packages/settlrl-engine/tools/profile_env.py`. It runs the
batched random-rollout loop under cProfile and prints the hottest frames, which
localises the cost to a specific call (including hidden device→host syncs that a
wall-clock timer would silently fold into `step`):

```bash
# cProfile breakdown of a batch-100 rollout
uv run --package settlrl-engine python packages/settlrl-engine/tools/profile_env.py \
    --batch-size 100 --steps 40

# on-device (XLA op) trace for TensorBoard
uv run --package settlrl-engine python packages/settlrl-engine/tools/profile_env.py \
    --batch-size 100 --trace /tmp/settlrl-trace
```
