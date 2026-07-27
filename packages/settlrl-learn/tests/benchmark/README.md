# Benchmarks

Micro-level timings of the training loop's units, built on
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/). All run on a
**random-init** `BoardGNN` (`gn_global` preset, width 96, layers 4 — the
pinned `az0_gnn96x4` config from `experiments/0004_alphazero`); the committed
training anchor lives in `experiments/`, out of reach of package tests, and
kernel timing does not depend on trained weights.

- **`test_net_forward[dev]`** — one jitted+vmapped `BoardGNN` forward at
  B=256.
- **`test_search_step[BN-dev]`** — one warmed dispatch of the vmapped net
  search (`make_net_search(64)`) on a mid-game batch, swept over batch sizes
  `N` in {64, 256}. This is `search_step_ms`, the unit the parallel-descent
  work moves; `search_step / (64 * net_fwd)` is the batching-headroom ratio.
- **`test_selfplay_window[dev]`** — `bench_selfplay` at a reduced budget
  (`samples=256, batch=64, repeats=1, warmup=1`); its reported rates
  (`samples_per_s` / `moves_per_s` / `sims_per_s`) surface via
  `benchmark.extra_info`.
- **`test_optimizer_step[dev]`** — one warmed `backend.make_step` dispatch on
  a broadcast zero batch at `batch_size=1024`.

All swept over devices: `cpu` always, plus `cuda` when an NVIDIA GPU is
usable (the workspace installs the CUDA jaxlib by default on Linux) —
otherwise the CUDA variants skip. Each variant pins its device explicitly,
and JIT is warmed up before every timed region.

These tests carry the `benchmark` marker and are **deselected from the
default `pytest` run**. Run them from the repo root via the shared wrapper
(extra arguments pass through to pytest-benchmark):

```bash
./run_benchmarks.sh                 # engine + agents + learn benchmarks
./run_benchmarks.sh -k cuda         # GPU variants only
./run_benchmarks.sh -k "search_step and cuda"
```

or directly:

```bash
uv run --package settlrl-learn pytest packages/settlrl-learn/tests/benchmark \
    -m benchmark --benchmark-only -n 0
```
