#!/usr/bin/env bash
# Run the throughput benchmarks: the engine's env rollouts, the agents'
# move/evaluate latencies, and the training loop's micro-timings. The flags
# re-select the benchmarks (deselected from the default test runs), run only
# them, and drop coverage. See the tests/benchmark/README.md in each package
# for options and examples.
set -euo pipefail

# -n 0 turns xdist back off: pytest-benchmark needs a single process for
# stable timings (the default addopts run the regular suites with -n auto).
# `|| [ $? -eq 5 ]` tolerates a package where a -k filter selects nothing.
for pkg in settlrl-engine settlrl-agents settlrl-learn; do
    uv run --package "$pkg" pytest "packages/$pkg/tests/benchmark" \
        -m benchmark --benchmark-only --no-cov -n 0 "$@" || [ "$?" -eq 5 ]
done
