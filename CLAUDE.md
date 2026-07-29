# settlrl-engine

## Rules

At the start of every session, read the base-game rulebook this engine
implements (the canonical source rules, kept as the external spec for rule
fidelity):
https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf

## Documentation

When you change code, check whether the relevant module-level docs (per-package `CLAUDE.md`) and READMEs still describe it accurately, and update them in the same change. Remove references that have gone stale.

**Code and types are documentation.** Never repeat in prose what a signature, type annotation, or name already states — if something is clearly understandable by reading the code, don't document it. Docs record only what code cannot express: invariants, design rationale, cross-module contracts, perf evidence, gotchas.

Array parameters and returns carry jaxtyping annotations. Reuse the shared alias — defined beside the constants that pin its dimensions — instead of bare `jax.Array` or a local redefinition; bare `jax.Array` is for the rare genuinely shape/dtype-polymorphic case. The test conftests turn these annotations into enforced runtime checks for the hooked modules, so they must be exact, not aspirational.

Keep docs concise. User-facing docs (READMEs) describe the **current structure** of the code — what each part is and how to use it — and nothing else. They are not a journal: no history or chronology, no "what we haven't done yet" / future work, no technical reasoning, hypotheses, or evidence (those belong in `CLAUDE.md`, which cites experiment numbers, or are omitted). Each section explains one thing, and explains it clearly. Keep abstractions clear and leave implementation details out.

Comments should be concise. Doc comments (docstrings) describe only the contract to callers — behavior not evident from the signature; no implementation detail, design motivation, or perf notes (those belong in the per-package `CLAUDE.md`, or are simply omitted).

## Conventions

- **Naming.** A name must read correctly at the call site without the
  context that defined it — spell out domain words instead of abbreviating.
  Single letters are for tight math indices only. If a name needs a comment
  to explain its role, rename it instead.
- **No speculative generality.** Build an abstraction when its second
  consumer exists, not before. No dummy/placeholder configs, no dead code.

## Experiments

ML experiments live in `experiments/` (contract: `experiments/README.md`).
Each numbered directory is an experiment *framework* — a class of related
experiments: `run.py [variant]` selects a config, framework-specific helpers
live in the same directory, outputs land in the git-ignored `runs/` (many
logs per framework), and `experiments/JOURNAL.md` indexes one verdict line
per concluded finding. Prefer extending a framework's variants over
scaffolding a new number (`uv run python experiments/new.py "<title>"` for
genuinely new classes). Strength claims gate through `settlrl-agents bench` or
an in-run match with the threshold asserted in code. Throughput claims gate
through experiment 0004's `bench_throughput` preset (pinned config + frozen
anchor) — quote `result.json` before/after at the same config. Record
evidence there, not in package docs — CLAUDE.md files cite experiment
numbers.

## Checks

Pre-commit hooks (ruff check/format, mypy over every package, the engine test
suite) run on each commit — `uv run pre-commit install` after a fresh clone.
CI (`.github/workflows/ci.yml`) runs the full gate on push/PR: lint, format
check, mypy, and every package's test suite (including settlrl-agents, whose
suite is too slow for a commit hook).

Before finishing any session, ensure the mypy checker passes:

```bash
uv run --package settlrl-engine mypy packages/settlrl-engine/src packages/settlrl-engine/tests
uv run --package settlrl-agents mypy packages/settlrl-agents/src packages/settlrl-agents/tests
```

When CUDA is available (check `jax.devices("cuda")` or `nvidia-smi`), always run benchmarks directly on the GPU (`-k cuda`) — skip the CPU benchmark runs. Without CUDA, run CPU-only (`JAX_PLATFORMS=cpu`, or `-k cpu` for the benchmarks).

## Parallel sessions

Multiple agents may work this repo at once. **Never run `git reset --hard`** (or
`git checkout -- .`, `git clean -fd`) to tidy *your own* working tree — it
silently destroys another session's uncommitted, unstaged edits, which are
unrecoverable. To set aside your own changes, use `git stash` (recoverable), and
only ever stash/reset paths you yourself touched. Commit green checkpoints
promptly so in-flight work survives someone else's mistake.
