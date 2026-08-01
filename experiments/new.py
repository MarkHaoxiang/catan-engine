"""Scaffold the next experiment::

    uv run python experiments/new.py "lookahead with a learned value"

allocates the next number and creates ``experiments/NNNN_<slug>/`` with
``run.py`` and ``report.md`` templates (the contract is in the README;
``0002_linear_value_fitting`` is a worked resolve-based example).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from string import Template

_HERE = Path(__file__).resolve().parent

_RUN_PY = Template('''"""$title.

Hypothesis: <what this run should show, in one sentence>.

    uv run python experiments/$dirname/run.py [variant] [key=value ...]
"""

import sys
from pathlib import Path

from settlrl_learn.experiment import Config, Run, start_run


class $cls(Config):
    seed: int = 0
    # ... typed knobs; defaults here, VARIANTS below are deltas onto them.


VARIANTS: dict[str, dict[str, object]] = {
    "default": {},
    "smoke": {},  # trivial budgets for the plumbing test (tests/test_smoke.py)
}


def run_experiment(run: Run, cfg: $cls) -> None:
    # run.log(...) per step, run.save_json(...) for artifacts. Gate any
    # strength claim through settlrl_agents.cli.bench and assert the threshold
    # here, not by eye.
    run.finish("fail", note="not implemented")


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "default"
    if variant not in VARIANTS:
        raise SystemExit(f"usage: run.py [{'|'.join(VARIANTS)}] [key=value ...]")
    cfg = $cls.resolve(VARIANTS[variant], overrides=sys.argv[2:])
    run_experiment(start_run(Path(__file__).parent, cfg.dump()), cfg)


if __name__ == "__main__":
    main()
''')

_REPORT_MD = Template("""# $num — $title

Status: open

## Hypothesis

## Setup

`uv run python experiments/$dirname/run.py` — config at the top of run.py.

## Results

<!-- the numbers, and the runs/ directory they came from -->

## Decision

<!-- adopted / falsified / parked, and what changes because of it; mirror the
one-line verdict into ../JOURNAL.md -->
""")


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    if not slug:
        raise SystemExit(f"cannot slugify {title!r}")
    return slug


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: uv run python experiments/new.py "<title>"')
    title = sys.argv[1]
    taken = (re.match(r"(\d{4})_", p.name) for p in _HERE.iterdir())
    num = max((int(m.group(1)) for m in taken if m), default=0) + 1
    slug = _slug(title)
    exp = _HERE / f"{num:04d}_{slug}"
    cls = "".join(part.capitalize() for part in slug.split("_")) + "Config"
    exp.mkdir()
    (exp / "run.py").write_text(
        _RUN_PY.substitute(title=title, dirname=exp.name, cls=cls)
    )
    (exp / "report.md").write_text(
        _REPORT_MD.substitute(num=f"{num:04d}", title=title, dirname=exp.name)
    )
    print(f"created {exp.relative_to(_HERE.parent)}")
    print("next: hypothesis + CONFIG in run.py, run it, write report.md,")
    print("then append the verdict line to experiments/JOURNAL.md")


if __name__ == "__main__":
    main()
