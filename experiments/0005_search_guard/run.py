"""Search guard: does a search change play stronger at a frozen net?

Hypothesis: a search-behavior change worth a full training A/B (two ~16 h runs)
first shows up as a play-strength gain at a *frozen* net — so a duel between two
search configurations over one checkpoint screens the expensive run for a GPU
hour. The search-side analogue of 0003's architecture guard.

The unit is a duel: the production-default ``incumbent`` and a ``challenger``
search play the same frozen anchor's net, seat-rotated, so search behavior is
the only difference between the sides. Both arms also play ``lookahead``, the
Elo-0 reference — a challenger that beats the incumbent while losing more to the
external arm is a red flag the head-to-head alone hides. Every arm's wall-clock
per move is measured, because an equal-simulation win is not an equal-wall-clock
win (``chance_nodes`` searches past every roll); ``wall_clock_matched`` adds a
third duel with the challenger's simulation count cut to the incumbent's
measured time.

A play-time null does not fully exclude training-time value — a better search
also makes better *targets*. It is still the right first question for these
flags: their standing rejections are play-time measurements under the
stationary heuristic leaf, and this asks again under the condition that
changed, a learned value.

    uv run python experiments/0005_search_guard/run.py [variant] [key=value ...]
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from duel import (
    GATE_ELO,
    Arm,
    MatchResult,
    anchor_semantics,
    arm_spec,
    duel,
    elo_delta,
    games_for_elo,
    load_anchor,
    seconds_per_move,
)
from pydantic import Field
from settlrl_learn.experiment import Config, Run, start_run

if TYPE_CHECKING:
    from settlrl_search.policy import OpponentSpec

REFERENCE_POLICY = "lookahead"
"""The external arm both sides play: the ``POLICIES`` entry pinned at Elo 0."""


class SearchArm(Config):
    """The search flags one side plays under (``SearchSettings``' own names).
    The defaults are production self-play's (0004's ``conf/search/scale.yaml``),
    which is what the incumbent must be for the duel to mean anything —
    :func:`run_experiment` checks them against the anchor's own sidecar."""

    expected_rolls: bool = False
    chance_nodes: bool = False
    dev_chance: bool = True
    ordered: bool = False


class SearchGuardConfig(Config):
    seed: int = 0
    anchor: str = "az2_hetero96x4"
    n_players: int = 2
    # The production self-play search scope -- what the flags would ship at.
    sims: int = 128
    considered: int = 16
    # Lanes past what a seating harvests only pay the start-up transient (no
    # lane finishes before a whole game), so `batch` stays under the games per
    # seating -- and 64 is where per-move cost bottoms out (../CLAUDE.md).
    batch: int = 64
    # Enough decided games for the screen to resolve the 35-Elo ship gate
    # (`duel.games_for_elo`): +-2 sigma of +-3.5 points, about +-24 Elo.
    games: int = 800
    reference_games: int = 80
    incumbent: SearchArm = Field(default_factory=SearchArm)
    challenger: SearchArm
    wall_clock_matched: bool = False
    timing_steps: int = 16
    timing_repeats: int = 3


VARIANTS: dict[str, dict[str, object]] = {
    # In-tree chance nodes for dice and dev draws: the search plans past a roll.
    # `dev_chance` is its sub-flag (already on by default) and does nothing
    # while `chance_nodes` is off, so the delta is the master flag alone.
    "chance": {"challenger": {"chance_nodes": True}},
    # Trivial budgets for a plumbing check (`sims=0` is the lookahead special
    # case -- cheap to compile).
    "smoke": {
        "challenger": {"chance_nodes": True},
        "sims": 0,
        "batch": 4,
        "games": 4,
        "reference_games": 4,
        "timing_steps": 2,
        "timing_repeats": 1,
        "wall_clock_matched": True,
    },
}


def guard_verdict(
    head_to_head: MatchResult, *, n_players: int
) -> tuple[str, dict[str, float]]:
    """The screen's three-way rule, read off the head-to-head's 2-sigma interval
    around the no-edge line (``1 / n_players``): the whole interval above it →
    ``"promising"`` (the change earns the training A/B), the whole interval
    below → ``"rejected"``, an interval spanning it (or no decided game at all)
    → ``"inconclusive"`` — the measurement failed, not the idea, and
    ``games_to_resolve`` is what would settle it.

    A screen's cost asymmetry is the reverse of the gate it screens for: a false
    negative shelves an idea permanently, a false positive costs one training
    A/B. So the rule never demands an edge *larger* than the 35-Elo ship
    threshold, and it separates "no edge" from "no measurement"."""
    lower = head_to_head.win_rate - 2 * head_to_head.standard_error
    upper = head_to_head.win_rate + 2 * head_to_head.standard_error
    no_edge = 1.0 / n_players
    diagnostics = {
        "lower_bound": lower,
        "upper_bound": upper,
        "no_edge": no_edge,
        "games_to_resolve": float(games_for_elo(GATE_ELO)),
    }
    if head_to_head.episodes == 0:
        return "inconclusive", diagnostics
    if lower > no_edge:
        return "promising", diagnostics
    if upper < no_edge:
        return "rejected", diagnostics
    return "inconclusive", diagnostics


def reference_gap(
    challenger_vs_reference: MatchResult, incumbent_vs_reference: MatchResult
) -> dict[str, float]:
    """The challenger's win rate against the external arm minus the incumbent's
    (negative = the challenger does worse outside), and that difference's SE
    with the two rates as independent binomials.

    Reported, never gating: at the reference budget its 2-sigma band spans
    roughly 90 Elo, so as a rule it would only ever fire on a collapse — but the
    number still shows a reader the head-to-head and the outside arm pointing
    opposite ways."""
    gap = challenger_vs_reference.win_rate - incumbent_vs_reference.win_rate
    standard_error = (
        challenger_vs_reference.standard_error**2
        + incumbent_vs_reference.standard_error**2
    ) ** 0.5
    return {"reference_gap": gap, "reference_gap_standard_error": standard_error}


def match_record(result: MatchResult, *, n_players: int) -> dict[str, float]:
    """One match's ``results.json`` entry. The Elo readout is pairwise, so it is
    reported at 2 players only."""
    record: dict[str, float] = {
        "wins": result.wins,
        "episodes": result.episodes,
        "win_rate": result.win_rate,
        "standard_error": result.standard_error,
    }
    if n_players == 2:
        record["elo_delta"], record["elo_standard_error"] = elo_delta(result)
    return record


def run_experiment(run: Run, cfg: SearchGuardConfig) -> None:
    from settlrl_agents import POLICIES

    pinned = anchor_semantics(cfg.anchor)
    playing = {flag: getattr(cfg.incumbent, flag) for flag in pinned}
    if playing != pinned:
        raise ValueError(
            f"incumbent plays {playing}, but anchor {cfg.anchor!r} recorded its "
            f"strength under {pinned} -- the duel's premise is that the "
            "incumbent is the search the frozen net was trained and gauntleted "
            "with"
        )

    net = load_anchor(cfg.anchor)
    arms = {
        name: Arm(name, num_simulations=cfg.sims, **flags.model_dump())
        for name, flags in (
            ("incumbent", cfg.incumbent),
            ("challenger", cfg.challenger),
        )
    }

    def spec_of(arm: Arm) -> "OpponentSpec":
        return arm_spec(arm, net, n_players=cfg.n_players, considered=cfg.considered)

    def price(arm: Arm, spec: "OpponentSpec") -> float:
        seconds = seconds_per_move(
            spec, n_players=cfg.n_players, batch=cfg.batch,
            steps=cfg.timing_steps, seed=cfg.seed, repeats=cfg.timing_repeats,
        )  # fmt: skip
        run.log(arm=arm.name, num_simulations=arm.num_simulations,
                seconds_per_move=seconds)  # fmt: skip
        return seconds

    def play(
        focus: "OpponentSpec", opponent: "OpponentSpec", games: int, seed: int
    ) -> MatchResult:
        return duel(
            focus, opponent, n_players=cfg.n_players, games=games,
            batch=cfg.batch, seed=seed,
        )  # fmt: skip

    specs = {name: spec_of(arm) for name, arm in arms.items()}
    seconds = {name: price(arms[name], spec) for name, spec in specs.items()}

    head_to_head = play(specs["challenger"], specs["incumbent"], cfg.games, cfg.seed)
    run.log(match="head_to_head", **match_record(head_to_head, n_players=cfg.n_players))

    reference: dict[str, MatchResult] = {}
    for name in arms:
        # Both arms start from the same seed, so they meet the reference from
        # the same initial boards -- but the two matches diverge at the first
        # differing decision (and auto-reset then regenerates boards at
        # different steps), so the two rates are independent binomials.
        reference[name] = play(specs[name], POLICIES[REFERENCE_POLICY],
                               cfg.reference_games, cfg.seed + 1000)  # fmt: skip
        run.log(match=f"{name}_vs_{REFERENCE_POLICY}",
                **match_record(reference[name], n_players=cfg.n_players))  # fmt: skip

    verdict, diagnostics = guard_verdict(head_to_head, n_players=cfg.n_players)
    gap = reference_gap(reference["challenger"], reference["incumbent"])
    results: dict[str, Any] = {
        "arms": {
            name: {**arm._asdict(), "seconds_per_move": seconds[name]}
            for name, arm in arms.items()
        },
        "head_to_head": match_record(head_to_head, n_players=cfg.n_players),
        f"vs_{REFERENCE_POLICY}": {
            name: match_record(result, n_players=cfg.n_players)
            for name, result in reference.items()
        },
        "diagnostics": {**diagnostics, **gap},
    }

    if cfg.wall_clock_matched:
        # The challenger at the simulation count its measured cost per move buys
        # for the incumbent's wall-clock, re-priced so the residual is visible.
        challenger = arms["challenger"]
        matched = challenger._replace(
            name="challenger_matched",
            num_simulations=max(
                1,
                round(
                    challenger.num_simulations
                    * seconds["incumbent"]
                    / seconds["challenger"]
                ),
            ),
        )
        matched_spec = spec_of(matched)
        matched_seconds = price(matched, matched_spec)
        matched_result = play(
            matched_spec, specs["incumbent"], cfg.games, cfg.seed + 5000
        )
        matched_verdict, matched_diagnostics = guard_verdict(
            matched_result, n_players=cfg.n_players
        )
        run.log(match="wall_clock_matched",
                **match_record(matched_result, n_players=cfg.n_players))  # fmt: skip
        results["wall_clock_matched"] = {
            "arm": {**matched._asdict(), "seconds_per_move": matched_seconds},
            "head_to_head": match_record(matched_result, n_players=cfg.n_players),
            "verdict": matched_verdict,
            "diagnostics": matched_diagnostics,
        }

    run.save_json("results.json", results)
    run.finish(
        verdict, anchor=cfg.anchor, sims=cfg.sims, n_players=cfg.n_players,
        head_to_head_win_rate=head_to_head.win_rate,
        head_to_head_standard_error=head_to_head.standard_error,
        seconds_per_move=seconds, **diagnostics, **gap,
    )  # fmt: skip


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "chance"
    if variant not in VARIANTS:
        raise SystemExit(f"usage: run.py [{'|'.join(VARIANTS)}] [key=value ...]")
    cfg = SearchGuardConfig.resolve(VARIANTS[variant], overrides=sys.argv[2:])
    run_experiment(start_run(Path(__file__).parent, cfg.dump()), cfg)


if __name__ == "__main__":
    main()
