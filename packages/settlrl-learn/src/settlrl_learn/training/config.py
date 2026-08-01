"""Typed, grouped configuration for the training loop.

The flat ``learn()`` keyword surface grouped into independently-constructible,
independently-validatable pydantic units (``extra="forbid"`` -- a typo'd knob
fails loudly). :class:`LearnConfig` is the whole loop contract; ``learn`` takes
one. :class:`SearchSettings` subclasses settlrl-search's ``SearchConfig`` to add
training defaults while inheriting its exclusive-rolls validator.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator
from settlrl_search.ismcts import SearchConfig


class _Group(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchSettings(SearchConfig):
    """settlrl-search's ``SearchConfig`` with training defaults. ``value_scale``
    is the *net* leaf's logit scale (``tanh(logit/2) = 2P-1``); the heuristic
    teacher search keeps the factory default (its own calibration)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_simulations: int = 64
    max_depth: int = 12
    max_considered: int = 16
    value_scale: float = 2.0
    expected_rolls: bool = True
    chance_nodes: bool = False
    dev_chance: bool = True
    ordered: bool = False
    fused_leaf: bool = True


class SelfPlayConfig(_Group):
    samples: int = 2048
    batch: int = 64
    temperature: float = 1.0
    temperature_moves: int = 0
    """> 0: a lane samples at `temperature` for its first this many recorded
    moves of the current game, then argmax (0 keeps `temperature` flat)."""
    max_steps: int = 100_000
    max_game_len: int = 800
    persistent: bool = False
    """Keep the self-play env, its games in flight and their RNG alive across
    calls (a ``SelfPlayCarry``) instead of rebuilding per call, so games finish
    across iteration boundaries instead of being discarded at them."""
    checkpoint_pad: int | None = Field(default=None, ge=1)
    """Bound the persistent carry's per-lane checkpoint pad below
    ``max_game_len`` (``None`` pads to ``max_game_len`` in full). A lane whose
    pending exceeds the bound at save time keeps only its most recent rows, so
    a resume then diverges from the uninterrupted run on that lane -- set it
    far above the typical game length."""
    pcr_full_prob: float = 1.0
    """Playout-cap randomization (KataGo): probability a self-play step runs the
    full ``num_simulations`` search and trains policy on its positions; the rest
    run ``pcr_fast_sims`` (value-only). 1.0 disables it (every position trains
    policy)."""
    pcr_fast_sims: int = 16

    @model_validator(mode="after")
    def _pad_covers_the_anneal_window(self) -> SelfPlayConfig:
        if (
            self.checkpoint_pad is not None
            and self.checkpoint_pad < self.temperature_moves
        ):
            raise ValueError(
                "checkpoint_pad < temperature_moves: truncation clamps a lane's "
                "pending_len below the anneal window, so the resumed lane would "
                "re-enter tempered sampling mid-game"
            )
        return self


class OptimConfig(_Group):
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    train_steps: int = 200
    reuse: float = 0.0
    """> 0 caps updates/iter at ``reuse * fresh / batch_size`` (the AZ sample-reuse
    factor) instead of a fixed ``train_steps``."""
    grad_clip: float = 1.0
    """> 0 wraps adamw in ``clip_by_global_norm`` at this cap (0 disables). Stateless,
    so toggling within a run is fine but a checkpoint's opt-state structure assumes
    its own setting -- resume an unclipped run with ``grad_clip=0``."""


class ReplayConfig(_Group):
    buffer_max: int = 50_000
    buffer_min: int = 256


class TeacherConfig(_Group):
    """Warm-start: the first ``iters`` iterations draw moves + policy targets from
    a fixed strong search (``sims`` simulations) over the code-supplied teacher
    value. ``enabled`` is the experiment-layer switch for passing that value."""

    enabled: bool = False
    iters: int = 0
    sims: int = 32


class ValueBlendConfig(_Group):
    """Canopy ``(1-a)z + a*q``: ``a`` ramps 0 -> ``max`` over ``ramp`` iters."""

    max: float = 0.0
    ramp: int = 10


class EvalConfig(_Group):
    """Periodic generalization check: every ``every`` iterations the first
    ``samples`` positions of that iteration's fresh self-play batch are scored for
    the ``val_*`` metrics *before* the net trains on them (a valid
    held-out-in-time signal -- the net generated them but hasn't fit them yet),
    then the whole batch trains as normal (100% of data used). ``every`` = 0
    disables it (no ``val_*``)."""

    every: int = 0
    samples: int = 2048


class ArenaConfig(_Group):
    """Periodic strength check. The chance/ordering *semantics* come from the
    backend (it carries them for the play agent); only the *budget* lives here.

    ``anchor_elos`` pins each anchor opponent's Elo on a fixed scale (``lookahead``
    = the heuristic gate at 0; ``random`` well below); the net's ``arena_elo`` is
    the MLE on that scale (:mod:`settlrl_learn.training.elo`). Anchors must stay
    frozen for a run -- changing them silently shifts every historical number.
    The defaults are calibration-scoped: valid only under the search settings
    (sims/considered/chance_nodes/dev_chance/ordered) the calibration ran with
    (JOURNAL.md scale-reset entry, 2026-07-29)."""

    games: int = 0
    every: int = 1
    batch: int = 16
    sims: int = 48
    considered: int = 16
    opponents: list[str] = Field(default_factory=lambda: ["lookahead", "random"])
    anchor_elos: dict[str, float] = Field(
        default_factory=lambda: {"lookahead": 0.0, "random": -1115.0}
    )
    opponent_every: dict[str, int] = Field(default_factory=dict)
    """Opponent name -> play it only every Nth arena round (absent, or 1, plays
    every round). ``run_arena``'s ``round_index`` counts arena invocations, not
    training iterations."""


class LearnConfig(_Group):
    """The complete net-agnostic ``learn`` configuration (one nested object)."""

    n_iterations: int
    seed: int = 0
    checkpoint_every: int = 1
    search: SearchSettings = Field(default_factory=SearchSettings)
    selfplay: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    value_blend: ValueBlendConfig = Field(default_factory=ValueBlendConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    arena: ArenaConfig = Field(default_factory=ArenaConfig)
