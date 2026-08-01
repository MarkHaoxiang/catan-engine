"""Batched self-play: the driver (:mod:`play`) and the persistent-pool carry
(:mod:`carry`)."""

from settlrl_learn.training.selfplay.play import Samples, SelfPlayStats, self_play

__all__ = [
    "Samples",
    "SelfPlayStats",
    "self_play",
]
