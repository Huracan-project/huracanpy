"""Huracanpy module for tracks diagnostics"""

__all__ = [
    "simple_global_histogram",
    "duration",
    "gen_vals",
    "extremum_vals",
    "ace_by_track",
    "pace_by_track",
    "translation_speed",
    "time_from_genesis",
    "time_from_extremum",
    "rate",
    "freq",
    "TC_days",
    "ACE",
]

from ._track_density import simple_global_histogram
from ._track_stats import duration, gen_vals, extremum_vals, ace_by_track, pace_by_track
from ._lifecycle import time_from_genesis, time_from_extremum
from ._rates import rate
from ._climato import freq, TC_days, ACE
