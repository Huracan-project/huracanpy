import numpy as np
import pytest
import xarray as xr

import huracanpy

from ..test_huracanpy import _load_with_checked_warnings


def test_time_from_genesis():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    t = huracanpy.calc.time_from_genesis(data.time, data.track_id)
    assert t.min() == 0
    assert (t.max() / np.timedelta64(1, "D")) == 13.5


@pytest.mark.parametrize("stat, t_min, t_max", [("max", -270, 102), ("min", -180, 252)])
def test_time_from_apex(stat, t_min, t_max):
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    t = huracanpy.calc.time_from_apex(data.time, data.track_id, data.wind10, stat)
    t_hours = t / np.timedelta64(1, "h")
    assert t_hours.min() == t_min
    assert t_hours.max() == t_max


def test_time_from_apex_fails(tracks_csv):
    with pytest.raises(NotImplementedError, match="stat not recognized"):
        huracanpy.calc.time_from_apex(
            tracks_csv.time, tracks_csv.track_id, tracks_csv.wind10, "nonsense"
        )


# Example tracks where apex_vals gives a different answer to argmin
# Multiple points with the same minimum pressure
# - argmin always gives the first
# - apex_vals used to give inconsistent results. Sometimes the last but also sometimes
#   in the middle


@pytest.mark.parametrize(
    "track_ids, variable, stat",
    [
        (["1980051S12102", "1980171N06142", "1980140N16116"], "slp", "min"),
        (["1980005S14120", "1980052S16155", "1980177N13259"], "wind", "max"),
    ],
)
def test_apex_vals_first_point(track_ids, variable, stat):
    tracks = _load_with_checked_warnings(None, source="ibtracs").hrcn.sel_id(track_ids)

    apex = tracks.hrcn.get_apex_vals(variable, stat=stat)

    if stat == "min":
        apex_arg = tracks.groupby("track_id").map(
            lambda x: x.isel(record=(x[variable].argmin(dim="record"))).set_coords(
                "track_id"
            )
        )
    else:
        apex_arg = tracks.groupby("track_id").map(
            lambda x: x.isel(record=(x[variable].argmax(dim="record"))).set_coords(
                "track_id"
            )
        )

    xr.testing.assert_identical(apex, apex_arg)
