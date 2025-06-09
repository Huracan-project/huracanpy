import numpy as np
import pytest

import huracanpy


def test_time_from_genesis():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    t = huracanpy.calc.time_from_genesis(data.time, data.track_id)
    assert t.min() == 0
    assert t.max() == 1166400000000000


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
