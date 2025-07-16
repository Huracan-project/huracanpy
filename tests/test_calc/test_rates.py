import pytest
import numpy as np
import pint
from metpy.units import units

import huracanpy


@pytest.mark.parametrize(("centering",), [("forward",), ("backward",)])
@pytest.mark.parametrize(("unit",), [(None,), ("m s-1",)])
@pytest.mark.parametrize(
    ("var", "track_id", "expected"),
    [
        # Test with only delta_var
        # With wind
        ("wind10", None, 0.089352551),
        # With slp
        ("slp", None, -23.8743878),
        # With time
        ("time", None, 21600.0),
        # Test with track_ids
        ("wind10", "track_id", 0.0546914583),
        ("time", "track_id", 23625.0),
    ],
)
def test_get_delta(tracks_csv, var, track_id, expected, unit, centering):
    var = tracks_csv[var]

    if unit is not None:
        var.attrs["units"] = unit

    if track_id is None:
        with pytest.warns(UserWarning, match="track_id is not provided"):
            delta = huracanpy.calc.delta(var, track_id, centering=centering)
    else:
        track_id = tracks_csv[track_id]
        delta = huracanpy.calc.delta(var, track_id, centering=centering)

    assert len(delta) == len(var)
    np.testing.assert_approx_equal(delta.mean(), expected, significant=6)
    assert np.isnan(delta).sum() == len(np.unique(track_id))
    assert not isinstance(delta.data, pint.Quantity)

    if var.name == "time":
        assert units.Quantity(1, "s") == units.Quantity(1, delta.attrs["units"])
    elif unit is not None:
        assert units.Quantity(1, var.attrs["units"]) == units.Quantity(
            1, delta.attrs["units"]
        )


def test_delta_centred(tracks_csv):
    diff_fwd = huracanpy.calc.delta(
        tracks_csv.wind10, tracks_csv.track_id, centering="forward"
    )
    diff_bwd = huracanpy.calc.delta(
        tracks_csv.wind10, tracks_csv.track_id, centering="backward"
    )
    diff_cntr = huracanpy.calc.delta(
        tracks_csv.wind10, tracks_csv.track_id, centering="centre"
    )
    diff_adpt = huracanpy.calc.delta(
        tracks_csv.wind10, tracks_csv.track_id, centering="adaptive"
    )

    np.testing.assert_array_equal(diff_fwd[:-1], diff_bwd[1:])
    np.testing.assert_array_equal(0.5 * (diff_fwd + diff_bwd), diff_cntr)

    diff_cntr[[30, 50, 98]] = diff_bwd[[30, 50, 98]]
    diff_cntr[[0, 31, 51]] = diff_fwd[[0, 31, 51]]

    np.testing.assert_array_equal(diff_cntr, diff_adpt)


def test_delta_fails(tracks_csv):
    with pytest.raises(ValueError, match="Option align='nonsense'"):
        huracanpy.calc.delta(
            tracks_csv.wind10, tracks_csv.track_id, centering="nonsense"
        )


def test_get_rate():
    data = huracanpy.load(huracanpy.example_csv_file)

    data.wind10.attrs["units"] = "m / s"

    intensification_rate_wind = huracanpy.calc.rate(
        data.wind10, data.time, data.track_id
    )
    np.testing.assert_approx_equal(
        intensification_rate_wind.mean(), 2.76335962e-06, significant=6
    )
    assert intensification_rate_wind.metpy.units == units("m/s^2")

    data.slp.attrs["units"] = "hPa"
    intensification_rate_slp = huracanpy.calc.rate(
        data.slp, data.time, data.track_id
    ).metpy.convert_units("hectopascals/hour")
    np.testing.assert_approx_equal(
        intensification_rate_slp.min(), -124.115, significant=6
    )


def test_rate_warns(tracks_csv):
    with pytest.warns(UserWarning, match="track_id is not provided"):
        huracanpy.calc.rate(tracks_csv.wind10, tracks_csv.time)
