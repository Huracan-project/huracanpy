import xarray as xr

import huracanpy


def test_accessor(tracks_csv):
    # Check that the method on the tracks is identical to the accessor method
    ds1 = tracks_csv.huracanpy.trackswhere(lambda x: (x.track_id == 1).all())
    ds2 = huracanpy.subset.trackswhere(tracks_csv, lambda x: (x.track_id == 1).all())

    xr.testing.assert_identical(ds1, ds2)
