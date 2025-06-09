import numpy as np
import pytest

import huracanpy
from huracanpy._concat import _reset_track_id


def test_reset_track_id(tracks_year):
    tracks = _reset_track_id(tracks_year, tracks_year.track_id, keep_original=True)

    # Check that the old array has been kept
    np.testing.assert_array_equal(tracks.track_id_original, tracks_year.track_id)

    # Check that the new IDs have been assigned correctly
    _, new_track_ids = np.unique(tracks.track_id_original, return_inverse=True)
    np.testing.assert_array_equal(tracks.track_id, new_track_ids)

    # Check that the numbers are as expected
    assert (
        tracks.track_id.sum()
        == (tracks.track_id_original - tracks.track_id_original[0]).sum()
    )


@pytest.mark.parametrize("prefix", [None, "{}_"])
def test_concat_tracks(tracks_csv, prefix):
    tracks = huracanpy.concat_tracks([tracks_csv, tracks_csv], prefix=prefix)

    assert len(tracks.time) == 198
    assert len(tracks.groupby("track_id")) == 6
