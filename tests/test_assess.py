import numpy as np
import pytest

import huracanpy


# TODO properly test tracks1_is_ref with a set of tracks that will give a different
#  answer
@pytest.mark.parametrize("tracks1_is_ref", [True, False])
def test_match(tracks1_is_ref):
    with (
        pytest.warns(
            UserWarning, match="This offline function loads a light version of IBTrACS"
        ),
        pytest.warns(UserWarning, match="You are loading the IBTrACS-WMO subset"),
    ):
        ib = huracanpy.load(source="ibtracs", ibtracs_subset="wmo")
    ref_1996 = ib.where(ib.time.dt.year == 1996, drop=True)
    uz = huracanpy.load(huracanpy.example_year_file)
    uz1 = uz.where(uz.track_id.isin([1207, 1208, 1210, 1212, 1220, 1238]), drop=True)
    uz2 = uz.where(uz.track_id.isin([1207, 1208, 1209, 1211, 1220, 1238]), drop=True)
    matches = huracanpy.assess.match(
        [uz1, uz2, ref_1996], ["UZ1", "UZ2", "ib"], tracks1_is_ref=tracks1_is_ref
    )

    data1, data2, data3 = uz1, uz2, ref_1996
    n1 = len(np.unique(data1.track_id.values))  # Number of tracks in dataset 1
    n2 = len(np.unique(data2.track_id.values))  # Number of tracks in dataset 2
    n3 = len(np.unique(data3.track_id.values))  # Number of tracks in dataset 3

    m_not1 = len(matches[matches.iloc[:, 0].isna()])
    m_not2 = len(matches[matches.iloc[:, 1].isna()])
    m_not3 = len(matches[matches.iloc[:, 2].isna()])
    m_all = len(matches[matches.isna().sum(axis=1) == 0])

    assert len(matches) == 6
    assert (n1, n2, n3) == (6, 6, 118)
    assert (m_not1, m_not2, m_not3, m_all) == (1, 1, 1, 3)


@pytest.mark.parametrize("distance_method", ["haversine", "geodesic"])
@pytest.mark.parametrize("shift, n_matches", [(1, 3), (4, 0)])
def test_match_shifted(tracks_csv, distance_method, shift, n_matches):
    tracks_shifted = tracks_csv.copy()
    tracks_shifted["lat"] = tracks_shifted.lat + shift

    matches = huracanpy.assess.match(
        [tracks_csv, tracks_shifted],
        ["a", "b"],
        distance_method=distance_method,
    )

    assert len(matches) == n_matches


def test_match_pair_empty(tracks_csv, tracks_year):
    matches = huracanpy.assess.match([tracks_csv, tracks_year], ["a", "b"])
    assert matches.size == 0


def test_match_multiple_empty(tracks_year):
    tracks = [
        tracks_year.hrcn.sel_id(track_ids)
        for track_ids in [
            list(range(1207, 1217)),
            list(range(1217, 1227)),
            list(range(1227, 1237)),
        ]
    ]
    with pytest.raises(
        NotImplementedError,
        match="For the moment, the case where two datasets have no match is not",
    ):
        huracanpy.assess.match(tracks, ["a", "b", "c"])


@pytest.mark.parametrize(
    "tracksets, message",
    [([], "You must provide at least two"), ([1, 2, 3], "Number of names provided")],
)
def test_match_fails(tracksets, message):
    with pytest.raises(ValueError, match=message):
        huracanpy.assess.match(tracksets)


def test_scores():
    with (
        pytest.warns(
            UserWarning, match="This offline function loads a light version of IBTrACS"
        ),
        pytest.warns(UserWarning, match="You are loading the IBTrACS-WMO subset"),
    ):
        ib = huracanpy.load(source="ibtracs", ibtracs_subset="wmo")
    ref_1996 = ib.where(ib.time.dt.year == 1996, drop=True)
    uz = huracanpy.load(huracanpy.example_year_file)
    matches = huracanpy.assess.match([uz, ref_1996], ["UZ", "ib"])

    pod = huracanpy.assess.pod(matches, ref_1996, ref_name="ib")
    far = huracanpy.assess.far(matches, uz, detected_name="UZ")

    assert 0.63 < pod < 0.64
    assert 0.14 < far < 0.15
