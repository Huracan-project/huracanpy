import numpy as np

import huracanpy


def test_match():
    ib = huracanpy.load(source="ibtracs", ibtracs_subset="wmo")
    ref_1996 = ib.where(ib.time.dt.year == 1996, drop=True)
    UZ = huracanpy.load(huracanpy.example_year_file)
    UZ1 = UZ.where(UZ.track_id.isin([1207, 1208, 1210, 1212, 1220, 1238]), drop=True)
    UZ2 = UZ.where(UZ.track_id.isin([1207, 1208, 1209, 1211, 1220, 1238]), drop=True)
    M = huracanpy.assess.match([UZ1, UZ2, ref_1996], ["UZ1", "UZ2", "ib"])

    data1, data2, data3 = UZ1, UZ2, ref_1996
    N1 = len(np.unique(data1.track_id.values))  # Number of tracks in dataset 1
    N2 = len(np.unique(data2.track_id.values))  # Number of tracks in dataset 2
    N3 = len(np.unique(data3.track_id.values))  # Number of tracks in dataset 3

    M_not1 = len(M[M.iloc[:, 0].isna()])
    M_not2 = len(M[M.iloc[:, 1].isna()])
    M_not3 = len(M[M.iloc[:, 2].isna()])
    M_all = len(M[M.isna().sum(axis=1) == 0])

    assert len(M) == 6
    assert (N1, N2, N3) == (6, 6, 118)
    assert (M_not1, M_not2, M_not3, M_all) == (1, 1, 1, 3)


def test_scores():
    ib = huracanpy.load(source="ibtracs", ibtracs_subset="wmo")
    ref_1996 = ib.where(ib.time.dt.year == 1996, drop=True)
    UZ = huracanpy.load(huracanpy.example_year_file)
    M = huracanpy.assess.match([UZ, ref_1996], ["UZ", "ib"])

    POD = huracanpy.assess.pod(M, ref_1996, ref_name="ib")
    FAR = huracanpy.assess.far(M, UZ, detected_name="UZ")

    assert 0.63 < POD < 0.64
    assert 0.14 < FAR < 0.15
