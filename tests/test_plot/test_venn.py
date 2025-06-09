import pytest

import huracanpy


def test_venn2(tracks_year):
    t1 = tracks_year.hrcn.sel_id([1207, 1208, 1210, 1212, 1220, 1238])
    t2 = tracks_year.hrcn.sel_id([1207, 1208, 1209, 1211, 1220, 1238])
    matches = huracanpy.assess.match([t1, t2], ["UZ1", "UZ2"])

    huracanpy.plot.venn([t1, t2], matches, labels=["UZ1", "UZ2"])


def test_venn3(tracks_year):
    t1 = tracks_year.hrcn.sel_id([1207, 1208, 1210, 1212, 1220, 1238])
    t2 = tracks_year.hrcn.sel_id([1207, 1208, 1209, 1211, 1220, 1238])
    matches = huracanpy.assess.match([t1, t2, tracks_year], ["UZ1", "UZ2", "UZ3"])

    huracanpy.plot.venn([t1, t2, tracks_year], matches, labels=["UZ1", "UZ2", "UZ3"])


@pytest.mark.parametrize(
    "datasets, labels, colors, error, message",
    [
        (
            list(range(4)),
            None,
            None,
            NotImplementedError,
            "We cannot plot Venn diagrams for more than 3",
        ),
        (
            list(range(2)),
            list(range(3)),
            None,
            ValueError,
            "datasets and labels must have the same length",
        ),
        (
            list(range(2)),
            list(range(2)),
            list(range(3)),
            ValueError,
            "datasets and colors must have the same length",
        ),
    ],
)
def test_venn_fails(datasets, labels, colors, message, error):
    with pytest.raises(error, match=message):
        huracanpy.plot.venn(datasets, None, labels, colors=colors)
