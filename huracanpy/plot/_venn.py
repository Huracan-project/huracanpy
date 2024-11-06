"""Venn diagrams for tracks matching visualisation"""

from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
import numpy as np


def venn(datasets, match, labels, colors=None, circle_color="k"):
    """
    Plot venn diagram to compare the datasets.

    Parameters
    ----------
    datasets : list of xr.dataset
        list of the datasets compared.
    match : pd.DataFrame
        match dataframe issued from match_pair or match_multiple.
    labels : list of str
        labels of the datasets.
    colors : list of str, optional
        list of colors to be used for each dataset. The default is None.
    circle_color : str, optional
        color of the overlaid circles. The default is "k".

    Raises
    ------
    NotImplementedError
        If more than three or less than two datasets are given.

    Returns
    -------
    None.

    """
    if len(datasets) == 2:
        f = _venn_2datasets
    elif len(datasets) == 3:
        f = _venn_3datasets
    else:
        raise NotImplementedError(
            "We cannot plot Venn diagrams for more than 3 datasets."
        )

    if len(datasets) != len(labels):
        raise ValueError("datasets and labels must have the same length")

    if colors is None:
        colors = ["w"] * len(datasets)
    else:
        if len(colors) != len(datasets):
            raise ValueError("datasets and colors must have the same length")
    f(*datasets, match, colors, labels, circle_color)


def _venn_2datasets(data1, data2, match, colors, labels=None, circle_color="k"):
    N1 = len(np.unique(data1.track_id.values))  # Number of tracks in dataset 1
    N2 = len(np.unique(data2.track_id.values))  # Number of tracks in dataset 2
    m = len(match)  # Number of tracks matching
    venn2((N1 - m, N2 - m, m), set_colors=colors, set_labels=labels)
    venn2_circles((N1 - m, N2 - m, m), color=circle_color)


def _venn_3datasets(data1, data2, data3, M, colors, labels=None, circle_color="k"):
    N1 = len(np.unique(data1.track_id.values))  # Number of tracks in dataset 1
    N2 = len(np.unique(data2.track_id.values))  # Number of tracks in dataset 2
    N3 = len(np.unique(data3.track_id.values))  # Number of tracks in dataset 3

    M_not1 = len(M[M.iloc[:, 0].isna()])
    M_not2 = len(M[M.iloc[:, 1].isna()])
    M_not3 = len(M[M.iloc[:, 2].isna()])
    M_all = len(M[M.isna().sum(axis=1) == 0])

    subsets = (
        (N1 - M_all - M_not2 - M_not3),
        (N2 - M_all - M_not1 - M_not3),
        M_not3,
        (N3 - M_all - M_not1 - M_not2),
        M_not2,
        M_not1,
        M_all,
    )

    venn3(
        subsets,
        set_labels=labels,
        set_colors=colors,
    )

    venn3_circles(subsets, color=circle_color)
