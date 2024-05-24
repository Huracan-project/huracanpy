"""Venn diagrams for tracks matching visualisation"""

from matplotlib_venn import venn2, venn2_circles
import numpy as np


def plot_venn_2datasets(
    data1, data2, match, colors=("w", "w"), labels=None, circle_color="k"
):
    N1 = len(np.unique(data1.track_id.values))  # Number of tracks in dataset 1
    N2 = len(np.unique(data2.track_id.values))  # Number of tracks in dataset 2
    m = len(match)  # Number of tracks matching
    venn2((N1 - m, N2 - m, m), set_colors=colors, set_labels=labels)
    venn2_circles((N1 - m, N2 - m, m), color=circle_color)


# TODO : créer une fonction venn qui détecte le nombre de datasets
