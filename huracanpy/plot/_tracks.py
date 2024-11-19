"""
Functions to plot the tracks
"""

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs


def tracks(
    lon,
    lat,
    intensity_var=None,
    subplot_kws=dict(projection=ccrs.PlateCarree(180)),
    fig_kws=dict(figsize=(10, 10)),
    scatter_kws=dict(palette="turbo", color="k"),
):
    """Create a map of all lon/lat points

    Parameters
    ----------
    lon : array_like
    lat : array_like
    intensity_var : array_like, optional
        Colour the individual points by
    subplot_kws : dict, optional
        Keywords passed to the `subplot_kw` argument of `matplotlib.pyplot.subplots`
    fig_kws : dict, optional
        Keywords passed to `matplotlib.pyplot.subplots`
    scatter_kws : dict, optional
        Keywords passed to `seaborn.scatterplot`

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes instances created for the plot

    """
    fig, ax = plt.subplots(subplot_kw=subplot_kws, **fig_kws)
    ax.coastlines()
    sns.scatterplot(
        x=lon,
        y=lat,
        hue=intensity_var,
        ax=ax,
        **scatter_kws,
        transform=ccrs.PlateCarree(),
    )

    return fig, ax
