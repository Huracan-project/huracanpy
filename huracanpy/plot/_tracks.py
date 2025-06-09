"""
Functions to plot the tracks
"""

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes


def tracks(
    lon,
    lat,
    intensity_var=None,
    ax=None,
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
    ax :  matplotlib.axes.Axes, optional
        The axes to draw the figure on. A new figure is created if ax is None
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
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=subplot_kws, **fig_kws)
    else:
        fig = ax.get_figure()

    if isinstance(ax, GeoAxes):
        if "transform" not in scatter_kws:
            scatter_kws["transform"] = ccrs.PlateCarree()
        ax.coastlines()

    sns.scatterplot(
        x=lon,
        y=lat,
        hue=intensity_var,
        ax=ax,
        **scatter_kws,
    )

    return fig, ax
