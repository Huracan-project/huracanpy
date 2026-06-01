"""
Functions to plot the tracks
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import seaborn as sns
from cartopy.mpl.geoaxes import GeoAxes

from .._util import combine_kws

_subplot_kws_default = dict(projection=ccrs.PlateCarree(180))
_fig_kws_default = dict(figsize=(10, 10))
_scatter_kws_default = dict(palette="turbo", color="k")


def tracks(
    lon,
    lat,
    intensity_var=None,
    ax=None,
    subplot_kws=None,
    fig_kws=None,
    scatter_kws=None,
):
    """Create a map of all lon/lat points

    Parameters
    ----------
    lon : array_like
        Longitude points
    lat : array_like
        Latitude points
    intensity_var : array_like, optional
        Colour the individual points by
    ax :  matplotlib.axes.Axes, optional
        The axes to draw the figure on. A new figure is created if ax is None
    subplot_kws : dict, optional
        Keywords passed to the `subplot_kw` argument of
        :func:`matplotlib.pyplot.subplots`
    fig_kws : dict, optional
        Keywords passed to :func:`matplotlib.pyplot.subplots`
    scatter_kws : dict, optional
        Keywords passed to :func:`seaborn.scatterplot`

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes instances created for the plot

    """
    subplot_kws = combine_kws(subplot_kws, _subplot_kws_default)
    fig_kws = combine_kws(fig_kws, _fig_kws_default)
    scatter_kws = combine_kws(scatter_kws, _scatter_kws_default)

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
