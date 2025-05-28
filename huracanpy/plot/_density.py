"""
Functions to plot track/genesis/whatever density

To compute the density, see huracanpy.diags.track_density
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes


def density(
    d,
    ax=None,
    contourf_kws=dict(cmap="magma_r", levels=10),
    subplot_kws=dict(projection=ccrs.PlateCarree(180)),
    fig_kws=dict(),
    cbar_kwargs={"label": ""},
):
    """Create a map showing the input density

    Parameters
    ----------
    d : xarray.Dataset
        Density map from :func:`huracanpy.calc.density`
    ax :  matplotlib.axes.Axes, optional
        The axes to draw the figure on. A new figure is created if ax is None
    contourf_kws : dict, optional
        Arguments to be passed to :func:`matplotlib.pyplot.contourf`
    subplot_kws : dict, optional
        Arguments to be passed to :func:`matplotlib.pyplot.subplots`' subplot_kw argument
    fig_kws : dict, optional
        Arguments to be passed to :func:`matplotlib.pyplot.subplots`
    cbar_kwargs: dict, optional
        Arguments to be passed to :func:`matplotlib.pyplot.colorbar`

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
        if "transform" not in contourf_kws:
            contourf_kws["transform"] = ccrs.PlateCarree()
        ax.coastlines()
        ax.gridlines(draw_labels=True)

    d.plot.contourf(ax=ax, cbar_kwargs=cbar_kwargs, **contourf_kws)

    return fig, ax
