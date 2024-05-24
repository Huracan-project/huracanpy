import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patheffects import Stroke
from cartopy.crs import Geodetic
from cartopy.mpl.geoaxes import GeoAxes


def _map_values(values, vmin, vmax, vrange, clip):
    # Allow a single value to be set for all the line segments
    if np.size(values) == 1:
        return values

    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()

    norm = plt.Normalize(vmin, vmax, clip)
    return vrange[0] + norm(values) * (vrange[1] - vrange[0])


def fancyline(
    x,
    y,
    colors=None,
    vmin=None,
    vmax=None,
    cmap=None,
    clip_colors=False,
    linewidths=None,
    wmin=None,
    wmax=None,
    wrange=(1, 5),
    clip_linewidths=True,
    alphas=None,
    amin=None,
    amax=None,
    arange=(0, 1),
    clip_alphas=True,
    linestyles=None,
    ax=None,
    transform=None,
    autoscale=True,
):
    """
    A line plot of x vs y that can show extra information by having a variable color,
    linewidth, alpha, linestyle, or any combination of the four.

    For variable color set a vmin, vmax, and cmap (similar to pcolormesh/contourf)

    >>> fancyline(x, y, colors=z, vmin=0, vmax=10, cmap="viridis")

    For variable linewidth, set a wmin, wmax, and wrange. Like with vmin/vmax, wmin/wmax
    specify the data values for the smallest and largest linewidth with variables
    outside that range being clipping to wmin/wmax. The wrange specifies the range of
    linewidths that you actually want to display

    >>> fancyline(x, y, linewidths=z, wmin=0, wmax=10, wrange=(1, 5))

    For a variable alpha, set an amin, amax, and arange

    >>> fancyline(x, y, alphas=z, amin=0, amax=10, arange=(0, 1))

    For a variable linestyle, pass an array of linestyles specified as strings. e.g. to
    show whether z is above a threshold

    >>> linestyles = ["--" if value < 5 else "-" for value in z]
    >>> fancyline(x, y, linestyles=linestyles)

    https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line

    Parameters
    ----------
    x, y : array_like
        The points to plot
    colors : array_like or str, optional
        The data used to set the colors along the line or a string of the color to set
        the line to a uniform color
    vmin, vmax : scalar, optional
        The bounding values of the colors array for the colormap. If not set, will use
        the min and max of the colors array
    cmap : str, optional
        Specify the colormap. Will default to the matplotlib default (viridis)
    clip_colors : bool, default=False
        Clips values outside vmin/vmax to vmin/vmax. The default of False is the
        standard behaviour for matplotlib and sets the colours to the outside colors
        from the colormap
    linewidths : array_like or scalar, optional
        The data used to set the width along the line or a single value for the whole
        line
    wmin, wmax : scalar, optional
        The bounding values for the linewidths. If not set, will use the min and max of
        the linewidths array
    wrange : tuple, default=(1, 5)
        The range of linewidths to map the data to
    clip_linewidths : bool, default=True
        Clips values outside wmin/wmax to wmin/wmax. Unlike clip_colors, this is set to
        True by default because negative linewidths are interpreted as positive and to
        avoid very large linewidths on the figure.
    alphas : array_like or scalar, optional
        The data used to set the alpha along the line or a single value for the whole
        line
    amin, amax : float, optional
        The bounding values for the alphas. If not set, will use the min and max of
        the alphas array
    arange : tuple, default=(0, 1)
        The range of alphas to map the data to
    clip_alphas : bool, default=True
        Clips values outside amin/amax to amin/amax. Unlike clip_colors, this is set to
        True by default because alphas outside the range 0-1 will raise an error
    linestyles : array_like or str, optional
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes, optional
        The axes to add the line to. If not given, it will be added to the most recent
        axes
    transform : cartopy coordinate reference system, optional
        If the axes being drawn on are `GeoAxes` then the data will be converted from
        the data coordinates, given by the transform, to the projection coordinates. If
        a transform is not given, then `cartopy.crs.Geodetic` will be used with default
        arguments
    autoscale : bool, default=True
        Determines whether to call ax.autoscale() after adding the lines to the plot
        because the x/y limits are not automatically adjusted otherwise. If you are
        overlaying the line on an existing plot you may want to avoid doing this by
        setting to False.

    Returns
    -------
    matplotlib.collections.LineCollection:
        The plotted LineCollection. Required as argument to `matplotlib.pyplot.colorbar`
    """
    if ax is None:
        ax = plt.gca()

    # Deal with cartopy transforms
    if isinstance(ax, GeoAxes):
        if transform is None:
            transform = Geodetic()

        xyz = ax.projection.transform_points(transform, x, y)
        x = xyz[:, 0]
        y = xyz[:, 1]

    # Break the xy points up in to line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Collect the line segments
    lc = LineCollection(segments, path_effects=[Stroke(capstyle="round")])

    if colors is not None:
        if np.size(colors) == 1:
            lc.set_colors(colors)
        else:
            lc.set_array(colors)
            if vmin is None:
                vmin = colors.min()
            if vmax is None:
                vmax = colors.max()
            lc.set_norm(plt.Normalize(vmin, vmax, clip_colors))
            lc.set_cmap(cmap)

    if linewidths is not None:
        lc.set_linewidth(_map_values(linewidths, wmin, wmax, wrange, clip_linewidths))

    if alphas is not None:
        lc.set_alpha(_map_values(alphas, amin, amax, arange, clip_alphas))

    if linestyles is not None:
        lc.set_linestyle(linestyles)

    # Add the colored line to the existing plot
    ax.add_collection(lc)

    if autoscale:
        ax.autoscale()

    return lc
