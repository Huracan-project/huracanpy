from matplotlib.collections import PathCollection
from metpy.xarray import preprocess_and_wrap
import numpy as np
import seaborn as sb

from .._metpy import validate_units
from .. import tc
from ..tc._conventions import _thresholds

_jointgrid_kws_defaults = dict(marginal_ticks=True)
_scatterplot_kws_defaults = dict(alpha=0.1, marker=".")
_lineplot_kws_defaults = dict()
_histplot_kws_defaults = dict(element="step", fill=False)
_pressure_wind_model_kws_defaults = dict()


@preprocess_and_wrap()
def pressure_wind_relation(
    pressure=None,
    wind=None,
    bins_pressure=None,
    bins_wind=None,
    pressure_units="hPa",
    wind_units="m s-1",
    grid=None,
    color=None,
    label=None,
    mslp_convention="Klotzbach",
    wind_convention="10min",
    jointgrid_kws=None,
    scatterplot_kws=None,
    lineplot_kws=None,
    histplot_kws=None,
    pressure_wind_model_kwargs=None,
    category_color="darkgrey",
    category_linestyle="--",
):
    """Plot a pressure wind relation

    Creates a figure with 3 axes. A central pressure wind scatter plot with a line of
    best fit (by default a quadratic least squares fit), and a histogram at the edges
    of the central axis showing pressure and wind separately. The central plot also
    includes horizontal and vertical lines to show the boundaries of intensity
    categories for both pressure and wind.

    To overlay multiple datasets, call this function multiple times and pass the
    returned grid (and optionally bins) from the first call.

    Parameters
    ----------
    pressure : array_like
        Cyclone pressure minima
    wind : array_like
        Cyclone wind maxima
    bins_pressure : array_like, optional
        The bin edges to use for the marginal histogram of pressure. Uses 10 equally
        spaced bins across the range of data if not specified
    bins_wind : array_like, optional
        The bin edges to use for the marginal histogram of wind. Uses 10 equally
        spaced bins across the range of data if not specified
    pressure_units : str, default="hPa"
        Units of the input pressure data, if not already specified as part of the input
    wind_units : str, default="m s-1"
        Units of the input wind data, if not already specified as part of the input
    grid : seaborn.JointGrid, optional
        The grid on which to put the plots. Use if you want to overlay multiple
        pressure-wind plots on the same grid by passing the grid returned from the first
        call of this function
    color : str, optional
        The colour to use for each plot (scatter, best fit, and histograms). Will use
        the next colour in the matplotlib colour cycle if not specified
    label : str, optional
        Labels the line of best fit. For using with :py:func:`matplotlib.pyplot.legend`
    mslp_convention : str, default="Klotzbach"
        The convention used to add a set of lines marking the boundaries between
        pressure categories. The other option is "Simpson"
    wind_convention : str, default="10min"
        The convention used to add a set of lines marking the boundaries between
        wind categories. It is based on the Saffir-Simpson scale and the time period
        used for sustained winds. The other option is "1min"
    jointgrid_kws : dict, optional
        Passed to :py:class:`seaborn.JointGrid` when creating a new figure. By default,
        this function changes `marginal_ticks` to `True`, but this can be overwritten
    scatterplot_kws : dict, optional
        Passed to :py:func:`seaborn.scatterplot` to plot all pressure/wind points.
        By default, this function changes `alpha` to `0.1` and `marker` to `.`, but this
        can be overwritten
    lineplot_kws : dict, optional
        Passed to :py:func:`seaborn.lineplot` to plot the best fit line
    histplot_kws : dict, optional
        Passed to :py:func:`seaborn.histplot` to plot the marginal 1d histograms.
        By default, this function changes `element` to `"step"` and `fill` to `False`,
        but this can be overwritten
    pressure_wind_model_kwargs : dict, optional
        Passed to :py:func:`huracanpy.tc.pressure_wind_relation` to calculate the best
        fit line for the data
    category_color : str, default="darkgrey"
        The colour of the horizontal and vertical lines showing the boundaries of the
        intensity categories
    category_linestyle : str, default="--"
        The linestyle of the horizontal and vertical lines showing the boundaries of the
        intensity categories
    Returns
    -------
    tuple[seaborn.JointGrid, array_like, array_like]
    """
    pressure = validate_units(pressure, pressure_units)
    wind = validate_units(wind, wind_units)

    jointgrid_kws = _combine_kws(jointgrid_kws, _jointgrid_kws_defaults)
    scatterplot_kws = _combine_kws(scatterplot_kws, _scatterplot_kws_defaults)
    lineplot_kws = _combine_kws(lineplot_kws, _lineplot_kws_defaults)
    histplot_kws = _combine_kws(histplot_kws, _histplot_kws_defaults)
    pressure_wind_model_kwargs = _combine_kws(
        pressure_wind_model_kwargs, _pressure_wind_model_kws_defaults
    )

    if bins_pressure is None:
        bins_pressure = np.linspace(np.min(pressure), np.max(pressure), 10)

    if bins_wind is None:
        bins_wind = np.linspace(np.min(wind), np.max(wind), 10)

    if grid is None:
        grid = _setup_grid(
            ylabel_xpos=bins_wind[-1],
            xlabel_ypos=bins_pressure[-1],
            jointgrid_kws=jointgrid_kws,
            mslp_convention=mslp_convention,
            wind_convention=wind_convention,
            mslp_units=pressure.units,
            wind_units=wind.units,
            category_color=category_color,
            category_linestyle=category_linestyle,
        )

    sb.scatterplot(
        x=wind,
        y=pressure,
        ax=grid.ax_joint,
        **scatterplot_kws,
    )

    # Match color on other axes to scatter points
    if color is None:
        pc = [c for c in grid.ax_joint.get_children() if isinstance(c, PathCollection)][
            -1
        ]
        color = pc.get_facecolor()[0][:3]

    model = tc.pressure_wind_relation(pressure, wind, **pressure_wind_model_kwargs)
    sb.lineplot(
        x=model.predict(bins_pressure),
        y=bins_pressure,
        ax=grid.ax_joint,
        color=color,
        label=label,
        **lineplot_kws,
    )

    sb.histplot(
        x=wind,
        bins=np.asarray(bins_wind),
        color=color,
        ax=grid.ax_marg_x,
        **histplot_kws,
    )
    grid.ax_marg_x.set_ylabel("")

    sb.histplot(
        y=pressure,
        bins=np.asarray(bins_pressure),
        color=color,
        ax=grid.ax_marg_y,
        **histplot_kws,
    )
    grid.ax_marg_y.set_xlabel("")

    grid.figure.tight_layout()

    return grid, bins_pressure, bins_wind


def _combine_kws(kws, kws_default):
    if kws is None:
        return kws_default.copy()
    else:
        # Overwrite default arguments with explicit arguments
        return {**kws_default, **kws}


def _setup_grid(
    xlabel_ypos,
    ylabel_xpos,
    jointgrid_kws,
    mslp_convention="Klotzbach",
    wind_convention="Saffir-Simpson",
    mslp_units="hPa",
    wind_units="m s-1",
    category_color="darkgrey",
    category_linestyle="--",
):
    grid = sb.JointGrid(**jointgrid_kws)
    categories_mslp = _thresholds[mslp_convention]
    bins_mslp = categories_mslp["bins"].to(mslp_units)
    categories_vmax = _thresholds[wind_convention]
    bins_vmax = categories_vmax["bins"].to(wind_units)

    for y in bins_mslp:
        if not np.isinf(y):
            grid.ax_joint.axhline(
                y=y, color=category_color, linestyle=category_linestyle, linewidth=0.75
            )
            grid.ax_marg_y.axhline(
                y=y, color=category_color, linestyle=category_linestyle, linewidth=0.75
            )

    for n, label in enumerate(categories_mslp["labels"]):
        if label >= 0:
            y = _get_label_position(bins_mslp, n)
            grid.ax_joint.text(ylabel_xpos, y, label, color=category_color)

    for x in bins_vmax:
        if not np.isinf(x):
            grid.ax_joint.axvline(
                x=x, color=category_color, linestyle=category_linestyle, linewidth=0.75
            )
            grid.ax_marg_x.axvline(
                x=x, color=category_color, linestyle=category_linestyle, linewidth=0.75
            )

    for n, label in enumerate(categories_vmax["labels"]):
        if label >= 0:
            x = _get_label_position(bins_vmax, n)
            grid.ax_joint.text(x, xlabel_ypos, label, color=category_color)

    grid.ax_marg_x.set(yscale="log", xlabel=None, ylabel=None)
    grid.ax_marg_y.set(xscale="log", xlabel=None, ylabel=None)

    grid.ax_joint.set(
        xticks=[round(b) for b in bins_vmax if not np.isinf(b)],
        yticks=[round(b) for b in bins_mslp if not np.isinf(b)],
        xlabel=r"$v_\mathrm{max}$" + f" ({bins_vmax.units:~P})",
        ylabel=r"$p_\mathrm{min}$" + f" ({bins_mslp.units:~P})",
    )

    return grid


def _get_label_position(bins, n):
    if np.isinf(bins[n]):
        return bins[n + 1] - 0.5 * (bins[n + 2] - bins[n + 1])
    elif np.isinf(bins[n + 1]):
        return bins[n] + 0.5 * (bins[n] - bins[n - 1])
    else:
        return 0.5 * (bins[n] + bins[n + 1])
