import matplotlib.pyplot as plt


def doughnut(values, reference_total, ax=None, **kwargs):
    """Plot a pie chart with a doughnut shape with thickness showing the total number of
    points relative to a reference total

    Based on Figs. 1/2 from
    Roberts et al. (2020) - Impact of Model Resolution on Tropical Cyclone Simulation
    Using the HighResMIP–PRIMAVERA Multimodel Ensemble
    https://doi.org/10.1175/JCLI-D-19-0639.1

    Parameters
    ----------
    values : array_like
        The values for each individual section of the doughnut
    reference_total : scalar
        The value to compare against the total of `values`. If the total of `values` is
        larger than `reference_total` the doughnut will be thicker, and if the total of
        `values` is smaller than `reference_total`, the doughnut will be thinner
    ax : matplotlib.axes.Axes, optional
        Axes to draw the doughnut on. Default will use the most recent axes
    **kwargs
        Remaining arguments are passed to :py:func:`matplotlib.pyplot.pie`

    Returns
    -------
    tuple
        Returns the three variables returned from :py:func:`matplotlib.pyplot.pie`.
        Unlike :py:func:`matplotlib.pyplot.pie`, autotexts will return an empty list if
        no labels are specified, so the length of the tuple is always three
    """
    if ax is None:
        ax = plt.gca()

    ratio = sum(values) / reference_total
    radius = 1 - 1 / (3 * ratio)

    radius = max(radius, 0.05)
    radius = min(radius, 1.0)

    if "wedgeprops" not in kwargs:
        kwargs["wedgeprops"] = dict()
    kwargs["wedgeprops"]["width"] = radius
    if "autopct" in kwargs:
        patches, texts, autotexts = ax.pie(values, **kwargs)
    else:
        patches, texts = ax.pie(values, **kwargs)
        autotexts = []

    return patches, texts, autotexts
