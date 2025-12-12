"""
Module containing functions to compute ACE
"""

import warnings

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from pint.errors import UnitStrippedWarning
from metpy.xarray import preprocess_and_wrap
from metpy.units import units

from .._metpy import dequantify_results, validate_units


def ace(
    wind,
    sum_by=None,
    threshold=34 * units("knots"),
    wind_units="m s-1",
):
    r"""Calculate accumulate cyclone energy (ACE)

    .. math:: \mathrm{ACE} =
        10^{-4} \sum v_\mathrm{max}^2 \quad (v_\mathrm{max} \ge 34 \mathrm{kn})

    By default, this function will return the "ACE" for each individual point in `wind`.
    To calculate more useful quantities of ACE, use the `sum_by` keyword.

    For example, to calculate the ACE of each individual track, doing

    >>> ace_by_track = huracanpy.tc.ace(tracks.wind, sum_by=tracks.track_id)

    will return a DataArray with track_id as a coordinate and the sum of ACE for each
    track as the data. Note that this is equivalent to using groupby:

    >>> ace_by_point = huracanpy.tc.ace(tracks.wind)
    >>> ace_by_track = ace_by_point.groupby(tracks.track_id).sum()

    To calculate the average ACE by track, you can do

    >>> ace_by_track_mean = ace_by_track.mean()

    Similarly to calculate a climatological mean ACE by year, run

    >>> climatological_ace = huracanpy.tc.ace(
    >>>    tracks.wind, sum_by=tracks.time.dt.year
    >>> ).mean()

    Parameters
    ----------
    wind : array_like
        Maximum velocity of a tropical cyclone associated with the tracks dataset
    sum_by : array_like
        Variable to take the sum of ACE values across. Must have the same length as wind
    threshold : scalar, default=34 knots
        ACE is set to zero below this threshold wind speed. The default argument is in
        knots. To pass an argument with units, use :py:mod:`metpy.units`, otherwise any
        non-default argument will be assumed to have the units of "wind_units" which is
        "m s-1" by default.
    wind_units : str, default="m s-1"
        If the units of wind are not specified in the attributes then the function will
        assume it is in these units before converting to knots

    Returns
    -------
    array_like
        The ACE for each track in wind

    """

    ace_values = _ace(wind, threshold, wind_units)

    if sum_by is not None:
        ace_values = ace_values.groupby(sum_by).sum()

    return ace_values


@dequantify_results
@preprocess_and_wrap(wrap_like="wind")
def _ace(wind, threshold=34 * units("knots"), wind_units="m s-1"):
    wind = validate_units(wind, wind_units)
    wind = wind.to(units("knots"))

    if threshold is not None:
        threshold = validate_units(threshold, wind_units)

        wind[wind < threshold] = 0 * units("knots")

    ace_values = (wind**2.0) * 1e-4

    return ace_values


def pace(
    pressure,
    wind=None,
    model=None,
    sum_by=None,
    threshold_wind=None,
    threshold_pressure=None,
    wind_units="m s-1",
    pressure_units="hPa",
    **kwargs,
):
    """Calculate a pressure-based accumulated cyclone energy (PACE) for each individual
       point

    PACE is calculated the same way as ACE, but the wind is derived from fitting a
    pressure-wind relationship and calculating wind values from pressure using this fit

    Example
    -------
    This function can be called in two ways

    1. Pass the pressure and wind to fit a pressure-wind relationship to the data and
    then calculate pace from the winds derived from this fit

    >>> pace, pw_model = get_pace(pressure, wind)

    The default model to fit is a quadratic polynomial
    (:py:class:`numpy.polynomial.polynomial.Polynomial` with `deg=2`)

    2. Pass just the pressure and an already fit model to calculate the wind speeds from
    this model

    >>> pace, _ = get_pace(pressure, model=pw_model)

    Parameters
    ----------
    pressure : array_like
        Cyclone minimum sea-level pressure
    wind : array_like, optional
        Cyclone wind. Only include if you want to train a model to fit the
        pressure-wind relation
    model : str, class, or object, optional
        The model to fit the pressure wind relation or a model with preset parameters to
        derive wind from pressure. Can also be set to "z2021" or "holland" for those
        preset models. The object must have a `fit` function that returns a trained
        model, consistent with numpy and scikit-learn models.
        Default is py:class:`numpy.polynomial.polynomial.Polynomial` with `deg=2`
    sum_by : array_like
        Variable to take the sum of PACE values across. Must have the same length as
        pressure/wind. For examples, see the documentation for `huracanpy.tc.ace`
    threshold_wind : scalar, optional
        PACE is set to zero below this threshold wind speed
    threshold_pressure : scalar, optional
        Similar to threshold wind, set PACE to zero where pressure is above this
        threshold
    wind_units : str, default="m s-1"
        If the units of wind are not specified in the attributes then the function will
        assume it is in these units before converting to knots
    pressure_units : str, default="Pa"
        If the units of pressure are not specified in the attributes then the function
        will assume it is in these units
    **kwargs
        Remaining keywords are passed to the `fit` function of the model

    Returns
    -------
    tuple (array_like, object) :
        Array of pace_values
        model : object

    """
    pace_values, model = _pace(
        pressure,
        wind=wind,
        model=model,
        threshold_wind=threshold_wind,
        threshold_pressure=threshold_pressure,
        wind_units=wind_units,
        pressure_units=pressure_units,
        **kwargs,
    )

    if sum_by is not None:
        pace_values = pace_values.groupby(sum_by).sum()

    return pace_values, model


@dequantify_results
@preprocess_and_wrap(wrap_like=("pressure", None))
def _pace(
    pressure,
    wind=None,
    model=None,
    threshold_wind=None,
    threshold_pressure=None,
    wind_units="m s-1",
    pressure_units="hPa",
    **kwargs,
):
    pressure = validate_units(pressure, pressure_units)
    pressure = pressure.to("hPa")
    if wind is not None:
        wind = validate_units(wind, wind_units)
        wind = wind.to(units("knots"))

    model_wind, model = _pressure_wind_relation(
        pressure, wind=wind, model=model, **kwargs
    )
    pace_values = _ace(model_wind, threshold=threshold_wind, wind_units="knots")

    if threshold_pressure is not None:
        threshold_pressure = validate_units(threshold_pressure, pressure_units)

        pace_values[pressure > threshold_pressure] = 0.0 * pace_values.units

    return pace_values, model


def _pressure_wind_relation(pressure, wind=None, model=None, **kwargs):
    if isinstance(model, str):
        if model.lower() == "z2021":
            model = pw_z2021
        elif model.lower() == "holland":
            model = pw_holland

    elif wind is not None:
        if model is None:
            # Here, we calculate a quadratic P/W fit based off of the "control"
            if "deg" not in kwargs:
                kwargs["deg"] = 2
            model = Polynomial.fit(pressure.magnitude, wind.magnitude, **kwargs)

        else:
            model = model.fit(pressure.magnitude, wind.magnitude, **kwargs)

    elif model is None:
        raise ValueError(
            "Need to specify either wind or model to calculate pressure-wind relation"
        )

    wind_from_fit = _get_wind_from_model(pressure.magnitude, model) * units("knots")

    return wind_from_fit, model


def _get_wind_from_model(pressure, model):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UnitStrippedWarning,
            message="The unit of the quantity is stripped when downcasting to ndarray.",
        )

        result = np.asarray(model(pressure))
    return result


# Pre-determined pressure-wind relationships
# Input pressure in hPa, output wind in knots
_z2021 = Polynomial([1.43290190e01, 5.68356519e-01, -1.05371378e-03])


def pw_z2021(pressure):
    return _z2021(1010.0 - pressure)


def pw_holland(pressure):
    return 2.3 * (1010.0 - pressure) ** 0.76
