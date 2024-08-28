"""
Module containing functions to compute ACE
"""

from numpy.polynomial.polynomial import Polynomial
import xarray as xr
import pint
from metpy.xarray import preprocess_and_wrap
from metpy.units import units


def ace_by_point(wind, threshold=34 * units("knots"), wind_units="m s-1"):
    """Calculate accumulate cyclone energy (ACE) for each individual point

    Parameters
    ----------
    wind : array_like
        Maximum velocity of a tropical cyclone
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
        The ACE at each point in wind

    """
    ace_values = _ace_by_point(wind, threshold, wind_units)

    # The return value has units so stays as a pint.Quantity
    # This can be annoying if you still want to do other things with the array
    # Metpy dequantify keeps the units as an attribute so it can still be used later
    # TODO - extend preprocess_and_wrap to include this if it is needed for more
    #  functions
    if isinstance(ace_values, xr.DataArray) and isinstance(
        ace_values.data, pint.Quantity
    ):
        ace_values = ace_values.metpy.dequantify()

    return ace_values


@preprocess_and_wrap(wrap_like="wind")
def _ace_by_point(wind, threshold=34 * units("knots"), wind_units="m s-1"):
    if not isinstance(wind, pint.Quantity) or wind.unitless:
        wind = wind * units(wind_units)
    wind = wind.to(units("knots"))

    if threshold is not None:
        if not isinstance(threshold, pint.Quantity) or threshold.unitless:
            threshold = threshold * units(wind_units)

        wind[wind < threshold] = 0 * units("knots")

    ace_values = (wind**2.0) * 1e-4

    return ace_values


def pace_by_point(
    pressure,
    wind=None,
    model=None,
    threshold_wind=None,
    threshold_pressure=None,
    wind_units="m s-1",
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

    >>> pace, pw_model = pace_by_point(pressure, wind)

    The default model to fit is a quadratic polynomial
    (:py:class:`numpy.polynomial.polynomial.Polynomial` with `deg=2`)

    2. Pass just the pressure and an already fit model to calculate the wind speeds from
       this model

    >>> pace, _ = pace_by_point(pressure, model=pw_model)

    Parameters
    ----------
    pressure : array_like
    wind : array_like, optional
    model : str or class, optional
    threshold_wind : scalar, optional
    threshold_pressure : scalar, optional
    wind_units : str, default="m s-1"
    **kwargs

    Returns
    -------
    pace_values : array_like

    model : object

    """
    model_wind, model = pressure_wind_relationship(
        pressure, wind=wind, model=model, **kwargs
    )
    pace_values = ace_by_point(
        model_wind, threshold=threshold_wind, wind_units=wind_units
    )

    if threshold_pressure is not None:
        pace_values[pressure > threshold_pressure] = 0.0

    return pace_values, model


def pressure_wind_relationship(pressure, wind=None, model=None, **kwargs):
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
            model = Polynomial.fit(pressure, wind, **kwargs)

        else:
            model = model.fit(pressure, wind, **kwargs)

    elif model is None:
        raise ValueError(
            "Need to specify either wind or model to calculate pressure-wind relation"
        )

    wind_from_fit = model(pressure)

    return wind_from_fit, model


# Pre-determined pressure-wind relationships
_z2021 = Polynomial([1.43290190e01, 5.68356519e-01, -1.05371378e-03])


def pw_z2021(pressure):
    return _z2021(1010.0 - pressure)


def pw_holland(pressure):
    return 2.3 * (1010.0 - pressure) ** 0.76
