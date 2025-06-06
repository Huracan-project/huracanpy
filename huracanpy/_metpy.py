import functools

from metpy.units import units
import pint
import xarray as xr


def dequantify_results(original_function):
    @functools.wraps(original_function)
    def wrapped_function(*args, **kwargs):
        result = original_function(*args, **kwargs)

        if isinstance(result, tuple):
            return tuple(_dequantify_result(r) for r in result)
        else:
            return _dequantify_result(result)

    return wrapped_function


def _dequantify_result(result):
    if isinstance(result, xr.DataArray) and isinstance(result.data, pint.Quantity):
        return result.metpy.dequantify()
    elif isinstance(result, pint.Quantity) and result.unitless:
        return result.magnitude
    else:
        return result


def validate_units(variable, expected_units):
    if not isinstance(variable, pint.Quantity) or variable.unitless:
        if callable(expected_units):
            expected_units = expected_units(variable)
        variable = variable * units(expected_units)

    return variable
