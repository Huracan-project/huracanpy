import functools

import xarray as xr
import pint


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
    else:
        return result
