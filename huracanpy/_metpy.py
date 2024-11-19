import xarray as xr
import pint


def dequantify_results(original_function):
    def wrapped_function(*args, **kwargs):
        result = original_function(*args, **kwargs)

        if isinstance(result, xr.DataArray) and isinstance(result.data, pint.Quantity):
            result = result.metpy.dequantify()

        return result

    return wrapped_function
