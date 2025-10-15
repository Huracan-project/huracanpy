"""huracanpy module for saving tracks data"""

from . import _netcdf


def save(dataset, filename, **kwargs):
    """
    Save dataset as filename.
    The file type (NetCDF or csv supported) is detected based on filename extension.

    Parameters
    ----------
    dataset : xarray.Dataset
    filename : str
        Must end in ".nc" or ".csv"
    **kwargs: Remaining keywords are passed to the save function one of
        - `pandas.DataFrame.to_csv`
        - `xarray.Dataset.to_netcdf`

    """
    if filename.split(".")[-1] == "nc":
        _netcdf.save(dataset, filename, **kwargs)
    elif filename.split(".")[-1] == "csv":
        dataset.to_dataframe().to_csv(filename, index=False, **kwargs)
    else:
        raise NotImplementedError(
            "File format not recognized. Please use one of {.nc, .csv}"
        )
