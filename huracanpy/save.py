"""huracanpy module for saving tracks data"""

from .load import _netcdf


def save(dataset, filename):
    if filename.split(".")[-1] == "nc":
        _netcdf.save(dataset, filename)
    elif filename.split(".")[-1] == "csv":
        dataset.to_dataframe().to_csv(filename, index=False)
    else:
        raise NotImplementedError(
            "File format not recognized. Please use one of {.nc, .csv}"
        )
