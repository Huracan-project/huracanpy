import numpy as np
import xarray as xr


# http://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#_contiguous_ragged_array_representation_of_trajectories
# The CF convention for saving the data is that the track_id or equivalent variable has
# length of number of tracks (trajectories) and is labelled with cf_role="trajectory"
# and a rowSize variable with the attribute sample_dimension="obs" (or equivalent of
# "obs") to tell us how to split apart the tracks.
# For working on track data we extend the track_id to the full length of all the
# trajectories to allow us to use groupby() and sel() with track_id to work by
# individual tracks. So we need to replace the "track_id" or equivalent variable when
# loading or saving the data
def load(filename, **kwargs):
    dataset = xr.open_dataset(filename, **kwargs)

    # Check which type of netCDF we have (2d, ragged, or CSV-like)
    track_id = _find_trajectory_id(dataset)
    time = dataset.time

    if time.dims != track_id.dims:
        # If time and track_id don't have the same dimension it could be 2d or ragged
        # Track ID should always be 1d
        if track_id.ndim != 1:
            raise ValueError(
                f"File has a track ID with {track_id.ndim} dimensions. Should be 1d"
            )
        dims = [track_id.dims[0]] + [
            dim for dim in time.dims if dim not in track_id.dims
        ]
        # If any variables have a time and track_id dimension it is 2d
        vars_2d = [var for var in dataset if sorted(dims) == sorted(dataset[var].dims)]

        if len(vars_2d) > 0:
            return as1d(dataset, dims, track_id, vars_2d)
        else:
            # Otherwise ragged array
            return stretch_trid(dataset, track_id)
    else:
        # Otherwise it is in the CSV format used by huracanpy, so just return the
        # dataset
        return dataset


def save(dataset, filename):
    # Find the variable with cf_role=trajectory_id
    trajectory_id = _find_trajectory_id(dataset)

    # Get the name of the sample dimension. The name "record" has been used in the load
    # functions, but we don't need to assume that is the name. It may be different when
    # loaded from other netCDF files
    sample_dimension = trajectory_id.dims

    if len(sample_dimension) == 1:
        sample_dimension = sample_dimension[0]
    else:
        raise ValueError(
            f"{trajectory_id.name} spans multiple dimensions, should be 1d"
        )

    # Sort by trajectory_id so each track can be described by the first index and
    # number of elements of the unique trajectory id
    if not (trajectory_id[1:] >= trajectory_id[:-1]).all():
        dataset = dataset.sortby(trajectory_id.name)
    else:
        dataset = dataset.copy()

    trajectory_ids, rowsize = np.unique(trajectory_id, return_counts=True)

    dataset[trajectory_id.name] = ("trajectory", trajectory_ids)
    dataset[trajectory_id.name].attrs = trajectory_id.attrs
    dataset["rowSize"] = ("trajectory", rowsize)
    dataset["rowSize"].attrs["sample_dimension"] = sample_dimension

    dataset.to_netcdf(filename)


def stretch_trid(dataset, trajectory_id):
    # Find the trajectory_id and rowSize variables from their CF labels
    rowsize = _find_rowsize(dataset)
    sample_dimension = rowsize.attrs["sample_dimension"]

    # Stretch the trajectory_id out along the sample dimension
    trajectory_id_stretched = []
    for npoints, tr_id in zip(rowsize.data, trajectory_id.data):
        trajectory_id_stretched.extend([tr_id] * npoints)

    dataset = dataset.drop_vars([trajectory_id.name, rowsize.name])

    dataset["track_id"] = (sample_dimension, trajectory_id_stretched)
    # Keep attributes (add cf_role if not already there)
    dataset["track_id"].attrs = trajectory_id.attrs

    return dataset


def as1d(dataset, dims, track_id, vars_2d):
    # Stack 2d dimensions into a record dimension
    dataset = dataset.stack(record=dims)

    # Record only as an index, not a coordinate
    record = dataset.record
    dataset = dataset.drop_vars(["record", *dims])

    # Set old dims as variables
    for dim in dims:
        # If it is a dimension, add it as a variable
        dataset[dim] = ("record", record[dim].values)

    # Remove data that is only nans
    # Use the combination of floating point variables
    vars_2d_floats = [
        var for var in vars_2d if np.issubdtype(dataset[var].dtype, np.floating)
    ]
    nans = np.array([np.isnan(dataset[var]) for var in vars_2d_floats]).all(axis=0)
    dataset = dataset.isel(record=np.where(~nans)[0])

    # Add cf role to track_id
    if track_id.name != "track_id":
        dataset = dataset.rename({track_id.name: "track_id"})

    return dataset


_trajectory_id_names = [
    # Default name for HuracanPy
    "track_id",
    # TRACK
    "TRACK_ID",
    # MIT netCDF
    "n_track",
    # CHAZ
    "stormID",
    # IBTrACS netCDF
    "storm",
]


def _find_trajectory_id(dataset):
    # Find the variable with cf_role=trajectory_id
    trajectory_id = [
        dataset[var]
        for var in dataset.variables
        if "cf_role" in dataset[var].attrs
        and dataset[var].attrs["cf_role"] == "trajectory_id"
    ]

    if len(trajectory_id) == 1:
        return trajectory_id[0]
    else:
        for name in _trajectory_id_names:
            if name in dataset or name in dataset.dims:
                trajectory_id = dataset[name]
                trajectory_id.attrs["cf_role"] = "trajectory_id"
                return trajectory_id

        raise ValueError(
            f"Found {len(trajectory_id)} variables with cf_role=trajectory_id."
            f"Should be exactly one."
        )


def _find_rowsize(dataset):
    # Find the variable with sample_dimension="obs" (or equivalent)
    rowsize = [
        dataset[var]
        for var in dataset.variables
        if "sample_dimension" in dataset[var].attrs
    ]

    if len(rowsize) == 1:
        return rowsize[0]
    else:
        raise ValueError(
            f"Found {len(rowsize)} variables to map to row size. Should be exactly one"
        )
