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

    # Find the trajectory_id and rowSize variables from their CF labels
    trajectory_id = _find_trajectory_id(dataset)
    rowsize = _find_rowsize(dataset)
    sample_dimension = rowsize.attrs["sample_dimension"]

    # Stretch the trajectory_id out along the sample dimension
    trajectory_id_stretched = []
    for npoints, tr_id in zip(rowsize.data, trajectory_id.data):
        trajectory_id_stretched.extend([tr_id] * npoints)

    dataset[trajectory_id.name] = (sample_dimension, trajectory_id_stretched)
    # Keep attributes (including cf_role)
    dataset[trajectory_id.name].attrs = trajectory_id.attrs

    return dataset.drop_vars(rowsize.name)


def save(dataset, filename):
    # Find the variable with cf_role=trajectory_id
    trajectory_id = _find_trajectory_id(dataset)

    # Get the name of the sample dimension. The name "obs" has been used in the load
    # functions, but we don't need to assume that is the name. It may be different when
    # loaded from other netCDF files
    sample_dimension = trajectory_id.dims

    if len(sample_dimension) == 1:
        sample_dimension = sample_dimension[0]
    else:
        raise ValueError(
            f"{trajectory_id.name} spans multiple dimensions, should be 1d"
        )

    trajectory_ids = np.unique(trajectory_id)
    rowsize = [np.count_nonzero(trajectory_id == x) for x in trajectory_ids]

    dataset[trajectory_id.name] = ("trajectory", trajectory_ids)
    dataset[trajectory_id.name].attrs = trajectory_id.attrs
    dataset["rowSize"] = ("trajectory", rowsize)
    dataset["rowSize"].attrs["sample_dimension"] = sample_dimension

    dataset.to_netcdf(filename)


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
        if "track_id" in dataset:
            dataset["track_id"].attrs["cf_role"] = "trajectory_id"
            return dataset["track_id"]
        else:
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
