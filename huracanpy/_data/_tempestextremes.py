import datetime

import numpy as np
import xarray as xr


def load(
    filename,
    variable_names=None,
    tempest_extremes_unstructured=False,
    tempest_extremes_header_str="start",
):
    output = list()

    with open(filename, "r") as f:
        data = f.readlines()

    lineno = 0
    # TempestExtremes ASCII does not have a track_id, so just use a counter variable
    track_id = 0

    # Just in case there are any empty lines at the start of the file
    # This can probably be deleted
    while not data[lineno].split()[0] == tempest_extremes_header_str:
        lineno += 1

    nfields = len(data[lineno + 1].split())

    # First three or four variables are grid index and lon,lat
    # i, j for structures grid. Single index for unstructured
    # Last four variables are year, month, day, hour
    varnames = ["i", "lon", "lat"]
    if not tempest_extremes_unstructured:
        varnames.insert(1, "j")

    if variable_names is None:
        for i in range(nfields - len(varnames) - 4):
            varnames.append(f"feature_{i}")
    else:
        nvars = nfields - len(varnames) - 4
        if len(variable_names) != nvars:
            raise ValueError(
                f"Number of variable names does not match expected number of variables:"
                f"{nvars}"
            )
        varnames += variable_names

    while lineno < len(data):
        start, npoints, year, month, day, hour = data[lineno].split()
        npoints = int(npoints)

        # Create the empty data for the new track
        times = [None] * npoints
        track_data = {label: ("record", np.zeros(npoints)) for label in varnames}

        track_id_array = np.array([track_id] * npoints)
        track_data["track_id"] = ("record", track_id_array)

        # Populate time and data line by line
        for m in range(npoints):
            line = data[lineno + 1 + m].split()

            for i, name in enumerate(varnames):
                track_data[name][1][m] = float(line[i])

            year, month, day, hour = line[-4:]
            time = datetime.datetime(int(year), int(month), int(day), int(hour))
            times[m] = time

        # Return a dataset for the individual track
        # Add time separately so xarray can deal with the awkward np.datetime64
        # format
        ds = xr.Dataset(track_data)
        ds["time"] = ("record", times)
        output.append(ds)

        track_id += 1
        lineno += npoints + 1

    output = xr.concat(output, dim="record")
    output.track_id.attrs["cf_role"] = "trajectory_id"

    return output
