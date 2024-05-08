import gzip

import datetime
import cftime
import numpy as np
import xarray as xr

from parse import parse

# Use align specifications (^, <, >) to allow variable whitespace in headers
# Left aligned (<) for "nvars" so nfields takes all whitespace between in case there is
# only one space
# Right aligned (>) for variables at the end since lines are stripped of whitespace
# prior to parsing
header_fmt = "TRACK_NUM{ntracks:^d}ADD_FLD{nfields:^d}{nvars:<d}&{var_has_coords}"
track_header_fmt = "TRACK_ID{track_id:>d}"
track_header_fmt_new = "TRACK_ID{track_id:^d}START_TIME{start_time:>}"
track_info_fmt = "POINT_NUM{npoints:>d}"


def _parse(fmt, string, **kwargs):
    # Call parse but raise an error if None is returned
    result = parse(fmt, string, **kwargs)

    if result is None:
        raise ValueError(f"Format {fmt} does not match string {string}")

    return result


def parse_date(date, calendar=None):
    if len(date) == 10:  # i.e., YYYYMMDDHH
        if calendar == "netcdftime":
            yr = int(date[0:4])
            mn = int(date[4:6])
            dy = int(date[6:8])
            hr = int(date[8:10])
            return cftime.datetime(yr, mn, dy, hour=hr, calendar="360_day")
        else:
            return datetime.datetime.strptime(date.strip(), "%Y%m%d%H")
    else:
        return int(date)


def load(filename, calendar=None, variable_names=None):
    """Load ASCII TRACK data as an xarray.Dataset

    Parameters
    ----------
    filename: str
    calendar : optional
    variable_names : list of str, optional
        TRACK

    Returns
    -------
    xarray.Dataset
    """
    if filename.split(".")[-1] == "gz":
        open_func = gzip.open
    else:
        open_func = open

    output = list()
    with open_func(filename, "rt") as f:
        # The first lines can contain extra information bounded by two extra lines
        # Just skip to the main header line for now
        line = ""
        while not line.startswith("TRACK_NUM"):
            line = f.readline().strip()

        # Load information about tracks from header line
        # If there are no added variables the line ends at the "&"
        try:
            header = _parse(header_fmt, line).named
        except ValueError:
            header = _parse(header_fmt.split("&")[0] + "&", line).named
            header["var_has_coords"] = ""

        ntracks = header["ntracks"]
        nfields = header["nfields"]
        nvars = header["nvars"]
        has_coords = [int(x) == 1 for x in header["var_has_coords"]]

        # Check header data is consistent
        assert len(has_coords) == nfields
        assert sum([3 if x == 1 else 1 for x in has_coords]) == nvars

        # Create a list of variables stored in each track
        # Generic names for variables as there is currently no information otherwise
        var_labels = ["lon", "lat", "vorticity"]
        if variable_names is None:
            variable_names = [f"feature_{n}" for n in range(nfields)]
        else:
            assert len(variable_names) == nfields
        for n, variable_name in enumerate(variable_names):
            if has_coords[n]:
                var_labels.append(f"{variable_name}_longitude")
                var_labels.append(f"{variable_name}_latitude")
            var_labels.append(variable_name)

        # Read in each track as an xarray dataset with time as the coordinate
        for n in range(ntracks):
            # Read individual track header (two lines)
            line = f.readline().strip()
            try:
                track_info = _parse(track_header_fmt, line).named
            except ValueError:
                track_info = _parse(track_header_fmt_new, line).named

            line = f.readline().strip()
            npoints = _parse(track_info_fmt, line)["npoints"]

            # Generate arrays for time coordinate and variables
            # Time is a list because it will hold datetime or cftime objects
            # Other variables are a dictionary mapping variable name to a tuple of
            # (time, data_array) as this is what is passed to xarray.Dataset
            times = [None] * npoints
            track_data = {label: ("obs", np.zeros(npoints)) for label in var_labels}

            # Add track ID as a variable along the obs dimension so that it can be used
            # for groupby
            track_id = np.array([track_info["track_id"]] * npoints)
            track_data["track_id"] = ("obs", track_id)

            # Populate time and data line by line
            for m in range(npoints):
                line = f.readline().strip().split("&")
                time, lon, lat, vorticity = line[0].split()
                times[m] = parse_date(time, calendar=calendar)
                track_data["lon"][1][m] = float(lon)
                track_data["lat"][1][m] = float(lat)
                track_data["vorticity"][1][m] = float(vorticity)

                for i, label in enumerate(var_labels[3:], start=1):
                    track_data[label][1][m] = float(line[i])

            # Return a dataset for the individual track
            # Add time separately so xarray can deal with the awkward np.datetime64
            # format
            ds = xr.Dataset(track_data)
            ds["time"] = ("obs", times)
            output.append(ds)

    output = xr.concat(output, dim="obs")
    output.track_id.attrs["cf_role"] = "trajectory_id"

    return output
