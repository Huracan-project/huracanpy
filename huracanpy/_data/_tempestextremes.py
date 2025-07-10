from io import StringIO

from . import _csv


def load(
    filename,
    variable_names=None,
    tempest_extremes_unstructured=False,
    tempest_extremes_header_str="start",
):
    with open(filename, "r") as f:
        data = f.readlines()

    lineno = 0

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
        varnames += [f"feature_{i}" for i in range(nfields - len(varnames) - 4)]
    else:
        nvars = nfields - len(varnames) - 4
        if len(variable_names) != nvars:
            raise ValueError(
                f"Number of variable names does not match expected number of variables:"
                f"{nvars}"
            )
        varnames += variable_names

    # TempestExtremes ASCII does not have a track_id, so just use a counter variable
    track_id = 0
    # Last four columns are always year, month, day, hour
    varnames = ["track_id"] + varnames + ["year", "month", "day", "hour"]

    output = [",".join(varnames)]
    while lineno < len(data):
        start, npoints, year, month, day, hour = data[lineno].split()
        npoints = int(npoints)

        # Populate time and data line by line
        for m in range(npoints):
            output.append(",".join([str(track_id)] + data[lineno + 1 + m].split()))

        track_id += 1
        lineno += npoints + 1

    return _csv.load(StringIO("\n".join(output)), index_col=False)
