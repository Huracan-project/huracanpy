from datetime import datetime
from io import StringIO

from . import _csv


time_vars = ["year", "month", "timestep"]


def load(filename, iris_timestep, **kwargs):
    with open(filename, "rt") as f:
        # First line is variable names. Rename track_id
        header = f.readline().strip().replace("#tcid", "track_id").split()

        # Time is split into year, month, timestep replace these columns with a single
        # time column. Retain the indices to convert
        indices = [header.index(var) for var in time_vars]
        for idx in reversed(indices):
            del header[idx]

        header.append("time")

        output = [",".join(header)]

        # Format each line as a CSV with time values replace
        for line in f:
            line = line.split()
            year, month, timestep = [int(line[idx]) for idx in indices]
            time = datetime(year, ((month - 1) % 12) + 1, 1) + timestep * iris_timestep

            for idx in reversed(indices):
                del line[idx]

            line.append(time.isoformat())

            output.append(",".join(line))

    # Use existing CSV load function
    return _csv.load(StringIO("\n".join(output)), index_col=False, **kwargs)
