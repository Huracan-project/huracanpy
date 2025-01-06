from io import StringIO

from . import _csv


def load(filename):
    with open(filename, "rt") as f:
        line = ""
        while "DATE" not in line:
            line = f.readline().strip()

        # Change first two named columns DATE->time and INDEX->track_id
        # Read others as is (but lower case)
        varnames = ["time", "track_id"] + [var.lower() for var in line.split()][2:]

        # Reformat the file in place like a CSV
        # Header line
        output = [",".join(varnames)]

        # Skip track header lines and collect data as CSV
        output += [",".join(line.split()) for line in f if not line.startswith("Event")]

    # Use existing CSV load function
    return _csv.load(StringIO("\n".join(output)), index_col=False)
