import warnings

import pandas as pd


def parse_any_line(line):
    line_number = line[:5]
    content = line[6:]
    return line_number, content


def identify_line_type(content):
    if len(content) == 28:
        return "track_header"
    elif len(content) == 38:
        return "track_point"
    elif len(content) == 3:
        return "track_type"
    else:
        warnings.warn("Line type not recognized")
        return None


def parse_track_header(content):
    date = content[:10]
    track_length = int(content[13:15])
    track_id = int(content[-4:])
    return date, track_length, track_id


def parse_track_point(content):
    time = content[:13]
    lat = int(content[14:17]) / 10
    lon = int(content[17:21]) / 10
    wind = int(content[22:25])
    pres = int(content[26:30])
    lat_wind = int(content[31:34]) / 10
    lon_wind = int(content[34:38]) / 10
    return time, lat, lon, wind, pres, lat_wind, lon_wind


def load(filename):
    with open(filename, "rt") as f:
        # Read file
        lines = f.readlines()
        # Remove escape character
        lines = [line[:-1] if line.endswith("\n") else line for line in lines]

    # Parse through file
    c, track_length = 0, 0
    L = []
    while len(lines) > 0:
        line = lines.pop(0)
        nb, content = parse_any_line(line)
        line_type = identify_line_type(content)
        if line_type == "track_header":
            # Check that previous track was finished
            if c != track_length:
                raise ValueError("Previous track not finished")
            # Start new track
            c = 0
            date, track_length, track_id = parse_track_header(content)
        elif line_type == "track_point":
            c += 1  # Count point
            time, lat, lon, wind, pres, lat_wind, lon_wind = parse_track_point(content)
            L.append([track_id, time, lat, lon, wind, pres, lat_wind, lon_wind])
        elif line_type == "track_type":
            pass

    # Format
    df = pd.DataFrame(
        L,
        columns=[
            "track_id",
            "time",
            "lat",
            "lon",
            "wind",
            "pres",
            "lat_wind",
            "lon_wind",
        ],
    )
    df["time"] = pd.to_datetime(df.time)

    # Return as xarray
    return df.to_xarray().rename({"index": "record"}).drop_vars("record")
