import xarray as xr

from ._track_stats import get_track_duration
from metpy.units import units


def get_freq(track_ids):
    return xr.DataArray(track_ids.hrcn.nunique())

    # TODO: groupby & norm by for the accessor only as it is much easier to write then


def get_tc_days(time, track_ids):
    durations = get_track_duration(time, track_ids)
    durations = durations * units(durations.attrs["units"])
    durations = durations.metpy.convert_units("day")
    return durations.sum()
