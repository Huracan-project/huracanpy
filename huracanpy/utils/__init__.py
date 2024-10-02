"""Huracanpy module for useful auxilliary functions"""

__all__ = [
    "get_hemisphere",
    "get_basin",
    "get_country",
    "get_continent",
    "get_land_or_ocean",
    "get_category",
    "get_sshs_cat",
    "get_pressure_cat",
    "get_time",
    "get_season",
    "interp_time",
    "get_ace",
    "get_pace",
    "get_pressure_wind_relation",
]

from .geography import (
    get_hemisphere,
    get_basin,
    get_country,
    get_continent,
    get_land_or_ocean,
)
from .category import get_category, get_sshs_cat, get_pressure_cat
from .time import get_time, get_season
from .interp import interp_time
from .ace import get_ace, get_pace, get_pressure_wind_relation


def add_all_info(
    data,
    lat_name="lat",
    lon_name="lon",
    track_id_name="track_id",
    time_name="time",
    year_name="year",
    month_name="month",
    day_name="day",
    hour_name="hour",
    wind_name="wind10",
    wind_units="m s-1",
    slp_name="slp",
    slp_units="hPa",
    sshs_convention="Saffir-Simpson",
    pres_cat_convention="Klotzbach",
    season_convention="short",
):
    """


    Parameters
    ----------
    data : the tracks xarray object
    lat_name : str, optional
        Name of the latitude variable. The default is "lat".
    lon_name : str, optional
        Name of the longitude variable. The default is "lon".
    track_id_name : str, optional
        Name of the track ID variable.. The default is "track_id".
    time_name : str, optional
        Name of the time variable. If set to None, no time-related attributes are computed. The default is "time".
    year_name : str, optional
        Name of the year variable. The default is "year".
    month_name : str, optional
        Name of the month variable. The default is "month".
    day_name : str, optional
        Name of the day variable. The default is "day".
    hour_name : str, optional
        Name of the hour variable. The default is "hour".
    wind_name : str, optional
        Name of the winf variable. The default is "wind10".
    wind_units : str, optional
        The default is "m s-1".
    slp_name : str, optional
        Name of the SLP variable. The default is "slp".
    slp_units : str, optional
        The default is "hPa".
    sshs_convention : str, optional
        The default is "Saffir-Simpson".
    pres_cat_convention : str, optional
        The default is "Klotzbach".

    Returns
    -------
    data : xr.DataSet
        Input dataset with additional info.

    """

    # Geographical
    if lat_name is not None:
        data["hemisphere"] = get_hemisphere(data[lat_name])
        if lon_name is not None:
            data["basin"] = get_basin(data[lon_name], data[lat_name])
            data["is_ocean"] = get_land_or_ocean(data[lon_name], data[lat_name])
            data["country"] = get_country(data[lon_name], data[lat_name])
            data["continent"] = get_continent(data[lon_name], data[lat_name])

    # Time
    if time_name is not None:
        if time_name not in list(data) + list(data.coords):
            data[time_name] = get_time(
                data[year_name],
                data[month_name],
                data[day_name],
                data[hour_name],
            )
        data["season"] = get_season(
            data[track_id_name],
            data[lat_name],
            data[time_name],
            convention=season_convention,
        )

    # Category
    if wind_name is not None:
        data["sshs"] = get_sshs_cat(data[wind_name], sshs_convention, wind_units)
    if slp_name is not None:
        data["pres_cat"] = get_pressure_cat(
            data[slp_name], pres_cat_convention, slp_units
        )

    return data
