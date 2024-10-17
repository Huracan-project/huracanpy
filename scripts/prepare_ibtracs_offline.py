import sys
import warnings

import numpy as np

import huracanpy
from huracanpy._data import ibtracs


def prepare_offline(wmo=True, usa=True):
    ib = ibtracs.online("since1980", "tmp/ibtracs.csv")

    # Remove season with tracks that are still provisional
    first_season_provi = ib.where(
        ib.track_type == "PROVISIONAL", drop=True
    ).season.min()
    ib = ib.where(ib.season < first_season_provi, drop=True)

    # Remove spur tracks
    ib = ib.where(ib.track_type == "main", drop=True)  # 348MB

    # - WMO subset
    if wmo:
        print("... WMO ...")
        ## Select WMO variables
        ib_wmo = ib[
            ["sid", "season", "basin", "time", "lon", "lat", "wmo_wind", "wmo_pres"]
        ].rename({"sid": "track_id", "wmo_wind": "wind", "wmo_pres": "slp"})  # 19MB

        ## Select only 6-hourly time steps
        ib_wmo = ib_wmo.where(ib_wmo.time.dt.hour % 6 == 0, drop=True)  # 9MB

        ## Deal with var types to reduce size ( at the moment, reduces by 42% )
        for var in ["lat", "lon", "slp", "wind"]:
            ib_wmo[var] = ib_wmo[var].astype(np.float16)
        for var in [
            "season",
        ]:
            ib_wmo[var] = ib_wmo[var].astype(np.int16)

        ## Save WMO file
        huracanpy.save(ib_wmo, ibtracs.wmo_file)

    if usa:
        # - USA subset
        print("... USA ...")
        ## Select USA variables
        ib_usa = ib[
            [
                "sid",
                "season",
                "basin",
                "time",
                "usa_lat",
                "usa_lon",
                "usa_status",
                "usa_wind",
                "usa_pres",
                "usa_sshs",
            ]
        ].rename(
            {
                "sid": "track_id",
                "usa_lat": "lat",
                "usa_lon": "lon",
                "usa_status": "status",
                "usa_wind": "wind",
                "usa_pres": "slp",
                "usa_sshs": "sshs_cat",
            }
        )  # 23MB

        ## Select only 6-hourly time steps
        ib_usa = ib_usa.where(ib_usa.time.dt.hour % 6 == 0, drop=True)  # 11MB

        ## Remove lines with no data
        ib_usa = ib_usa.where(~np.isnan(ib_usa.lon), drop=True)

        ## Deal with var types to reduce size ( at the moment, reduces by 25% ) -> TODO : Manage wind and slp data...
        for var in ["lat", "lon", "wind", "slp"]:
            ib_usa[var] = ib_usa[var].astype(np.float16)
        for var in ["season"]:
            ib_usa[var] = ib_usa[var].astype(np.int16)
        for var in ["sshs_cat"]:
            ib_usa[var] = ib_usa[var].astype(np.int8)

        ## Save
        huracanpy.save(ib_usa, ibtracs.usa_file)

    warnings.warn(
        "If you just updated the offline files within the package, do not forget to update information in offline loader warnings"
    )


if __name__ == "__main__":
    args = [arg.lower() for arg in sys.argv[1:]]
    wmo = "wmo" in args
    usa = "usa" in args

    prepare_offline(wmo=wmo, usa=usa)
