import xarray as xr

from ._data._save import save

# from huracanpy.subset import trackswhere
from .utils.geography import (
    get_hemisphere,
    get_basin,
    get_land_or_ocean,
    get_country,
    get_continent,
)
# from .time import get_time, get_season
# from .interp import interp_time


@xr.register_dataset_accessor("hrcn")
class HuracanPyAccessor:
    def __init__(self, dataset):
        self._dataset = dataset.copy()

    def save(self, filename):
        """
        Save dataset as filename.
        The file type (NetCDF or csv supported) is detected based on filename extension.

        Parameters
        ----------
        filename : str
            Must end in ".nc" or ".csv"

        Returns
        -------
        None.

        """

        save(self._dataset, filename)

    # ---- utils ----
    # geography
    def get_hemisphere(self, lat_name="lat"):
        return get_hemisphere(self._dataset[lat_name])

    def add_hemisphere(self, lat_name="lat"):
        self._dataset["hemisphere"] = self.get_hemisphere(lat_name=lat_name)
        return self._dataset

    def get_basin(self, lon_name="lon", lat_name="lat", convention="WMO", crs=None):
        return get_basin(
            self._dataset[lon_name],
            self._dataset[lat_name],
            convention=convention,
            crs=crs,
        )

    def add_basin(self, lon_name="lon", lat_name="lat", convention="WMO", crs=None):
        self._dataset["basin"] = self.get_basin(lon_name, lat_name, convention, crs)
        return self._dataset

    def get_land_or_ocean(
        self, lon_name="lon", lat_name="lat", resolution="10m", crs=None
    ):
        return get_land_or_ocean(
            self._dataset[lon_name],
            self._dataset[lat_name],
            resolution=resolution,
            crs=crs,
        )

    def add_land_or_ocean(
        self, lon_name="lon", lat_name="lat", resolution="10m", crs=None
    ):
        self._dataset["land_or_ocean"] = self.get_land_or_ocean(
            lon_name, lat_name, resolution, crs
        )
        return self._dataset

    def get_country(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        return get_country(
            self._dataset[lon_name],
            self._dataset[lat_name],
            resolution=resolution,
            crs=crs,
        )

    def add_country(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        self._dataset["country"] = self.get_country(lon_name, lat_name, resolution, crs)
        return self._dataset

    def get_continent(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        return get_continent(
            self._dataset[lon_name],
            self._dataset[lat_name],
            resolution=resolution,
            crs=crs,
        )

    def add_continent(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        self._dataset["continent"] = self.get_continent(
            lon_name, lat_name, resolution, crs
        )
        return self._dataset


#    # time
#    def get_season(self, track_id_name = "track_id", lat_name = "lat", time_name = "time", convention="short"):
#        return get_season(
#            self._dataset[track_id_name],
#            self._dataset[lat_name],
#            self._dataset[time_name],
#            convention = convention
#            )
#
#    # interp
#    def interp_time(self, freq="1h", track_id_name="track_id", prog_bar=False,):
#        return interp_time(self._dataset, freq, track_id_name, prog_bar)


# track density
#    def simple_track_density(
#        self,
#        lon_name="lon",
#        lat_name="lat",
#        bin_size=5,
#        N_seasons=1,
#    ):
#        return simple_global_histogram(
#            self._dataset[lon_name],
#            self._dataset[lat_name],
#            bin_size=bin_size,
#            N_seasons=N_seasons,
#        )

#    def duration(self, time_name="time", track_id_name="track_id"):
#        return duration(self._dataset[time_name], self._dataset[track_id_name])
