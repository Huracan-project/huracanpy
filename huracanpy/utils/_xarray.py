import xarray as xr

from huracanpy.subset import trackswhere
from .geography import (
    get_hemisphere,
    get_basin,
    get_land_or_ocean,
    get_country,
    get_continent,
)


@xr.register_dataset_accessor("huracanpy")
class HuracanpyAccessor:
    def __init__(self, dataset):
        self._dataset = dataset

    def trackswhere(self, condition):
        return trackswhere(self._dataset, condition)

    def get_hemisphere(self, lat_name="lat"):
        return get_hemisphere(self._dataset[lat_name])

    def get_basin(self, lon_name="lon", lat_name="lat", convention="WMO", crs=None):
        return get_basin(
            self._dataset[lon_name],
            self._dataset[lat_name],
            convention=convention,
            crs=crs,
        )

    def get_land_or_ocean(
        self, lon_name="lon", lat_name="lat", resolution="10m", crs=None
    ):
        return get_land_or_ocean(
            self._dataset[lon_name],
            self._dataset[lat_name],
            resolution=resolution,
            crs=crs,
        )

    def get_country(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        return get_country(
            self._dataset[lon_name],
            self._dataset[lat_name],
            resolution=resolution,
            crs=crs,
        )

    def get_continent(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        return get_continent(
            self._dataset[lon_name],
            self._dataset[lat_name],
            resolution=resolution,
            crs=crs,
        )
