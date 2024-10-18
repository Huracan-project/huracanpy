import xarray as xr
from metpy.units import units

from ._data._save import save

# from huracanpy.subset import trackswhere
from .utils.geography import (
    get_hemisphere,
    get_basin,
    get_land_or_ocean,
    get_country,
    get_continent,
)
from .utils.ace import get_ace, get_pace
from .utils.time import get_time_components, get_season
from .utils.category import get_category, get_pressure_cat, get_sshs_cat
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

    # ==== utils ====
    # ---- geography ----
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

    # ---- ACE & PACE ----
    def get_ace(
        self, wind_name="wind", threshold=34 * units("knots"), wind_units="m s-1"
    ):
        """
        Calculate Accumulated Cyclone Energy (ACE) for each point.
        """
        return get_ace(
            self._dataset[wind_name], threshold=threshold, wind_units=wind_units
        )

    def add_ace(
        self, wind_name="wind", threshold=34 * units("knots"), wind_units="m s-1"
    ):
        """
        Add ACE calculation to the dataset.
        """
        self._dataset["ace"] = self.get_ace(
            wind_name, threshold=threshold, wind_units=wind_units
        )
        return self._dataset

    def get_pace(
        self,
        pressure_name="slp",
        wind_name=None,
        model=None,
        threshold_wind=None,
        threshold_pressure=None,
        wind_units="m s-1",
        **kwargs,
    ):
        """
        Calculate Pressure-based Accumulated Cyclone Energy (PACE) for each point.
        """
        pace_values, model = get_pace(
            self._dataset[pressure_name],
            wind=self._dataset[wind_name] if wind_name else None,
            model=model,
            threshold_wind=threshold_wind,
            threshold_pressure=threshold_pressure,
            wind_units=wind_units,
            **kwargs,
        )
        return pace_values

    def add_pace(
        self,
        pressure_name="slp",
        wind_name=None,
        model=None,
        threshold_wind=None,
        threshold_pressure=None,
        wind_units="m s-1",
        **kwargs,
    ):
        """
        Add PACE calculation to the dataset.
        """
        pace_values = self.get_pace(
            pressure_name=pressure_name,
            wind_name=wind_name,
            model=model,
            threshold_wind=threshold_wind,
            threshold_pressure=threshold_pressure,
            wind_units=wind_units,
            **kwargs,
        )
        self._dataset["pace"] = pace_values
        return self._dataset

    # ---- time ----
    def get_time_components(self, time_name="time"):
        """
        Expand the time variable into year, month, day, and hour.
        """
        return get_time_components(self._dataset[time_name])

    def add_time_components(self, time_name="time"):
        """
        Add year, month, day, and hour as new variables to the dataset.
        """
        year, month, day, hour = self.get_time_components(time_name)
        self._dataset["year"] = year
        self._dataset["month"] = month
        self._dataset["day"] = day
        self._dataset["hour"] = hour
        return self._dataset

    def get_season(
        self,
        track_id_name="track_id",
        lat_name="lat",
        time_name="time",
        convention="short",
    ):
        """
        Derive the season for each track based on latitude and time.
        """
        return get_season(
            self._dataset[track_id_name],
            self._dataset[lat_name],
            self._dataset[time_name],
            convention=convention,
        )

    def add_season(
        self,
        track_id_name="track_id",
        lat_name="lat",
        time_name="time",
        convention="short",
    ):
        """
        Add the season as a new variable to the dataset.
        """
        self._dataset["season"] = self.get_season(
            track_id_name, lat_name, time_name, convention
        )
        return self._dataset

    # --- category ----
    def get_category(
        self,
        variable_name,
        bins=None,
        labels=None,
        convention=None,
        variable_units=None,
    ):
        """
        Calculate a generic category from a variable and a set of thresholds.
        """
        return get_category(
            self._dataset[variable_name],
            bins=bins,
            labels=labels,
            convention=convention,
            variable_units=variable_units,
        )

    def add_category(
        self,
        variable_name,
        new_var_name,
        bins=None,
        labels=None,
        convention=None,
        variable_units=None,
    ):
        """
        Add a generic category to the dataset as a new variable.
        """
        self._dataset[new_var_name] = self.get_category(
            variable_name,
            bins=bins,
            labels=labels,
            convention=convention,
            variable_units=variable_units,
        )
        return self._dataset

    def get_sshs_cat(
        self, wind_name="wind", convention="Saffir-Simpson", wind_units="m s-1"
    ):
        """
        Determine the Saffir-Simpson Hurricane Scale (SSHS) category.
        """
        return get_sshs_cat(
            self._dataset[wind_name],
            convention=convention,
            wind_units=wind_units,
        )

    def add_sshs_cat(
        self, wind_name="wind", convention="Saffir-Simpson", wind_units="m s-1"
    ):
        """
        Add the SSHS category to the dataset.
        """
        self._dataset["sshs_cat"] = self.get_sshs_cat(wind_name, convention, wind_units)
        return self._dataset

    def get_pressure_cat(self, slp_name="slp", convention="Klotzbach", slp_units="hPa"):
        """
        Determine the pressure category based on the selected convention.
        """
        return get_pressure_cat(
            self._dataset[slp_name],
            convention=convention,
            slp_units=slp_units,
        )

    def add_pressure_cat(self, slp_name="slp", convention="Klotzbach", slp_units="hPa"):
        """
        Add the pressure category to the dataset.
        """
        self._dataset["pressure_cat"] = self.get_pressure_cat(
            slp_name, convention, slp_units
        )
        return self._dataset


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
