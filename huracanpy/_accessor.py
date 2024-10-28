import xarray as xr
from metpy.units import units

from ._data._save import save


from . import utils, diags, plot


@xr.register_dataset_accessor("hrcn")
class HuracanPyAccessor:
    def __init__(self, dataset):
        self._dataset = dataset.copy()

    # %% Save
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

    # %% utils
    # ---- geography
    def get_hemisphere(self, lat_name="lat"):
        return utils._geography.get_hemisphere(self._dataset[lat_name])

    def add_hemisphere(self, lat_name="lat"):
        self._dataset["hemisphere"] = self.get_hemisphere(lat_name=lat_name)
        return self._dataset

    def get_basin(self, lon_name="lon", lat_name="lat", convention="WMO", crs=None):
        return utils._geography.get_basin(
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
        return utils._geography.get_land_or_ocean(
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
        return utils._geography.get_country(
            self._dataset[lon_name],
            self._dataset[lat_name],
            resolution=resolution,
            crs=crs,
        )

    def add_country(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        self._dataset["country"] = self.get_country(lon_name, lat_name, resolution, crs)
        return self._dataset

    def get_continent(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        return utils._geography.get_continent(
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

    # ---- ACE & PACE
    def get_ace(
        self, wind_name="wind", threshold=34 * units("knots"), wind_units="m s-1"
    ):
        """
        Calculate Accumulated Cyclone Energy (ACE) for each point.
        """
        return utils._ace.get_ace(
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
        pace_values, model = utils._ace.get_pace(
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

    # ---- time
    def get_time_components(self, time_name="time"):
        """
        Expand the time variable into year, month, day, and hour.
        """
        return utils._time.get_time_components(self._dataset[time_name])

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
        return utils._time.get_season(
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

    # --- category
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
        return utils._category.get_category(
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
        return utils._category.get_sshs_cat(
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
        return utils._category.get_pressure_cat(
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

    # ---- translation
    def get_distance(
        self,
        lon_name="lon",
        lat_name="lat",
        track_id_name="track_id",
        method="geod",
        ellps="WGS84",
    ):
        """
        Compute the distance between points along a track.
        """
        if track_id_name in list(self._dataset.variables):
            return utils._translation.get_distance(
                self._dataset[lon_name],
                self._dataset[lat_name],
                track_id=self._dataset[track_id_name],
                method=method,
                ellps=ellps,
            )
        if (track_id_name is None) or (
            track_id_name not in list(self._dataset.variables)
        ):
            return utils._translation.get_distance(
                self._dataset[lon_name],
                self._dataset[lat_name],
                track_id=None,
                method=method,
                ellps=ellps,
            )

    def add_distance(
        self,
        lon_name="lon",
        lat_name="lat",
        track_id_name="track_id",
        method="geod",
        ellps="WGS84",
    ):
        """
        Add the distance calculation to the dataset.
        """
        self._dataset["distance"] = self.get_distance(
            lon_name, lat_name, track_id_name, method, ellps
        )
        return self._dataset

    def get_translation_speed(
        self,
        lon_name="lon",
        lat_name="lat",
        time_name="time",
        track_id_name="track_id",
        method="geod",
        ellps="WGS84",
    ):
        """
        Compute the translation speed along tracks.
        """
        if track_id_name in list(self._dataset.variables):
            return utils._translation.get_translation_speed(
                self._dataset[lon_name],
                self._dataset[lat_name],
                self._dataset[time_name],
                track_id=self._dataset[track_id_name],
                method=method,
                ellps=ellps,
            )
        if (track_id_name is None) or (
            track_id_name not in list(self._dataset.variables)
        ):
            return utils._translation.get_translation_speed(
                self._dataset[lon_name],
                self._dataset[lat_name],
                self._dataset[time_name],
                track_id=None,
                method=method,
                ellps=ellps,
            )

    def add_translation_speed(
        self,
        lon_name="lon",
        lat_name="lat",
        time_name="time",
        track_id_name="track_id",
        method="geod",
        ellps="WGS84",
    ):
        """
        Add the translation speed calculation to the dataset.
        """
        self._dataset["translation_speed"] = self.get_translation_speed(
            lon_name, lat_name, time_name, track_id_name, method, ellps
        )
        return self._dataset

    # ---- rates
    def get_delta(self, var_name="wind10", track_id_name="track_id", **kwargs):
        if track_id_name in list(self._dataset.variables):
            return utils.get_delta(
                self._dataset[var_name],
                track_ids=self._dataset[track_id_name],
                **kwargs,
            )
        if (track_id_name is None) or (
            track_id_name not in list(self._dataset.variables)
        ):
            return utils.get_delta(self._dataset[var_name], track_ids=None, **kwargs)

    def add_delta(self, var_name="wind10", track_id_name="track_id", **kwargs):
        """
        Add the distance calculation to the dataset.
        """
        self._dataset["delta_" + var_name] = self.get_delta(
            var_name, track_id_name, **kwargs
        )
        return self._dataset

    def get_rate(
        self, var_name="wind10", time_name="time", track_id_name="track_id", **kwargs
    ):
        if track_id_name in list(self._dataset.variables):
            return utils.get_rate(
                self._dataset[var_name],
                self._dataset[time_name],
                track_ids=self._dataset[track_id_name],
                **kwargs,
            )
        if (track_id_name is None) or (
            track_id_name not in list(self._dataset.variables)
        ):
            return utils.get_rate(
                self._dataset[var_name],
                self._dataset[time_name],
                track_id=None,
                **kwargs,
            )

    def add_rate(
        self, var_name="wind10", time_name="time", track_id_name="track_id", **kwargs
    ):
        self._dataset["rate_" + var_name] = self.get_rate(
            var_name, time_name, track_id_name, **kwargs
        )
        return self._dataset

    # ---- interp
    def interp_time(self, freq="1h", track_id_name="track_id", prog_bar=False):
        """
        Interpolate track data at a given frequency.
        """
        return utils._interp.interp_time(
            self._dataset, freq=freq, track_id_name=track_id_name, prog_bar=prog_bar
        )

    # ---- lifecycle
    def get_time_from_genesis(self, time_name="time", track_id_name="track_id"):
        return utils._lifecycle.get_time_from_genesis(
            self._dataset[time_name], self._dataset[track_id_name]
        )

    def add_time_from_genesis(self, time_name="time", track_id_name="track_id"):
        self._dataset["time_from_genesis"] = self.get_time_from_genesis(
            time_name, track_id_name
        )
        return self._dataset

    def get_time_from_apex(
        self,
        time_name="time",
        track_id_name="track_id",
        intensity_var_name="wind",
        stat="max",
    ):
        return utils._lifecycle.get_time_from_apex(
            self._dataset[time_name],
            self._dataset[track_id_name],
            self._dataset[intensity_var_name],
            stat=stat,
        )

    def add_time_from_apex(
        self,
        time_name="time",
        track_id_name="track_id",
        intensity_var_name="wind",
        stat="max",
    ):
        self._dataset["time_from_apex"] = self.get_time_from_apex(
            time_name, track_id_name, intensity_var_name, stat
        )
        return self._dataset

    # %% plot
    def plot_tracks(
        self, lon_name="lon", lat_name="lat", intensity_var_name=None, **kwargs
    ):
        if intensity_var_name in list(self._dataset.variables):
            intensity_var = self._dataset[intensity_var_name]
        else:
            intensity_var = None

        return plot.tracks(
            self._dataset[lon_name], self._dataset[lat_name], intensity_var, **kwargs
        )

    def plot_density(
        self, lon_name="lon", lat_name="lat", density_kws=dict(), **kwargs
    ):
        d = self.get_density(**density_kws)
        return plot.density(d, **kwargs)

    # %% diags
    # ---- density
    def get_density(self, lon_name="lon", lat_name="lat", method="histogram", **kwargs):
        return diags.density(
            self._dataset[lon_name], self._dataset[lat_name], method=method, **kwargs
        )

    # ---- track stats
    def get_track_duration(self, time_name="time", track_id_name="track_id"):
        return diags.get_track_duration(
            self._dataset[time_name], self._dataset[track_id_name]
        )

    def get_track_ace(self, wind_name="wind", track_id_name="track_id", **kwargs):
        return diags.get_track_ace(
            self._dataset[wind_name], self._dataset[track_id_name], **kwargs
        )

    def get_track_pace(
        self, pressure_name="slp", track_id_name="track_id", wind_name=None, **kwargs
    ):
        if wind_name is None:
            wind = None
        else:
            wind = self._dataset[wind_name]
        return diags.get_track_pace(
            self._dataset[pressure_name], self._dataset[track_id_name], wind, **kwargs
        )

    def get_gen_vals(self, time_name="time", track_id_name="track_id"):
        return diags.get_gen_vals(self._dataset, time_name, track_id_name)

    def get_apex_vals(self, varname, track_id_name="track_id", stat="max"):
        return diags.get_apex_vals(
            self._dataset, varname, track_id_name=track_id_name, stat=stat
        )

    # ---- climato
