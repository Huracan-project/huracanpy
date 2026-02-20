import xarray as xr
from metpy.units import units
import pandas as pd

from . import (
    plot,
    tc,
    info,
    calc,
    save,
    interp_time,
    sel_id,
    trackswhere,
)


@xr.register_dataarray_accessor("hrcn")
class HuracanPyDataArrayAccessor:
    def __init__(self, dataarray):
        self._dataarray = dataarray.copy()

    def nunique(self):
        """
        Method to count number of unique element in a DataArray

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return pd.Series(self._dataarray).nunique()


@xr.register_dataset_accessor("hrcn")
class HuracanPyDatasetAccessor:
    def __init__(self, dataset):
        self._dataset = dataset

    # %% Save
    def save(self, filename, **kwargs):
        """
        Save dataset as filename.
        The file type (NetCDF or csv supported) is detected based on filename extension.
        """

        save(self._dataset, filename, **kwargs)

    def sel_id(self, track_id, track_id_name="track_id"):
        return sel_id(self._dataset, self._dataset[track_id_name], track_id)

    def trackswhere(self, condition, track_id_name="track_id"):
        return trackswhere(self._dataset, self._dataset[track_id_name], condition)

    # %% utils
    # ---- geography
    def get_hemisphere(self, lat_name="lat"):
        return info.hemisphere(self._dataset[lat_name])

    def add_hemisphere(self, lat_name="lat"):
        self._dataset["hemisphere"] = self.get_hemisphere(lat_name=lat_name)
        return self._dataset

    def get_basin(self, lon_name="lon", lat_name="lat", convention="WMO-TC", crs=None):
        return info.basin(
            self._dataset[lon_name],
            self._dataset[lat_name],
            convention=convention,
            crs=crs,
        )

    def add_basin(self, lon_name="lon", lat_name="lat", convention="WMO-TC", crs=None):
        self._dataset["basin"] = self.get_basin(lon_name, lat_name, convention, crs)
        return self._dataset

    def get_is_land(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        return info.is_land(
            self._dataset[lon_name],
            self._dataset[lat_name],
            resolution=resolution,
            crs=crs,
        )

    def add_is_land(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        self._dataset["is_land"] = self.get_is_land(lon_name, lat_name, resolution, crs)
        return self._dataset

    def get_is_ocean(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        return info.is_ocean(
            self._dataset[lon_name],
            self._dataset[lat_name],
            resolution=resolution,
            crs=crs,
        )

    def add_is_ocean(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        self._dataset["is_ocean"] = self.get_is_ocean(
            lon_name, lat_name, resolution, crs
        )
        return self._dataset

    def get_country(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        return info.country(
            self._dataset[lon_name],
            self._dataset[lat_name],
            resolution=resolution,
            crs=crs,
        )

    def add_country(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        self._dataset["country"] = self.get_country(lon_name, lat_name, resolution, crs)
        return self._dataset

    def get_continent(self, lon_name="lon", lat_name="lat", resolution="10m", crs=None):
        return info.continent(
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

    def get_landfall_points(
        self,
        lon_name="lon",
        lat_name="lat",
        track_id_name="track_id",
        resolution="10m",
        crs=None,
    ):
        return info.landfall_points(
            self._dataset[lon_name],
            self._dataset[lat_name],
            self._dataset[track_id_name],
            resolution=resolution,
            crs=crs,
        )

    # ---- ACE & PACE
    def get_ace(
        self,
        wind_name="wind",
        sum_by=None,
        threshold=34 * units("knots"),
        wind_units="m s-1",
    ):
        """
        Calculate Accumulated Cyclone Energy (ACE) for each point.
        """
        if sum_by is not None:
            sum_by = self._dataset[sum_by]
        return tc.ace(
            self._dataset[wind_name],
            sum_by=sum_by,
            threshold=threshold,
            wind_units=wind_units,
        )

    def add_ace(
        self, wind_name="wind", threshold=34 * units("knots"), wind_units="m s-1"
    ):
        """
        Add ACE calculation to the dataset.
        """
        self._dataset["ace"] = self.get_ace(
            wind_name, sum_by=None, threshold=threshold, wind_units=wind_units
        )
        return self._dataset

    def get_pace(
        self,
        pressure_name="slp",
        wind_name=None,
        model=None,
        sum_by=None,
        threshold_wind=None,
        threshold_pressure=None,
        wind_units="m s-1",
        **kwargs,
    ):
        """
        Calculate Pressure-based Accumulated Cyclone Energy (PACE) for each point.
        """
        pace_values, model = tc.pace(
            self._dataset[pressure_name],
            wind=self._dataset[wind_name] if wind_name else None,
            model=model,
            sum_by=self._dataset[sum_by] if sum_by else None,
            threshold_wind=threshold_wind,
            threshold_pressure=threshold_pressure,
            wind_units=wind_units,
            **kwargs,
        )
        return pace_values, model

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
        pace_values, model = self.get_pace(
            pressure_name=pressure_name,
            wind_name=wind_name,
            model=model,
            sum_by=None,
            threshold_wind=threshold_wind,
            threshold_pressure=threshold_pressure,
            wind_units=wind_units,
            **kwargs,
        )
        self._dataset["pace"] = pace_values
        return self._dataset, model

    def get_pressure_wind_relation(
        self, pressure_name="slp", wind_name="wind", model=None, **kwargs
    ):
        return tc.pressure_wind_relation(
            self._dataset[pressure_name],
            self._dataset[wind_name],
            model=model,
            **kwargs,
        )

    # ---- time
    def get_timestep(self, time_name="time", track_id_name="track_id"):
        return info.timestep(self._dataset[time_name], self._dataset[track_id_name])

    def get_time_components(self, time_name="time"):
        """
        Expand the time variable into year, month, day, and hour.
        """
        return info.time_components(self._dataset[time_name])

    def add_time_components(self, time_name="time"):
        """
        Add year, month, day, and hour as new variables to the dataset.
        """
        components = self.get_time_components(time_name)

        return xr.merge([self._dataset, *components])

    def get_season(
        self,
        track_id_name="track_id",
        lat_name="lat",
        time_name="time",
        convention="tc-short",
    ):
        """
        Derive the season for each track based on latitude and time.
        """
        return info.season(
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
        convention="tc-short",
    ):
        """
        Add the season as a new variable to the dataset.
        """
        self._dataset["season"] = self.get_season(
            track_id_name, lat_name, time_name, convention
        )
        return self._dataset

    # --- utils
    def get_inferred_track_id(self, *variable_names):
        return info.inferred_track_id(*[self._dataset[var] for var in variable_names])

    def add_inferred_track_id(self, *variable_names, track_id_name="track_id"):
        self._dataset[track_id_name] = self.get_inferred_track_id(*variable_names)
        return self._dataset

    # --- category
    def get_category(
        self,
        var_name,
        bins=None,
        labels=None,
        variable_units=None,
    ):
        """
        Calculate a generic category from a variable and a set of thresholds.
        """
        return info.category(
            self._dataset[var_name],
            bins=bins,
            labels=labels,
            variable_units=variable_units,
        )

    def add_category(
        self,
        var_name,
        new_var_name=None,
        bins=None,
        labels=None,
        variable_units=None,
    ):
        """
        Add a generic category to the dataset as a new variable.
        """
        if new_var_name is None:
            new_var_name = f"category_{var_name}"
        self._dataset[new_var_name] = self.get_category(
            var_name,
            bins=bins,
            labels=labels,
            variable_units=variable_units,
        )
        return self._dataset

    def get_beaufort_category(self, wind_name="wind", wind_units="m s-1"):
        return info.beaufort_category(self._dataset[wind_name], wind_units=wind_units)

    def add_beaufort_category(self, wind_name="wind", wind_units="m s-1"):
        """
        Add the SSHS category to the dataset.
        """
        self._dataset["beaufort_category"] = self.get_beaufort_category(
            wind_name, wind_units
        )
        return self._dataset

    def get_saffir_simpson_category(self, wind_name="wind", wind_units="m s-1"):
        """
        Determine the Saffir-Simpson Hurricane Scale (SSHS) category.
        """
        return tc.saffir_simpson_category(
            self._dataset[wind_name], wind_units=wind_units
        )

    def add_saffir_simpson_category(self, wind_name="wind", wind_units="m s-1"):
        """
        Add the SSHS category to the dataset.
        """
        self._dataset["saffir_simpson_category"] = self.get_saffir_simpson_category(
            wind_name, wind_units
        )
        return self._dataset

    def get_pressure_category(
        self, slp_name="slp", convention="Klotzbach", slp_units=None
    ):
        """
        Determine the pressure category based on the selected convention.
        """
        return tc.pressure_category(
            self._dataset[slp_name], convention=convention, slp_units=slp_units
        )

    def add_pressure_category(
        self, slp_name="slp", convention="Klotzbach", slp_units=None
    ):
        """
        Add the pressure category to the dataset.
        """
        self._dataset["pressure_category"] = self.get_pressure_category(
            slp_name, convention, slp_units
        )
        return self._dataset

    def get_beta_drift(self, lat_name="lat", wind_name="wind", rmw_name="rmw"):
        return tc.beta_drift(
            self._dataset[lat_name], self._dataset[wind_name], self._dataset[rmw_name]
        )

    def add_beta_drift(self, lat_name="lat", wind_name="wind", rmw_name="rmw"):
        v_drift, theta_drift = self.get_beta_drift(lat_name, wind_name, rmw_name)

        self._dataset["v_drift"] = v_drift
        self._dataset["theta_drift"] = theta_drift
        return self._dataset

    # ---- translation
    def get_azimuth(
        self,
        lon_name="lon",
        lat_name="lat",
        track_id_name="track_id",
        ellps="WGS84",
        centering="forward",
    ):
        """
        Compute the azimuth between points along a track
        """
        if track_id_name in list(self._dataset.variables):
            return calc.azimuth(
                self._dataset[lon_name],
                self._dataset[lat_name],
                track_id=self._dataset[track_id_name],
                ellps=ellps,
                centering=centering,
            )
        if (track_id_name is None) or (
            track_id_name not in list(self._dataset.variables)
        ):
            return calc.azimuth(
                self._dataset[lon_name],
                self._dataset[lat_name],
                track_id=None,
                ellps=ellps,
                centering=centering,
            )

    def add_azimuth(
        self,
        lon_name="lon",
        lat_name="lat",
        track_id_name="track_id",
        ellps="WGS84",
        centering="forward",
    ):
        """
        Add the azimuth calculation to the dataset.
        """
        self._dataset["azimuth"] = self.get_azimuth(
            lon_name, lat_name, track_id_name, ellps, centering
        )
        return self._dataset

    def get_distance(
        self,
        lon_name="lon",
        lat_name="lat",
        track_id_name="track_id",
        method="geod",
        ellps="WGS84",
        centering="forward",
    ):
        """
        Compute the distance between points along a track.
        """
        if track_id_name in list(self._dataset.variables):
            return calc.distance(
                self._dataset[lon_name],
                self._dataset[lat_name],
                track_id=self._dataset[track_id_name],
                method=method,
                ellps=ellps,
                centering=centering,
            )
        if (track_id_name is None) or (
            track_id_name not in list(self._dataset.variables)
        ):
            return calc.distance(
                self._dataset[lon_name],
                self._dataset[lat_name],
                track_id=None,
                method=method,
                ellps=ellps,
                centering=centering,
            )

    def add_distance(
        self,
        lon_name="lon",
        lat_name="lat",
        track_id_name="track_id",
        method="geod",
        ellps="WGS84",
        centering="forward",
    ):
        """
        Add the distance calculation to the dataset.
        """
        self._dataset["distance"] = self.get_distance(
            lon_name, lat_name, track_id_name, method, ellps, centering
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
        centering="forward",
    ):
        """
        Compute the translation speed along tracks.
        """
        if track_id_name in list(self._dataset.variables):
            return calc.translation_speed(
                self._dataset[lon_name],
                self._dataset[lat_name],
                self._dataset[time_name],
                track_id=self._dataset[track_id_name],
                method=method,
                ellps=ellps,
                centering=centering,
            )
        if (track_id_name is None) or (
            track_id_name not in list(self._dataset.variables)
        ):
            return calc.translation_speed(
                self._dataset[lon_name],
                self._dataset[lat_name],
                self._dataset[time_name],
                track_id=None,
                method=method,
                ellps=ellps,
                centering=centering,
            )

    def add_translation_speed(
        self,
        lon_name="lon",
        lat_name="lat",
        time_name="time",
        track_id_name="track_id",
        method="geod",
        ellps="WGS84",
        centering="forward",
    ):
        """
        Add the translation speed calculation to the dataset.
        """
        self._dataset["translation_speed"] = self.get_translation_speed(
            lon_name,
            lat_name,
            time_name,
            track_id_name,
            method,
            ellps,
            centering,
        )
        return self._dataset

    def get_corral_radius(
        self,
        lon_name="lon",
        lat_name="lat",
        time_name="time",
        track_id_name="track_id",
        window=None,
        min_points=None,
    ):
        return calc.corral_radius(
            self._dataset[lon_name],
            self._dataset[lat_name],
            time=self._dataset[time_name],
            track_id=self._dataset[track_id_name],
            window=window,
            min_points=min_points,
        )

    def add_corral_radius(
        self,
        lon_name="lon",
        lat_name="lat",
        time_name="time",
        track_id_name="track_id",
        window=None,
        min_points=None,
    ):
        self._dataset["corral_radius"] = self.get_corral_radius(
            lon_name,
            lat_name,
            time_name=time_name,
            track_id_name=track_id_name,
            window=window,
            min_points=min_points,
        )

        return self._dataset

    # ---- rates
    def get_delta(self, var_name="wind10", track_id_name="track_id", **kwargs):
        if track_id_name in list(self._dataset.variables):
            return calc.delta(
                self._dataset[var_name],
                track_ids=self._dataset[track_id_name],
                **kwargs,
            )
        if (track_id_name is None) or (
            track_id_name not in list(self._dataset.variables)
        ):
            return calc.delta(self._dataset[var_name], track_ids=None, **kwargs)

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
            return calc.rate(
                self._dataset[var_name],
                self._dataset[time_name],
                track_ids=self._dataset[track_id_name],
                **kwargs,
            )
        if (track_id_name is None) or (
            track_id_name not in list(self._dataset.variables)
        ):
            return calc.rate(
                self._dataset[var_name],
                self._dataset[time_name],
                track_ids=None,
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
        return interp_time(
            self._dataset, self._dataset[track_id_name], freq=freq, prog_bar=prog_bar
        )

    # ---- lifecycle
    def get_time_from_genesis(self, time_name="time", track_id_name="track_id"):
        return calc.time_from_genesis(
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
        return calc.time_from_apex(
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
        d = self.get_density(lon_name=lon_name, lat_name=lat_name, **density_kws)
        return plot.density(d, **kwargs)

    def plot_fancyline(
        self,
        lon_name="lon",
        lat_name="lat",
        track_id_name="track_id",
        colors=None,
        linewidths=None,
        alphas=None,
        linestyles=None,
        **kwargs,
    ):
        extra_names = dict(
            colors=colors, linewidths=linewidths, alphas=alphas, linestyles=linestyles
        )
        output = []
        for track_id, track in self._dataset.groupby(track_id_name):
            # Allow the other variables to be passed as variable names or constant
            # strings
            # e.g. colors can be a variable on the track or could just be "red"
            extra_variables = {
                key: (track[name] if name in track else name)
                for key, name in extra_names.items()
            }

            output.append(
                plot.fancyline(
                    track[lon_name], track[lat_name], **extra_variables, **kwargs
                )
            )

        return output

    def plot_pressure_wind_relation(
        self,
        pressure_name="",
        wind_name="",
        **kwargs,
    ):
        return plot.pressure_wind_relation(
            self._dataset[pressure_name],
            self._dataset[wind_name],
            **kwargs,
        )

    # %% diags
    # ---- density
    def get_density(self, lon_name="lon", lat_name="lat", method="histogram", **kwargs):
        return calc.density(
            self._dataset[lon_name], self._dataset[lat_name], method=method, **kwargs
        )

    # ---- track stats
    def get_track_duration(self, time_name="time", track_id_name="track_id"):
        return calc.track_duration(
            self._dataset[time_name], self._dataset[track_id_name]
        )

    def get_gen_vals(self, time_name="time", track_id_name="track_id"):
        return calc.gen_vals(
            self._dataset, self._dataset[time_name], self._dataset[track_id_name]
        )

    def get_apex_vals(self, var_name, track_id_name="track_id", stat="max"):
        return calc.apex_vals(
            self._dataset,
            variable=self._dataset[var_name],
            track_id=self._dataset[track_id_name],
            stat=stat,
        )
