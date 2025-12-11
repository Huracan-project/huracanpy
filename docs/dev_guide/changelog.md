# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.4.0
### Added
- `huracanpy.info.beaufort_category`
- `huracanpy.calc.corral_radius`

### Changed
- `huracanpy.calc.apex_vals` always gives the first point when there a multiple minima/maxima with the same values. Consistent with the behaviour of argmin/argmax

## v1.3.1
### Fixed
- `calc.density` with `crop=True` will only crop the region outside of where there are tracks rather than all empty longitudes/latitudes

## v1.3.0
### Added
- Include missing accessor functions
  - `get_beta_drift` and `add_beta_drift`
  - `plot_fancyline`
- `plot.tracks` and `plot.density` can be placed on existing axes if they are explicitly passed to the function
- Load tracks from the Imperial College Storm Model (IRIS)
- `huracanpy.infer_track_id` to generate a track ID from unique combinations of variables. e.g. "year" and "storm number in year".
  - Also add as an option to `huracanpy.load`
- Can use `huracanpy.load` with a list of files
- `huracanpy.concat_tracks` to concatenate tracks while keeping the track_id as a unique identifier

### Changed
- Reduced unnecessary warning messages
- Always use `var_name` as the accessor keyword for choosing a specific variable. Previously some functions used `varname`
- track density kde function integrates to the total number of points being passed, so that numbers are reasonable.

### Fixed
- Consistent use of track_id in `delta` calculations
- Correctly convert units in `category` functions
- Always do the `beta_drift` calculation with wind speed in metres per second
- Allow requesting other time components than year/month/day/hour from `add_time_components`, consistent with `info.time_components`
- `get_azimuth` was calling the wrong function
- Correctly use wind in "knots" and pressure in "hPa" when using pre-calculated pressure-wind relations ("holland", "z2021") in `tc.pace`
- Use a lower precision `datetime64` when times are outside the default representable range

## v1.2.0
### Added
- Modify `huracanpy.load` to load TRACK files with timesteps instead of dates  (e.g. `track_calendar=("1940-01-01", 6)` to specify start time and timestep length in hours)
- Add "Old HURDAT"/ECMWF track data reader
- Add function to compute azimuth
- Allow `huracanpy.sel_id` to be used to select multiple track IDs

### Changed
- Update python versions supported (python 3.9-3.13, changed from 3.8-3.12)

## v1.1.0 
### Added
- Support for WiTRACK text files in huracanpy.load
- Beta-drift computation (along with RMW support within distance function)
- Full documentation incuding new examples
- kde option for densities
- Reference track set option for matching
- Basin definitions from E. Sainsbury's papers

### Fixes
- Accessor was not behaving in the right way when the object was modified after the accessor's first call;
- Allow for ISO time to be called "isotime" in csv/parquet

## v1.0.0
### Added
- xarray Dataset accessor (`hrcn`)
- Extra keyword arguments to `huracanpy.load`
  - `rename`
  - `units`
  - `baselon`
- `sel_id` function to select a single track by track_id from a `Dataset` but faster than using `groupby`

### Changed
- Simplified module namespaces
  - `utils.{module}.{function}` -> `info.{function}`
  - `diags.{module}.{function}` -> `calc.{function}` or `tc.{function}`
  - `subset.trackswhere` -> `trackswhere`
- Simplified function naming
  - `plot_` prefix removed functions in `plot` module
- Modified arguments to `load`
  - keyword `tracker` renamed `source` to reflect not all tracks are from trackers, e.g. IBTrACS or statistical-dynamical downscaling models
  - Use `ibtracs_subset` to determine whether the subset is online or not, removing `ibtracs_online` keyword
  - When `filename` is specified for an online IBTrACS dataset, save the downloaded data to that file, rather than using `ibtracs_clean=False` and a default filename
- Use a single matching function `assess.match` for 2 or more datasets instead of `assess.match_pair` and `assess.match_multiple`
- Remove `get_` from functions in `info`. Instead this syntax is used to differentiate `get_` and `add_` functions in the `hrcn` accessor
- Split `info.get_land_or_ocean` into `is_land` and `is_ocean`
- Renamed `tc.sshs_cat` to `tc.saffir_simpson_category` and `tc.pres_cat` to `tc.pressure_category`
- Improved support for calculations with units (using metpy style functions)
- Updated IBTrACS data

### Removed
- `add_info` argument from `huracanpy.load`
- `add_all_info` function. Instead use `add_` functions on `hrcn` accessor
- `get_time`. Functionality already covered by `pandas.to_datetime`

### Fixed
- `huracanpy.info.season` now works with `cftime.datetime`
