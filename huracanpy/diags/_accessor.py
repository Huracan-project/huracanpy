import xarray as xr

from .track_density import simple_global_histogram


@xr.register_dataset_accessor("diags")
class DiagsAccessor:
    def __init__(self, dataset):
        self._dataset = dataset

    # track density
    def simple_track_density(
        self,
        lon_name="lon",
        lat_name="lat",
        bin_size=5,
        N_seasons=1,
    ):
        return simple_global_histogram(
            self._dataset[lon_name],
            self._dataset[lat_name],
            bin_size=bin_size,
            N_seasons=N_seasons,
        )
