import xarray as xr

from huracanpy.subset import trackswhere
from .geography import get_hemisphere


@xr.register_dataset_accessor("huracanpy")
class HuracanpyAccessor:
    def __init__(self, dataset):
        self._dataset = dataset

    def trackswhere(self, condition):
        return trackswhere(self._dataset, condition)

    def get_hemisphere(self, lat_name="lat"):
        return get_hemisphere(self._dataset[lat_name])
