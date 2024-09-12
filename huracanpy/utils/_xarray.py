import xarray as xr

from huracanpy.subset import trackswhere


@xr.register_dataset_accessor("huracanpy")
class HuracanpyAccessor:
    def __init__(self, dataset):
        self._dataset = dataset

    def trackswhere(self, condition):
        return trackswhere(self._dataset, condition)
