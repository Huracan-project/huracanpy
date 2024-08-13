import pandas as pd
import xarray as xr


def nunique(self):
    return pd.Series(self).nunique()


xr.DataArray.nunique = nunique


def freq(self, by="season", track_id_name="track_id"):
    return xr.DataArray(self[track_id_name].nunique() / self[by].nunique())
