import pandas as pd
import xarray as xr


def nunique(self):
    return pd.Series(self).nunique()


xr.DataArray.nunique = nunique
