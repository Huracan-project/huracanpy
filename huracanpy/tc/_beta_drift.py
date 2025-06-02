import numpy as np
from metpy.constants import Re, omega
from metpy.units import units
from metpy.xarray import preprocess_and_wrap

from .._metpy import dequantify_results, validate_units


omega = omega / units("radian")


@dequantify_results
@preprocess_and_wrap(wrap_like=("lat", "lat"))
def beta_drift(
    lat,
    wind_max,
    radius_wind_max,
):
    """
    Based on Smith, 1993: https://journals.ametsoc.org/view/journals/atsc/50/18/1520-0469_1993_050_3213_ahbdl_2_0_co_2.xml?tab_body=pdf

    Parameters
    ----------
    lat : TYPE
        DESCRIPTION.
    wind_max : TYPE
        DESCRIPTION.
    radius_wind_max : TYPE
        DESCRIPTION.

    Returns
    -------
    V_drift : TYPE
        DESCRIPTION.
    theta_drift : TYPE
        DESCRIPTION.

    """

    # Treat input
    # Convert lat to rad
    lat = validate_units(
        lat,
        # We assume lats are in degrees if they exceed pi
        expected_units=lambda x: "degrees" if np.abs(x).max() > np.pi else "radians",
    )
    lat = lat.to("radians")

    # Assume max wind is in m/s if not given
    wind_max = validate_units(wind_max, expected_units="m s-1")

    # Convert rmw to m
    radius_wind_max = validate_units(
        radius_wind_max,
        # We assume rmw are in km if they are below 10,000
        expected_units=lambda x: "km" if x.max() < 10000 else "m",
    )
    radius_wind_max = radius_wind_max.to("m")

    # Coriolis parameter
    beta = 2 * omega * np.cos(lat) / Re  # s-1 m-1

    # Beta-drift parameters
    V_char = (radius_wind_max**2) * beta  # m/s
    B = V_char / wind_max  # non-dimensionnal
    V_drift_adim = 0.72 * B ** (-0.54)  # non-dmensionnal
    V_drift = V_drift_adim * V_char  # m/s
    theta_drift = 308 - 9.6 * np.log(B)  # degrees

    return V_drift, theta_drift
