from cartopy.crs import Geodetic
from geopandas import GeoDataFrame
import numpy as np
from shapely.geometry import Point, LineString


def to_geodataframe(lon, lat, track_id=None, *, crs=None):
    if crs is None:
        crs = Geodetic()

    xyz = Geodetic().transform_points(crs, lon, lat)

    # Geodetic transform treats -180 as different to 180 which can lead to a wrap around
    # the globe when it should be not moving. Use -180 as an arbitrary convention
    xyz[:, 0][xyz[:, 0] == 180] = -180

    # Convert tracks to a dictionary with lon, lat points as a geometry
    # Create the Shapely geometry from lon, lat points
    if track_id is None:
        tracks_dict = dict(geometry=[Point(xy) for xy in xyz[:, :2]])

    else:
        track_id = np.asarray(track_id)
        track_id, index, npoints = np.unique(
            track_id, return_index=True, return_counts=True
        )

        tracks_dict = dict(
            track_id=track_id,
            geometry=[
                LineString(xyz[idx : idx + n, :2]) if n > 1 else Point(xyz[idx, :2])
                for idx, n in zip(index, npoints)
            ],
        )

    # Return as a GeoDataFrame
    return GeoDataFrame(tracks_dict, crs=Geodetic())
