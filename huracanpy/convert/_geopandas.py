from cartopy.crs import Geodetic
from geopandas import GeoDataFrame
import numpy as np
from shapely.geometry import Point, LineString


def to_geodataframe(lon, lat, track_id=None, *, crs=None):
    if crs is None:
        crs = Geodetic()

    xyz = Geodetic().transform_points(crs, lon, lat)

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
                LineString(xyz[idx : idx + n, :2]) for idx, n in zip(index, npoints)
            ],
        )

    # Return as a GeoDataFrame
    return GeoDataFrame(tracks_dict, crs=Geodetic())
