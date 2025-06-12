# coding=utf8
"""
Copyright (C) 2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import os
from pathlib import Path
from typing import Mapping

from pyproj import CRS
import xarray as xr

# import numpy as np
import grass_session  # noqa: F401
import grass.script as gs

from xarray_grass.grass_interface import GrassInterface
from xarray_grass.xarray_grass import dir_is_grass_mapset, dir_is_grass_project
# from xarray_grass.coord_utils import get_region_from_xarray

# Default dimension names
default_dims = {
    "start_time": "start_time",
    "end_time": "end_time",
    "x": "x",
    "y": "y",
    "latitude": "latitude",
    "longitude": "longitude",
    "x_3d": "x_3d",
    "y_3d": "y_3d",
    "latitude_3d": "latitude_3d",
    "longitude_3d": "longitude_3d",
    "z": "z",
}


def to_grass(
    dataset: xr.Dataset | xr.DataArray,
    mapset: str | Path,
    dims: Mapping[str, str] = None,
    create: bool = True,
) -> None:
    """Convert an xarray.Dataset or xarray.DataArray to GRASS GIS maps.

    This function handles the setup of the GRASS environment and session
    management. It can create a new mapset if specified and not already
    existing. It then calls the appropriate internal functions to perform
    the conversion of the xarray object's data variables into GRASS raster,
    raster 3D, STRDS, or STR3DS object.


    Parameters
    ----------
    dataset : xr.Dataset | xr.DataArray
        The xarray object to convert. If a Dataset, each data variable
        will be converted.
    mapset : str | Path
        Path to the target GRASS mapset.
    dims : Mapping[str, str], optional
        A mapping from standard dimension names
        ('start_time', 'end_time', 'x', 'y', 'latitude', 'longitude',
        'x_3d', 'y_3d', 'latitude_3d', 'longitude_3d', 'z',)
        to the actual dimension names in the dataset. For example, if your 3D dataset
        longitude coordinate is named 'lon', you would pass `dims={'longitude_3d': 'lon'}`.
        Defaults to None, which implies standard dimension names are used.
    create : bool, optional
        If True (default), the mapset will be created if it does not exist.
        The parent directory of the mapset path must be a valid GRASS project
        (location).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the provided `mapset` path is invalid, not a GRASS mapset,
        or if its parent directory is not a valid GRASS project when
        `create` is False or the mapset doesn't exist.
        If the target mapset is not accessible from an existing GRASS session.
    """
    mapset_path = Path(mapset)
    mapset = mapset_path.stem
    project_name = mapset_path.parent.stem
    project_path = mapset_path.parent
    gisdb = project_path.parent
    if mapset_path.is_dir() and not dir_is_grass_mapset(mapset_path):
        raise ValueError(f"{mapset} is not a valid GRASS mapset.")
    if not mapset_path.is_dir() and not dir_is_grass_project(project_path):
        raise ValueError(
            f"{mapset} not found and {project_path} is not a valid GRASS project."
        )
    # if the path is not a mapset, create one
    if not mapset_path.is_dir() and dir_is_grass_project(project_path) and create:
        gs.run_command(
            "g.mapset", mapset=mapset_path.name, project=project_name, flags="c"
        )

    # set the dimensions dict
    if dims:
        filtered_user_dims = {k: v for k, v in dims.items() if k in default_dims}
        dims = default_dims.copy().update(filtered_user_dims)
    else:
        # Use default dimension names if not provided
        dims = default_dims.copy()

    # Check if we're already in a GRASS session
    session = None
    if "GISRC" not in os.environ:
        # No existing session, create a new one
        session = grass_session.Session(
            gisdb=str(gisdb), location=str(project_name), mapset=str(mapset)
        )
        session.__enter__()
        gi = GrassInterface()

    else:
        # We're in an existing session, check if it matches the requested path
        gi = GrassInterface()
        gisenv = gi.get_gisenv()
        current_gisdb = gisenv["GISDBASE"]
        current_location = gisenv["LOCATION_NAME"]
        accessible_mapsets = gi.get_accessible_mapsets()

        requested_path = Path(gisdb) / Path(project_name)
        current_path = Path(current_gisdb) / Path(current_location)

        if requested_path != current_path or str(mapset) not in accessible_mapsets:
            raise ValueError(
                f"Cannot access {mapset_path} "
                f"from current GRASS session in project {current_path}. "
                f"Accessible mapsets: {accessible_mapsets}."
            )
    try:
        xarray_to_grass(dataset, gi, dims)
    finally:
        if session is not None:
            session.__exit__(None, None, None)


def xarray_to_grass(
    dataset: xr.Dataset | xr.DataArray,
    gi: GrassInterface,
    dims: Mapping[str, str] = None,
) -> None:
    """Convert an xarray Dataset or DataArray to GRASS maps.
    This function validates the CRS and pass the individual DataArrays to the
    `datarray_to_grass` function"""
    grass_crs = CRS(gi.get_crs_wkt_str())
    dataset_crs = CRS(dataset.attrs["crs_wkt"])
    # TODO: reproj if not same crs
    # TODO: handle no CRS with for xy locations
    if grass_crs != dataset_crs:
        raise ValueError(
            f"CRS mismatch: GRASS project CRS is {grass_crs}, "
            f"but dataset CRS is {dataset_crs}."
        )
    try:
        for var_name, data in dataset.data_vars.items():
            datarray_to_grass(data, gi, dims)
    except AttributeError:
        datarray_to_grass(dataset, gi, dims)


def dims_ok(data: xr.DataArray, gi: GrassInterface) -> bool:
    """Check if the dimensions of the DataArray are valid for GRASS.
    TODO: finish implementation."""
    # time_dim = next([s for s in data.dims if "time" in s.lower()], None)
    x_dim = next([s for s in data.dims if s.lower().startswith("x")], None)
    y_dim = next([s for s in data.dims if s.lower().startswith("y")], None)
    # z_dim = next([s for s in data.dims if s.lower().startswith("z")], None)
    lat_dim = next([s for s in data.dims if s.lower().startswith("lat")], None)
    lon_dim = next([s for s in data.dims if s.lower().startswith("lon")], None)

    if any((x_dim, y_dim)) and any((lon_dim, lat_dim)):
        raise ValueError(
            f"DataArray {data.name} has conflicting x, y and lat, lon dimensions."
        )
    if any((x_dim, y_dim)) and not all((x_dim, y_dim)):
        raise ValueError(f"DataArray {data.name} must have both x and y dimensions")
    if any((lat_dim, lon_dim)) and not all((lat_dim, lon_dim)):
        raise ValueError(f"DataArray {data.name} must have both lat and lon dimensions")
    if not all((x_dim, y_dim)) or all((lat_dim, lon_dim)):
        raise ValueError(
            f"DataArray {data.name} must have either x and y or lat and lon dimensions."
        )


def datarray_to_grass(
    data: xr.DataArray,
    gi: GrassInterface,
    dims: Mapping[str, str] = None,
) -> None:
    """Convert an xarray DataArray to GRASS maps."""
    if len(data.dims) > 4 or len(data.dims) < 2:
        raise ValueError(
            f"Only DataArray with 2 to 4 dimensions are supported. "
            f"Found {len(data.dims)} dimension(s)."
        )

    # Determine the type of GRASS dataset based on dimensions
    actual_dims = set(data.dims)  # For efficient lookup

    # Check for 2D spatial dimensions (e.g., latitude, longitude or x, y)
    has_latlon_2d = dims["latitude"] in actual_dims and dims["longitude"] in actual_dims
    has_xy_2d = dims["x"] in actual_dims and dims["y"] in actual_dims
    is_spatial_2d = has_latlon_2d or has_xy_2d

    # Check for 3D spatial dimensions (e.g., latitude_3d, longitude_3d, z or x_3d, y_3d, z)
    has_latlon_3d = (
        dims["latitude_3d"] in actual_dims
        and dims["longitude_3d"] in actual_dims
        and dims["z"] in actual_dims
    )
    has_xy_3d = (
        dims["x_3d"] in actual_dims
        and dims["y_3d"] in actual_dims
        and dims["z"] in actual_dims
    )
    is_spatial_3d = has_latlon_3d or has_xy_3d

    # Check for time dimension
    has_time = dims["start_time"] in actual_dims
    # Note: 'end_time' is also a potential temporal dimension but GRASS STRDS/STR3DS
    # are typically defined by a start time.
    # For simplicity 'start_time' is the primary indicator here.

    # Determine dataset type based on number of dimensions and identified dimension types
    is_raster = len(data.dims) == 2 and is_spatial_2d
    is_raster_3d = len(data.dims) == 3 and is_spatial_3d
    is_strds = len(data.dims) == 3 and has_time and is_spatial_2d
    is_str3ds = len(data.dims) == 4 and has_time and is_spatial_3d

    error_msg = (
        f"DataArray {data.name} does not match any supported GRASS dataset type. "
        f"Expected 2D, 3D, STRDS, or STR3DS."
    )

    if is_raster or is_strds:
        # set region
        pass
    elif is_raster_3d or is_str3ds:
        # set 3D region
        pass
    else:
        raise ValueError(error_msg)

    if is_raster:
        # write raster map
        pass
    elif is_strds:
        # write STRDS
        pass
    elif is_raster_3d:
        # write raster 3D map
        pass
    elif is_str3ds:
        # write STR3DS
        pass
    else:
        # This should have been catch before
        raise ValueError(error_msg)
