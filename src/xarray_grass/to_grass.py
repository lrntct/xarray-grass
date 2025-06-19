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
import numpy as np
import pandas as pd
import grass_session  # noqa: F401

from xarray_grass.grass_interface import GrassInterface
from xarray_grass.xarray_grass import dir_is_grass_mapset, dir_is_grass_project
from xarray_grass.coord_utils import get_region_from_xarray


# Default dimension names
default_dims = {
    "start_time": "start_time",
    "end_time": "end_time",
    "x": "x",
    "y": "y",
    "x_3d": "x_3d",
    "y_3d": "y_3d",
    "z": "z",
}


def to_grass(
    dataset: xr.Dataset | xr.DataArray,
    mapset: str | Path,
    dims: Mapping[str, str] = None,
    create: bool = False,
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
        ('start_time', 'end_time', 'x', 'y', 'x_3d', 'y_3d', 'z',)
        to the actual dimension names in the dataset. For example, if your 3D dataset
        east-west coordinate is named 'lon', you would pass `dims={'x_3d': 'lon'}`.
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

    if create:
        # Not until GRASS 8.5
        raise NotImplementedError("'create' not yet available.")

    if mapset_path.is_file():
        raise ValueError(f"Mapset path '{mapset_path}' is a file, not a directory.")

    # Prioritize check for create=False with a non-existent mapset
    if not mapset_path.is_dir() and not create:
        raise ValueError(
            f"Mapset path '{mapset_path}' is not a valid directory and create is False."
        )

    if mapset_path.is_dir() and not dir_is_grass_mapset(mapset_path):
        raise ValueError(
            f"Path '{mapset_path}' exists but is not a valid GRASS mapset."
        )

    if not mapset_path.is_dir() and create and not dir_is_grass_project(project_path):
        raise ValueError(
            f"Mapset '{mapset_path}' not found and its parent directory "
            f"'{project_path}' is not a valid GRASS project."
        )

    if not mapset_path.is_dir() and dir_is_grass_project(project_path) and create:
        # gs.run_command(
        #     "g.mapset", mapset=mapset_path.name, project=project_name, flags="c"
        # )
        # Skip until grass 8.5
        pass

    # set the dimensions dict
    if dims is not None:
        if not isinstance(dims, Mapping):
            raise TypeError("dims parameter must be a mapping (e.g., a dictionary).")
        # Start with a copy of defaults, then update with valid user-provided dims
        processed_dims = default_dims.copy()
        for k, v in dims.items():
            processed_dims[k] = v
        dims = processed_dims
    else:
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
    # TODO: handle no CRS for xy locations
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


def datarray_to_grass(
    data: xr.DataArray,
    gi: GrassInterface,
    dims: Mapping[str, str] = None,
) -> None:
    """Convert an xarray DataArray to GRASS maps.

    Uses standardized (x, y) dimension naming internally. For datasets with
    latitude/longitude dimensions, provide explicit mapping via dims parameter.
    """
    if len(data.dims) > 4 or len(data.dims) < 2:
        raise ValueError(
            f"Only DataArray with 2 to 4 dimensions are supported. "
            f"Found {len(data.dims)} dimension(s)."
        )

    # Check for 2D spatial dimensions
    is_spatial_2d = dims["x"] in data.dims and dims["y"] in data.dims

    # Check for 3D spatial dimensions
    is_spatial_3d = (
        dims["x_3d"] in data.dims
        and dims["y_3d"] in data.dims
        and dims["z"] in data.dims
    )

    # Check for time dimension
    has_time = dims["start_time"] in data.dims

    # Note: 'end_time' is also a potential temporal dimension but GRASS STRDS/STR3DS
    # are typically defined by a start time.
    # For simplicity 'start_time' is the primary indicator here.

    # Determine dataset type based on number of dimensions and identified dimension types
    is_raster = len(data.dims) == 2 and is_spatial_2d
    is_raster_3d = len(data.dims) == 3 and is_spatial_3d
    is_strds = len(data.dims) == 3 and has_time and is_spatial_2d
    is_str3ds = len(data.dims) == 4 and has_time and is_spatial_3d

    # Set temp region
    current_region = gi.get_region()
    temp_region = get_region_from_xarray(data, dims)
    gi.set_region(temp_region)
    #  TODO: reshape to match user dims
    try:
        if is_raster:
            gi.write_raster_map(data, data.name)
        elif is_strds:
            write_stds(data, gi, dims)
        elif is_raster_3d:
            gi.write_raster3d_map(data, data.name)
        elif is_str3ds:
            write_stds(data, gi, dims)
        else:
            raise ValueError(
                f"DataArray '{data.name}' does not match any supported GRASS dataset type. "
                f"Expected 2D, 3D, STRDS, or STR3DS."
            )
    finally:
        # Restore the original region
        gi.set_region(current_region)


def write_stds(data: xr.DataArray, gi: GrassInterface, dims: Mapping):
    # 1. Determine the temporal coordinate and type
    time_coord = data[dims["start_time"]]
    time_dtype = time_coord.dtype
    if isinstance(time_dtype, np.dtypes.DateTime64DType):
        temporal_type = "absolute"
    elif np.issubdtype(time_dtype, np.integer):
        temporal_type = "relative"
        time_unit = time_coord.attrs.get("units", None)
    else:
        raise ValueError(f"Temporal type not supported: {time_dtype}")
    # 2. Determine the semantic type
    # TODO: find actual type
    semantic_type = "mean"
    # 2.5 determine if 2D or 3D
    is_3d = False
    stds_type = "strds"
    if len(data.isel({dims["start_time"]: 0}).dims) == 3:
        is_3d = True
        stds_type = "str3ds"

    # 3. Loop through the time dim:
    map_list = []
    for index, time in enumerate(time_coord):
        darray = data.sel({dims["start_time"]: time})
        nd_array = darray.values
        # 3.1 Write each map individually
        raster_name = f"{data.name}_{temporal_type}_{index}"
        if not is_3d:
            gi.write_raster_map(arr=nd_array, rast_name=raster_name)
        else:
            gi.write_raster3d_map(arr=nd_array, rast_name=raster_name)
        # 3.2 populate an iterable[tuple[str, datetime | timedelta]]
        time_value = time.values.item()
        if temporal_type == "absolute":
            absolute_time = pd.Timestamp(time_value)
            map_list.append((raster_name, absolute_time.to_pydatetime()))
        else:
            relative_time = pd.Timedelta(time_value, unit=time_unit)
            map_list.append((raster_name, relative_time.to_pytimedelta()))
    # 4. Create STDS and register the maps in it
    gi.register_maps_in_stds(
        stds_title="",
        stds_name=data.name,
        stds_desc="",
        map_list=map_list,
        semantic=semantic_type,
        t_type=temporal_type,
        stds_type=stds_type,
    )
